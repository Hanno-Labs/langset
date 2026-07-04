"""Trainer: fit the LLM emitter so `emit(input_text)` lands where `emit(target_text)` does — a native
self-contrastive objective (both views in the model's own space, in-batch negatives). The target text DEFINES
the geometry. Two light aux terms keep it grounded and spread; selection is collapse-aware.

Dataset rows: `input_text` (what you have at inference) + `target_text` (a description of the same item that
defines where it should land). Pass a `datasets.Dataset` or `list[dict]`; use `column_mapping` to rename.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F

from langset import selection
from langset.modeling import LangSetModel
from langset.training_args import TrainingArguments

_RECON_K = 8            # soft-prompt tokens the latent expands into for the recon decoder
_RECON_MAXLEN = 128     # target_text tokens the recon aux reconstructs
_LEARN_TGT = 160        # [LEARN] rows: max target (substance) tokens generated under next-token CE
_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4   # collapse below this isn't penalized; above it, selection is tanked


def _columns(dataset: Any) -> dict[str, list[Any]]:
    if hasattr(dataset, "column_names"):                       # datasets.Dataset
        return {c: list(dataset[c]) for c in dataset.column_names}
    rows = list(dataset)                                       # list[dict]
    return {k: [r[k] for r in rows] for k in rows[0]}


def supcon_loss(z: torch.Tensor, labels: list[str], tau: float) -> torch.Tensor:
    """Supervised-contrastive (Khosla et al.) over emitted latents: same-label items are positives (pulled together),
    all others negatives (pushed apart) — a few group labels SHAPE the geometry into SEPARATE REGIONS. Being a proper
    contrastive loss (each anchor has both positives and negatives) it separates without collapsing. Items whose label
    is ''/'unknown'/'none'/'nan' are dropped. Returns 0 if fewer than two labelled items or no positive pair exists."""
    dev = z.device
    keep = [k for k, l in enumerate(labels) if str(l).strip().lower() not in ("", "unknown", "none", "nan")]
    if len(keep) < 2:
        return z.new_zeros(())
    zz = F.normalize(z[keep], p=2, dim=-1)
    lab = [labels[k] for k in keep]
    b = len(keep)
    sim = (zz @ zz.t() / tau).masked_fill(torch.eye(b, device=dev, dtype=torch.bool), -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = torch.tensor([[1.0 if (i != j and lab[i] == lab[j]) else 0.0 for j in range(b)] for i in range(b)],
                       device=dev)
    npos = pos.sum(1)
    has = npos > 0
    if not bool(has.any()):
        return z.new_zeros(())
    return (-(pos * logp).sum(1)[has] / npos[has]).mean()


class Trainer:
    def __init__(self, model: LangSetModel, args: TrainingArguments, train_dataset: Any,
                 eval_dataset: Optional[Any] = None, column_mapping: Optional[dict[str, str]] = None,
                 on_checkpoint: Optional[Callable[[], None]] = None) -> None:
        self.model = model
        self.args = args
        # if set, the best-so-far model is written to output_dir on every improvement and this is called after
        # (e.g. modal Volume.commit) so another process can eval the live best checkpoint mid-training.
        self.on_checkpoint = on_checkpoint
        # ONE switch routes the whole trainer: a multi_latent model emits a VARIABLE-LENGTH latent set, so it reads
        # a `target_texts` (list[str] per row) column and runs the FSQ token-native loop; otherwise the single-latent
        # self-contrastive path (byte-for-byte unchanged) reads a scalar `target_text` column.
        self.multi_latent = bool(model.head.multi_latent)
        cols = _columns(train_dataset)
        inv = {v: k for k, v in (column_mapping or {}).items()}   # user-col -> canonical
        get = lambda canon: cols[inv.get(canon, canon)]           # type: ignore[index]  # noqa: E731
        self.input_text = [str(x) for x in get("input_text")]
        if self.multi_latent:
            raw_tt = get("target_texts")                          # per row: a non-empty list of target descriptions
            self.target_texts: list[list[str]] = []
            for i, v in enumerate(raw_tt):
                if not isinstance(v, (list, tuple)) or len(v) == 0:
                    raise ValueError(
                        f"multi_latent Trainer needs a 'target_texts' column of non-empty lists; row {i} = {v!r}")
                self.target_texts.append([str(x) for x in v])
            # optional MULTI-latent hard negatives: a per-row LIST of texts the emitted latents must be pushed AWAY
            # from (batch-pooled InfoNCE bank, weight lam_hard_neg). Empty/None rows contribute no negatives.
            self.hard_neg_texts: Optional[list[list[str]]] = None
            if getattr(args, "hard_neg_field", None) is not None:
                raw_hn = cols[inv.get(args.hard_neg_field, args.hard_neg_field)]
                self.hard_neg_texts = [
                    [str(x) for x in (v if isinstance(v, (list, tuple)) else ([v] if v not in (None, "") else []))]
                    for v in raw_hn
                ]
            # optional MULTI-latent supervised-contrastive: a per-row LIST of group labels aligned 1:1 with
            # target_texts (each emitted item's stage/group). Shapes emissions into separate regions (weight lam_sup).
            self.sup_labels: Optional[list[list[str]]] = None
            if getattr(args, "sup_field", None) is not None:
                raw_sup = cols[inv.get(args.sup_field, args.sup_field)]
                self.sup_labels = [
                    [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])] for v in raw_sup
                ]
            if args.verbose:
                hn = "" if not self.hard_neg_texts else " (+hard-neg)"
                sp = "" if not self.sup_labels else " (+supcon)"
                print(f"[langset] {len(self.input_text)} rows (multi-latent){hn}{sp}", flush=True)
            return
        self.target_text = [str(x) for x in get("target_text")]
        # optional false-negative masking: per-row set of facet keys; in-batch pairs sharing any key are masked.
        self.mask_keys: Optional[list[frozenset[str]]] = None
        if args.mask_field is not None:
            raw = cols[inv.get(args.mask_field, args.mask_field)]
            self.mask_keys = [
                frozenset(v if isinstance(v, (list, tuple, set)) else [v]) if v not in (None, "") else frozenset()
                for v in raw
            ]
        # optional hard negatives: a mined near-miss target per row (encoded as an extra negative each step).
        self.hard_neg_text: Optional[list[str]] = None
        if getattr(args, "hard_neg_field", None) is not None:
            raw = cols[inv.get(args.hard_neg_field, args.hard_neg_field)]
            self.hard_neg_text = [str(v) if v not in (None, "") else "" for v in raw]
        # optional knowledge-injection: rows tagged "learn" train next-token CE (input_text -> target_text) instead
        # of contrastive; they're pulled OUT of the contrastive split and fed as a separate learn pool.
        self.is_learn: list[bool] = [False] * len(self.input_text)
        if getattr(args, "learn_field", None) is not None and args.learn_ratio > 0:
            raw = cols[inv.get(args.learn_field, args.learn_field)]
            self.is_learn = [str(v).lower() == "learn" for v in raw]
        n_learn = sum(self.is_learn)
        if args.verbose:
            masked = "" if self.mask_keys is None else " (+false-neg mask)"
            hn = "" if self.hard_neg_text is None else " (+hard-neg)"
            lr = "" if n_learn == 0 else f" (+{n_learn} learn @ratio {args.learn_ratio})"
            print(f"[langset] {len(self.input_text)} rows{masked}{hn}{lr}", flush=True)

    def train(self) -> LangSetModel:
        if self.multi_latent:
            return self._train_multi()
        a, m = self.args, self.model
        dev = m.device
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        tok = m.tokenizer

        def tok_to(texts: list[str], mx: int) -> tuple[torch.Tensor, torch.Tensor]:
            e = tok(texts, padding=True, truncation=True, max_length=mx, return_tensors="pt")
            return e["input_ids"].to(dev), e["attention_mask"].to(dev)

        ids, mask = tok_to(self.input_text, a.max_len)            # input view
        t2_ids, t2_mask = tok_to(self.target_text, a.max_len)     # target view (self-contrastive target)
        tr_ids, tr_mask = tok_to(self.target_text, _RECON_MAXLEN)  # target tokens for the recon aux
        hn_ids = hn_mask = None
        if self.hard_neg_text is not None:                        # hard-neg view (empty "" rows tokenize fine, masked below)
            hn_ids, hn_mask = tok_to([t or " " for t in self.hard_neg_text], a.max_len)

        # knowledge-injection: learn rows go to a SEPARATE next-token-CE pool; the contrastive split is embed-only.
        learn_pool = [i for i in range(len(self.input_text)) if self.is_learn[i]]
        ln_doc_ids = ln_doc_mask = ln_tgt_ids = ln_tgt_mask = None
        if learn_pool:
            ln_doc_ids, ln_doc_mask = tok_to([self.input_text[i] for i in learn_pool], a.max_len)      # instruction+case
            ln_tgt_ids, ln_tgt_mask = tok_to([self.target_text[i] for i in learn_pool], _LEARN_TGT)    # german substance

        n = len(self.input_text)
        embed_all = np.array([i for i in range(n) if not self.is_learn[i]])
        embed_perm = embed_all[rng.permutation(len(embed_all))]
        n_val = max(4, int(len(embed_perm) * a.val_frac))
        val_idx, tr_idx = embed_perm[:n_val], embed_perm[n_val:]

        # recon aux: latent -> K soft tokens -> backbone decodes target_text (token CE). Grounds the latent in the
        # text. `connector` is TRAINING-ONLY scaffolding (not saved; inference just emits the latent).
        hsz, vsz = m.h, m.vocab_size
        connector = torch.nn.Linear(m.latent_dim, _RECON_K * hsz).to(dev)

        def recon_loss(latent: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            ti, tm = tr_ids[rows], tr_mask[rows]
            temb = m.embed(ti)
            soft = connector(latent).view(latent.size(0), _RECON_K, hsz).to(temb.dtype)
            seq = torch.cat([soft, temb], dim=1)
            am = torch.cat([torch.ones(latent.size(0), _RECON_K, device=dev, dtype=tm.dtype), tm], dim=1)
            out = m._run_backbone(seq, am, ti, _RECON_K)     # soft tokens synthetic; real target tokens at [K:]
            sl = slice(_RECON_K - 1, _RECON_K - 1 + ti.size(1))
            lg = getattr(out, "logits", None)
            if lg is not None:                          # model exposes an lm_head
                pred_lg = lg[:, sl, :].float()
            else:                                       # text tower (no lm_head): project only the recon positions
                hid = m._last_hidden(out)[:, sl, :]     # via the tied input embedding -> avoids full-seq 262k OOM
                pred_lg = F.linear(hid.float(), m.embed.weight.float())
            return F.cross_entropy(pred_lg.reshape(-1, vsz), ti.masked_fill(tm == 0, -100).reshape(-1),
                                   ignore_index=-100)

        def learn_loss(pos: torch.Tensor) -> torch.Tensor:
            # [LEARN] rows: teacher-forced causal LM. Condition on the case (instruction+case), CE ONLY on the German
            # substance tokens -> forces the backbone's hidden states to REPRESENT the substance (builds the axis the
            # probe found missing). Projection via the tied embedding on the target span only (no lm_head, no OOM).
            di, dm = ln_doc_ids[pos], ln_doc_mask[pos]                       # type: ignore[index]
            ti, tm = ln_tgt_ids[pos], ln_tgt_mask[pos]                       # type: ignore[index]
            seq = torch.cat([di, ti], dim=1)
            am = torch.cat([dm, tm], dim=1)
            hid = m._last_hidden(m._run_backbone(m.embed(seq), am, seq, 0))  # all real tokens -> real_start=0
            sd = di.size(1)
            ph = hid[:, sd - 1: sd - 1 + ti.size(1), :]                     # hidden that predicts each target token
            lg = F.linear(ph.float(), m.embed.weight.float())               # [B, St, vocab]
            return F.cross_entropy(lg.reshape(-1, vsz), ti.masked_fill(tm == 0, -100).reshape(-1),
                                   ignore_index=-100)

        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad] + list(connector.parameters()),
                                lr=a.lr)
        run = None
        if a.report_to == "wandb":
            import wandb  # type: ignore[import-untyped]
            run = wandb.init(project=a.wandb_project, config=vars(a))

        best_score, best_state, no_improve = -1e9, None, 0
        for ep in range(a.epochs):
            m.train()
            order = tr_idx[rng.permutation(len(tr_idx))]
            tot = nb = 0.0
            for i in range(0, len(order), a.batch_size):
                if learn_pool and rng.random() < a.learn_ratio:   # KNOWLEDGE step: teach substance before the retrieval step
                    lp = torch.tensor(rng.choice(len(learn_pool), size=min(a.batch_size, len(learn_pool)),
                                                 replace=False), device=dev)
                    lloss = learn_loss(lp)
                    opt.zero_grad(); lloss.backward(); opt.step()
                idx = torch.tensor(order[i:i + a.batch_size], device=dev)
                pred = m(ids[idx], mask[idx])
                target = m(t2_ids[idx], t2_mask[idx])            # self-contrastive: emit(target_text), same space
                tmat = target
                if hn_ids is not None:                          # HARD NEGATIVES: mined near-miss targets as extra
                    with torch.no_grad():                       # negative columns. no_grad -> memory-safe (no 4th backward);
                        hn = m(hn_ids[idx], hn_mask[idx])        # gradient still flows to `pred`, pushing it off the hard negs.
                    tmat = torch.cat([target, hn], dim=0)        # [2B, d]: cols [0:B]=in-batch targets, [B:2B]=hard negs
                logits = (pred @ tmat.t()) / a.tau              # in-batch negatives force separation (no collapse)
                B = len(idx)
                neg_mask = torch.zeros(B, logits.size(1), dtype=torch.bool, device=dev)   # NB: not `mask` (that's the attn mask)
                if self.mask_keys is not None:                  # in-batch block: drop same-issue false negatives
                    bkeys = [self.mask_keys[j] for j in idx.tolist()]
                    for r in range(B):
                        kr = bkeys[r]
                        if not kr:
                            continue
                        for c in range(B):
                            if r != c and (kr & bkeys[c]):
                                neg_mask[r, c] = True
                if hn_ids is not None:
                    # PER-ANCHOR-ONLY hard neg: anchor i sees ONLY its own mined hard neg (col B+i), never the other
                    # anchors' (a batch-SHARED negative cluster herds every emit off one region -> geometry collapse,
                    # observed: collapse 0.03 -> 0.28/0.57). Keep the valid diagonal of the hard-neg block; mask rest.
                    valid = [bool(self.hard_neg_text[j]) for j in idx.tolist()]
                    for r in range(B):
                        for c in range(B):
                            if not (r == c and valid[c]):
                                neg_mask[r, B + c] = True
                if bool(neg_mask.any()):
                    logits = logits.masked_fill(neg_mask, float("-inf"))   # diagonal (positive) always kept
                loss = F.cross_entropy(logits, torch.arange(B, device=dev))                 # primary
                loss = loss + a.lam_recon * recon_loss(pred, idx)                            # aux: grounding
                if a.lam_uniform > 0 and len(idx) > 1:                                       # aux: uniformity
                    sq = torch.pdist(F.normalize(pred, p=2, dim=-1), p=2).pow(2)
                    loss = loss + a.lam_uniform * sq.mul(-2.0).exp().mean().log()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1

            if ep % a.eval_every:
                continue
            # validate in the CURRENT geometry: input-view vs target-view retrieval + collapse + held-out recon.
            emit_in = np.asarray(m.encode([self.input_text[j] for j in val_idx], normalize_embeddings=True))
            emit_tg = np.asarray(m.encode([self.target_text[j] for j in val_idx], normalize_embeddings=True))
            mrr = selection.retrieval_mrr(emit_in, emit_tg)["mrr"]
            collapse = selection.collapse_score(emit_in)
            with torch.no_grad():
                rv, tot_v = 0.0, 0
                for s in range(0, len(val_idx), a.batch_size):
                    vb = torch.tensor(val_idx[s:s + a.batch_size], device=dev)
                    rv += float(recon_loss(m(ids[vb], mask[vb]), vb)) * len(vb); tot_v += len(vb)
                recon_val = rv / tot_v
            # recon_val is teacher-forced -> blind to collapse; hard-penalize high collapse so a collapsed epoch
            # can never win.
            sel_score = -recon_val - _COLLAPSE_PENALTY * max(0.0, collapse - _COLLAPSE_FLOOR)
            if a.verbose:
                print(f"ep{ep:02d} loss={tot/nb:.3f} mrr={mrr:.3f} collapse={collapse:.3f} "
                      f"recon_val={recon_val:.3f} sel={sel_score:.3f}", flush=True)
            if run is not None:
                run.log({"loss": tot / nb, "mrr": mrr, "collapse": collapse, "recon_val": recon_val,
                         "sel_score": sel_score, "epoch": ep})

            if sel_score > best_score:
                best_score = sel_score
                best_state = {"head": {k: v.detach().cpu().clone() for k, v in m.head.state_dict().items()},
                              "lora": {k: v.detach().cpu().clone()
                                       for k, v in m.backbone.state_dict().items() if "lora" in k}}
                no_improve = 0
                if self.on_checkpoint is not None:            # persist best-so-far + notify (live checkpoint)
                    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
                    m.save_pretrained(a.output_dir)
                    self.on_checkpoint()
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    if a.verbose:
                        print(f"[langset] early stop at ep{ep} (best {best_score:.3f})", flush=True)
                    break

        if best_state is not None:                            # restore best
            m.head.load_state_dict(best_state["head"])
            m.backbone.load_state_dict(best_state["lora"], strict=False)
        m.eval()
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(a.output_dir)
        if run is not None:
            run.finish()
        if a.verbose:
            print(f"[langset] done. best={best_score:.3f} -> {a.output_dir}", flush=True)
        return m

    def _train_multi(self) -> LangSetModel:
        """Multi-latent (variable-length FSQ latent-set) training. `input_text` seeds an autoregressive emission of a
        latent per `target_texts` item, terminated by a learned STOP. Each target latent is supplied by a stop-grad
        EMA twin (BYOL/JEPA) — MANDATORY here: with the online model emitting its own target both sides move and the
        geometry collapses. Objective = per-dim FSQ digit CE + a folded-in STOP CE + a cosine reconstruction; one
        loss, no lam_* knobs. Selection = retrieval MRR against the row's own targets, with a non-collapse diversity
        count logged as the anti-collapse guard. (Generalized from the validated regulatory seed-token loop.)"""
        a, m = self.args, self.model
        dev = m.device
        tok = m.tokenizer
        head = m.head
        d = int(m.latent_dim)
        fsq_dim = int(head.fsq_dim)
        fsq_levels = int(head.fsq_levels)
        stop_idx = fsq_levels                                     # STOP is the extra class folded into dim-0's softmax
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)

        seeds = self.input_text
        futs = [lst[:a.max_target_items] for lst in self.target_texts]   # cap targets per row
        n = len(seeds)
        perm = rng.permutation(n)
        cut = max(1, int(n * (1 - a.val_frac)))
        tr_idx = perm[:cut].tolist()
        val_idx = perm[cut:].tolist() or perm[:1].tolist()        # never-empty val (a tiny smoke can fill train)

        # EMA target twin (stop-grad): supplies the target latents so both sides don't move together and collapse.
        ema_model: Any = copy.deepcopy(m)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        ema_o = [po for po in m.parameters() if po.requires_grad]
        ema_e = [pe for pe, po in zip(ema_model.parameters(), m.parameters()) if po.requires_grad]

        def ema_update() -> None:
            with torch.no_grad():
                torch._foreach_mul_(ema_e, a.ema_m)
                torch._foreach_add_(ema_e, ema_o, alpha=1.0 - a.ema_m)

        def emit_texts(texts: list[str], mdl: Any) -> torch.Tensor:
            """Single-latent emission of each text -> [N, d] on device (normalized), no_grad — the target latents."""
            e = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            with torch.no_grad():
                z = mdl(e["input_ids"], e["attention_mask"])
            return F.normalize(z.float(), dim=-1)

        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr)
        run = None
        if a.report_to == "wandb":
            import wandb  # type: ignore[import-untyped]
            run = wandb.init(project=a.wandb_project, config=vars(a))

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            """Free-roll each val seed -> emitted latents; decode each by nearest-neighbor against an EMA-emitted bank
            of the val `target_texts`. Reports (a) retrieval MRR vs the chain's OWN targets and (b) a NON-COLLAPSE
            diversity count = distinct nearest-bank items produced (FSQ must not mean-collapse to one mode)."""
            m.eval()
            bank_texts: list[str] = []
            bank_chain: list[int] = []
            for ci in val_idx:
                for t in futs[ci]:
                    bank_texts.append(t)
                    bank_chain.append(ci)
            if not bank_texts:
                m.train()
                return {"retr_mrr": 0.0, "purity": 0.0, "n_distinct": 0, "avg_emitted": 0.0}
            zb = F.normalize(ema_model.emit(bank_texts).to(dev).float(), dim=-1)   # [Nbank, d] target-space bank
            chain_t = torch.tensor(bank_chain, device=dev)
            rr: list[float] = []
            produced: set[int] = set()
            n_emit = 0
            emit_vecs: list[torch.Tensor] = []                       # emitted val latents (for stage kNN-purity)
            emit_labs: list[str] = []                                # position-aligned sup label of each emission
            for i in range(0, len(val_idx), a.batch_size):
                chunk = val_idx[i:i + a.batch_size]
                out = m.rollout([seeds[c] for c in chunk], max_steps=a.max_steps, return_lengths=True)
                lats, lens = cast("tuple[torch.Tensor, torch.Tensor]", out)   # list input => (lat [B,Lmax,d], len [B])
                for kk, ci in enumerate(chunk):
                    own = chain_t == ci
                    for j in range(int(lens[kk])):
                        v = F.normalize(lats[kk, j].float(), dim=-1)
                        sims = zb @ v                                 # [Nbank]
                        produced.add(int(sims.argmax()))
                        n_emit += 1
                        if self.sup_labels is not None and j < len(self.sup_labels[ci]):
                            emit_vecs.append(v.detach().cpu())        # emission j <- true stage of target item j
                            emit_labs.append(self.sup_labels[ci][j])
                        if bool(own.any()):                          # MRR: rank of the best OWN-chain target
                            order = torch.argsort(sims, descending=True)
                            hit = torch.nonzero(own[order], as_tuple=False)
                            if hit.numel() > 0:
                                rr.append(1.0 / (int(hit[0].item()) + 1))
            m.train()
            purity = (selection.knn_purity(torch.stack(emit_vecs).numpy(), emit_labs)
                      if len(emit_vecs) > 6 else 0.0)                # stage-separation of the emitted geometry
            return {"retr_mrr": float(np.mean(rr)) if rr else 0.0, "purity": purity,
                    "n_distinct": len(produced), "avg_emitted": n_emit / max(len(val_idx), 1)}

        rng_t = torch.Generator().manual_seed(a.seed)
        best = -1.0
        best_state: Optional[dict[str, Any]] = None
        metrics: dict[str, float] = {}
        for ep in range(a.epochs):
            m.train()
            order = torch.randperm(len(tr_idx), generator=rng_t).tolist()
            tot = 0.0
            nb = 0
            agg = {"loss_stop": 0.0, "loss_dims": 0.0, "recon_loss": 0.0}
            for i in range(0, len(order), a.batch_size):
                bidx = [tr_idx[k] for k in order[i:i + a.batch_size]]
                se = tok([seeds[k] for k in bidx], padding=True, truncation=True, max_length=a.max_len,
                         padding_side="left", return_tensors="pt").to(dev)   # left-pad: hid[s_len-1] = last real token
                ent_lists = [list(futs[k]) for k in bidx]
                lmax = max(len(x) for x in ent_lists)
                flat_texts = [txt for lst in ent_lists for txt in lst]
                flat_tgt = emit_texts(flat_texts, ema_model)         # [ΣL, d] stop-grad EMA targets
                b = len(bidx)
                target_lat = torch.zeros(b, lmax, d, device=dev)
                valid = torch.zeros(b, lmax, dtype=torch.bool, device=dev)
                lens_l: list[int] = []
                k = 0
                for r, lst in enumerate(ent_lists):
                    nl = len(lst)
                    lens_l.append(nl)
                    target_lat[r, :nl] = flat_tgt[k:k + nl]
                    valid[r, :nl] = True
                    k += nl
                # token-native FSQ: predict each item's per-dim digits, then a STOP folded into dim-0's softmax.
                dim_lg, stop_lg, digits, recon = m.rollout_train_codebook(
                    se["input_ids"], se["attention_mask"], target_lat, a.tau)
                dim0 = torch.cat([dim_lg[:, :, 0, :], stop_lg], -1)  # [b, lmax+1, L+1] — digit-0 + STOP
                lab0 = torch.full((b, lmax + 1), -100, dtype=torch.long, device=dev)
                lab_rest = torch.full((b, lmax, fsq_dim - 1), -100, dtype=torch.long, device=dev)
                for r, nl in enumerate(lens_l):
                    lab0[r, :nl] = digits[r, :nl, 0]
                    lab0[r, nl] = stop_idx                           # emit digit-0 per item, then STOP after the last
                    lab_rest[r, :nl] = digits[r, :nl, 1:]
                loss_stop = F.cross_entropy(dim0.reshape(-1, fsq_levels + 1), lab0.reshape(-1), ignore_index=-100)
                loss_dims = F.cross_entropy(dim_lg[:, :lmax, 1:, :].reshape(-1, fsq_levels),
                                            lab_rest.reshape(-1), ignore_index=-100)
                recon_loss = (1.0 - F.cosine_similarity(recon[valid], target_lat[valid], dim=-1)).mean()
                loss = loss_stop + loss_dims + recon_loss           # base objective (+ optional hard-neg InfoNCE)
                if self.hard_neg_texts is not None and a.lam_hard_neg > 0:
                    hn_flat = [t for k in bidx for t in self.hard_neg_texts[k]]
                    if hn_flat:                                     # each emitted recon: own EMA target vs a shared bank
                        hn_bank = emit_texts(hn_flat, ema_model)    # [Nhn, d] stop-grad normalized hard-neg latents
                        rv = F.normalize(recon[valid], dim=-1)      # [Nvalid, d] emitted reconstructions
                        pos = (rv * target_lat[valid]).sum(-1, keepdim=True)   # [Nvalid, 1] cos to own target
                        neg = rv @ hn_bank.t()                      # [Nvalid, Nhn] cos to every hard neg
                        logits_hn = torch.cat([pos, neg], dim=1) / a.tau
                        loss_hn = F.cross_entropy(
                            logits_hn, torch.zeros(logits_hn.size(0), dtype=torch.long, device=dev))
                        loss = loss + a.lam_hard_neg * loss_hn
                        agg["loss_hard_neg"] = agg.get("loss_hard_neg", 0.0) + float(loss_hn.detach())
                if self.sup_labels is not None and a.lam_sup > 0:
                    # per-item group labels flattened in the SAME row-major order as recon[valid] (row r, items 0..nl)
                    sup_flat = [(self.sup_labels[k][j] if j < len(self.sup_labels[k]) else "unknown")
                                for r, k in enumerate(bidx) for j in range(lens_l[r])]
                    loss_sup = supcon_loss(recon[valid], sup_flat, a.sup_tau)   # pull same-stage, push different-stage
                    loss = loss + a.lam_sup * loss_sup
                    agg["loss_sup"] = agg.get("loss_sup", 0.0) + float(loss_sup.detach())
                opt.zero_grad()
                loss.backward()
                opt.step()
                ema_update()                                        # EMA twin tracks the online model
                tot += float(loss.detach())
                nb += 1
                agg["loss_stop"] += float(loss_stop.detach())
                agg["loss_dims"] += float(loss_dims.detach())
                agg["recon_loss"] += float(recon_loss.detach())

            if ep % a.eval_every:
                continue
            metrics = evaluate()
            row = {"loss": tot / max(nb, 1), **{kk: vv / max(nb, 1) for kk, vv in agg.items()}}
            mrr, pur = metrics["retr_mrr"], metrics.get("purity", 0.0)
            sel = pur if a.select == "purity" else (mrr + pur) if a.select == "blend" else mrr
            if a.verbose:
                hn_s = f" hn={row['loss_hard_neg']:.3f}" if "loss_hard_neg" in row else ""
                sup_s = f" sup={row['loss_sup']:.3f}" if "loss_sup" in row else ""
                pur_s = f" purity={pur:.3f}" if self.sup_labels is not None else ""
                print(f"ep{ep:02d} loss={row['loss']:.3f} stop={row['loss_stop']:.3f} dims={row['loss_dims']:.3f} "
                      f"recon={row['recon_loss']:.3f}{hn_s}{sup_s} | retr_mrr={mrr:.3f}{pur_s} "
                      f"[sel:{a.select}={sel:.3f}] distinct={metrics['n_distinct']} "
                      f"avg_emit={metrics['avg_emitted']:.2f}", flush=True)
            if run is not None:
                run.log({**row, "epoch": ep, "eval/retr_mrr": mrr, "eval/purity": pur, "eval/sel": sel,
                         "eval/n_distinct": metrics["n_distinct"], "eval/avg_emitted": metrics["avg_emitted"]})
            if sel > best:
                best = sel
                best_state = {"head": {kk: vv.detach().cpu().clone() for kk, vv in m.head.state_dict().items()},
                              "lora": {kk: vv.detach().cpu().clone()
                                       for kk, vv in m.backbone.state_dict().items() if "lora" in kk}}
                Path(a.output_dir).mkdir(parents=True, exist_ok=True)
                m.save_pretrained(a.output_dir)                     # persist best-so-far (live checkpoint)
                if self.on_checkpoint is not None:
                    self.on_checkpoint()
                if a.verbose:
                    print(f"        <- best {a.select}={best:.3f}, saved to {a.output_dir}", flush=True)

        if best_state is not None:                                  # restore best into memory (matches single-latent)
            m.head.load_state_dict(best_state["head"])
            m.backbone.load_state_dict(best_state["lora"], strict=False)
        m.eval()
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(a.output_dir)
        if run is not None:
            run.finish()
        if a.verbose:
            print(f"[langset] done (multi-latent). best {a.select}={best:.3f} -> {a.output_dir}", flush=True)
        return m
