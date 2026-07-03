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
_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4   # collapse below this isn't penalized; above it, selection is tanked


def _columns(dataset: Any) -> dict[str, list[Any]]:
    if hasattr(dataset, "column_names"):                       # datasets.Dataset
        return {c: list(dataset[c]) for c in dataset.column_names}
    rows = list(dataset)                                       # list[dict]
    return {k: [r[k] for r in rows] for k in rows[0]}


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
            if args.verbose:
                print(f"[langset] {len(self.input_text)} rows (multi-latent)", flush=True)
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
        if args.verbose:
            masked = "" if self.mask_keys is None else " (+false-neg mask)"
            print(f"[langset] {len(self.input_text)} rows{masked}", flush=True)

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

        n = len(self.input_text)
        perm = rng.permutation(n)
        n_val = max(4, int(n * a.val_frac))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

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
                idx = torch.tensor(order[i:i + a.batch_size], device=dev)
                pred = m(ids[idx], mask[idx])
                target = m(t2_ids[idx], t2_mask[idx])            # self-contrastive: emit(target_text), same space
                logits = (pred @ target.t()) / a.tau            # in-batch negatives force separation (no collapse)
                if self.mask_keys is not None:                  # drop false negatives: same-issue pairs aren't negatives
                    bkeys = [self.mask_keys[j] for j in idx.tolist()]
                    fn = torch.zeros(len(idx), len(idx), dtype=torch.bool, device=dev)
                    for r in range(len(idx)):
                        kr = bkeys[r]
                        if not kr:
                            continue
                        for c in range(len(idx)):
                            if r != c and (kr & bkeys[c]):
                                fn[r, c] = True
                    logits = logits.masked_fill(fn, float("-inf"))   # diagonal (positive) always kept
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))          # primary
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
                return {"retr_mrr": 0.0, "n_distinct": 0, "avg_emitted": 0.0}
            zb = F.normalize(ema_model.emit(bank_texts).to(dev).float(), dim=-1)   # [Nbank, d] target-space bank
            chain_t = torch.tensor(bank_chain, device=dev)
            rr: list[float] = []
            produced: set[int] = set()
            n_emit = 0
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
                        if bool(own.any()):                          # MRR: rank of the best OWN-chain target
                            order = torch.argsort(sims, descending=True)
                            hit = torch.nonzero(own[order], as_tuple=False)
                            if hit.numel() > 0:
                                rr.append(1.0 / (int(hit[0].item()) + 1))
            m.train()
            return {"retr_mrr": float(np.mean(rr)) if rr else 0.0,
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
                loss = loss_stop + loss_dims + recon_loss           # one objective — no lam_* knobs
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
            sel = metrics["retr_mrr"]
            if a.verbose:
                print(f"ep{ep:02d} loss={row['loss']:.3f} stop={row['loss_stop']:.3f} dims={row['loss_dims']:.3f} "
                      f"recon={row['recon_loss']:.3f} | retr_mrr={sel:.3f} distinct={metrics['n_distinct']} "
                      f"avg_emit={metrics['avg_emitted']:.2f}", flush=True)
            if run is not None:
                run.log({**row, "epoch": ep, "eval/retr_mrr": sel,
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
                    print(f"        <- best retr_mrr={best:.3f}, saved to {a.output_dir}", flush=True)

        if best_state is not None:                                  # restore best into memory (matches single-latent)
            m.head.load_state_dict(best_state["head"])
            m.backbone.load_state_dict(best_state["lora"], strict=False)
        m.eval()
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(a.output_dir)
        if run is not None:
            run.finish()
        if a.verbose:
            print(f"[langset] done (multi-latent). best retr_mrr={best:.3f} -> {a.output_dir}", flush=True)
        return m
