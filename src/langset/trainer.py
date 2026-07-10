"""Trainer: fit the LLM emitter so `emit(input_text)` lands where `emit(target_text)` does — a native
self-contrastive objective (both views in the model's own space, in-batch negatives). The target text DEFINES
the geometry. Two light aux terms keep it grounded and spread; selection is collapse-aware.

Dataset rows: `input_text` (what you have at inference) + `target_text` (a description of the same item that
defines where it should land). Pass a `datasets.Dataset` or `list[dict]`; use `column_mapping` to rename.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F

from langset import selection
from langset.modeling import LangSetModel
from langset.strategies import MultiStepCtx     # the aux-term step context; trainer builds it each step. The concrete
#   strategies (FSQObjective/EMATwinTarget/...) are injected via TrainingArguments and read off `a`, not imported here.
from langset.training_args import TrainingArguments

if TYPE_CHECKING:                              # only for local annotations of the injected strategy instances
    from langset.strategies import _EmissionObjective, _TargetSource

_RECON_K = 8            # soft-prompt tokens the latent expands into for the recon decoder
_RECON_MAXLEN = 128     # target_text tokens the recon aux reconstructs
_LEARN_TGT = 160        # [LEARN] rows: max target (substance) tokens generated under next-token CE
_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4   # collapse below this isn't penalized; above it, selection is tanked


def _wandb_config(a: TrainingArguments) -> dict[str, Any]:
    """`vars(a)` for wandb, with the injected strategy fields (classes/callables — not JSON-serializable) rendered
    as their names, so the config still LOGS which strategy each run used (e.g. target_source="SIGRegTarget")."""
    cfg = dict(vars(a))
    for k, v in cfg.items():
        if callable(v):
            cfg[k] = getattr(v, "__name__", repr(v))
    return cfg


def _columns(dataset: Any) -> dict[str, list[Any]]:
    if hasattr(dataset, "column_names"):                       # datasets.Dataset
        return {c: list(dataset[c]) for c in dataset.column_names}
    rows = list(dataset)                                       # list[dict]
    return {k: [r[k] for r in rows] for k in rows[0]}


def _fuse_views(ids_a: torch.Tensor, mask_a: torch.Tensor, ids_b: torch.Tensor, mask_b: torch.Tensor,
                pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FUSE two views for ONE forward: pad both to the batch's common real max-len, stack on the batch dim.
    RIGHT-padded -> real tokens lead; the padded columns are masked so each row's emit is IDENTICAL to a separate
    per-view forward. Halves kernel launches and grad-ckpt recomputes. Split the output back at row B."""
    L = int(max(mask_a.sum(dim=1).max().item(), mask_b.sum(dim=1).max().item()))

    def _fit(x: torch.Tensor, mk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cur = x.size(1)
        if cur == L:
            return x, mk
        if cur > L:
            return x[:, :L], mk[:, :L]
        xp = x.new_full((x.size(0), L - cur), pad_id)
        mp = mk.new_zeros((mk.size(0), L - cur))
        return torch.cat([x, xp], dim=1), torch.cat([mk, mp], dim=1)

    ia, ma = _fit(ids_a, mask_a)
    ib, mb = _fit(ids_b, mask_b)
    return torch.cat([ia, ib], dim=0), torch.cat([ma, mb], dim=0)


def _dyn_trim(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DYNAMIC PADDING: sequences are pre-tokenized/padded to the DATASET max, so short batches still forward at
    full width. Trim each batch to its own real max length (RIGHT-padded, so real tokens are the leading columns).
    The dropped columns are pure padding the attention mask already zeros -> forward output on the kept tokens is
    IDENTICAL. Big saving when doc lengths vary (legal corpora do). Guarded by test_train_identity."""
    n = int(mask.sum(dim=1).max().item())
    if n <= 0 or n >= ids.size(1):
        return ids, mask
    return ids[:, :n], mask[:, :n]


# ---- single-latent step engines -----------------------------------------------------------------
# The real axis of the single-latent path is "where do pred/target/hard-neg features come from, and how is the
# step run": the LIVE backbone (default) vs. FROZEN-POOL cached vectors (pool_mode="last" + frozen backbone). Both
# obey ONE interface; train() picks the engine ONCE, so the epoch loop and eval block have no per-feature `if`s.

class _StepEngine:
    """Strategy for producing the contrastive features of one step. `supports_recon` gates the recon aux — the
    frozen-pool engine has no backbone in the loop, so recon (which decodes via the backbone) is unavailable there."""

    supports_recon: bool = True

    def precompute(self) -> None:
        """Called ONCE before the epoch loop (frozen-pool caches all view features here; backbone impl = no-op)."""

    def featurize(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """-> (pred [B,d] grad flows, target [B,d] honoring stop_grad_target, hard_neg [Hn,d] or None)."""
        raise NotImplementedError

    def recon(self, pred: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Recon aux scalar. Only defined/called when supports_recon."""
        raise NotImplementedError

    def val_embeddings(self, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (emit_in, emit_tg) numpy for the eval MRR/retrieval block."""
        raise NotImplementedError


class BackboneStepEngine(_StepEngine):
    """DEFAULT: runs the LIVE backbone. Owns the fuse_views / stop_grad_target / _dyn_trim variations internally
    (they are "how to run the backbone", cohesive here). Byte-identical to the historical default path."""

    supports_recon = True

    def __init__(self, model: LangSetModel, args: TrainingArguments, tok: Any,
                 ids: torch.Tensor, mask: torch.Tensor, t2_ids: torch.Tensor, t2_mask: torch.Tensor,
                 hn_ids: Optional[torch.Tensor], hn_mask: Optional[torch.Tensor],
                 recon_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 input_text: list[str], target_text: list[str]) -> None:
        self.m, self.a, self.tok = model, args, tok
        self.ids, self.mask, self.t2_ids, self.t2_mask = ids, mask, t2_ids, t2_mask
        self.hn_ids, self.hn_mask = hn_ids, hn_mask
        self._recon_fn = recon_fn
        self.input_text, self.target_text = input_text, target_text

    def featurize(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m, a = self.m, self.a
        if a.fuse_views and not a.stop_grad_target:      # FUSE input+target in ONE forward (1 launch + 1 recompute);
            fi, fm = _fuse_views(self.ids[idx], self.mask[idx], self.t2_ids[idx], self.t2_mask[idx],
                                 self.tok.pad_token_id)  # math identical (padding masked); split back at row B
            both = m(fi, fm); _nb = len(idx)
            pred, target = both[:_nb], both[_nb:]
        else:
            pred = m(*_dyn_trim(self.ids[idx], self.mask[idx]))
            if a.stop_grad_target:                       # BYOL/MoCo: target anchors geometry, no backward
                with torch.no_grad():
                    target = m(*_dyn_trim(self.t2_ids[idx], self.t2_mask[idx]))
            else:
                target = m(*_dyn_trim(self.t2_ids[idx], self.t2_mask[idx]))   # self-contrastive: emit(target_text)
        hn: Optional[torch.Tensor] = None
        if self.hn_ids is not None:                      # HARD NEGATIVES: mined near-miss targets; no_grad (memory-safe,
            with torch.no_grad():                        # no 4th backward) — gradient still flows to `pred`, off the negs.
                hn = m(*_dyn_trim(self.hn_ids[idx], self.hn_mask[idx]))
        return pred, target, hn

    def recon(self, pred: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self._recon_fn(pred, idx)

    def val_embeddings(self, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emit_in = np.asarray(self.m.encode([self.input_text[j] for j in val_idx], normalize_embeddings=True))
        emit_tg = np.asarray(self.m.encode([self.target_text[j] for j in val_idx], normalize_embeddings=True))
        return emit_in, emit_tg


class FrozenPoolStepEngine(_StepEngine):
    """FROZEN-POOL: backbone frozen + pool_mode="last" -> features are STATIC. Encode every view ONCE in precompute()
    and train only the head on the cached vectors (no backbone in the step loop) => epochs run in seconds. No recon
    (it needs the frozen-out backbone), so select by retrieval MRR."""

    supports_recon = False

    def __init__(self, model: LangSetModel, args: TrainingArguments,
                 ids: torch.Tensor, mask: torch.Tensor, t2_ids: torch.Tensor, t2_mask: torch.Tensor,
                 hn_ids: Optional[torch.Tensor], hn_mask: Optional[torch.Tensor]) -> None:
        self.m, self.a = model, args
        self.ids, self.mask, self.t2_ids, self.t2_mask = ids, mask, t2_ids, t2_mask
        self.hn_ids, self.hn_mask = hn_ids, hn_mask
        self.feat_in: Optional[torch.Tensor] = None
        self.feat_tg: Optional[torch.Tensor] = None
        self.feat_hn: Optional[torch.Tensor] = None

    def precompute(self) -> None:
        import time as _time
        m, a = self.m, self.a
        enc_bs = max(8, min(2048, 1_000_000 // max(1, a.max_len)))   # no_grad+frozen -> big encode batch, LENGTH-aware to fill the

        def _pool_all(pid: torch.Tensor, pmask: torch.Tensor) -> torch.Tensor:   # GPU (train batch can stay small)
            outs = []
            with torch.no_grad():
                for s in range(0, pid.size(0), enc_bs):
                    outs.append(m._pool_hidden(*_dyn_trim(pid[s:s + enc_bs], pmask[s:s + enc_bs])).half())
            return torch.cat(outs, 0)
        m.eval(); _t = _time.time()
        self.feat_in = _pool_all(self.ids, self.mask)
        self.feat_tg = _pool_all(self.t2_ids, self.t2_mask)
        self.feat_hn = _pool_all(self.hn_ids, self.hn_mask) if self.hn_ids is not None else None
        m.train()
        if a.verbose:
            fi = self.feat_in
            print(f"[langset] CACHED {fi.size(0)} frozen features ({fi.size(1)}d, "
                  f"{fi.element_size()*fi.nelement()/1e6:.0f}MB/view) in {_time.time()-_t:.1f}s "
                  f"-> head-only training, no backbone in loop", flush=True)

    def featurize(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m, a = self.m, self.a
        assert self.feat_in is not None and self.feat_tg is not None
        pred = m.head_project(self.feat_in[idx].float())
        if a.stop_grad_target:
            with torch.no_grad():
                target = m.head_project(self.feat_tg[idx].float())
        else:
            target = m.head_project(self.feat_tg[idx].float())
        hn: Optional[torch.Tensor] = None
        if self.feat_hn is not None:
            with torch.no_grad():
                hn = m.head_project(self.feat_hn[idx].float())
        return pred, target, hn

    def val_embeddings(self, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.feat_in is not None and self.feat_tg is not None
        with torch.no_grad():
            vi = torch.tensor(val_idx, device=self.m.device)
            emit_in = self.m.head_project(self.feat_in[vi].float()).cpu().numpy()
            emit_tg = self.m.head_project(self.feat_tg[vi].float()).cpu().numpy()
        return emit_in, emit_tg


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
            # optional Exp-B CoT: a per-row reasoning STRING the model learns to GENERATE before the latents and
            # conditions the emission on (via seed_builder=cot_seed_texts + loss_terms=build_cot_loss_terms). Absent
            # column -> empty strings, so the default (non-CoT) strategies stay byte-identical: multi_seed_texts uses
            # raw seeds and CoTGenTerm — even if injected — self-skips on all-empty reasoning.
            cot_key = inv.get("cot_text", "cot_text")          # honor column_mapping (a renamed reasoning column)
            self.cot_texts: list[str] = (
                [("" if x is None else str(x)) for x in cols[cot_key]]   # None/absent -> "" (not the literal "None")
                if cot_key in cols else [""] * len(self.input_text))
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
            # optional FSQ LABEL SUBSPACE: per-item label columns -> reserved-dim codeword targets (a formal, head-free
            # label space in the emitted code). Builds a class->codeword map per facet and a flat "plan" of which
            # rest-dim column carries which facet's k-th codeword digit.
            self.label_cols: Optional[dict[str, list[list[str]]]] = None
            self.label_codewords: dict[str, dict[str, list[int]]] = {}
            self.label_plan: Optional[list[tuple[int, str, int]]] = None   # (rest_col = dim-1, field, digit_pos)
            if getattr(args, "label_dims", None):
                fsq_levels = int(model.head.fsq_levels)
                self.label_cols = {}
                plan: list[tuple[int, str, int]] = []
                for field, dims in args.label_dims.items():
                    raw = cols[inv.get(field, field)]
                    seqs = [[str(x) for x in (v if isinstance(v, (list, tuple)) else [v])] for v in raw]
                    self.label_cols[field] = seqs
                    classes = sorted({c for s in seqs for c in s
                                      if str(c).lower() not in ("", "unknown", "none", "na", "nan")})
                    m_dims = len(dims)
                    if len(classes) > fsq_levels ** m_dims:
                        raise ValueError(f"label_dims[{field}]: {len(classes)} classes > {fsq_levels}^{m_dims} "
                                         f"codewords — reserve more digits")
                    cw: dict[str, list[int]] = {}
                    for ci, c in enumerate(classes):
                        x, digs = ci, []
                        for _ in range(m_dims):
                            digs.append(x % fsq_levels); x //= fsq_levels    # little-endian base-fsq_levels codeword
                        cw[c] = digs
                    self.label_codewords[field] = cw
                    for pos, dd in enumerate(dims):
                        if int(dd) < 1:
                            raise ValueError(f"label_dims dim {dd} must be >=1 (dim 0 is STOP-coupled)")
                        plan.append((int(dd) - 1, field, pos))
                self.label_plan = plan
                if args.verbose:
                    print(f"[langset] FSQ label subspace: "
                          + "; ".join(f"{f}->{args.label_dims[f]} ({len(self.label_codewords[f])} cls)"
                                      for f in args.label_dims), flush=True)
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
            run = wandb.init(project=a.wandb_project, config=_wandb_config(a))

        best_score, best_state, no_improve = -1e9, None, 0

        # ---- preempt-resume: reload full training state from a durable checkpoint if one exists (else start fresh) ----
        start_ep = 0
        _ckpt = (Path(a.resume_dir) / "resume.pt") if a.resume_dir else None
        ck = torch.load(_ckpt, map_location="cpu") if (_ckpt is not None and _ckpt.exists()) else None
        if ck is not None and a.run_sig is not None and ck.get("run_sig") != a.run_sig:
            print(f"[langset] IGNORING {_ckpt}: run_sig mismatch (ckpt={ck.get('run_sig')!r} != this run "
                  f"{a.run_sig!r}) -> starting FRESH", flush=True)   # a DIFFERENT model/data/config can never resume us
            ck = None
        if ck is not None:
            _params = dict(m.named_parameters())
            for nm, t in ck["trainable"].items():
                if nm in _params:
                    _params[nm].data.copy_(t.to(_params[nm].device, _params[nm].dtype))
            connector.load_state_dict({k: v.to(dev) for k, v in ck["connector"].items()})
            opt.load_state_dict(ck["opt"])
            for stt in opt.state.values():                     # optimizer state tensors must live on the model's device
                for k, v in stt.items():
                    if torch.is_tensor(v):
                        stt[k] = v.to(dev)
            start_ep = int(ck["ep"]); best_score = float(ck["best_score"]); no_improve = int(ck["no_improve"])
            best_state = ck.get("best_state")
            try:                                               # rng restore is best-effort (robust, not bit-exact)
                rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"])
            except Exception:
                pass
            print(f"[langset] RESUMED from {_ckpt} -> start ep{start_ep}/{a.epochs} best={best_score:.3f}", flush=True)

        def save_resume(next_ep: int) -> None:
            """Atomically persist FULL training state (weights+opt+connector+epoch+best+rng) so a preempt/retry resumes
            from the last epoch boundary instead of ep0. tmp+rename => a mid-write preempt cannot corrupt the good file."""
            if not a.resume_dir:
                return
            d = Path(a.resume_dir); d.mkdir(parents=True, exist_ok=True)
            tmp = d / "resume.pt.tmp"
            torch.save({
                "trainable": {nm: p.detach().cpu() for nm, p in m.named_parameters() if p.requires_grad},
                "connector": {k: v.detach().cpu() for k, v in connector.state_dict().items()},
                "opt": opt.state_dict(),
                "ep": int(next_ep), "best_score": float(best_score), "no_improve": int(no_improve),
                "best_state": best_state, "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "run_sig": a.run_sig,          # identity fingerprint: resume REFUSES to load this into a different run
            }, tmp)
            tmp.replace(d / "resume.pt")
            if self.on_checkpoint is not None:
                self.on_checkpoint()                           # e.g. Volume.commit() -> durable across preempt

        # ---- pick the step engine ONCE (see the _StepEngine classes above): FROZEN-POOL (backbone frozen +
        # pool_mode="last" -> features static, cached once, head-only training in seconds) vs the LIVE backbone
        # (default). The epoch loop + eval below then run ONE path with no per-feature `if`s. ----
        if getattr(m, "pool_mode", "") == "last" and getattr(m, "_frozen_bb", False):
            engine: _StepEngine = FrozenPoolStepEngine(m, a, ids, mask, t2_ids, t2_mask, hn_ids, hn_mask)
        else:
            engine = BackboneStepEngine(m, a, tok, ids, mask, t2_ids, t2_mask, hn_ids, hn_mask,
                                        recon_loss, self.input_text, self.target_text)
        engine.precompute()          # no-op for the backbone; frozen-pool encodes+caches every view here

        # OPTIONAL step-level diagnostics (env-gated, OFF by default = byte-identical). Mirrors the multi-latent path's
        # profiler harness for the single-latent trainer: LANGSET_PROFILE_STEPS=N captures a torch profiler over the
        # first N steps then dumps a CUDA-time table and EXITS; LANGSET_MEM_PRINT=N prints the MEASURED VRAM peak for
        # the first N steps (for right-sizing batch/seq-len instead of guessing at OOMs).
        import os as _os
        import time as _time
        _prof_n = int(_os.environ.get("LANGSET_PROFILE_STEPS", "0"))   # diagnostic: profile N single-latent steps then EXIT
        _prof = None
        _prof_t0 = 0.0
        _gstep = 0
        _mem_n = int(_os.environ.get("LANGSET_MEM_PRINT", "0"))        # diagnostic: print MEASURED VRAM peak for first N steps
        _mem_step = 0
        _cuda = torch.cuda.is_available()                             # profile whatever device is present (CPU-only is valid)
        if _prof_n > 0:
            from torch.profiler import ProfilerActivity as _PA, profile as _tp_profile
            _acts = [_PA.CPU] + ([_PA.CUDA] if _cuda else [])          # no CUDA activity on a CPU box
            _prof = _tp_profile(activities=_acts, record_shapes=False, with_stack=False)
            _prof.__enter__()
            _prof_t0 = _time.perf_counter()
            print(f"[PROFILE] capturing {_prof_n} single-latent steps (bs={a.batch_size} ml={a.max_len} "
                  f"grad_ckpt={getattr(m, '_grad_ckpt', '?')} attn={getattr(m, '_attn_impl', '?')}) then exiting ...", flush=True)

        for ep in range(start_ep, a.epochs):
            m.train()
            order = tr_idx[rng.permutation(len(tr_idx))]
            if a.max_steps_per_epoch:                          # SMALL epochs: cap steps so each <= ~30min (natural save pt)
                order = order[:a.max_steps_per_epoch * a.batch_size]
            tot = nb = 0.0
            for i in range(0, len(order), a.batch_size):
                if learn_pool and rng.random() < a.learn_ratio:   # KNOWLEDGE step: teach substance before the retrieval step
                    lp = torch.tensor(rng.choice(len(learn_pool), size=min(a.batch_size, len(learn_pool)),
                                                 replace=False), device=dev)
                    lloss = learn_loss(lp)
                    opt.zero_grad(); lloss.backward(); opt.step()
                idx = torch.tensor(order[i:i + a.batch_size], device=dev)
                pred, target, hn = engine.featurize(idx)        # engine owns WHERE features come from (backbone vs cached)
                # HARD NEGATIVES (if any): mined near-miss targets as extra negative columns [B:2B] (in-batch targets
                # stay cols [0:B]); the engine computed them under no_grad so gradient still flows only to `pred`.
                tmat = target if hn is None else torch.cat([target, hn], dim=0)
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
                if engine.supports_recon and a.lam_recon > 0:                                # aux: grounding. At 0 the
                    loss = loss + a.lam_recon * engine.recon(pred, idx)                      # term is zero anyway; SKIP so
                    #  recon's fp32 full-vocab ([B,S,vocab]) projection graph is NOT built every step (that graph, not
                    #  the stripped lm_head, is what OOM'd a 0.6B at 84GB without grad_ckpt). Also lets frozen-pool run
                    #  (recon needs the backbone that pool_mode deliberately freezes out). Default 0.3 -> unchanged.
                if a.lam_uniform > 0 and len(idx) > 1:                                       # aux: uniformity
                    sq = torch.pdist(F.normalize(pred, p=2, dim=-1), p=2).pow(2)
                    loss = loss + a.lam_uniform * sq.mul(-2.0).exp().mean().log()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1

                if _mem_n and _mem_step < _mem_n and torch.cuda.is_available():   # MEASURED peak (not an estimate)
                    torch.cuda.synchronize()
                    _peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    _cur = torch.cuda.memory_allocated() / 2 ** 30
                    print(f"[MEM] step {_mem_step}: peak={_peak:.1f}GiB current={_cur:.1f}GiB "
                          f"(bs={a.batch_size} ml={a.max_len} lora_top_k={getattr(m, '_lora_top_k', '?')} "
                          f"grad_ckpt={getattr(m, '_grad_ckpt', '?')} sgt={a.stop_grad_target})", flush=True)
                    _mem_step += 1

                if _prof is not None:                              # profiling: sync for honest timing, dump + EXIT at N
                    if _cuda:
                        torch.cuda.synchronize()                   # only meaningful (and only valid) with a CUDA device
                    _gstep += 1
                    if _gstep >= _prof_n:
                        _wall = _time.perf_counter() - _prof_t0
                        _prof.__exit__(None, None, None)
                        print(f"[PROFILE] SUMMARY {_prof_n} steps: wall={_wall:.1f}s = {_wall / _prof_n:.3f}s/step", flush=True)
                        _sort = "cuda_time_total" if _cuda else "cpu_time_total"   # cuda_time column is absent on CPU
                        print(_prof.key_averages().table(sort_by=_sort, row_limit=25), flush=True)
                        import sys as _sys
                        _sys.exit(0)

            if ep % a.eval_every:
                continue
            # validate in the CURRENT geometry: input-view vs target-view retrieval + collapse + held-out recon.
            emit_in, emit_tg = engine.val_embeddings(val_idx)   # engine owns HOW val embeddings are produced
            mrr = selection.retrieval_mrr(emit_in, emit_tg)["mrr"]
            collapse = selection.collapse_score(emit_in)
            if not engine.supports_recon or a.lam_recon == 0.0:  # recon not the objective (frozen-pool OR lam_recon=0) ->
                recon_val = 0.0                                 # select by retrieval MRR, and skip the wasteful recon-val
                sel_score = mrr - _COLLAPSE_PENALTY * max(0.0, collapse - _COLLAPSE_FLOOR)   # fp32 vocab projection
            else:
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
            save_resume(ep + 1)          # epoch boundary: durable full-state checkpoint so a preempt resumes HERE, not ep0

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
        fsq_levels = int(head.fsq_levels)                        # only for MultiStepCtx; the emission objective derives
        #                                                         its own fsq_dim/fsq_levels/stop_idx off model.head
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)

        seeds = self.input_text
        seed_texts = a.seed_builder(self, seeds, a)              # seed-builder strategy (INJECTED): what the emission reads
        futs = [lst[:a.max_target_items] for lst in self.target_texts]   # cap targets per row
        if a.emit_seed:
            # PHASE-0 as an emitted node: prepend each seed's OWN text as target position 0, so the emitter learns to
            # produce its start-state latent before the futures. Everything downstream (digits/STOP/recon/phase head/
            # eval bank) shifts by one automatically; sup_labels gets a leading "phase0" class. Must happen HERE —
            # before evaluate() closes over `futs` and before the phase_head label set is built from self.sup_labels.
            futs = [[seeds[i], *futs[i]] for i in range(len(futs))]
            if self.sup_labels is not None:
                self.sup_labels = [["phase0", *self.sup_labels[i]] for i in range(len(self.sup_labels))]
        n = len(seeds)
        perm = rng.permutation(n)
        cut = max(1, int(n * (1 - a.val_frac)))
        tr_idx = perm[:cut].tolist()
        val_idx = perm[cut:].tolist() or perm[:1].tolist()        # never-empty val (a tiny smoke can fill train)

        # Target source (stop-grad EMA twin by default): supplies the target latents so both sides don't move
        # together and collapse. Selected ONCE (see the _TargetSource seam).
        target_source: _TargetSource = a.target_source(m, a, tok, dev)   # target strategy (INJECTED), built ONCE

        # PHASE HEAD (the non-collapsing alternative to SupCon): a transient linear classifier hidden->phase trained
        # with CE on the emitted reconstruction. CE only needs a separating hyperplane, so it makes phase LINEARLY
        # decodable WITHOUT pulling same-phase events together (SupCon's identity collapse). Grad flows into up/down_proj
        # + LoRA, shaping the FSQ geometry to be phase-separable while retr_mrr (event identity) survives. Not persisted
        # (its job is to inject phase gradient; eval re-fits its own probe on the now phase-informative emissions).
        phase_head: Optional[torch.nn.Module] = None
        phase_ids: dict[str, int] = {}
        if self.sup_labels is not None and a.lam_phase > 0:
            labs = sorted({lb for row2 in self.sup_labels for lb in row2
                           if lb and lb.lower() not in ("unknown", "none", "nan", "")})
            phase_ids = {lb: i for i, lb in enumerate(labs)}
            phase_head = torch.nn.Linear(d, len(phase_ids)).to(dev)
        params = [p for p in m.parameters() if p.requires_grad]
        if phase_head is not None:
            params = params + list(phase_head.parameters())
        opt = torch.optim.AdamW(params, lr=a.lr)
        run = None
        if a.report_to == "wandb":
            import wandb  # type: ignore[import-untyped]
            run = wandb.init(project=a.wandb_project, config=_wandb_config(a))

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            """Free-roll each val seed -> emitted latents; decode each by nearest-neighbor against an EMA-emitted bank
            of the val `target_texts`. Reports (a) retrieval MRR vs the chain's OWN targets and (b) a NON-COLLAPSE
            diversity count = distinct nearest-bank items produced (FSQ must not mean-collapse to one mode)."""
            m.eval()
            import time as _et
            _ev_t0 = _et.perf_counter()
            veval = val_idx[:a.eval_max_chains] if a.eval_max_chains else val_idx   # bound eval cost to a fixed cohort
            bank_texts: list[str] = []
            bank_chain: list[int] = []
            for ci in veval:
                for t in futs[ci]:
                    bank_texts.append(t)
                    bank_chain.append(ci)
            if not bank_texts:
                m.train()
                return {"retr_mrr": 0.0, "purity": 0.0, "n_distinct": 0, "avg_emitted": 0.0}
            zb = F.normalize(target_source.twin.emit(bank_texts).to(dev).float(), dim=-1)   # [Nbank, d] target-space bank
            chain_t = torch.tensor(bank_chain, device=dev)
            rr: list[float] = []
            produced: set[int] = set()
            n_emit = 0
            emit_vecs: list[torch.Tensor] = []                       # emitted val latents (for stage kNN-purity)
            emit_labs: list[str] = []                                # position-aligned sup label of each emission
            for i in range(0, len(veval), a.batch_size):
                chunk = veval[i:i + a.batch_size]
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
            print(f"[EVAL] {_et.perf_counter() - _ev_t0:.1f}s | {len(veval)} chains, {len(bank_texts)} bank, "
                  f"{n_emit} emissions", flush=True)
            return {"retr_mrr": float(np.mean(rr)) if rr else 0.0, "purity": purity,
                    "n_distinct": len(produced), "avg_emitted": n_emit / max(len(veval), 1)}

        rng_t = torch.Generator().manual_seed(a.seed)
        best = -1.0
        best_state: Optional[dict[str, Any]] = None
        metrics: dict[str, float] = {}

        # ---- preempt-resume (multi-latent): reload full state (LoRA+head+phase_head+opt+epoch+best+rng) if present ----
        start_ep = 0
        _ckpt = (Path(a.resume_dir) / "resume.pt") if a.resume_dir else None
        ck = torch.load(_ckpt, map_location="cpu") if (_ckpt is not None and _ckpt.exists()) else None
        if ck is not None and a.run_sig is not None and ck.get("run_sig") != a.run_sig:
            print(f"[langset] IGNORING {_ckpt}: run_sig mismatch (ckpt={ck.get('run_sig')!r} != this run "
                  f"{a.run_sig!r}) -> starting FRESH", flush=True)
            ck = None
        if ck is not None:
            _params = dict(m.named_parameters())
            for nm, t in ck["trainable"].items():
                if nm in _params:
                    _params[nm].data.copy_(t.to(_params[nm].device, _params[nm].dtype))
            if phase_head is not None and ck.get("phase_head") is not None:
                phase_head.load_state_dict({k: v.to(dev) for k, v in ck["phase_head"].items()})
            opt.load_state_dict(ck["opt"])
            for stt in opt.state.values():                     # optimizer state tensors must live on the model's device
                for k, v in stt.items():
                    if torch.is_tensor(v):
                        stt[k] = v.to(dev)
            start_ep = int(ck["ep"]); best = float(ck["best"]); best_state = ck.get("best_state")
            try:                                               # rng restore is best-effort (robust, not bit-exact)
                rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"]); rng_t.set_state(ck["gen_rng"])
            except Exception:
                pass
            print(f"[langset] RESUMED (multi) from {_ckpt} -> start ep{start_ep}/{a.epochs} best={best:.3f}", flush=True)

        def save_resume(next_ep: int) -> None:
            """Atomically persist FULL multi-latent training state so a preempt/retry resumes from the last epoch
            boundary instead of ep0. tmp+rename => a mid-write preempt cannot corrupt the good file."""
            if not a.resume_dir:
                return
            d = Path(a.resume_dir); d.mkdir(parents=True, exist_ok=True)
            tmp = d / "resume.pt.tmp"
            payload: dict[str, Any] = {
                "trainable": {nm: p.detach().cpu() for nm, p in m.named_parameters() if p.requires_grad},
                "opt": opt.state_dict(),
                "ep": int(next_ep), "best": float(best), "best_state": best_state,
                "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "gen_rng": rng_t.get_state(), "run_sig": a.run_sig,
            }
            if phase_head is not None:
                payload["phase_head"] = {k: v.detach().cpu() for k, v in phase_head.state_dict().items()}
            torch.save(payload, tmp)
            tmp.replace(d / "resume.pt")
            if self.on_checkpoint is not None:
                self.on_checkpoint()                           # e.g. Volume.commit() -> durable across preempt

        import os as _os
        import time as _time
        from contextlib import nullcontext as _nullctx
        _prof_steps = int(_os.environ.get("LANGSET_PROFILE_STEPS", "0"))   # diagnostic: profile N steps then STOP
        _prof = None
        _gstep = 0
        _prof_t0 = 0.0
        _rfn = None
        if _prof_steps > 0:
            from torch.profiler import profile as _tp_profile, ProfilerActivity as _PA, record_function as _rfn
            acts = [_PA.CPU] + ([_PA.CUDA] if torch.cuda.is_available() else [])
            _prof = _tp_profile(activities=acts, record_shapes=False, with_stack=False)
            _prof.__enter__()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _prof_t0 = _time.perf_counter()
            print(f"[PROFILE] capturing {_prof_steps} training steps then exiting ...", flush=True)

        def _rf(name: str) -> Any:                             # named phase range when profiling, no-op otherwise
            return _rfn(name) if _prof is not None else _nullctx()

        objective: _EmissionObjective = a.emission(m, a, dev, self)   # emission strategy (INJECTED), built ONCE
        loss_terms = a.loss_terms(a)                           # aux separation/shaping terms (INJECTED), built ONCE
        for ep in range(start_ep, a.epochs):
            m.train()
            order = a.epoch_order(tr_idx, rng_t, a, seeds)     # epoch ordering strategy (INJECTED)
            if a.max_steps_per_epoch:                          # SMALL epochs: cap steps so each <= ~30min (natural save pt)
                order = order[:a.max_steps_per_epoch * a.batch_size]
            tot = 0.0
            nb = 0
            agg = {"loss_stop": 0.0, "loss_dims": 0.0, "recon_loss": 0.0}
            for i in range(0, len(order), a.batch_size):
                bidx = [tr_idx[k] for k in order[i:i + a.batch_size]]
                se = tok([seed_texts[k] for k in bidx], padding=True, truncation=True, max_length=a.max_len,
                         padding_side="left", return_tensors="pt").to(dev)   # left-pad: hid[s_len-1] = last real token
                ent_lists = [list(futs[k]) for k in bidx]
                lmax = max(len(x) for x in ent_lists)
                flat_texts = [txt for lst in ent_lists for txt in lst]
                with _rf("ema_encode"):
                    flat_tgt = target_source.encode(flat_texts)      # [ΣL, d] stop-grad EMA targets
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
                with _rf("rollout"):
                    em = objective.emit(se, target_lat, valid, lens_l, bidx, b, lmax, ep)
                recon, dim_lg, lab_label = em.recon, em.dim_lg, em.lab_label
                loss_stop, loss_dims, recon_loss = em.logs["loss_stop"], em.logs["loss_dims"], em.logs["recon_loss"]
                loss = em.base_loss                                 # base objective (emission + STOP + recon); separation below
                c = MultiStepCtx(                                    # everything the aux terms read this step
                    trainer=self, args=a, model=m, dev=dev, bidx=bidx, lens_l=lens_l, flat_texts=flat_texts,
                    valid=valid, target_lat=target_lat, recon=recon, dim_lg=dim_lg, lmax=lmax,
                    fsq_levels=fsq_levels, lab_label=lab_label, target_source=target_source,
                    phase_head=phase_head, phase_ids=phase_ids)
                for term in loss_terms:                             # aux separation/shaping terms (fixed order; each self-skips)
                    if term.isolated_backward:                      # isolated terms run AFTER the main backward (below)
                        continue
                    contrib = term.contribute(c)
                    if contrib is not None:
                        _k, _raw, _w = contrib
                        loss = loss + _w * _raw
                        agg[_k] = agg.get(_k, 0.0) + float(_raw.detach())
                if target_source.wants_regularizer and int(valid.sum()) > 1:   # e.g. SIGReg anti-collapse penalty
                    z_pred, z_tgt = objective.z_for_reg(em, target_lat, valid, lmax)
                    loss_sig = target_source.regularizer(z_pred, z_tgt)
                    loss = loss + a.sigreg_lambda * loss_sig
                    agg["loss_sig"] = agg.get("loss_sig", 0.0) + float(loss_sig.detach())
                opt.zero_grad()
                with _rf("backward"):
                    loss.backward()                                 # frees the main graph before any isolated term runs
                for term in loss_terms:                             # isolated terms: own forward+backward, grads ACCUMULATE
                    if not term.isolated_backward:
                        continue
                    contrib = term.contribute(c)
                    if contrib is not None:
                        _k, _raw, _w = contrib
                        (_w * _raw).backward()
                        agg[_k] = agg.get(_k, 0.0) + float(_raw.detach())
                with _rf("opt_ema"):
                    opt.step()
                    target_source.update()                          # EMA twin tracks the online model
                tot += float(loss.detach())
                nb += 1
                agg["loss_stop"] += float(loss_stop.detach())
                agg["loss_dims"] += float(loss_dims.detach())
                agg["recon_loss"] += float(recon_loss.detach())
                if _prof is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _gstep += 1
                    if _gstep >= _prof_steps:
                        _wall = _time.perf_counter() - _prof_t0
                        _prof.__exit__(None, None, None)
                        _ka = _prof.key_averages()
                        def _sc(e: Any) -> float:            # self-CUDA us across torch versions
                            return float(getattr(e, "self_cuda_time_total", 0) or getattr(e, "self_device_time_total", 0) or 0)
                        _cuda_s = sum(_sc(e) for e in _ka) / 1e6
                        _busy = 100.0 * _cuda_s / _wall if _wall > 0 else 0.0
                        _phase = {n: sum(_sc(e) for e in _ka if e.key == n) / 1e6
                                  for n in ("ema_encode", "rollout", "backward", "opt_ema")}
                        print(f"[PROFILE] ===== SUMMARY over {_prof_steps} steps =====", flush=True)
                        print(f"[PROFILE] wall={_wall:.1f}s  {_wall/_prof_steps:.2f}s/step  |  GPU-busy={_cuda_s:.1f}s = "
                              f"{_busy:.0f}% of wall  ({'OVERHEAD-BOUND' if _busy < 65 else 'COMPUTE-BOUND'})", flush=True)
                        _named = sum(_phase.values())
                        for _n, _v in _phase.items():
                            print(f"[PROFILE]   {_n:11s} GPU {_v:5.1f}s = {100*_v/_cuda_s if _cuda_s else 0:4.0f}% of GPU-busy", flush=True)
                        print(f"[PROFILE]   residual(CPU/py) ~{max(0.0, _wall - _named):.1f}s of wall not in named GPU phases", flush=True)
                        for _sk in ("self_cuda_time_total", "self_device_time_total", "cuda_time_total", "cpu_time_total"):
                            try:
                                print(f"[PROFILE] === sort_by={_sk} ===\n"
                                      + _ka.table(sort_by=_sk, row_limit=30), flush=True)
                            except Exception:
                                continue
                        return   # diagnostic run: stop after profiling

            # per-epoch ONLINE-weights snapshot to {output_dir}_ep{N,2N,...} (1-based) — INDEPENDENT of the eval
            # cadence (a trajectory to eval offline, separate from the best-so-far restore). snapshot_every=0 = off.
            if getattr(a, "snapshot_every", 0) and (ep + 1) % a.snapshot_every == 0:
                snap = f"{a.output_dir}_ep{ep + 1}"
                Path(snap).mkdir(parents=True, exist_ok=True)
                m.save_pretrained(snap)
                if self.on_checkpoint is not None:
                    self.on_checkpoint()
                if a.verbose:
                    print(f"        <- snapshot ep{ep + 1} -> {snap}", flush=True)

            # eval/select cadence — but a selector flagged `needs_final_epoch` (e.g. last_epoch_selector) must SEE the
            # final epoch, else with eval_every>1 the last epoch is skipped and an earlier one is kept instead.
            if ep % a.eval_every and not (ep == a.epochs - 1 and getattr(a.selector, "needs_final_epoch", False)):
                continue
            metrics = evaluate()
            row = {"loss": tot / max(nb, 1), **{kk: vv / max(nb, 1) for kk, vv in agg.items()}}
            mrr, pur = metrics["retr_mrr"], metrics.get("purity", 0.0)
            sel = a.selector(a.select, mrr, pur, ep)             # checkpoint-selection strategy (INJECTED)
            if a.verbose:
                hn_s = f" hn={row['loss_hard_neg']:.3f}" if "loss_hard_neg" in row else ""
                sup_s = f" sup={row['loss_sup']:.3f}" if "loss_sup" in row else ""
                ph_s = f" phase={row['loss_phase']:.3f}" if "loss_phase" in row else ""
                pur_s = f" purity={pur:.3f}" if self.sup_labels is not None else ""
                print(f"ep{ep:02d} loss={row['loss']:.3f} stop={row['loss_stop']:.3f} dims={row['loss_dims']:.3f} "
                      f"recon={row['recon_loss']:.3f}{hn_s}{sup_s}{ph_s} | retr_mrr={mrr:.3f}{pur_s} "
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
            save_resume(ep + 1)          # epoch boundary: durable full-state checkpoint so a preempt resumes HERE, not ep0

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
