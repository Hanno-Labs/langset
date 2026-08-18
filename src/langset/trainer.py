"""Training loops for single- and multi-latent LangSet models.

Single-latent training aligns ``input_text`` embeddings with corresponding ``target_text`` embeddings using a
self-contrastive objective. Multi-latent training consumes ``target_texts`` lists and delegates emission and
target construction to the configured strategy classes.

Datasets may be ``datasets.Dataset`` instances or lists of dictionaries. Use ``column_mapping`` to map custom
column names to LangSet's canonical fields.
"""

from __future__ import annotations

import random
from contextlib import AbstractContextManager
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F

from langset import selection
from langset.heads import Head, RtHead, resolve_head
from langset.loss import (
    LearnLossContext,
    ReconstructionLossContext,
    SLContext,
    learn_loss,
    recon_loss,
    sl_loss,
)
from langset.modeling import LangSetModel
from langset.strategies import (
    MultiStepCtx,
)  # the aux-term step context; trainer builds it each step. The concrete

#   strategies (emission/target strategies/...) are injected via TrainingArguments and read off `a`, not imported here.
from langset.training_args import TrainingArguments

if TYPE_CHECKING:  # only for local annotations of the injected strategy instances
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

    from langset.strategies import _EmissionObjective, _TargetSource

_RECON_K = 8  # soft-prompt tokens the latent expands into for the recon decoder
_RECON_MAXLEN = 128  # target_text tokens the recon aux reconstructs
_LEARN_TGT = 160  # [LEARN] rows: max target (substance) tokens generated under next-token CE
_COLLAPSE_PENALTY = 3.0
_COLLAPSE_FLOOR = 0.4  # collapse below this isn't penalized; above it, selection is tanked


def _wandb_config(a: TrainingArguments) -> dict[str, Any]:
    """Return a wandb-safe argument mapping with strategy classes and callables rendered by name."""
    cfg = dict(vars(a))
    for k, v in cfg.items():
        if callable(v):
            cfg[k] = getattr(v, "__name__", repr(v))
    return cfg


def _columns(dataset: Dataset | list[dict[str, Any]]) -> dict[str, list[Any]]:
    if hasattr(dataset, "column_names"):  # datasets.Dataset
        ds = cast("Dataset", dataset)
        return {c: list(ds[c]) for c in ds.column_names}
    rows = list(dataset)  # list[dict]
    return {k: [r[k] for r in rows] for k in rows[0]}


def _fuse_views(
    ids_a: torch.Tensor,
    mask_a: torch.Tensor,
    ids_b: torch.Tensor,
    mask_b: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two right-padded views into one backbone forward.

    Both views are resized to the batch's common maximum real length and stacked along the batch dimension. Only
    masked padding columns are added or removed, so each row has the same retained-token output as a separate
    per-view forward. The caller splits the output at the original batch size.
    """
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
    """Trim a right-padded batch to its longest non-padding sequence.

    Because only masked padding columns are removed, outputs for retained positions are unchanged.
    """
    n = int(mask.sum(dim=1).max().item())
    if n <= 0 or n >= ids.size(1):
        return ids, mask
    return ids[:, :n], mask[:, :n]


# ---- text replay (rehearsal) — shared by the single- and multi-latent learn paths ---------------
def _require_emit_rows(is_learn: list[bool], learn_field: Optional[str]) -> None:
    """Require at least one emission row when text-replay rows are separated from the training split."""
    if is_learn and all(is_learn):
        raise ValueError(
            f"learn_field '{learn_field}' tagged all {len(is_learn)} rows as 'learn' — no rows left for the emit "
            "objective. Leave some rows untagged: they carry the latent geometry; 'learn' rows only rehearse text."
        )


def _tokenize_replay(
    tok: PreTrainedTokenizerBase, texts: list[str], max_len: int, side: str, dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize replay text with an explicit padding side.

    Conditioning documents must be left-padded so their final real token predicts the first target token in
    ``learn_loss``. Targets are right-padded, and their padding is excluded from cross-entropy.
    """
    e = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        padding_side=side,
        return_tensors="pt",
    )
    return e["input_ids"].to(dev), e["attention_mask"].to(dev)


def _snapshot_best(m: LangSetModel) -> dict[str, Any]:
    """Snapshot all state required to restore the best epoch.

    Pretrained models whose base weights can be reconstructed store the head and LoRA parameters. Randomly
    initialized and fully fine-tuned models store the complete backbone. Persisted auxiliary heads are included.
    """
    snap: dict[str, Any] = {
        "head": {k: v.detach().cpu().clone() for k, v in m.head.state_dict().items()}
    }
    if m._pretrained and not getattr(m, "_full_ft", False):
        snap["lora"] = {
            k: v.detach().cpu().clone() for k, v in m.backbone.state_dict().items() if "lora" in k
        }
    else:
        snap["backbone"] = {k: v.detach().cpu().clone() for k, v in m.backbone.state_dict().items()}
    if len(
        m.aux_heads
    ):  # PERSISTED auxiliary heads (langset.heads): restore the best epoch's readouts too
        snap["aux_heads"] = {
            name: {k: v.detach().cpu().clone() for k, v in mod.state_dict().items()}
            for name, mod in m.aux_heads.items()
        }
    return snap


def _restore_best(m: LangSetModel, best_state: dict[str, Any]) -> None:
    """Restore a snapshot created by `_snapshot_best`, including legacy LoRA-only snapshots."""
    m.head.load_state_dict(best_state["head"])
    if "backbone" in best_state:  # random-init: full backbone
        m.backbone.load_state_dict(best_state["backbone"], strict=False)
    else:  # pretrained (or legacy checkpoint): LoRA only
        m.backbone.load_state_dict(best_state["lora"], strict=False)
    for name, sd in best_state.get("aux_heads", {}).items():  # PERSISTED auxiliary heads
        m.aux_heads[name].load_state_dict(sd)


# ---- single-latent step engines -----------------------------------------------------------------
# Single-latent steps obtain prediction, target, and hard-negative features either from the trainable backbone or
# from cached vectors produced by a frozen backbone. Both implementations expose the same step interface.


class _StepEngine:
    """Interface for producing contrastive features for one single-latent step.

    ``supports_recon`` indicates whether the backbone is available for reconstruction loss.
    """

    supports_recon: bool = True

    def precompute(self) -> None:
        """Prepare reusable features before the epoch loop; the live-backbone implementation is a no-op."""

    def featurize(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Build the input, target, and optional hard-negative features for a batch.

        Returns ``(prediction, target, hard_negatives)``. Prediction and target have shape
        ``[batch, latent_dim]``. Hard negatives are either ``None`` or a tensor of negative features. Whether
        gradients flow through the target is controlled by ``stop_grad_target``.
        """
        raise NotImplementedError

    def val_embeddings(self, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return NumPy input and target embeddings for retrieval evaluation."""
        raise NotImplementedError


class BackboneStepEngine(_StepEngine):
    """Produce contrastive features by running the trainable backbone for each step.

    This is the default single-latent engine. It computes input, target, and
    optional hard-negative features and supports reconstruction auxiliaries;
    runtime options such as view fusion and target stop-gradient are handled
    inside `featurize`.
    """

    supports_recon = True

    def __init__(
        self,
        model: LangSetModel,
        args: TrainingArguments,
        tok: PreTrainedTokenizerBase,
        ids: torch.Tensor,
        mask: torch.Tensor,
        t2_ids: torch.Tensor,
        t2_mask: torch.Tensor,
        hn_ids: Optional[torch.Tensor],
        hn_mask: Optional[torch.Tensor],
        input_text: list[str],
        target_text: list[str],
    ) -> None:
        self.m, self.a, self.tok = model, args, tok
        self.ids, self.mask, self.t2_ids, self.t2_mask = ids, mask, t2_ids, t2_mask
        self.hn_ids, self.hn_mask = hn_ids, hn_mask
        self.input_text, self.target_text = input_text, target_text

    def featurize(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m, a = self.m, self.a
        if (
            a.fuse_views and not a.stop_grad_target
        ):  # FUSE input+target in ONE forward (1 launch + 1 recompute);
            fi, fm = _fuse_views(
                self.ids[idx],
                self.mask[idx],
                self.t2_ids[idx],
                self.t2_mask[idx],
                cast("int", self.tok.pad_token_id),  # tokenizer always has a pad id at train time
            )  # math identical (padding masked); split back at row B
            both = m(fi, fm)
            _nb = len(idx)
            pred, target = both[:_nb], both[_nb:]
        else:
            pred = m(*_dyn_trim(self.ids[idx], self.mask[idx]))
            if a.stop_grad_target:  # BYOL/MoCo: target anchors geometry, no backward
                with torch.no_grad():
                    target = m(*_dyn_trim(self.t2_ids[idx], self.t2_mask[idx]))
            else:
                target = m(
                    *_dyn_trim(self.t2_ids[idx], self.t2_mask[idx])
                )  # self-contrastive: emit(target_text)
        hn: Optional[torch.Tensor] = None
        if (
            self.hn_ids is not None
        ):  # HARD NEGATIVES: mined near-miss targets; no_grad (memory-safe,
            with (
                torch.no_grad()
            ):  # no 4th backward) — gradient still flows to `pred`, off the negs.
                assert self.hn_mask is not None  # populated alongside hn_ids
                hn = m(*_dyn_trim(self.hn_ids[idx], self.hn_mask[idx]))
        return pred, target, hn

    def val_embeddings(self, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emit_in = np.asarray(
            self.m.encode([self.input_text[j] for j in val_idx], normalize_embeddings=True)
        )
        emit_tg = np.asarray(
            self.m.encode([self.target_text[j] for j in val_idx], normalize_embeddings=True)
        )
        return emit_in, emit_tg


class FrozenPoolStepEngine(_StepEngine):
    """Cache frozen-backbone features and train only the projection head.

    This engine is used when ``pool_mode="last"`` and the backbone is frozen. It precomputes input, target, and
    optional hard-negative features before the epoch loop. Reconstruction loss is unavailable because the
    backbone does not participate in training.
    """

    supports_recon = False

    def __init__(
        self,
        model: LangSetModel,
        args: TrainingArguments,
        ids: torch.Tensor,
        mask: torch.Tensor,
        t2_ids: torch.Tensor,
        t2_mask: torch.Tensor,
        hn_ids: Optional[torch.Tensor],
        hn_mask: Optional[torch.Tensor],
    ) -> None:
        self.m, self.a = model, args
        self.ids, self.mask, self.t2_ids, self.t2_mask = ids, mask, t2_ids, t2_mask
        self.hn_ids, self.hn_mask = hn_ids, hn_mask
        self.feat_in: Optional[torch.Tensor] = None
        self.feat_tg: Optional[torch.Tensor] = None
        self.feat_hn: Optional[torch.Tensor] = None

    def precompute(self) -> None:
        import time as _time

        m, a = self.m, self.a
        enc_bs = max(
            8, min(2048, 1_000_000 // max(1, a.max_len))
        )  # no_grad+frozen -> big encode batch, LENGTH-aware to fill the

        def _pool_all(
            pid: torch.Tensor, pmask: torch.Tensor
        ) -> torch.Tensor:  # GPU (train batch can stay small)
            outs = []
            with torch.no_grad():
                for s in range(0, pid.size(0), enc_bs):
                    outs.append(
                        m._pool_hidden(
                            *_dyn_trim(pid[s : s + enc_bs], pmask[s : s + enc_bs])
                        ).half()
                    )
            return torch.cat(outs, 0)

        m.eval()
        _t = _time.time()
        self.feat_in = _pool_all(self.ids, self.mask)
        self.feat_tg = _pool_all(self.t2_ids, self.t2_mask)
        self.feat_hn = (
            _pool_all(self.hn_ids, cast("torch.Tensor", self.hn_mask))
            if self.hn_ids is not None
            else None
        )
        m.train()
        if a.verbose:
            fi = self.feat_in
            print(
                f"[langset] CACHED {fi.size(0)} frozen features ({fi.size(1)}d, "
                f"{fi.element_size() * fi.nelement() / 1e6:.0f}MB/view) in {_time.time() - _t:.1f}s "
                f"-> head-only training, no backbone in loop",
                flush=True,
            )

    def featurize(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
    """Train a `LangSetModel` on text-to-latent alignment data.

    The trainer selects the single- or multi-latent path from the model configuration. Single-latent rows require
    ``input_text`` and ``target_text``; multi-latent rows require ``input_text`` and a non-empty ``target_texts``
    list. Optional columns configure hard negatives, false-negative masks, concepts, states, auxiliary heads, and
    text replay according to `TrainingArguments`.

    ``eval_dataset`` is currently reserved and ignored; validation rows are split from ``train_dataset``.
    ``on_checkpoint``, when provided, is called after the checkpoint is written locally so callers can perform
    external persistence or synchronization.
    """

    def __init__(
        self,
        model: LangSetModel,
        args: TrainingArguments,
        train_dataset: Dataset | list[dict[str, Any]],
        eval_dataset: Optional[Dataset | list[dict[str, Any]]] = None,
        column_mapping: Optional[dict[str, str]] = None,
        on_checkpoint: Optional[Callable[[], None]] = None,
    ) -> None:
        self.model = model
        self.args = args
        # When set, this callback runs after a best-so-far checkpoint is written, allowing an external system to
        # publish or evaluate the durable checkpoint.
        self.on_checkpoint = on_checkpoint
        # A multi-latent model reads a per-row `target_texts` list and uses the selected emission strategy. The
        # single-latent path reads one `target_text` value per row.
        self.multi_latent = bool(model.head.multi_latent)
        # Multi-latent inference feeds emitted states back autoregressively. Require positive scheduled sampling so
        # training exposes the model to its own prior emissions. An unspecified probability defaults to 0.25.
        if self.multi_latent:
            if args.emission is None:
                raise ValueError(
                    "multi_latent=True requires an explicit emission strategy: ConceptObjective, "
                    "StateResidualObjective, CodeSoftmaxObjective, or QueryBridgeEmission"
                )
            if args.ss_prob is None:
                args.ss_prob = 0.25
                print(
                    "[langset] multi_latent + ss_prob unset -> ss_prob=0.25 (rollout must be trained; set ss_prob "
                    "explicitly to override, and ss_warmup>0 for deep train_hops)",
                    flush=True,
                )
            elif args.ss_prob <= 0.0:
                raise ValueError(
                    "multi_latent=True with ss_prob=0 trains PURELY teacher-forced but rolls out AUTOREGRESSIVELY at "
                    "inference -> exposure bias: the rollout is never trained. Set ss_prob>0 (e.g. 0.25) so the "
                    "emitter learns to consume its own predictions, or use multi_latent=False. "
                    "You cannot roll out what you did not train."
                )
        elif args.ss_prob is None:
            args.ss_prob = 0.0  # single-latent never rolls; teacher-forced is correct
        # PLUGGABLE AUXILIARY HEADS are multi-latent only (they generalize the phase head, which lives only there).
        if getattr(args, "heads", None) and not self.multi_latent:
            raise ValueError(
                "TrainingArguments(heads=...) is supported on the MULTI-LATENT path only (build the model with "
                "multi_latent=True). The heads generalize the phase head, which is multi-latent-only."
            )
        cols = _columns(train_dataset)
        inv = {v: k for k, v in (column_mapping or {}).items()}  # user-col -> canonical
        get = lambda canon: cols[inv.get(canon, canon)]  # type: ignore[index]  # noqa: E731
        # JEPA masked-self-prediction: the caller gives a RAW `text` column; the Trainer masks it FRESH EVERY
        # EPOCH in train() (target = the full text, the EMA-twin teacher; input = a masked view). Nothing is
        # pre-masked here — we only stash the raw texts + resolved masker. Auto-activates when the dataset has a
        # `text` column and no `input_text`, or when `args.masker` is set explicitly.
        _has = lambda c: inv.get(c, c) in cols  # noqa: E731
        self._masking = (getattr(args, "masker", None) is not None) or (
            _has("text") and not _has("input_text")
        )
        self._masker: Optional[Any] = None
        self._mask_texts: list[str] = []
        if self._masking:
            if self.multi_latent:
                raise ValueError(
                    "masked mode (JEPA) is single-latent only; use a non-multi_latent model"
                )
            from langset.masking import mask_view, resolve_masker

            self._masker = resolve_masker(
                getattr(args, "masker", None), getattr(args, "mask_ratio", 0.15)
            )
            self._mask_view = mask_view  # bound for per-epoch re-masking in train()
            self._mask_texts = [str(x) for x in get("text")]
            if not self._mask_texts:
                raise ValueError("masked mode needs a non-empty `text` column")
            _init = self._mask_view(self._mask_texts, self._masker, random.Random(args.seed))
            cols = {**cols, "input_text": _init, "target_text": list(self._mask_texts)}
            # input_text/target_text are now SYNTHESIZED under their canonical keys -> read them directly (any
            # column_mapping for them is meaningless: their source is `text`, not a real input/target column). But
            # KEEP `inv` intact so the optional fields below (mask_field, hard_neg_field, learn_field)
            # still honor column_mapping exactly as the non-masked path does -> same dataset+mapping, same behavior.
            get = lambda canon: cols[canon]  # type: ignore[index]  # noqa: E731
        self.input_text = [str(x) for x in get("input_text")]
        if self.multi_latent:
            raw_tt = get("target_texts")  # per row: a non-empty list of target descriptions
            self.target_texts: list[list[str]] = []
            for i, v in enumerate(raw_tt):
                if not isinstance(v, (list, tuple)) or len(v) == 0:
                    raise ValueError(
                        f"multi_latent Trainer needs a 'target_texts' column of non-empty lists; row {i} = {v!r}"
                    )
                self.target_texts.append([str(x) for x in v])
            # Optional per-row reasoning text used by the CoT loss and seed strategies. Missing values become empty
            # strings; the default strategies ignore them, and CoTGenTerm skips batches without reasoning text.
            cot_key = inv.get(
                "cot_text", "cot_text"
            )  # honor column_mapping (a renamed reasoning column)
            self.cot_texts: list[str] = (
                [
                    ("" if x is None else str(x)) for x in cols[cot_key]
                ]  # None/absent -> "" (not the literal "None")
                if cot_key in cols
                else [""] * len(self.input_text)
            )
            # QueryBridge may consume a per-row list of hard-negative texts in its own contrastive denominator.
            self.hard_neg_texts: Optional[list[list[str]]] = None
            hn_field = getattr(args, "hard_neg_field", None)
            if hn_field is not None:
                raw_hn = cols[inv.get(hn_field, hn_field)]
                self.hard_neg_texts = [
                    [
                        str(x)
                        for x in (
                            v
                            if isinstance(v, (list, tuple))
                            else ([v] if v not in (None, "") else [])
                        )
                    ]
                    for v in raw_hn
                ]
            # Optional supervised-contrastive labels, aligned one-to-one with each row's target texts.
            self.sup_labels: Optional[list[list[str]]] = None
            sup_field = getattr(args, "sup_field", None)
            if sup_field is not None:
                raw_sup = cols[inv.get(sup_field, sup_field)]
                self.sup_labels = [
                    [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])] for v in raw_sup
                ]
            # ConceptObjective accepts one facet mapping per target step. Discover a deterministic alphabet from
            # those mappings, then let the configured code source construct the fixed codebook.
            self.concept_alphabet: Optional[dict[str, list[str]]] = None
            self.concept_laws: Optional[list[list[dict[int, dict[int, float]]]]] = None
            concept_field = getattr(args, "concept_field", None)
            if concept_field is not None:
                from langset.strategies import discover_concept_alphabet, parse_concepts

                raw_c = list(cols[inv.get(concept_field, concept_field)])
                self.concept_alphabet = discover_concept_alphabet(raw_c)
                facets = list(self.concept_alphabet)
                fidx = {f: i for i, f in enumerate(facets)}
                midx = {f: {n: i for i, n in enumerate(self.concept_alphabet[f])} for f in facets}
                laws = []
                for raw in raw_c:
                    ticks = raw if isinstance(raw, (list, tuple)) else [raw]
                    per_tick = []
                    for tick in list(ticks)[: args.max_target_items]:
                        d: dict[int, dict[int, float]] = {}
                        for f, w in parse_concepts(tick).items():
                            if f in fidx:
                                d[fidx[f]] = {midx[f][n]: v for n, v in w.items() if n in midx[f]}
                        per_tick.append(d)
                    laws.append(per_tick)
                self.concept_laws = laws
                sizes = {f: len(v) for f, v in self.concept_alphabet.items()}
                print(
                    f"[concepts] discovered {sum(sizes.values())} concepts across {len(sizes)} facets: "
                    f"{sizes}",
                    flush=True,
                )
            # StateResidualObjective reads per-step member indices directly as supervision for its named component.
            self.state_labels: Optional[list[list[list[int]]]] = None
            state_field = getattr(args, "state_field", None)
            if state_field is not None:
                raw_st = cols[inv.get(state_field, state_field)]
                self.state_labels = [
                    [[int(i) for i in tick] for tick in row[: args.max_target_items]]
                    for row in raw_st
                ]
            # Read each auxiliary head's target column. Runtime resolution later converts these raw row values into
            # class IDs or numeric tensors according to the configured loss.
            self.head_cols: dict[str, list[object]] = {}
            for h in getattr(args, "heads", []):
                self.head_cols[h.name] = list(cols[inv.get(h.target, h.target)])
            # Mark rows used for multi-latent text replay. This branch returns before the single-latent replay setup.
            self.is_learn: list[bool] = [False] * len(self.input_text)
            learn_field = getattr(args, "learn_field", None)
            if learn_field is not None and args.learn_ratio > 0:
                raw = cols[inv.get(learn_field, learn_field)]
                self.is_learn = [str(v).lower() == "learn" for v in raw]
            _require_emit_rows(self.is_learn, args.learn_field)
            if args.verbose:
                hn = "" if not self.hard_neg_texts else " (+hard-neg)"
                sp = "" if not self.sup_labels else " (+supcon)"
                lr = (
                    ""
                    if sum(self.is_learn) == 0
                    else f" (+{sum(self.is_learn)} learn @ratio {args.learn_ratio})"
                )
                print(
                    f"[langset] {len(self.input_text)} rows (multi-latent){hn}{sp}{lr}", flush=True
                )
            return
        self.target_text = [str(x) for x in get("target_text")]
        # optional false-negative masking: per-row set of facet keys; in-batch pairs sharing any key are masked.
        self.mask_keys: Optional[list[frozenset[str]]] = None
        if args.mask_field is not None:
            raw = cols[inv.get(args.mask_field, args.mask_field)]
            self.mask_keys = [
                frozenset(v if isinstance(v, (list, tuple, set)) else [v])
                if v not in (None, "")
                else frozenset()
                for v in raw
            ]
        # optional hard negatives: a mined near-miss target per row (encoded as an extra negative each step).
        self.hard_neg_text: Optional[list[str]] = None
        hn_field = getattr(args, "hard_neg_field", None)
        if hn_field is not None:
            raw = cols[inv.get(hn_field, hn_field)]
            self.hard_neg_text = [str(v) if v not in (None, "") else "" for v in raw]
        # optional knowledge-injection: rows tagged "learn" train next-token CE (input_text -> target_text) instead
        # of contrastive; they're pulled OUT of the contrastive split and fed as a separate learn pool.
        self.is_learn: list[bool] = [False] * len(self.input_text)
        learn_field = getattr(args, "learn_field", None)
        if learn_field is not None and args.learn_ratio > 0:
            raw = cols[inv.get(learn_field, learn_field)]
            self.is_learn = [str(v).lower() == "learn" for v in raw]
        _require_emit_rows(self.is_learn, args.learn_field)
        n_learn = sum(self.is_learn)
        if args.verbose:
            masked = "" if self.mask_keys is None else " (+false-neg mask)"
            hn = "" if self.hard_neg_text is None else " (+hard-neg)"
            lr = "" if n_learn == 0 else f" (+{n_learn} learn @ratio {args.learn_ratio})"
            jepa = (
                ""
                if not self._masking
                else (
                    f" (JEPA masked-self: {type(self._masker).__name__} @{getattr(args, 'mask_ratio', 0.15)}, "
                    f"fresh mask/epoch)"
                )
            )
            print(f"[langset] {len(self.input_text)} rows{jepa}{masked}{hn}{lr}", flush=True)

    def train(self) -> LangSetModel:
        """Train the configured model and return the best restored model.

        Dispatches to the multi-latent strategy loop when the model uses a multi-latent head; otherwise runs
        single-latent contrastive training. Checkpointing, early stopping, text replay, and auxiliary losses are
        controlled by `TrainingArguments`.
        """
        if self.multi_latent:
            return self._train_multi()
        a, m = self.args, self.model
        dev = m.device
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        tok = m.tokenizer

        def tok_to(texts: list[str], mx: int) -> tuple[torch.Tensor, torch.Tensor]:
            """Tokenize text into input-ID and attention-mask tensors on the model device."""
            e = tok(texts, padding=True, truncation=True, max_length=mx, return_tensors="pt")
            return e["input_ids"].to(dev), e["attention_mask"].to(dev)

        ids, mask = tok_to(self.input_text, a.max_len)  # input view
        t2_ids, t2_mask = tok_to(
            self.target_text, a.max_len
        )  # target view (self-contrastive target)
        tr_ids, tr_mask = tok_to(self.target_text, _RECON_MAXLEN)  # target tokens for the recon aux
        hn_ids = hn_mask = None
        if (
            self.hard_neg_text is not None
        ):  # hard-neg view (empty "" rows tokenize fine, masked below)
            hn_ids, hn_mask = tok_to([t or " " for t in self.hard_neg_text], a.max_len)

        # texts associated with knowledge injection
        learn_pool = [i for i in range(len(self.input_text)) if self.is_learn[i]]
        ln_doc_ids = ln_doc_mask = ln_tgt_ids = ln_tgt_mask = None
        if learn_pool:
            ln_doc_ids, ln_doc_mask = _tokenize_replay(
                tok, [self.input_text[i] for i in learn_pool], a.max_len, "left", dev
            )  # doc LEFT-pad
            ln_tgt_ids, ln_tgt_mask = _tokenize_replay(
                tok, [self.target_text[i] for i in learn_pool], _LEARN_TGT, "right", dev
            )  # target RIGHT-pad

        # shuffle indices for training/validation split
        n = len(self.input_text)
        embed_all = np.array([i for i in range(n) if not self.is_learn[i]])
        embed_perm = embed_all[rng.permutation(len(embed_all))]
        n_val = max(4, int(len(embed_perm) * a.val_frac))
        val_idx, tr_idx = embed_perm[:n_val], embed_perm[n_val:]

        connector = torch.nn.Linear(m.latent_dim, _RECON_K * m.h).to(dev)

        opt = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad] + list(connector.parameters()),
            lr=a.lr,
        )
        run = None
        if a.report_to == "wandb":
            wandb = import_module("wandb")
            run = wandb.init(project=a.wandb_project, config=_wandb_config(a))

        best_score, best_state, no_improve = -1e9, None, 0

        # ---- preempt-resume: reload full training state from a durable checkpoint if one exists (else start fresh) ----
        start_ep = 0
        _ckpt = (Path(a.resume_dir) / "resume.pt") if a.resume_dir else None
        ck = (
            torch.load(_ckpt, map_location="cpu")
            if (_ckpt is not None and _ckpt.exists())
            else None
        )
        if ck is not None and a.run_sig is not None and ck.get("run_sig") != a.run_sig:
            print(
                f"[langset] IGNORING {_ckpt}: run_sig mismatch (ckpt={ck.get('run_sig')!r} != this run "
                f"{a.run_sig!r}) -> starting FRESH",
                flush=True,
            )  # a DIFFERENT model/data/config can never resume us
            ck = None

        # code associated with resuming from a checkpoint
        if ck is not None:
            _params = dict(m.named_parameters())
            for nm, t in ck["trainable"].items():
                if nm in _params:
                    _params[nm].data.copy_(t.to(_params[nm].device, _params[nm].dtype))
            connector.load_state_dict({k: v.to(dev) for k, v in ck["connector"].items()})
            opt.load_state_dict(ck["opt"])
            for (
                stt
            ) in opt.state.values():  # optimizer state tensors must live on the model's device
                for k, v in stt.items():
                    if torch.is_tensor(v):
                        stt[k] = v.to(dev)
            start_ep = int(ck["ep"])
            best_score = float(ck["best_score"])
            no_improve = int(ck["no_improve"])
            best_state = ck.get("best_state")
            try:  # rng restore is best-effort (robust, not bit-exact)
                rng.bit_generator.state = ck["np_rng"]
                torch.set_rng_state(ck["torch_rng"])
            except Exception:
                pass
            print(
                f"[langset] RESUMED from {_ckpt} -> start ep{start_ep}/{a.epochs} best={best_score:.3f}",
                flush=True,
            )

        def save_resume(next_ep: int) -> None:
            """Save model, optimizer, selection, and RNG state through an atomic file replacement."""
            if not a.resume_dir:
                return
            d = Path(a.resume_dir)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "resume.pt.tmp"
            torch.save(
                {
                    "trainable": {
                        nm: p.detach().cpu() for nm, p in m.named_parameters() if p.requires_grad
                    },
                    "connector": {k: v.detach().cpu() for k, v in connector.state_dict().items()},
                    "opt": opt.state_dict(),
                    "ep": int(next_ep),
                    "best_score": float(best_score),
                    "no_improve": int(no_improve),
                    "best_state": best_state,
                    "np_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "run_sig": a.run_sig,  # identity fingerprint: resume REFUSES to load this into a different run
                },
                tmp,
            )
            tmp.replace(d / "resume.pt")
            if self.on_checkpoint is not None:
                self.on_checkpoint()  # Notify the caller after the checkpoint becomes durable.

        # Select the feature engine once. Frozen last-token pooling caches backbone features; other configurations
        # compute features through the backbone during each step.
        if getattr(m, "pool_mode", "") == "last" and getattr(m, "_frozen_bb", False):
            engine: _StepEngine = FrozenPoolStepEngine(
                m, a, ids, mask, t2_ids, t2_mask, hn_ids, hn_mask
            )
        else:
            engine = BackboneStepEngine(
                m,
                a,
                tok,
                ids,
                mask,
                t2_ids,
                t2_mask,
                hn_ids,
                hn_mask,
                self.input_text,
                self.target_text,
            )
        engine.precompute()  # The frozen-pool engine caches all view features here.

        # Environment-gated diagnostics. LANGSET_PROFILE_STEPS profiles the requested number of steps and exits;
        # LANGSET_MEM_PRINT reports CUDA memory usage for the requested number of steps.
        import os as _os
        import time as _time

        _prof_n = int(
            _os.environ.get("LANGSET_PROFILE_STEPS", "0")
        )  # diagnostic: profile N single-latent steps then EXIT
        _prof = None
        _prof_t0 = 0.0
        _gstep = 0
        _mem_n = int(
            _os.environ.get("LANGSET_MEM_PRINT", "0")
        )  # diagnostic: print MEASURED VRAM peak for first N steps
        _mem_step = 0
        _cuda = torch.cuda.is_available()  # profile whatever device is present (CPU-only is valid)
        if _prof_n > 0:
            from torch.profiler import ProfilerActivity as _PA
            from torch.profiler import profile as _tp_profile

            _acts = [_PA.CPU] + ([_PA.CUDA] if _cuda else [])  # no CUDA activity on a CPU box
            _prof = _tp_profile(activities=_acts, record_shapes=False, with_stack=False)
            _prof.__enter__()
            _prof_t0 = _time.perf_counter()
            print(
                f"[PROFILE] capturing {_prof_n} single-latent steps (bs={a.batch_size} ml={a.max_len} "
                f"grad_ckpt={getattr(m, '_grad_ckpt', '?')} attn={getattr(m, '_attn_impl', '?')}) then exiting ...",
                flush=True,
            )

        def _grad_cache_step(idx: torch.Tensor) -> float:
            """Compute a full-batch contrastive gradient with chunk-sized backbone activation memory.

            The first pass computes detached batch embeddings and their loss gradients. Each chunk is then
            recomputed with autograd enabled, and the cached embedding gradients are propagated through the
            backbone. Embedding-slot heads receive gradients from the first pass. Deterministic forwards, including
            zero dropout, are required.
            """
            ch = a.gc_chunk or a.batch_size
            chunks = [idx[j : j + ch] for j in range(0, len(idx), ch)]
            preds: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            hns: list[torch.Tensor] = []
            with torch.no_grad():  # PHASE 1: embeddings only; each chunk's activations freed
                for c in chunks:
                    p, t, h = engine.featurize(c)
                    preds.append(p)
                    targets.append(t)
                    if h is not None:
                        hns.append(h)
            pf = torch.cat(preds).detach().requires_grad_(True)
            # the target participates in autograd UNLESS it's a stop-grad target (BYOL/MoCo-style): then it is a
            # frozen key, gradient reaches only `pred` — mirror the direct path, which featurizes it under no_grad.
            tf = torch.cat(targets).detach()
            if not a.stop_grad_target:
                tf.requires_grad_(True)
            hf = torch.cat(hns) if hns else None  # hard negs stay no_grad (as in featurize)
            loss = sl_loss(
                SLContext(
                    model=m,
                    args=a,
                    pred=pf,
                    target=tf,
                    hn=hf,
                    idx=idx,
                    mask_keys=self.mask_keys,
                    hard_neg_text=self.hard_neg_text,
                )
            ).to_tensor()  # full-batch loss -> cached rep grads (+ emb_slot head grads)
            opt.zero_grad()
            loss.backward()  # fills pf.grad (+ tf.grad unless stop-grad, + slot-head params); backbone NOT in this graph
            gp = pf.grad
            gt = tf.grad if tf.requires_grad else None
            assert gp is not None  # loss.backward just populated it
            off = 0
            for c in chunks:  # PHASE 2: re-forward WITH grad, inject cached grads -> backbone param grads accumulate
                cn = len(c)
                p, t, _ = engine.featurize(c)
                tensors, grads = [p], [gp[off : off + cn]]
                if (
                    gt is not None and t.requires_grad
                ):  # skip the target when it's a stop-grad (no-grad) key
                    tensors.append(t)
                    grads.append(gt[off : off + cn])
                torch.autograd.backward(tensors, grads)
                off += cn
            opt.step()
            return float(loss.detach())

        if a.grad_cache:
            assert a.lam_recon == 0, (
                "grad_cache requires lam_recon==0 (recon needs the per-token backbone graph, not just the "
                "pooled embedding); set TrainingArguments(lam_recon=0)"
            )
            assert float(m.head.drop.p) == 0.0, (
                "grad_cache requires dropout==0: phase-1 and phase-2 forwards of a chunk must be identical, but "
                "dropout randomizes them so the cached embedding grads no longer match the re-forward. Rebuild "
                "the model with dropout=0 (this also zeros lora_dropout, which is driven by the same arg)."
            )
            print(
                f"[langset] GRADCACHE ON: effective batch={a.batch_size}, gc_chunk={a.gc_chunk or a.batch_size} "
                "(peak activation = one chunk; big in-batch-negative batch decoupled from memory)",
                flush=True,
            )

        for ep in range(start_ep, a.epochs):
            m.train()
            # JEPA: RE-MASK the raw text fresh this epoch (new random holes every epoch -> the model never sees
            # the same (visible, hidden) split twice; target view t2 stays the full text and is untouched).
            if self._masking:
                masker = self._masker
                assert masker is not None  # set whenever _masking is on
                new_in = self._mask_view(
                    self._mask_texts, masker, random.Random(a.seed + 1000 + ep)
                )
                self.input_text = new_in  # val eval re-encodes from this
                ids, mask = tok_to(new_in, a.max_len)
                engine.ids, engine.mask = ids, mask  # backbone re-featurizes from these each step
            order = tr_idx[rng.permutation(len(tr_idx))]
            if a.max_steps_per_epoch:  # SMALL epochs: cap steps so each <= ~30min (natural save pt)
                order = order[: a.max_steps_per_epoch * a.batch_size]
            tot = nb = 0.0
            for i in range(0, len(order), a.batch_size):
                if (
                    learn_pool and rng.random() < a.learn_ratio
                ):  # KNOWLEDGE step: teach substance before the retrieval step
                    lp = torch.tensor(
                        rng.choice(
                            len(learn_pool), size=min(a.batch_size, len(learn_pool)), replace=False
                        ),
                        device=dev,
                    )
                    assert ln_doc_ids is not None and ln_doc_mask is not None
                    assert ln_tgt_ids is not None and ln_tgt_mask is not None
                    lloss = learn_loss(
                        LearnLossContext(
                            model=m,
                            args=a,
                            pos=lp,
                            ln_doc_ids=ln_doc_ids,
                            ln_doc_mask=ln_doc_mask,
                            ln_tgt_ids=ln_tgt_ids,
                            ln_tgt_mask=ln_tgt_mask,
                        )
                    ).to_tensor()
                    opt.zero_grad()
                    lloss.backward()
                    opt.step()
                idx = torch.tensor(order[i : i + a.batch_size], device=dev)
                if (
                    a.grad_cache
                ):  # GradCache: big in-batch-negative batch, peak activation capped at gc_chunk
                    tot += _grad_cache_step(idx)
                    nb += 1
                else:
                    pred, target, hn = engine.featurize(
                        idx
                    )  # engine owns WHERE features come from (backbone vs cached)
                    # Hard negatives are extra contrastive columns; gradients reach only ``pred``.
                    loss = sl_loss(
                        SLContext(
                            model=m,
                            args=a,
                            pred=pred,
                            target=target,
                            hn=hn,
                            idx=idx,
                            mask_keys=self.mask_keys,
                            hard_neg_text=self.hard_neg_text,
                        )
                    ).to_tensor()
                    if (
                        engine.supports_recon and a.lam_recon > 0
                    ):  # aux: grounding. At 0 the term is zero anyway;
                        loss = (
                            loss
                            + a.lam_recon
                            * recon_loss(
                                ReconstructionLossContext(
                                    model=m,
                                    args=a,
                                    latent=pred,
                                    rows=idx,
                                    tr_ids=tr_ids,
                                    tr_mask=tr_mask,
                                    connector=connector,
                                )
                            ).to_tensor()
                        )
                        # Building reconstruction only when enabled avoids its full-vocabulary projection graph.
                        # GradCache does not support this per-token backbone graph and requires `lam_recon == 0`.
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach())
                    nb += 1

                if (
                    _mem_n and _mem_step < _mem_n and torch.cuda.is_available()
                ):  # MEASURED peak (not an estimate)
                    torch.cuda.synchronize()
                    _peak = torch.cuda.max_memory_allocated() / 2**30
                    _cur = torch.cuda.memory_allocated() / 2**30
                    print(
                        f"[MEM] step {_mem_step}: peak={_peak:.1f}GiB current={_cur:.1f}GiB "
                        f"(bs={a.batch_size} ml={a.max_len} lora_top_k={getattr(m, '_lora_top_k', '?')} "
                        f"grad_ckpt={getattr(m, '_grad_ckpt', '?')} sgt={a.stop_grad_target})",
                        flush=True,
                    )
                    _mem_step += 1

                if _prof is not None:  # profiling: sync for honest timing, dump + EXIT at N
                    if _cuda:
                        torch.cuda.synchronize()  # only meaningful (and only valid) with a CUDA device
                    _gstep += 1
                    if _gstep >= _prof_n:
                        _wall = _time.perf_counter() - _prof_t0
                        _prof.__exit__(None, None, None)
                        print(
                            f"[PROFILE] SUMMARY {_prof_n} steps: wall={_wall:.1f}s = {_wall / _prof_n:.3f}s/step",
                            flush=True,
                        )
                        _sort = (
                            "cuda_time_total" if _cuda else "cpu_time_total"
                        )  # cuda_time column is absent on CPU
                        print(_prof.key_averages().table(sort_by=_sort, row_limit=25), flush=True)
                        import sys as _sys

                        _sys.exit(0)

            if ep % a.eval_every:
                continue
            # validate in the CURRENT geometry: input-view vs target-view retrieval + collapse + held-out recon.
            emit_in, emit_tg = engine.val_embeddings(
                val_idx
            )  # engine owns HOW val embeddings are produced
            mrr = selection.retrieval_mrr(emit_in, emit_tg)["mrr"]
            collapse = selection.collapse_score(emit_in)
            if not engine.supports_recon or a.lam_recon == 0.0:
                # Without a reconstruction objective, select by collapse-penalized retrieval and skip reconstruction.
                recon_val = 0.0
                sel_score = mrr - _COLLAPSE_PENALTY * max(
                    0.0, collapse - _COLLAPSE_FLOOR
                )  # fp32 vocab projection
            else:
                with torch.no_grad():
                    rv, tot_v = 0.0, 0
                    for s in range(0, len(val_idx), a.batch_size):
                        vb = torch.tensor(val_idx[s : s + a.batch_size], device=dev)
                        rv += float(
                            recon_loss(
                                ReconstructionLossContext(
                                    model=m,
                                    args=a,
                                    latent=m(ids[vb], mask[vb]),
                                    rows=vb,
                                    tr_ids=tr_ids,
                                    tr_mask=tr_mask,
                                    connector=connector,
                                )
                            ).to_tensor()
                        ) * len(vb)
                        tot_v += len(vb)
                    recon_val = rv / tot_v
                # recon_val is teacher-forced -> blind to collapse; hard-penalize high collapse so a collapsed epoch
                # can never win.
                sel_score = -recon_val - _COLLAPSE_PENALTY * max(0.0, collapse - _COLLAPSE_FLOOR)
            if a.verbose:
                print(
                    f"ep{ep:02d} loss={tot / nb:.3f} mrr={mrr:.3f} collapse={collapse:.3f} "
                    f"recon_val={recon_val:.3f} sel={sel_score:.3f}",
                    flush=True,
                )
            if run is not None:
                run.log(
                    {
                        "loss": tot / nb,
                        "mrr": mrr,
                        "collapse": collapse,
                        "recon_val": recon_val,
                        "sel_score": sel_score,
                        "epoch": ep,
                    }
                )

            if sel_score > best_score:
                best_score = sel_score
                best_state = _snapshot_best(
                    m
                )  # LoRA-only if pretrained, FULL backbone if random-init
                no_improve = 0
                if self.on_checkpoint is not None:  # persist best-so-far + notify (live checkpoint)
                    Path(a.output_dir).mkdir(parents=True, exist_ok=True)
                    m.save_pretrained(a.output_dir)
                    self.on_checkpoint()
            else:
                no_improve += 1
                if no_improve >= a.patience:
                    if a.verbose:
                        print(f"[langset] early stop at ep{ep} (best {best_score:.3f})", flush=True)
                    break
            save_resume(
                ep + 1
            )  # epoch boundary: durable full-state checkpoint so a preempt resumes HERE, not ep0

        if best_state is not None:  # restore best
            _restore_best(m, best_state)
        m.eval()
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(a.output_dir)
        if run is not None:
            run.finish()
        if a.verbose:
            print(f"[langset] done. best={best_score:.3f} -> {a.output_dir}", flush=True)
        return m

    def _train_multi(self) -> LangSetModel:
        """Train an explicitly selected multi-vector emission objective.

        Named-state objectives autoregressively feed each committed concept mixture back and learn an independent
        STOP decision. QueryBridge emits an unordered continuous vector set in one pass. A target-source strategy
        supplies the comparison geometry, while optional auxiliary terms shape it without choosing the emitter."""
        a, m = self.args, self.model
        dev = m.device
        tok = m.tokenizer
        d = int(m.latent_dim)
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)

        seeds = self.input_text
        seed_texts = a.seed_builder(
            self, seeds, a
        )  # seed-builder strategy (INJECTED): what the emission reads
        futs = [lst[: a.max_target_items] for lst in self.target_texts]  # cap targets per row
        if a.emit_seed:
            # PHASE-0 as an emitted node: prepend each seed's OWN text as target position 0, so the emitter learns to
            # produce its start-state latent before the futures. Everything downstream (state/STOP/recon/phase head/
            # eval bank) shifts by one automatically; sup_labels gets a leading "phase0" class. Must happen HERE —
            # before evaluate() closes over `futs` and before the phase_head label set is built from self.sup_labels.
            futs = [[seeds[i], *futs[i]] for i in range(len(futs))]
            if self.sup_labels is not None:
                self.sup_labels = [
                    ["phase0", *self.sup_labels[i]] for i in range(len(self.sup_labels))
                ]
        n = len(seeds)
        _emb = np.array(
            [i for i in range(n) if not getattr(self, "is_learn", [False] * n)[i]]
        )  # replay rows: not embedded
        perm = _emb[
            rng.permutation(len(_emb))
        ]  # learn-tagged rows are rehearsed as text only, kept OUT of the
        cut = max(
            1, int(len(perm) * (1 - a.val_frac))
        )  # latent split (matches the single-latent path)
        tr_idx = perm[:cut].tolist()
        val_idx = (
            perm[cut:].tolist() or perm[:1].tolist()
        )  # never-empty val (a tiny smoke can fill train)

        # Build the configured target source once for the training run.
        target_source: _TargetSource = a.target_source(m, a, tok, dev)

        # Resolve auxiliary heads before constructing the optimizer. The phase compatibility option becomes a
        # transient classification head; user-configured heads follow. Persisted heads share their module with the
        # model so save_pretrained() and head_output() can retain and query them.
        phase_head: Optional[torch.nn.Module] = None
        phase_ids: dict[str, int] = {}
        _spec_values: list[tuple[Head, list[object]]] = []
        if a.lam_phase > 0 and self.sup_labels is not None:
            _spec_values.append(
                (
                    Head.phase_shim(cast(str, a.sup_field), a.lam_phase),
                    cast("list[object]", self.sup_labels),
                )
            )
        _spec_values.extend((h, self.head_cols[h.name]) for h in a.heads)
        _names = [
            spec.name for spec, _ in _spec_values
        ]  # Head.name keys the log/agg entry, the checkpoint, and
        _dups = sorted(
            {n for n in _names if _names.count(n) > 1}
        )  # the head_output lookup -> must be unique
        if _dups:
            raise ValueError(
                f"duplicate Head name(s) {_dups}: each head name must be unique (it keys logging, the persisted "
                f"checkpoint, and head_output). Note lam_phase reserves the name 'phase'."
            )
        rt_heads: list[RtHead] = [
            resolve_head(spec, vals, d, int(m.h), dev) for spec, vals in _spec_values
        ]
        if a.grad_cache and any(h.spec.reads == "hidden" for h in rt_heads):
            raise ValueError(
                "grad_cache is incompatible with a reads='hidden' auxiliary head: it reads the seed's pooled "
                "backbone hidden (a separate forward), not the cached recon. Use grad_cache=False."
            )
        for h in (
            rt_heads
        ):  # persisted heads: register on the model so they serialize + are queryable at inference
            if not h.spec.transient:
                m.add_aux_head(h.module, h.spec_dict())
        # Build the emission strategy BEFORE the optimizer: a strategy may register its OWN trainable module on the
        # model (e.g. a parallel-query bridge over a frozen backbone), and `params` below must then capture it via
        # `m.parameters()`.
        assert (
            a.emission is not None
        )  # validated in __init__; narrows the injected callable for the type checker
        objective: _EmissionObjective = a.emission(
            m, a, dev, self
        )  # emission strategy (INJECTED), built ONCE
        params = [
            p for p in m.parameters() if p.requires_grad
        ]  # includes PERSISTED aux heads + any strategy module registered above
        # TRANSIENT aux-head params are trainer-owned (not on the model), so add them explicitly — exactly as the old
        # phase head did. Parameter order does not affect AdamW's per-parameter update.
        transient_head_params = [
            p for h in rt_heads if h.spec.transient for p in h.module.parameters()
        ]
        opt = torch.optim.AdamW(params + transient_head_params, lr=a.lr)
        run = None
        if a.report_to == "wandb":
            wandb = import_module("wandb")
            run = wandb.init(project=a.wandb_project, config=_wandb_config(a))

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            """Evaluate free-running validation emissions against the target-source retrieval bank.

            Reports retrieval MRR against each row's targets and the number of distinct nearest-bank items emitted.
            """
            m.eval()
            import time as _et

            _ev_t0 = _et.perf_counter()
            veval = (
                val_idx[: a.eval_max_chains] if a.eval_max_chains else val_idx
            )  # bound eval cost to a fixed cohort
            bank_texts: list[str] = []
            bank_chain: list[int] = []
            for ci in veval:
                for t in futs[ci]:
                    bank_texts.append(t)
                    bank_chain.append(ci)
            if not bank_texts:
                m.train()
                return {"retr_mrr": 0.0, "purity": 0.0, "n_distinct": 0, "avg_emitted": 0.0}
            eval_twin = target_source.twin
            assert eval_twin is not None  # every _TargetSource builds its twin in __init__
            zb = F.normalize(
                eval_twin.emit(bank_texts).to(dev).float(), dim=-1
            )  # [Nbank, d] target-space bank
            chain_t = torch.tensor(bank_chain, device=dev)
            rr: list[float] = []
            produced: set[int] = set()
            n_emit = 0
            emit_vecs: list[torch.Tensor] = []  # emitted val latents (for stage kNN-purity)
            emit_labs: list[str] = []  # position-aligned sup label of each emission
            for i in range(0, len(veval), a.batch_size):
                chunk = veval[i : i + a.batch_size]
                # emission-strategy owns inference: state delegates to the AR rollout; a parallel-query family emits
                # in ONE pass. Both return (lat [B,Lmax,d], len [B]).
                lats, lens = objective.emit_infer([seeds[c] for c in chunk], a.max_steps)
                for kk, ci in enumerate(chunk):
                    own = chain_t == ci
                    for j in range(int(lens[kk])):
                        v = F.normalize(lats[kk, j].float(), dim=-1)
                        sims = zb @ v  # [Nbank]
                        produced.add(int(sims.argmax()))
                        n_emit += 1
                        if self.sup_labels is not None and j < len(self.sup_labels[ci]):
                            emit_vecs.append(
                                v.detach().cpu()
                            )  # emission j <- true stage of target item j
                            emit_labs.append(self.sup_labels[ci][j])
                        if bool(own.any()):  # MRR: rank of the best OWN-chain target
                            order = torch.argsort(sims, descending=True)
                            hit = torch.nonzero(own[order], as_tuple=False)
                            if hit.numel() > 0:
                                rr.append(1.0 / (int(hit[0].item()) + 1))
            m.train()
            purity = (
                selection.knn_purity(torch.stack(emit_vecs).numpy(), emit_labs)
                if len(emit_vecs) > 6
                else 0.0
            )  # stage-separation of the emitted geometry
            print(
                f"[EVAL] {_et.perf_counter() - _ev_t0:.1f}s | {len(veval)} chains, {len(bank_texts)} bank, "
                f"{n_emit} emissions",
                flush=True,
            )
            return {
                "retr_mrr": float(np.mean(rr)) if rr else 0.0,
                "purity": purity,
                "n_distinct": len(produced),
                "avg_emitted": n_emit / max(len(veval), 1),
            }

        rng_t = torch.Generator().manual_seed(a.seed)
        best = -1.0
        best_state: Optional[dict[str, Any]] = None
        metrics: dict[str, float] = {}

        # ---- preempt-resume (multi-latent): reload full state (LoRA+head+phase_head+opt+epoch+best+rng) if present ----
        start_ep = 0
        _ckpt = (Path(a.resume_dir) / "resume.pt") if a.resume_dir else None
        ck = (
            torch.load(_ckpt, map_location="cpu")
            if (_ckpt is not None and _ckpt.exists())
            else None
        )
        if ck is not None and a.run_sig is not None and ck.get("run_sig") != a.run_sig:
            print(
                f"[langset] IGNORING {_ckpt}: run_sig mismatch (ckpt={ck.get('run_sig')!r} != this run "
                f"{a.run_sig!r}) -> starting FRESH",
                flush=True,
            )
            ck = None
        if ck is not None:
            _params = dict(m.named_parameters())
            for nm, t in ck["trainable"].items():
                if nm in _params:
                    _params[nm].data.copy_(t.to(_params[nm].device, _params[nm].dtype))
            for h in rt_heads:  # PLUGGABLE AUX HEADS (langset.heads): restore phase shim + user heads for exact resume
                hsd = ck.get("aux_heads", {}).get(h.spec.name)
                if hsd is not None:
                    h.module.load_state_dict({k: v.to(dev) for k, v in hsd.items()})
            opt.load_state_dict(ck["opt"])
            for (
                stt
            ) in opt.state.values():  # optimizer state tensors must live on the model's device
                for k, v in stt.items():
                    if torch.is_tensor(v):
                        stt[k] = v.to(dev)
            start_ep = int(ck["ep"])
            best = float(ck["best"])
            best_state = ck.get("best_state")
            try:  # rng restore is best-effort (robust, not bit-exact)
                rng.bit_generator.state = ck["np_rng"]
                torch.set_rng_state(ck["torch_rng"])
                rng_t.set_state(ck["gen_rng"])
            except Exception:
                pass
            print(
                f"[langset] RESUMED (multi) from {_ckpt} -> start ep{start_ep}/{a.epochs} best={best:.3f}",
                flush=True,
            )

        def save_resume(next_ep: int) -> None:
            """Save multi-latent training and RNG state through an atomic file replacement."""
            if not a.resume_dir:
                return
            d = Path(a.resume_dir)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "resume.pt.tmp"
            payload: dict[str, Any] = {
                "trainable": {
                    nm: p.detach().cpu() for nm, p in m.named_parameters() if p.requires_grad
                },
                "opt": opt.state_dict(),
                "ep": int(next_ep),
                "best": float(best),
                "best_state": best_state,
                "np_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "gen_rng": rng_t.get_state(),
                "run_sig": a.run_sig,
            }
            if rt_heads:  # PLUGGABLE AUX HEADS (langset.heads): phase shim + user heads (transient AND persisted)
                payload["aux_heads"] = {
                    h.spec.name: {k: v.detach().cpu() for k, v in h.module.state_dict().items()}
                    for h in rt_heads
                }
            torch.save(payload, tmp)
            tmp.replace(d / "resume.pt")
            if self.on_checkpoint is not None:
                self.on_checkpoint()  # e.g. Volume.commit() -> durable across preempt

        import os as _os
        import time as _time
        from contextlib import nullcontext as _nullctx

        _prof_steps = int(
            _os.environ.get("LANGSET_PROFILE_STEPS", "0")
        )  # diagnostic: profile N steps then STOP
        _prof = None
        _gstep = 0
        _prof_t0 = 0.0
        _rfn = None
        if _prof_steps > 0:
            from torch.profiler import (
                ProfilerActivity as _PA,
            )
            from torch.profiler import (
                profile as _tp_profile,
            )
            from torch.profiler import (
                record_function as _rfn,
            )

            acts = [_PA.CPU] + ([_PA.CUDA] if torch.cuda.is_available() else [])
            _prof = _tp_profile(activities=acts, record_shapes=False, with_stack=False)
            _prof.__enter__()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _prof_t0 = _time.perf_counter()
            print(f"[PROFILE] capturing {_prof_steps} training steps then exiting ...", flush=True)

        def _rf(
            name: str,
        ) -> AbstractContextManager[Any]:  # named phase range when profiling, no-op otherwise
            if _prof is not None:
                assert _rfn is not None  # imported together with _prof under LANGSET_PROFILE_STEPS
                return _rfn(name)
            return _nullctx()

        loss_terms = a.loss_terms(a)  # Build the configured auxiliary loss terms once.

        # Rows marked for text replay rehearse next-token prediction from input_text to the first target text. The
        # replay loss uses the tied input embedding because the language-model output head is not retained.
        learn_pool = [
            i for i in range(len(seeds)) if getattr(self, "is_learn", [False] * len(seeds))[i]
        ]
        ln_doc_ids = ln_doc_mask = ln_tgt_ids = ln_tgt_mask = None

        if learn_pool:
            ln_doc_ids, ln_doc_mask = _tokenize_replay(
                tok, [seeds[i] for i in learn_pool], a.max_len, "left", dev
            )  # doc LEFT-pad
            ln_tgt_ids, ln_tgt_mask = _tokenize_replay(
                tok,
                [(self.target_texts[i][0] if self.target_texts[i] else " ") for i in learn_pool],
                _LEARN_TGT,
                "right",
                dev,
            )  # target RIGHT-pad

        if a.grad_cache:  # multi-latent GradCache: reject the configs it cannot keep exact for the cross-batch term
            assert float(m.head.drop.p) == 0.0, (
                "grad_cache requires dropout==0: the phase-1 (no_grad, full batch) and phase-2 (grad, per chunk) "
                "rollouts must be identical, but dropout randomizes them so the cached recon grads no longer match. "
                "Rebuild the model with dropout=0."
            )
            assert not target_source.wants_regularizer, (
                "grad_cache is incompatible with a cross-batch regularizer (e.g. SIGRegTarget): it is computed "
                "over the whole batch's latents and is not cached. Use the EMA twin, or grad_cache=False."
            )
            assert not any(getattr(t, "isolated_backward", False) for t in loss_terms), (
                "grad_cache does not support isolated-backward loss terms."
            )
            print(
                f"[langset] GRADCACHE ON (multi): effective batch={a.batch_size}, gc_chunk={a.gc_chunk or a.batch_size} "
                "(cross-batch InfoNCE cached EXACT; base loss accumulated per-chunk). ss_prob kept via a shared mask.",
                flush=True,
            )

        def _multi_grad_cache_step(
            se: dict[str, torch.Tensor],
            target_lat: torch.Tensor,
            valid: torch.Tensor,
            lens_l: list[int],
            bidx: list[int],
            b: int,
            lmax: int,
            ep: int,
            flat_texts: list[str],
            agg: dict[str, float],
        ) -> float:
            """Compute multi-latent cross-batch gradients with chunk-sized activation memory.

            The detached full-batch rollout uses a shared scheduled-sampling mask, evaluates cross-batch terms, and
            caches gradients with respect to emitted latents. Each chunk is then rolled out again with autograd
            enabled to propagate the cached gradients and its row-weighted base loss. Cross-batch terms retain their
            full-batch semantics; base losses are accumulated per row.
            """
            ss_prob = a.ss_prob
            assert (
                ss_prob is not None
            )  # Trainer resolves the None sentinel to a float before any step
            eff_ss = ss_prob if a.ss_warmup <= 0 else ss_prob * min(1.0, ep / a.ss_warmup)
            H = lmax if a.train_hops is None else max(0, min(int(a.train_hops), lmax))
            ss_mask = (torch.rand(b, H, device=dev) < eff_ss) if (eff_ss > 0 and H > 0) else None

            # PHASE 1: full-batch rollout (no_grad) -> recon; cross-batch loss on cached recon -> recon.grad
            with torch.no_grad():
                em_full = objective.emit(
                    se, target_lat, valid, lens_l, bidx, b, lmax, ep, ss_mask=ss_mask
                )
            recon_rg = em_full.recon.detach().requires_grad_(True)
            c = MultiStepCtx(
                trainer=self,
                args=a,
                model=m,
                dev=dev,
                bidx=bidx,
                lens_l=lens_l,
                flat_texts=flat_texts,
                valid=valid,
                target_lat=target_lat,
                recon=recon_rg,
                lmax=lmax,
                target_source=target_source,
                phase_head=phase_head,
                phase_ids=phase_ids,
            )
            cross = recon_rg.new_zeros(())
            for term in loss_terms:  # cross-batch recon-pure terms (InfoNCE etc.)
                contrib = term.contribute(c)
                if contrib is not None:
                    _k, _raw, _w = contrib
                    cross = cross + _w * _raw
                    agg[_k] = agg.get(_k, 0.0) + float(_raw.detach())
            for h in (
                rt_heads
            ):  # PLUGGABLE AUX HEADS (recon read site only; "hidden" is rejected under grad_cache).
                # A recon head is a pure fn of the cached recon, so it belongs in the phase-1 cross accumulation.
                raw_h = h.loss_on(
                    h.module(recon_rg[valid].float()), h._flat_recon_targets(bidx, lens_l), dev
                )
                if raw_h is not None:
                    cross = cross + h.spec.eff_weight(ep) * raw_h
                    agg[h.spec.loss_key] = agg.get(h.spec.loss_key, 0.0) + float(raw_h.detach())
            opt.zero_grad()
            if cross.requires_grad:
                cross.backward()  # fills recon_rg.grad (+ any head params: phase_head, slot heads)
            gcache = (
                recon_rg.grad
            )  # [b, lmax, d] cached full-batch cross-batch grad (None if no cross term fired)

            # PHASE 2: re-roll each chunk WITH grad; base loss (row-weighted) + inject cached recon-grads
            ch = a.gc_chunk or a.batch_size
            base_tot = 0.0
            for j0 in range(0, b, ch):
                rows = list(range(j0, min(j0 + ch, b)))
                ri = torch.tensor(rows, device=dev)
                se_c = {k: v[ri] for k, v in se.items()}
                tgt_c, val_c = target_lat[ri], valid[ri]
                lens_c = [lens_l[r] for r in rows]
                bidx_c = [bidx[r] for r in rows]
                mask_c = ss_mask[ri] if ss_mask is not None else None
                em_c = objective.emit(
                    se_c, tgt_c, val_c, lens_c, bidx_c, len(rows), lmax, ep, ss_mask=mask_c
                )
                w = (
                    len(rows) / b
                )  # row-weight so the per-chunk base means sum ~ the batch mean (pragmatic)
                (em_c.base_loss * w).backward(retain_graph=gcache is not None)
                if (
                    gcache is not None
                ):  # inject the cached cross-batch grad through THIS chunk's recon
                    torch.autograd.backward(em_c.recon, gcache[ri])
                base_tot += float(em_c.base_loss.detach()) * w
                for (
                    _k,
                    _v,
                ) in (
                    em_c.logs.items()
                ):  # whatever THIS objective logs (objective-defined components)
                    agg[_k] = agg.get(_k, 0.0) + float(_v.detach()) * w
            opt.step()
            target_source.update()
            return base_tot + float(cross.detach())

        for ep in range(start_ep, a.epochs):
            m.train()
            order = a.epoch_order(tr_idx, rng_t, a, seeds)  # epoch ordering strategy (INJECTED)
            if a.max_steps_per_epoch:  # SMALL epochs: cap steps so each <= ~30min (natural save pt)
                order = order[: a.max_steps_per_epoch * a.batch_size]
            tot = 0.0
            nb = 0
            agg: dict[
                str, float
            ] = {}  # accumulates whatever the emission objective logs (its keys, not another objective's)
            for i in range(0, len(order), a.batch_size):
                if (
                    learn_pool and float(rng.random()) < a.learn_ratio
                ):  # REPLAY step: rehearse text, own opt.step
                    lp = torch.as_tensor(
                        rng.choice(
                            len(learn_pool), size=min(a.batch_size, len(learn_pool)), replace=False
                        ),
                        device=dev,
                        dtype=torch.long,
                    )
                    assert ln_doc_ids is not None and ln_doc_mask is not None
                    assert ln_tgt_ids is not None and ln_tgt_mask is not None
                    opt.zero_grad()
                    lloss = learn_loss(
                        LearnLossContext(
                            model=m,
                            args=a,
                            pos=lp,
                            ln_doc_ids=ln_doc_ids,
                            ln_doc_mask=ln_doc_mask,
                            ln_tgt_ids=ln_tgt_ids,
                            ln_tgt_mask=ln_tgt_mask,
                        )
                    ).to_tensor()
                    lloss.backward()
                    opt.step()
                    agg["learn_loss"] = agg.get("learn_loss", 0.0) + float(lloss.detach())
                bidx = [tr_idx[k] for k in order[i : i + a.batch_size]]
                se = tok(
                    [seed_texts[k] for k in bidx],
                    padding=True,
                    truncation=True,
                    max_length=a.max_len,
                    padding_side="left",
                    return_tensors="pt",
                ).to(dev)  # left-pad: hid[s_len-1] = last real token
                ent_lists = [list(futs[k]) for k in bidx]
                flat_texts = [txt for lst in ent_lists for txt in lst]
                with _rf("ema_encode"):
                    flat_tgt = target_source.encode(flat_texts)  # [ΣL, d] stop-grad EMA targets
                b = len(bidx)
                # target↔slot shaping now owned by the emission strategy (was inline here; default = positional TF)
                target_lat, valid, lens_l, lmax = objective.build_targets(
                    ent_lists, flat_tgt, d, dev
                )
                if (
                    a.grad_cache
                ):  # two-phase: cross-batch InfoNCE cached exact, base loss accumulated per chunk
                    tot += _multi_grad_cache_step(
                        se, target_lat, valid, lens_l, bidx, b, lmax, ep, flat_texts, agg
                    )
                    nb += 1
                    continue
                with _rf("rollout"):
                    em = objective.emit(se, target_lat, valid, lens_l, bidx, b, lmax, ep)
                recon = em.recon
                loss = em.base_loss  # base objective (emission + STOP + recon); separation below
                c = MultiStepCtx(  # everything the aux terms read this step
                    trainer=self,
                    args=a,
                    model=m,
                    dev=dev,
                    bidx=bidx,
                    lens_l=lens_l,
                    flat_texts=flat_texts,
                    valid=valid,
                    target_lat=target_lat,
                    recon=recon,
                    lmax=lmax,
                    target_source=target_source,
                    phase_head=phase_head,
                    phase_ids=phase_ids,
                )
                for (
                    term
                ) in loss_terms:  # aux separation/shaping terms (fixed order; each self-skips)
                    if term.isolated_backward:  # isolated terms run AFTER the main backward (below)
                        continue
                    contrib = term.contribute(c)
                    if contrib is not None:
                        _k, _raw, _w = contrib
                        loss = loss + _w * _raw
                        agg[_k] = agg.get(_k, 0.0) + float(_raw.detach())
                # Apply resolved auxiliary heads after the general loss terms and before target regularization.
                for h in rt_heads:
                    if (
                        h.spec.reads == "recon"
                    ):  # per emitted latent (grad shapes the emission geometry, like the phase head)
                        pred = h.module(recon[valid].float())
                        flat = h._flat_recon_targets(bidx, lens_l)
                    else:  # "hidden": pooled per-sequence seed hidden — a separate backbone forward whose grad shapes it
                        pred = h.module(
                            m.seed_hidden(se["input_ids"], se["attention_mask"]).float()
                        )
                        flat = [h.values[k] for k in bidx]
                    raw = h.loss_on(pred, flat, dev)
                    if raw is not None:
                        loss = loss + h.spec.eff_weight(ep) * raw
                        agg[h.spec.loss_key] = agg.get(h.spec.loss_key, 0.0) + float(raw.detach())
                if (
                    target_source.wants_regularizer and int(valid.sum()) > 1
                ):  # e.g. SIGReg anti-collapse penalty
                    z_pred, z_tgt = objective.z_for_reg(em, target_lat, valid, lmax)
                    loss_sig = target_source.regularizer(z_pred, z_tgt)
                    assert (
                        loss_sig is not None
                    )  # wants_regularizer=True sources return a tensor here
                    loss = loss + a.sigreg_lambda * loss_sig
                    agg["loss_sig"] = agg.get("loss_sig", 0.0) + float(loss_sig.detach())
                opt.zero_grad()
                with _rf("backward"):
                    loss.backward()  # frees the main graph before any isolated term runs
                for term in loss_terms:  # isolated terms: own forward+backward, grads ACCUMULATE
                    if not term.isolated_backward:
                        continue
                    contrib = term.contribute(c)
                    if contrib is not None:
                        _k, _raw, _w = contrib
                        (_w * _raw).backward()
                        agg[_k] = agg.get(_k, 0.0) + float(_raw.detach())
                with _rf("opt_ema"):
                    opt.step()
                    target_source.update()  # EMA twin tracks the online model
                tot += float(loss.detach())
                nb += 1
                for (
                    _k,
                    _v,
                ) in em.logs.items():  # emission-defined keys (objective-defined components)
                    agg[_k] = agg.get(_k, 0.0) + float(_v.detach())
                if _prof is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _gstep += 1
                    if _gstep >= _prof_steps:
                        _wall = _time.perf_counter() - _prof_t0
                        _prof.__exit__(None, None, None)
                        _ka = _prof.key_averages()

                        def _sc(e: object) -> float:  # self-CUDA us across torch versions
                            return float(
                                getattr(e, "self_cuda_time_total", 0)
                                or getattr(e, "self_device_time_total", 0)
                                or 0
                            )

                        _cuda_s = sum(_sc(e) for e in _ka) / 1e6
                        _busy = 100.0 * _cuda_s / _wall if _wall > 0 else 0.0
                        _phase = {
                            n: sum(_sc(e) for e in _ka if e.key == n) / 1e6
                            for n in ("ema_encode", "rollout", "backward", "opt_ema")
                        }
                        print(f"[PROFILE] ===== SUMMARY over {_prof_steps} steps =====", flush=True)
                        print(
                            f"[PROFILE] wall={_wall:.1f}s  {_wall / _prof_steps:.2f}s/step  |  GPU-busy={_cuda_s:.1f}s = "
                            f"{_busy:.0f}% of wall  ({'OVERHEAD-BOUND' if _busy < 65 else 'COMPUTE-BOUND'})",
                            flush=True,
                        )
                        _named = sum(_phase.values())
                        for _n, _v in _phase.items():
                            print(
                                f"[PROFILE]   {_n:11s} GPU {_v:5.1f}s = {100 * _v / _cuda_s if _cuda_s else 0:4.0f}% of GPU-busy",
                                flush=True,
                            )
                        print(
                            f"[PROFILE]   residual(CPU/py) ~{max(0.0, _wall - _named):.1f}s of wall not in named GPU phases",
                            flush=True,
                        )
                        for _sk in (
                            "self_cuda_time_total",
                            "self_device_time_total",
                            "cuda_time_total",
                            "cpu_time_total",
                        ):
                            try:
                                print(
                                    f"[PROFILE] === sort_by={_sk} ===\n"
                                    + _ka.table(sort_by=_sk, row_limit=30),
                                    flush=True,
                                )
                            except Exception:
                                continue
                        return None  # ty: ignore[invalid-return-type]  # diagnostic profiling run: bails early (LangSetModel path unaffected)

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
            if ep % a.eval_every and not (
                ep == a.epochs - 1 and getattr(a.selector, "needs_final_epoch", False)
            ):
                continue
            metrics = evaluate()
            row = {"loss": tot / max(nb, 1), **{kk: vv / max(nb, 1) for kk, vv in agg.items()}}
            mrr, pur = metrics["retr_mrr"], metrics.get("purity", 0.0)
            sel = a.selector(a.select, mrr, pur, ep)  # checkpoint-selection strategy (INJECTED)
            if a.verbose:
                hn_s = f" hn={row['loss_hard_neg']:.3f}" if "loss_hard_neg" in row else ""
                sup_s = f" sup={row['loss_sup']:.3f}" if "loss_sup" in row else ""
                ph_s = f" phase={row['loss_phase']:.3f}" if "loss_phase" in row else ""
                pur_s = f" purity={pur:.3f}" if self.sup_labels is not None else ""
                # the emission's own terms, in a stable order; unchanged for state (stop/dims/recon), and a
                # different objective prints ITS terms rather than KeyError-ing on another objective's.
                _names = {
                    "loss_stop": "stop",
                    "loss_dims": "dims",
                    "recon_loss": "recon",
                    "loss_code": "code",
                    "loss_state": "state",
                    "loss_concept": "concept",
                    "emit_cos": "emit_cos",
                }
                base_s = " ".join(f"{v}={row[k]:.3f}" for k, v in _names.items() if k in row)
                # per-facet concept losses are discovered from the data, so their keys are not known here
                facet_s = " ".join(
                    f"{k[2:]}={row[k]:.2f}" for k in sorted(row) if k.startswith("c_")
                )
                base_s = f"{base_s} {facet_s}".strip() if facet_s else base_s
                print(
                    f"ep{ep:02d} loss={row['loss']:.3f} {base_s}"
                    f"{hn_s}{sup_s}{ph_s} | retr_mrr={mrr:.3f}{pur_s} "
                    f"[sel:{a.select}={sel:.3f}] distinct={metrics['n_distinct']} "
                    f"avg_emit={metrics['avg_emitted']:.2f}",
                    flush=True,
                )
            if run is not None:
                run.log(
                    {
                        **row,
                        "epoch": ep,
                        "eval/retr_mrr": mrr,
                        "eval/purity": pur,
                        "eval/sel": sel,
                        "eval/n_distinct": metrics["n_distinct"],
                        "eval/avg_emitted": metrics["avg_emitted"],
                    }
                )
            if sel > best:
                best = sel
                best_state = _snapshot_best(
                    m
                )  # LoRA-only if pretrained, FULL backbone if random-init
                Path(a.output_dir).mkdir(parents=True, exist_ok=True)
                m.save_pretrained(a.output_dir)  # persist best-so-far (live checkpoint)
                if self.on_checkpoint is not None:
                    self.on_checkpoint()
                if a.verbose:
                    print(
                        f"        <- best {a.select}={best:.3f}, saved to {a.output_dir}",
                        flush=True,
                    )
            save_resume(
                ep + 1
            )  # epoch boundary: durable full-state checkpoint so a preempt resumes HERE, not ep0

        if best_state is not None:  # restore best into memory (matches single-latent)
            _restore_best(m, best_state)
        m.eval()
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        m.save_pretrained(a.output_dir)
        if run is not None:
            run.finish()
        if a.verbose:
            print(
                f"[langset] done (multi-latent). best {a.select}={best:.3f} -> {a.output_dir}",
                flush=True,
            )
        return m
