"""Pluggable AUXILIARY-SUPERVISED HEADS — the generalization of the hard-coded phase head.

The multi-latent trainer historically carried ONE domain concept baked into the library: a transient
`hidden->phase` CE classifier (`lam_phase`) that shaped the emitted geometry to be phase-separable. Baking
"phase" (and a would-be "value") into a domain-AGNOSTIC latent-prediction library is wrong: phase, value, time,
ordinal-rank are all the SAME mechanism — a small supervised head hung off the model that (a) injects a shaping
gradient into the backbone and/or (b) is itself a queryable readout. This module makes that mechanism a plug.

A `Head` declares the four axes the phase head implicitly hard-coded:
  1. READ SITE (`reads`):  "recon"  = one prediction per EMITTED latent, in the token stream (what the phase head
                                       reads — grad flows into the FSQ up/down-proj + LoRA, shaping the geometry).
                            "hidden" = the pooled/final backbone hidden, ONE prediction per sequence (a value/time
                                       readout of the current state; grad flows into the backbone/LoRA).
  2. TARGET (`target`):    a per-item dataset column supplying the supervision (like the existing `sup_field`).
  3. LOSS (`loss`):        "ce" (classification) | "mse" (regression) | a CUSTOM CALLABLE (pred, target) -> scalar
                            (the seam a censored survival/hazard loss plugs into).
  4. LIFECYCLE (`transient`): transient (NOT saved; its only job is the shaping gradient — eval re-fits its own
                            probe, like today's phase head) OR PERSISTED (saved with the checkpoint AND reloadable
                            at inference via `LangSetModel.head_output(name, ...)` — the value/time head you query).
Plus `weight` (the `lam`) and optional `warmup` (ramp the weight 0->w over the first N epochs — a hidden->scalar
head on garbage early emissions destabilizes, the same reason the phase head can stick at chance; mirrors
`ss_warmup`).

`lam_phase` is now a SHIM over this plug: `Head.phase_shim(sup_field, weight)` reconstructs the exact transient
recon+CE head, so the historical path is byte-identical (see the multi-latent golden, which trains with
lam_phase>0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union, cast

import torch
import torch.nn.functional as F

# A custom loss reads (pred [N, dim], target [N] or [N, k]) and returns a scalar. The target packs whatever the
# loss needs — e.g. a censored survival loss reads target[:, 0]=Δt, target[:, 1]=event-observed (dim can be 1).
HeadLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_READS = ("recon", "hidden")
_BUILTIN_LOSSES = ("ce", "mse")
# label tokens treated as "no supervision" for this item (mirrors the phase-head / emb_slots filters exactly).
_MISSING = ("", "unknown", "none", "nan")


@dataclass
class Head:
    """One auxiliary supervised head. See the module docstring for the four axes. Instantiate and pass a list via
    `TrainingArguments(heads=[Head(...), ...])`; each is built once and trained in every step alongside the
    emission objective. Domain-agnostic: the library never names "phase" or "value" — you do, per Head."""

    name: str  # log key (`loss_{name}`) and, when persisted, the inference lookup key for `head_output(name, ...)`
    reads: str  # "recon" (per emitted latent) | "hidden" (pooled per-sequence backbone hidden)
    target: str  # per-item dataset column supplying the labels/values (like `sup_field`)
    loss: Union[str, HeadLoss]  # "ce" | "mse" | a custom callable(pred, target) -> scalar
    dim: Optional[int] = (
        None  # output width; None => infer for "ce" (#classes), else REQUIRED (mse/custom)
    )
    transient: bool = (
        True  # True: not saved (shaping-gradient only). False: PERSISTED + queryable at inference.
    )
    weight: float = 1.0  # the `lam` — this head's loss weight in the total
    warmup: int = (
        0  # ramp `weight` 0->weight linearly over the first N epochs (0 = full weight from ep0)
    )

    # populated by resolution when `loss` is a custom callable, so the trainer can special-case target parsing
    # (a custom loss may want a per-item VECTOR target, e.g. (Δt, censored)). Not user-set.
    _custom: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reads not in _READS:
            raise ValueError(
                f"Head({self.name!r}).reads must be one of {_READS}; got {self.reads!r}"
            )
        if isinstance(self.loss, str):
            if self.loss not in _BUILTIN_LOSSES:
                raise ValueError(
                    f"Head({self.name!r}).loss str must be one of {_BUILTIN_LOSSES} or a callable; "
                    f"got {self.loss!r}"
                )
        elif callable(self.loss):
            self._custom = True
        else:
            raise ValueError(
                f"Head({self.name!r}).loss must be a str or a callable; got {self.loss!r}"
            )
        if self.loss != "ce" and self.dim is None:
            raise ValueError(
                f"Head({self.name!r}): dim is required for a {'custom' if self._custom else self.loss} "
                f"loss (only 'ce' can infer #classes)."
            )
        if self.weight < 0:
            raise ValueError(f"Head({self.name!r}).weight must be >= 0; got {self.weight}")
        if self.warmup < 0:
            raise ValueError(f"Head({self.name!r}).warmup must be >= 0; got {self.warmup}")

    @property
    def is_ce(self) -> bool:
        return self.loss == "ce"

    @property
    def loss_key(self) -> str:
        return f"loss_{self.name}"

    def eff_weight(self, ep: int) -> float:
        """The warmup-ramped weight for epoch `ep` (mirrors ss_warmup: linear 0->weight over the first N epochs)."""
        if self.warmup <= 0:
            return self.weight
        return self.weight * min(1.0, ep / self.warmup)

    @classmethod
    def phase_shim(cls, sup_field: str, weight: float) -> "Head":
        """Reconstruct the historical `lam_phase` head as a Head: a TRANSIENT recon+CE classifier over the
        `sup_field` per-item stage labels. The resolver builds its module and class map identically to the old
        inline phase head, so the training path stays byte-identical (guarded by the multi-latent golden)."""
        return cls(
            name="phase",
            reads="recon",
            target=sup_field,
            loss="ce",
            dim=None,  # inferred #classes — same sorted/filtered class set as the old phase head
            transient=True,
            weight=weight,
        )


def build_ce_classes(rows: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    """Class list + label->id map for a CE head, built EXACTLY like the old phase head: the sorted set of non-empty,
    non-missing labels across every item of every row. Items whose label is missing map to -100 (ignored) at loss
    time. Deterministic (sorted) so the head width and class indices are reproducible."""
    classes = sorted({lb for row in rows for lb in row if lb and lb.lower() not in _MISSING})
    return classes, {c: i for i, c in enumerate(classes)}


@dataclass
class RtHead:
    """A RESOLVED head: the user `Head` spec + its built `nn.Linear` module + (for CE) its class map + a reference
    to the per-row target values. Trainer-internal — one per `Head`, built once in `_train_multi`."""

    spec: Head
    module: torch.nn.Linear
    values: list[
        object
    ]  # per-row target values: a LIST per row for reads="recon", a scalar for reads="hidden"
    classes: Optional[list[str]] = (
        None  # CE only: index->label (also persisted so `head_output` can name argmaxes)
    )
    class_map: Optional[dict[str, int]] = None  # CE only: label->index

    @property
    def out_dim(self) -> int:
        return int(self.module.out_features)

    @property
    def in_dim(self) -> int:
        return int(self.module.in_features)

    def spec_dict(self) -> dict[str, object]:
        """Serializable metadata for a PERSISTED head — written to config.json so `LangSetModel.load` can rebuild
        the module and `head_output` knows how to read it back."""
        loss = self.spec.loss if isinstance(self.spec.loss, str) else "custom"
        return {
            "name": self.spec.name,
            "reads": self.spec.reads,
            "loss": loss,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "classes": self.classes,
        }

    # --- per-step loss ------------------------------------------------------------------------------------------
    def _flat_recon_targets(self, bidx: list[int], lens_l: list[int]) -> list[object]:
        """Per-emitted-latent target values in `recon[valid]` row-major order (row0's items, then row1's, ...),
        aligned exactly like the old phase head's `pf`: item j of row k is `values[k][j]`, or None past the row's
        list length (-> ignored)."""
        out: list[object] = []
        for r, k in enumerate(bidx):
            row = self.values[k]
            rowlist = list(row) if isinstance(row, (list, tuple)) else [row]
            for j in range(lens_l[r]):
                out.append(rowlist[j] if j < len(rowlist) else None)
        return out

    def _ce_ids(self, flat: list[object], dev: torch.device) -> torch.Tensor:
        assert self.class_map is not None
        return torch.tensor(
            [self.class_map.get(str(x), -100) if x is not None else -100 for x in flat], device=dev
        )

    def loss_on(
        self, pred: torch.Tensor, flat: list[object], dev: torch.device
    ) -> Optional[torch.Tensor]:
        """This head's loss for predictions `pred` [N, out_dim] against the flat per-prediction targets. Returns
        None when nothing in the batch supervises this head (all items missing) so the caller can skip it.
        CE ignores missing items (-100); MSE masks to finite targets; a CUSTOM loss gets (pred, target) with
        missing packed as NaN (its job to handle censoring). The CE branch is byte-identical to the old phase term."""
        if self.spec.is_ce:
            return F.cross_entropy(pred, self._ce_ids(flat, dev), ignore_index=-100)
        tgt = _as_float_target(flat, dev)  # [N] or [N, k]; NaN = missing
        if self.spec._custom:
            return cast_loss(self.spec.loss)(
                pred, tgt
            )  # custom loss owns masking/censoring over the full batch
        tgt2 = tgt.unsqueeze(1) if tgt.dim() == 1 else tgt
        keep = torch.isfinite(tgt2).all(dim=-1)
        if not bool(keep.any()):
            return None
        return F.mse_loss(pred[keep], tgt2[keep])


def cast_loss(loss: Union[str, HeadLoss]) -> HeadLoss:
    """Narrow a Head.loss to the callable branch (only called when _custom is True, i.e. loss is a callable)."""
    assert callable(loss)
    return cast(HeadLoss, loss)


def _as_float_target(flat: list[object], dev: torch.device) -> torch.Tensor:
    """Parse per-item target values to a float tensor [N] (scalars) or [N, k] (list/tuple items, e.g. a survival
    (Δt, event) pair). Missing / unparseable values become NaN so downstream masking (MSE) or the custom loss can
    drop them."""

    def _f(x: object) -> float:
        if x is None:
            return float("nan")
        s = str(x)
        if s == "" or s.lower() in _MISSING or s.lower() == "na":
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return float("nan")

    vals: list[object] = [
        [_f(e) for e in x] if isinstance(x, (list, tuple)) else _f(x) for x in flat
    ]
    return torch.tensor(vals, dtype=torch.float32, device=dev)


def resolve_head(
    spec: Head, values: list[object], latent_dim: int, hidden_dim: int, dev: torch.device
) -> RtHead:
    """Build the head's `nn.Linear` (its RNG draw happens HERE — for the phase shim this is at the exact point the
    old inline phase head was built, keeping init byte-identical) + its CE class map. `values` is the per-row target
    column (a list per row for reads="recon", a scalar per row for reads="hidden")."""
    in_dim = latent_dim if spec.reads == "recon" else hidden_dim
    classes: Optional[list[str]] = None
    class_map: Optional[dict[str, int]] = None
    out_dim = spec.dim
    if spec.is_ce:
        rows = [[str(e) for e in v] if isinstance(v, (list, tuple)) else [str(v)] for v in values]
        classes, class_map = build_ce_classes(rows)
        n_cls = len(classes)
        if n_cls == 0:  # else F.cross_entropy fails opaquely on a [N, 0] logit
            raise ValueError(
                f"CE head {spec.name!r} has no classes — all `target` labels are missing "
                f"(empty/unknown/none/nan). Supply real class labels, or use a non-CE loss."
            )
        if (
            spec.dim is not None and spec.dim != n_cls
        ):  # a CE head's width IS the class count — don't let it drift
            raise ValueError(
                f"CE head {spec.name!r}: dim={spec.dim} conflicts with the {n_cls} inferred classes "
                f"({classes}). Leave dim=None for a CE head — its width is inferred from the labels."
            )
        out_dim = n_cls  # CE width is always the inferred class count (phase shim keeps dim=None -> byte-identical)
    assert out_dim is not None  # __post_init__ guarantees dim is set for non-CE losses
    module = torch.nn.Linear(in_dim, out_dim).to(dev)
    return RtHead(spec, module, values, classes, class_map)
