"""Configurable auxiliary supervision for multi-latent training.

A `Head` attaches a linear prediction layer to one of two representations:

- `reads="recon"` applies the head to each emitted latent.
- `reads="hidden"` applies the head to the pooled backbone hidden state.

Each head selects a dataset column through `target` and uses classification (`loss="ce"`), regression
(`loss="mse"`), or a custom loss callable. `weight` scales the loss, and `warmup` can ramp that weight during early
epochs.

Transient heads participate in training but are not saved. Persisted heads (`transient=False`) are stored with
the model and can be queried through `LangSetModel.head_output()` after loading the checkpoint.

`Head.phase_shim()` provides compatibility with the `lam_phase` training option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union, cast

import torch
import torch.nn.functional as F

# A custom loss receives predictions shaped `[N, dim]` and targets shaped `[N]` or `[N, k]`, and returns a scalar.
HeadLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_READS = ("recon", "hidden")
_BUILTIN_LOSSES = ("ce", "mse")
# Labels treated as missing supervision.
_MISSING = ("", "unknown", "none", "nan")


@dataclass
class Head:
    """Configuration for an auxiliary supervised head.

    Args:
        name: Unique name used for logging and persisted-head lookup.
        reads: Representation to read. `"recon"` predicts once per emitted latent; `"hidden"` predicts once per
            input sequence.
        target: Dataset column containing labels or numeric target values.
        loss: `"ce"`, `"mse"`, or a callable accepting `(predictions, targets)` and returning a scalar tensor.
        dim: Output width. Classification heads infer this from their labels; MSE and custom-loss heads must
            specify it.
        transient: If true, train the head without saving it in the model. If false, persist it for use with
            `LangSetModel.head_output()`.
        weight: Multiplier applied to this head's loss.
        warmup: Number of epochs over which to ramp the effective weight from zero to `weight`.
    """

    name: str
    reads: str
    target: str
    loss: Union[str, HeadLoss]
    dim: Optional[int] = None  # Infer the output width for CE; required for MSE and custom losses.
    transient: bool = True  # Persist the head only when false.
    weight: float = 1.0
    warmup: int = 0  # Zero applies the full weight from epoch zero.

    # Set during validation when `loss` is a custom callable.
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
        """Return this head's effective loss weight for epoch `ep`.

        With a positive `warmup`, the weight increases linearly from zero at epoch zero to `weight` at epoch
        `warmup`.
        """
        if self.warmup <= 0:
            return self.weight
        return self.weight * min(1.0, ep / self.warmup)

    @classmethod
    def phase_shim(cls, sup_field: str, weight: float) -> "Head":
        """Create the transient classification head used by `lam_phase`.

        The returned head reads emitted latents, predicts labels from `sup_field`, and scales its cross-entropy
        loss by `weight`.
        """
        return cls(
            name="phase",
            reads="recon",
            target=sup_field,
            loss="ce",
            dim=None,  # Infer the class count from labels.
            transient=True,
            weight=weight,
        )


def build_ce_classes(rows: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    """Build a deterministic class vocabulary for categorical targets.

    Empty labels and labels matching the module's missing-value tokens are excluded case-insensitively.

    Returns:
        A sorted class list and a mapping from each class label to its integer ID.
    """
    classes = sorted({lb for row in rows for lb in row if lb and lb.lower() not in _MISSING})
    return classes, {c: i for i, c in enumerate(classes)}


@dataclass
class RtHead:
    """Resolved runtime state for a `Head`.

    This trainer-internal object combines the user specification, its linear module, optional classification
    metadata, and the source target values.
    """

    spec: Head
    module: torch.nn.Linear
    values: list[object]  # One list per row for `recon`; one scalar or vector per row for `hidden`.
    classes: Optional[list[str]] = None  # CE index-to-label mapping persisted with the head.
    class_map: Optional[dict[str, int]] = None  # CE label-to-index mapping used during training.

    @property
    def out_dim(self) -> int:
        return int(self.module.out_features)

    @property
    def in_dim(self) -> int:
        return int(self.module.in_features)

    def spec_dict(self) -> dict[str, object]:
        """Return the metadata needed to save and reconstruct a persisted head."""
        loss = self.spec.loss if isinstance(self.spec.loss, str) else "custom"
        return {
            "name": self.spec.name,
            "reads": self.spec.reads,
            "loss": loss,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "classes": self.classes,
        }

    # Per-step loss helpers.
    def _flat_recon_targets(self, bidx: list[int], lens_l: list[int]) -> list[object]:
        """Flatten selected row targets in the order used by `recon[valid]`.

        Targets missing beyond the end of a row's target list are represented by `None`.
        """
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
        """Compute this head's loss for flattened batch targets.

        For classification, missing targets are encoded as `-100` and ignored by cross-entropy. For MSE, rows
        containing non-finite targets are excluded and `None` is returned if no rows remain. Custom losses receive
        the complete prediction and target tensors, with missing or unparseable targets represented as `NaN`, and
        are responsible for their own masking.

        Args:
            pred: Predictions shaped `[N, out_dim]`.
            flat: One target value per prediction.
            dev: Device on which to construct the target tensor.
        """
        if self.spec.is_ce:
            return F.cross_entropy(pred, self._ce_ids(flat, dev), ignore_index=-100)
        tgt = _as_float_target(flat, dev)  # [N] or [N, k]; NaN = missing
        if self.spec._custom:
            return cast_loss(self.spec.loss)(pred, tgt)  # Custom losses handle their own masking.
        tgt2 = tgt.unsqueeze(1) if tgt.dim() == 1 else tgt
        keep = torch.isfinite(tgt2).all(dim=-1)
        if not bool(keep.any()):
            return None
        return F.mse_loss(pred[keep], tgt2[keep])


def cast_loss(loss: Union[str, HeadLoss]) -> HeadLoss:
    """Return `loss` narrowed to the custom callable type."""
    assert callable(loss)
    return cast(HeadLoss, loss)


def _as_float_target(flat: list[object], dev: torch.device) -> torch.Tensor:
    """Convert scalar or vector targets to a float tensor.

    Missing and unparseable values become `NaN`. Scalar inputs produce shape `[N]`; consistently sized list or
    tuple inputs produce shape `[N, k]`.
    """

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
    """Build the runtime module and metadata for a `Head`.

    For classification heads, classes are inferred from `values` and `dim`, when provided, must match the inferred
    class count. Other loss types use the explicitly configured `dim`.

    Args:
        spec: Head configuration to resolve.
        values: Per-row target values.
        latent_dim: Input width for `reads="recon"`.
        hidden_dim: Input width for `reads="hidden"`.
        dev: Device on which to create the linear module.

    Returns:
        The resolved runtime head.
    """
    in_dim = latent_dim if spec.reads == "recon" else hidden_dim
    classes: Optional[list[str]] = None
    class_map: Optional[dict[str, int]] = None
    out_dim = spec.dim
    if spec.is_ce:
        rows = [[str(e) for e in v] if isinstance(v, (list, tuple)) else [str(v)] for v in values]
        classes, class_map = build_ce_classes(rows)
        n_cls = len(classes)
        if n_cls == 0:  # Avoid constructing a classifier with zero output classes.
            raise ValueError(
                f"CE head {spec.name!r} has no classes — all `target` labels are missing "
                f"(empty/unknown/none/nan). Supply real class labels, or use a non-CE loss."
            )
        if (
            spec.dim is not None and spec.dim != n_cls
        ):  # CE width must equal the inferred class count.
            raise ValueError(
                f"CE head {spec.name!r}: dim={spec.dim} conflicts with the {n_cls} inferred classes "
                f"({classes}). Leave dim=None for a CE head — its width is inferred from the labels."
            )
        out_dim = n_cls
    assert out_dim is not None  # __post_init__ guarantees dim is set for non-CE losses
    module = torch.nn.Linear(in_dim, out_dim).to(dev)
    return RtHead(spec, module, values, classes, class_map)
