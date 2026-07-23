"""The generalized emission-strategy protocol — the abstraction target for EMISSION_PROTOCOL.md.

This file is a SPEC (the fuller, typed interface we are converging toward), separate from the concrete wiring in
this PR. It describes the interface a non-autoregressive, continuous, parallel-query emission family (a
DETR/Perceiver "query-bridge" over a frozen encoder) implements so it slots into the multi-latent trainer and
inherits GradCache / kv_cache / `_TargetSource` / `_LossTerm` / checkpoint selection for free. The first such
family, `QueryBridgeEmission` (bridge_emission.py), already ships in this PR against the EXISTING
`_EmissionObjective` seam (strategies.py); this spec captures the cleaner interface the seam should grow into next.

It generalizes today's `_EmissionObjective` (strategies.py:355) by moving three AR-FSQ assumptions out of the
interface:
  1. the emission FORWARD (train + infer) belongs to the strategy, not the model's hardcoded `rollout*`;
  2. length/validity are OUTPUTS the strategy decides (AR: STOP token; parallel: a validity head), not
     trainer-provided teacher-forcing inputs;
  3. target↔emission ASSIGNMENT is the strategy's business (AR: positional teacher-forcing; DETR: Hungarian
     matching), so `emit_train` takes the target SET, not a pre-aligned `target_lat`/`valid`/`lens`.

`FSQObjective` is the reference impl (it already satisfies most of this); `QueryBridgeEmission` is the future
second impl (see ii3 `bridge_lightning.py`). Persisting a strategy's own module reuses the 0.13.0 aux-head plug
(`modeling.py add_aux_head`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

if TYPE_CHECKING:  # keep this file import-light (no cycle with strategies/trainer/modeling)
    import torch
    from torch import nn

    from langset.strategies import EmissionOut


class SeedBatch(Protocol):
    """Tokenized input the emission reads. `input_ids`/`attention_mask` are on-device. `substrate` is an
    OPTIONAL cache of the frozen-encoder per-token hidden states — the kv_cache seam already forwards the prompt
    once, and a frozen-encoder bridge wants exactly that reuse."""

    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"
    substrate: "torch.Tensor | None"


class TargetSet(Protocol):
    """The per-row targets the `_TargetSource` produced for this batch: the target texts (row-major) and their
    encoded latents. The strategy decides how emissions bind to these (positional vs matched)."""

    texts: list[list[str]]  # per row: the row's target item texts
    latents: "torch.Tensor"  # [B, L, d] encoded targets (L = max items across the batch's rows)
    lens: list[int]  # per row: item count


@runtime_checkable
class EmissionStrategy(Protocol):
    """What the trainer would call instead of reaching into `rollout_train_codebook` / `rollout`.

    Contrast with `_EmissionObjective.emit(se, target_lat, valid, lens_l, ...)` (strategies.py:369): there the
    trainer pre-builds `valid`/`lens`/`lmax`/`target_lat` from the targets (trainer.py:1851) and the model owns
    the AR forward. Here the strategy receives the raw `SeedBatch` + `TargetSet` and returns everything — the
    emitted latents, the validity/lengths IT chose, and its base loss (with any matching done internally)."""

    #: True for a discrete/decodable codebook family (FSQ); False for a continuous-vector family (query-bridge).
    #: The free-run/eval path reads this to pick the emission head, same as `_EmissionObjective.codebook`.
    codebook: bool

    def parameters(self) -> Iterable["nn.Parameter"]:
        """Trainable params the strategy OWNS beyond the backbone (FSQ head; or bridge queries + cross-attn
        decoder + validity head). The trainer adds these to the optimizer; persist them via `add_aux_head`
        (`modeling.py:735`) so `save_pretrained` serializes them. Empty if the strategy only reuses `model.head`."""
        ...

    def emit_train(self, seeds: "SeedBatch", targets: "TargetSet", ep: int) -> "EmissionOut":
        """One training emission + base loss. The strategy runs its OWN forward, decides validity/lengths, and
        does its OWN target assignment:
          - FSQObjective: `rollout_train_codebook` (hops + feedback + scheduled-sampling) → positional teacher
            forcing → digit-CE + folded-STOP-CE + cosine recon.
          - QueryBridgeEmission: one cross-attention pass over `seeds.substrate` → N vectors + validity logits →
            Hungarian match to `targets` → per-fact InfoNCE (+ cross-batch negs) + validity BCE.
        Returns `EmissionOut` extended so `valid`/`lens` travel WITH it (strategy-determined), not as inputs.
        `ep` drives any warmup (FSQ scheduled-sampling; a bridge may ignore it)."""
        ...

    def emit_infer(self, texts: list[str], max_items: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """Inference emission: text → a variable-length latent SET per row. `model.rollout()` would delegate here.
          - FSQ: autoregressive rollout terminating on STOP (unbounded N up to `max_items`).
          - QueryBridge: one pass, keep queries whose validity > 0.5 (N ≤ n_queries).
        Returns `(lat [B, L, d], lens [B])`, zero-padding halted/invalid rows — same shape `rollout(return_lengths=True)`
        yields today, so downstream retrieval/eval is unchanged."""
        ...

    def z_for_reg(
        self, em: "EmissionOut", targets: "TargetSet"
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """(predicted, target) latents a `_TargetSource.regularizer` (e.g. SIGReg) constrains. FSQ uses the
        pre-quantization `z`; a continuous family uses the emitted vectors directly. Mirrors
        `_EmissionObjective.z_for_reg` (strategies.py:394)."""
        ...


# Conformance note (checked by intent, wired later):
#   FSQObjective already implements `codebook`, an `emit`, and `z_for_reg`. The refactor in EMISSION_PROTOCOL.md
#   step 2-4 renames/rehapes `emit` -> `emit_train` (validity/lens returned, not passed), adds `emit_infer`
#   (delegated from `model.rollout`), and adds `parameters()`. None of that changes FSQ TENSORS — it only lifts
#   the AR assumptions out of the interface so a second family can exist.  DO NOT implement the bridge here.
