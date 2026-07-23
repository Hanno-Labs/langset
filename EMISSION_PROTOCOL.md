# Emission strategy protocol — abstraction plan

Goal: widen langset's seams so a **non-autoregressive, continuous, parallel-query emission family** (a DETR/
Perceiver-style "query-bridge" over a frozen encoder) can be plugged in as a strategy — and *instantly* inherit
GradCache, kv_cache, the trainer, checkpoint selection, and the `_TargetSource` / `_LossTerm` seams.

**This doc defines the protocol and the port plan. It does NOT port the bridge.** The bridge stays a separate
proof (`~/dev/ii3/packages/fin-exp01-mteb/privcap/train/bridge_lightning.py`, validated: retrieval-preserving
multi-latent, beats single-vector on entity×time). This is the abstraction that would let it live in langset.

## Current emission architecture (as of 0.13.1 — `origin/main` `ca28ec2`)

> Correction: kv_cache is **released** (PR #16, on the 0.13.x line), not WIP. This doc was first drafted against a
> stale local `main` (0.12.0, 6 commits behind); it is now based on the real latest, `origin/main` = 0.13.1, which
> also carries the **pluggable aux-head "Head" plug** (PR #15) that matters for the biggest gap below.

| piece | file | role | pluggable? |
|---|---|---|---|
| `_EmissionObjective.emit()` | `strategies.py:355` (emit `:369`) | seed batch + targets → `EmissionOut` (emitted latents + base loss) | injected via `TrainingArguments.emission` ✅ |
| `FSQObjective` | `strategies.py:402` (emit `:418`) | the ONE impl — calls `m.rollout_train_codebook()`, computes digit-CE + STOP-CE + recon | — |
| `EmissionOut` | `strategies.py:335` | `{recon[B,L,d], base_loss, logs, dim_lg?, lab_label?}` (`dim_lg`/`lab_label` already `Optional`) | mostly generic ✅ |
| `_TargetSource` | `strategies.py:499` | targets the emissions chase (EMA twin / SIGReg / cached) | injected ✅ |
| `_LossTerm` | `strategies.py:110` | aux terms over `MultiStepCtx` | injected ✅ |
| **aux-head "Head" plug** | `modeling.py add_aux_head :735`, save `:1216` | register a PERSISTED trainable module; `save_pretrained` serializes it; module SHARED with trainer | **precedent for strategy-owned params ✅** |
| **AR rollout (train)** | `modeling.py rollout_train_codebook :1038` | the emission *structure* — sequential hops + feedback + STOP | **baked into the model** ❌ |
| **AR rollout (infer)** | `modeling.py rollout() :847` | inference emission | **model-owned, AR-only** ❌ |
| valid/lens/lmax/target_lat | `trainer.py:1851-1857` | built from TARGET texts, passed INTO `emit()` | **teacher-forced assumption** ❌ |

So the loss/target/aux layers are already strategy-shaped, and 0.13.0's `add_aux_head` already solves "a pluggable
module that persists" — the pattern a strategy's own params would reuse. The two hard ❌ rows are the real gap.

## The three coupling gaps

1. **Emission structure is model-owned, not strategy-owned.** `emit()` reaches into `m.rollout_train_codebook`,
   and inference calls `m.rollout()`. Both hardcode "sequential hops + `head.feedback` + STOP folded into FSQ
   dim-0". A parallel-query family has a different forward (N learned queries cross-attend the frozen encoder's
   token states in ONE pass) and its own trainable module (queries + a small decoder). Its *persistence* is
   already solved — `add_aux_head` (0.13.0) registers a module that `save_pretrained` serializes and that the
   trainer trains in place — so the module has a home; what's missing is that the *forward* is hardcoded AR.

2. **Teacher-forcing is in the interface.** The trainer computes `lmax`/`valid`/`lens_l`/`target_lat` from the
   per-row target item lists (`trainer.py:1942`) and passes them to `emit()`. That presumes the emission is
   reconstructing a KNOWN target set at KNOWN positions (AR teacher-forcing). A DETR family emits a fixed N (=
   `n_queries`), decides validity via a learned head, and assigns emissions↔targets by **Hungarian matching** —
   so length/validity are OUTPUTS of the strategy, and target-assignment is part of its loss, not a trainer input.

3. **Inference emission isn't behind the strategy.** `rollout()` is on the model. Whatever emits at train time
   must also emit at eval/inference; today that's two AR code paths in the model, not one strategy.

## Proposed protocol — `EmissionStrategy`

Generalize `_EmissionObjective` into a protocol that OWNS its module, its train-emission, AND its infer-emission,
and that RETURNS length/validity instead of receiving them. `FSQObjective` becomes the reference impl; a future
`QueryBridgeEmission` becomes a second impl. Sketch (see `emission_protocol.py` in this worktree):

```
class EmissionStrategy(Protocol):
    # own trainable params beyond the backbone (FSQ head, or bridge queries+decoder). Registered so the
    # optimizer + checkpoint see them. None if the strategy only reuses model.head.
    def parameters(self) -> Iterable[nn.Parameter]: ...

    # TRAIN: seeds + the per-row target set -> emitted latents, validity the STRATEGY decides, and the base loss
    # (FSQ: teacher-forced digit/STOP/recon; DETR: Hungarian-matched InfoNCE + validity BCE). The strategy does
    # its own target↔emission assignment; the trainer no longer pre-builds valid/lens.
    def emit_train(self, seeds: SeedBatch, targets: TargetSet, ctx: StepCfg) -> EmissionOut: ...

    # INFER: text -> a variable-length latent SET per row (AR: rollout w/ STOP; DETR: one pass + validity gate).
    def emit_infer(self, texts: list[str], max_items: int) -> tuple[Tensor, Tensor]:  # (lat[B,L,d], lens[B])
        ...

    # what SIGReg-style regularizers constrain (pre-quant z for FSQ; the raw emitted vecs for continuous)
    def z_for_reg(self, em: EmissionOut, targets: TargetSet) -> tuple[Tensor, Tensor]: ...
```

`EmissionOut` stays as-is (already generic enough); `valid`/`lens` move INTO it (strategy-determined) instead of
being trainer inputs. `SeedBatch` = tokenized input (+ optional cached frozen substrate — the kv_cache seam
already caches the prompt forward, which a frozen-encoder bridge wants too). `TargetSet` = the per-row target
texts/latents the `_TargetSource` produced.

## How each family satisfies it

| protocol method | `FSQObjective` (AR, discrete) | `QueryBridgeEmission` (parallel, continuous) |
|---|---|---|
| `parameters()` | `model.head` (FSQ down/up/emit_logits) | learned queries + 2-layer cross-attn decoder + validity head |
| `emit_train` | `rollout_train_codebook` (hops+feedback+ss) → digit/STOP/recon loss | one cross-attn pass → N vecs; Hungarian match to targets → InfoNCE + validity BCE |
| `emit_infer` | `rollout()` with STOP | one pass, keep validity>0.5 |
| `z_for_reg` | pre-quant `z = down_proj(·)` | the emitted vectors directly |
| length/validity | STOP token (unbounded N) | validity head (N ≤ n_queries) |

Both are then **the same to the trainer** — GradCache (`_multi_grad_cache_step`), kv_cache, `_TargetSource`
(swap EMA-twin → a frozen-retriever/query target), `_LossTerm`, and checkpoint selection all work unchanged.

## Port plan (do NOT execute yet — this doc + `emission_protocol.py` are the deliverable)

1. **Define `EmissionStrategy` Protocol** (`emission_protocol.py`) — done here as the abstraction target.
2. **Move `valid`/`lens`/`lmax` out of the trainer→emit inputs and into `EmissionOut`** (strategy-determined).
   `FSQObjective` keeps computing them from targets (byte-identical); the interface just stops assuming it.
   Guard with `tests/test_trainer_multi_characterization.py`.
3. **Introduce `emit_infer` on the strategy**; make `model.rollout()` delegate to the active strategy's
   `emit_infer` (FSQ path unchanged; the door opens for a non-AR one).
4. **Let a strategy register its own `parameters()`** into the optimizer + persistence. Largely DONE already:
   `add_aux_head` (`modeling.py:735`) registers a persisted module `save_pretrained` serializes (`:1216`) and the
   trainer trains in place — a strategy would register its queries/decoder the same way. Remaining work is only
   wiring the optimizer to pick up the strategy's params alongside `model.head`/LoRA.
5. Only THEN would a `QueryBridgeEmission` be ~200 lines implementing the four methods — and it inherits the
   whole trainer. Not part of this task.

## Risk / non-goals

- Byte-identical FSQ behavior is the invariant — every step above keeps `FSQObjective` producing the same tensors
  (characterization tests are the ratchet). This is a *seam-widening*, not a rewrite.
- Continuous (non-FSQ) emission departs from langset's token-native/decodable philosophy; that's a deliberate
  second family, opt-in via injection, not a replacement for FSQ.
