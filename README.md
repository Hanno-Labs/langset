# langset — a short path to a world model in your LLM

**langset few-shot fine-tunes a pretrained LLM to *predict in latent space*.** Bolt a tiny head onto the
backbone and it emits a *sequence* of latents in its own token stream — one per step, with a learned STOP —
where each latent holds a **calibrated superposition** of possible next states. That is a JEPA world model, and
the LLM **is** it: no separate world model bolted alongside, no pixel simulator, no new architecture. You get
there few-shot, by describing the states in words.

<p align="center">
  <img src="examples/maze-superposition/assets/maze-frontier.gif" alt="A trained langset world model flooding a maze: one latent per tick holding a set of frontier cells, with the model's P(solvable) readout firming up as the search advances." width="340">
</p>

<p align="center"><em>A trained langset world model rolling out a maze — each tick is one emitted latent holding
a whole <strong>set</strong> of next states (the lime frontier), the caption counting how many. It predicts the
distribution of where the search could be, not a single guess. See
<a href="examples/maze-superposition">the superposition example</a>.</em></p>

The *same* machinery with a **single** latent per input is a bespoke **embedding model**, SetFit-ready — that
tier still works and is documented [below](#also-a-bespoke-embedding-model). But the reason to reach for langset
is the world model.

## The idea

**You describe what a latent should mean, in words — and that description *is* the geometry.** Point the target
texts at the *next states* of a process and the model learns to **emit** them: given a state, predict the set of
admissible futures, with its own uncertainty calibrated to how open the future is. You don't gather labels or
hand-build a simulator — you write the states, and the LLM's world knowledge does the reading.

* 🔮 **It predicts, in latent space.** Emit a *set-valued* latent future with calibrated uncertainty — the JEPA
  property LeCun argues an LLM needs a *separate* world model for. Here the LLM does it itself.
* 🧭 **You design the state-space in words.** The target text defines what the world model tracks; rewrite it to
  re-steer the model — no architecture changes, no relabeling.
* 🧠 **World knowledge does the reading.** A generative LLM generalizes from *hundreds* of examples, not
  millions — it *reads* each state rather than pattern-matching surface tokens.
* 🪞 **The LLM is the world model.** Latents live in the model's own hidden space and (FSQ) its own token
  vocabulary, so text and latent prediction share one stream — not two networks.

## Install

```bash
pip install langset
```

## Quickstart — a world model

Rows are `input_text` (a state) → `target_texts` (the **set** of possible next states). The model learns to emit
that set, deciding *how many* via a learned STOP; at inference `rollout(..., return_soft=True)` reads the emitted
set back plus its per-step entropy — the model's own calibrated uncertainty.

```python
from langset import LangSetModel, Trainer, TrainingArguments

rows = [{"input_text": "<a state>", "target_texts": ["<next state A>", "<next state B>", "<next state C>"]},
        # ...one row per state; target_texts is the SET of admissible next states
       ]
model = LangSetModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", multi_latent=True)   # FSQ set-emission head
Trainer(model, TrainingArguments(epochs=15), rows).train()

lat, lengths, soft, ent = model.rollout("<a state>", return_soft=True)
# soft = the expected latent SET (the superposition); ent = per-step entropy (higher = a more open future)
```

**[examples/maze-superposition](examples/maze-superposition)** is the end-to-end reference: it trains a world
model on a maze search-frontier and *measures* the headline property with [`langset.probes`](src/langset/probes.py) —
the emitted latent's entropy tracks the true number of possible next states (a calibrated superposition, not one
guess).

## How it works — JEPA in the token stream

1. **Predict in latent space (JEPA).** Each emitted latent is trained to match the **stop-grad target latents**
   of its next states (in-batch negatives keep distinct states apart). Predicting a target-encoder's latents —
   not pixels, not tokens — is exactly the JEPA objective, here run *inside* a pretrained LLM.
2. **Token-native emission.** Each latent is finite-scalar-quantized (FSQ) into per-dimension digits the model
   predicts, with **STOP folded into dim-0's softmax**; every emitted latent is fed back into the stream so the
   next one is conditioned on those already emitted. The latent is literally a token — text and latents share
   one softmax/CE interface.
3. **Anti-collapse is the JEPA apparatus.** A stop-grad **EMA target twin** (BYOL/JEPA) supplies the targets by
   default; inject `SIGRegTarget` for the EMA-free **LeJEPA** alternative
   ([details below](#anti-collapse-ema-twin-default-vs-sigreg-lejepa)).
4. **Collapse-aware selection.** langset selects on held-out retrieval/reconstruction with a hard penalty on any
   collapse of the geometry — never on the training loss (which collapse can game).

## World-model knobs

Every knob below is a **strategy injected into `TrainingArguments`**, not a boolean on a monolith — the defaults
give you the FSQ + EMA-twin world model, and each injection swaps one piece.

### Anti-collapse: EMA twin (default) vs SIGReg (LeJEPA)

By default the multi-latent path prevents representation collapse with an **EMA target twin** — a stop-grad
copy of the model whose slowly-moving latents are the targets, so the online model can't trivially match a
target that moves with it. Injecting `target_source=SIGRegTarget` swaps this for **SIGReg** (Sketched Isotropic Gaussian
Regularization, from LeJEPA, [Balestriero & LeCun 2025, arXiv:2511.08544](https://arxiv.org/abs/2511.08544)):
no twin, no stop-grad — targets come from the *live* encoder, and collapse is prevented instead by
regularizing the pre-quantization latent `z = down_proj(·)` toward an isotropic Gaussian (an Epps–Pulley
goodness-of-fit test over random 1-D projections). The isotropic Gaussian is the distribution LeJEPA proves is
uniquely privileged for downstream identifiability ([arXiv:2605.26379](https://arxiv.org/abs/2605.26379)).

```python
from langset.strategies import SIGRegTarget
Trainer(model, TrainingArguments(target_source=SIGRegTarget, sigreg_lambda=0.3), rows).train()
```

Anti-collapse is chosen by *injecting a different target-source strategy*, not a boolean flag — the default
`target_source=EMATwinTarget` and `SIGRegTarget` are interchangeable implementations (see `strategies.py`).

**Trade-offs:**

| | EMA twin (default) | SIGReg (`target_source=SIGRegTarget`) |
|---|---|---|
| memory | a full frozen copy of the backbone in VRAM | none — no twin |
| per-step cost | one extra target forward | one Gaussian-regularizer pass (cheap) |
| anti-collapse | stop-grad target | isotropic-Gaussian penalty on pre-quant `z` |
| separation term | pairs with in-batch InfoNCE (`lam_multi_nce`) | replaces it (InfoNCE auto-gated off) |

Empirically SIGReg **matches or beats** the twin on *local* calibration but retains a small gap on *global*
structure at high `sigreg_lambda`; **tune `sigreg_lambda` ≈ 0.3** (over-diversification washes out global
structure; too small under-constrains). Implementation note (see `sigreg.py`): the test is **center-only, not
standardized** — standardizing per-dim before the Gaussianity test is scale-invariant and silently defeats
anti-collapse (a collapsed batch passes with zero gradient). SIGReg is research-grade; the EMA twin remains
the validated default.

### Continuous emission — `ContinuousObjective`

By default multi-latent quantizes each emission to FSQ digits (a discrete token). The continuous path keeps
the same variable-length, STOP-terminated, fed-back latent-set structure but emits a **raw continuous vector**
(`out_proj`), trained by cosine to the target with a BCE STOP head instead of digit cross-entropy.

Two pieces select it: the **model** is built with the continuous head (`continuous_emit=True` — a head-architecture
property, chosen at `from_pretrained`), and the **trainer** is given the continuous emission strategy by *injecting*
`emission=ContinuousObjective` (interchangeable with the default `FSQObjective`, see `strategies.py`):

```python
from langset.strategies import ContinuousObjective
model = LangSetModel.from_pretrained("...", multi_latent=True, continuous_emit=True)
Trainer(model, TrainingArguments(emission=ContinuousObjective), rows).train()
```

**Why:** a discrete argmax emission can only name *one* future, so when an input admits several plausible next
latents the digit head can't represent the *mixture*. A continuous emission can settle at the **centroid of the
admissible futures** — calibrated superposition, the one-to-many property that motivates Large Concept Models'
diffusion-over-next-concept ([Meta LCM, arXiv:2412.08821](https://arxiv.org/abs/2412.08821)); it is also the
regime the isotropic-Gaussian identifiability result assumes ([arXiv:2605.26379](https://arxiv.org/abs/2605.26379)),
which a bounded discrete lattice does not satisfy.

**Trade-offs:** you gain the ability to represent a calibrated distribution over futures, but you give up the
token-native discreteness — emissions are no longer in-vocabulary digits sharing the softmax/CE machinery, and
the FSQ grid no longer provides *free* anti-collapse, so lean on the EMA twin (or SIGReg) and watch the
`distinct` diversity count. Use `ContinuousObjective` when the target is genuinely one-to-many; stay on the default
FSQ when you want the discrete token interface (e.g. to co-train text and latents in one stream).

### CoT-conditioned emission — `build_cot_loss_terms` + `cot_seed_texts`

By default the model emits latents straight from the input seed. Injecting the CoT strategy pair inserts a
**chain-of-thought step**: the model is co-trained to *generate* a per-row reasoning string (a `cot_text` column)
before it emits the latents, and the latent forward is conditioned on `seed + CoT`. Two objectives share one
optimizer step — the FSQ latent loss, and a next-token cross-entropy on the CoT string (weight `lam_cot`) through
the tied embedding, i.e. the **same CE machinery the latents already use**.

It's selected by *injecting two strategies* (not a flag): `loss_terms=build_cot_loss_terms` adds the isolated
`CoTGenTerm`, and `seed_builder=cot_seed_texts` conditions the emission on the reasoning:

```python
from langset.strategies import build_cot_loss_terms, cot_seed_texts
rows = [{"input_text": "...", "target_texts": ["...", "..."], "cot_text": "step-by-step reasoning ..."}, ...]
model = LangSetModel.from_pretrained("Qwen/Qwen3-1.7B-Base", multi_latent=True)
Trainer(model, TrainingArguments(loss_terms=build_cot_loss_terms, seed_builder=cot_seed_texts, lam_cot=1.0),
        rows).train()
```

**Why:** some target latents aren't a direct function of the surface input — they need an intermediate
inference the model can do but doesn't surface in one hop. Letting the model *think in tokens first*, then emit,
lets that reasoning inform the latent, in the spirit of chain-of-thought reasoning
([Wei et al., arXiv:2201.11903](https://arxiv.org/abs/2201.11903)) — but here the reasoning and the latent are
co-trained in **one token stream** rather than the reasoning being an external prompt, and unlike COCONUT's
continuous latent thoughts ([Hao et al., arXiv:2412.06769](https://arxiv.org/abs/2412.06769)) the CoT stays in
readable token space sharing the softmax/CE interface.

**Trade-offs:** the two forward+backward passes run *separately* so their autograd graphs never coexist (peak
activation is `max(latent, cot)`, not the sum) — but you still pay a second forward per step, and CoT blocks are
long (train-time cost scales with CoT length, not the short latents). You need a `cot_text` column: at train
time it's a teacher-forcing target; the measured result is the *lift* from the model learning to produce its own
CoT (self-generated reasoning helps even when the CoT text itself came from a stronger teacher, which is not a
fair ceiling to compare against). Without the injected strategies (or with an absent `cot_text` column) the path
is byte-identical to the plain FSQ emission — `CoTGenTerm` self-skips on empty reasoning.

### Superposition — one seed, several alternative futures

When a single input seed admits **several alternative futures** — its `target_texts` are competing branches of
*one* state, not disjoint items — the default in-batch objective pushes those branches apart, forcing the
emitted latent to commit to one. Injecting the superposition strategy triple lets it instead represent the
calibrated **mixture** over branches (its uncertainty), the token-space analogue of predicting a distribution
over next states:

| injected strategy | effect |
|---|---|
| `epoch_order=grouped_epoch_order` | orders each epoch so a seed's branches are **contiguous**, so they *tend to share a batch* (guaranteed only when `batch_size` ≥ the per-seed branch count and the groups align — contiguity alone doesn't stop a group straddling a batch boundary); when they do co-occur, their per-target digit-CE sums within the batch ≈ a soft cross-entropy toward the branch mixture `P_mix` |
| `loss_terms=build_superposition_loss_terms` | adds `same_seed_mask` to the in-batch InfoNCE, treating two branches of the **same seed** as false-negatives (not pushed apart), so the emitted latent may settle at their **centroid** (the mixture) rather than being repelled from it |
| `selector=last_epoch_selector` | keeps the **last epoch** instead of early-stopping on `retr_mrr` (see below) |

```python
from langset.strategies import build_superposition_loss_terms, grouped_epoch_order, last_epoch_selector
Trainer(model, TrainingArguments(loss_terms=build_superposition_loss_terms,
                                 epoch_order=grouped_epoch_order,
                                 selector=last_epoch_selector), rows).train()
```

Without these injections the default strategies treat branches as independent items (byte-identical to the
standard multi-latent path). Use them only when branches of one seed genuinely share a state and you want the
emission to be a *distribution*, not a pick. This is the property a discrete FSQ argmax can only approximate as a
mixture of codes; see also [`ContinuousObjective`](#continuous-emission--continuousobjective) for a raw-vector centroid.

Why `last_epoch_selector`: the default checkpoint selection (`retr_mrr`) rewards a *collapsed* one-future-per-seed
geometry, which is exactly the wrong signal here — under superposition training you *want* retrieval MRR to fall
as the latent spreads over a seed's alternatives, so keep the last epoch rather than early-stopping on it. There
is also a plain `snapshot_every=N` scalar knob that saves the online weights to `{output_dir}_ep{N}`,
`{output_dir}_ep{2N}`, … after every N epochs — independent of the eval cadence, separate from the best-so-far
restore — to keep a checkpoint trajectory for offline evaluation.

### Text replay — `learn_field` / `learn_ratio`

Fine-tuning a backbone on the emit objective can slowly erode its plain next-token ability. **Text replay**
interleaves ordinary language-model steps into training to rehearse it: tag some rows `learn` (via a
`learn_field` column) and, with probability `learn_ratio` before each normal batch, the trainer runs a
next-token cross-entropy on those rows (`input_text` → target, through the tied input embedding — no separate
LM head) as its own optimizer step. It's the standard **rehearsal** remedy for catastrophic forgetting
([Robins 1995](https://doi.org/10.1080/09540099550039318)) applied to the backbone.

```python
rows = [
    {"input_text": "...", "target_texts": ["...", "..."]},          # normal emit rows
    {"input_text": "domain fact to keep fluent", "target_texts": ["..."] , "tag": "learn"},   # rehearsed as text
]
Trainer(model, TrainingArguments(learn_field="tag", learn_ratio=0.2), rows).train()
```

Works on **both** the single-latent and multi-latent paths (multi-latent uses `target_texts[0]` as the replay
target). `learn_ratio=0` or no `learn` rows = off (byte-identical). Use it when the backbone must stay fluent on
a domain while you retrain its emission geometry; leave it off for pure embedding tasks.

## Also: a bespoke embedding model

The **single-latent** path emits **one** vector per input instead of a set — a bespoke embedding model in a
geometry you define, and a drop-in Sentence-Transformer body for SetFit. Same one idea: the `target_text` *is*
the geometry (describe instrumentation and the space clusters by instrumentation; describe vocals and emotion and
it clusters by those). Rows are `input_text` → a single `target_text`:

```python
from langset import LangSetModel, Trainer, TrainingArguments

rows = [{"input_text": "an hour-long track of detuned riffs that never break stride, at the pace of continental drift",
         "target_text": "glacial detuned doom-metal, sludgy and hypnotic, buried roared vocals"},
        # ...
       ]
model = LangSetModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")   # any HF causal LM, single-latent
Trainer(model, TrainingArguments(), train_dataset=rows).train()

z = model.encode(["a wall of downtuned fuzz that buries the vocals under sheer volume"])
print(z.shape)   # (1, 576) — a latent in the backbone's own space
```

See [`examples/sounds_like/`](examples/sounds_like/) for the full reference task (album review → "how it sounds"
latent).

### With SetFit

The name is the chain: **lang·set·fit** — a *language* model emits into the *set* geometry (langset, usable on
its own), which then *fit*s a classifier. `model.as_sentence_transformer()` is a drop-in
[SetFit](https://github.com/huggingface/setfit) `model_body`. The clean distinction: **SetFit answers with a
*label*; langset answers with a *latent*.**

| | reach for **SetFit** | reach for **langset** |
|---|---|---|
| your answer is | a **label** (fixed classes) | a **point in a space** — retrieval, "find similar", ranking, clustering |
| you define the target by | enumerating classes | a **description** of the geometry ("how it sounds") |
| your input | text to classify | text *or an identifier* — leans on the LLM's world knowledge |

Use SetFit alone for plain few-shot classification; use langset when the answer is a geometry you'll retrieve /
rank / cluster in; chain **langset → SetFit** when a task-shaped body helps the classifier.

```bash
pip install "langset[setfit]"      # pins the verified composition window
```
```python
from sklearn.linear_model import LogisticRegression
from setfit import SetFitModel

clf = SetFitModel(model_body=langset_model.as_sentence_transformer(),
                  model_head=LogisticRegression(max_iter=2000),    # direct construction needs an explicit head
                  labels=[...])
clf.fit(x_train, y_train, num_epochs=1)        # frozen body + head — the robust path
clf.predict(["..."])
```

> **Dependency alignment.** `langset` runs modern backbones (transformers ≥4.41, incl. **Qwen3**).
> `langset[setfit]` pins **transformers 4.46.x / torch <2.5 / Python 3.10–3.12**: SetFit imports
> `training_args.default_logdir` (removed after 4.46), and 4.46 + torch ≥2.5 trips a `torch.distributed.tensor`
> bug. Use the frozen-body `SetFitModel.fit`/`predict` path above; the full `setfit.Trainer` (fine-tunes the
> body) is fragile in this window.

### A set of latents, outside world models

The variable-length **set** emission is useful even when the set isn't a set of *futures* — anywhere one input
carries an unknown number of things, and a single averaged vector would blur them together:

* **Multi-item extraction** — a latent per entity / keyphrase / skill / ingredient, retrieved against a reference
  bank ([`examples/ner-multi-latent/`](examples/ner-multi-latent/) does this for named entities).
* **Multi-vector retrieval** — represent a query or document as a *set* of latents (ColBERT-style late
  interaction) instead of one averaged vector.
* **Multi-aspect / multi-label** — one latent per facet (`{brand, category, material}`) or applicable label.

```python
from langset import LangSetModel
import torch.nn.functional as F

m = LangSetModel.load("path/to/checkpoint", device="cpu")
bank = ["PER: Barack Obama", "LOC: Berlin", "PER: Angela Merkel", "ORG: Apple", "LOC: California"]
zb = F.normalize(m.emit(bank).float(), dim=-1)
lat = F.normalize(m.rollout("Barack Obama visited Berlin to meet Angela Merkel.").float(), dim=-1)
for v in lat:                                                        # one latent per entity, count set by a learned STOP
    print(bank[int((v @ zb.T).argmax())])                           # PER: Barack Obama / LOC: Berlin / PER: Angela Merkel
```
