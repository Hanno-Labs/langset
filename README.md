# langset — read text, answer in a vector

**langset turns a language model into your own bespoke embedding space, few-shot.** Bolt a tiny vector head
onto a pretrained LLM, describe the axis you want in words, and it learns to *read text and emit a latent* into
a geometry defined by those descriptions — using the LLM's world knowledge to do the reading. The latent lives
in the model's own space (it's *your* embedding, not a re-projected off-the-shelf one), and it drops straight
into SetFit as a Sentence-Transformer body.

## The one idea

**The `target_text` *is* the geometry.** Whatever your target descriptions describe becomes the axis your
latent space measures — and nothing else. Describe instrumentation and the space clusters by instrumentation;
describe vocals and emotion and it clusters by vocals and emotion. You don't discover the geometry, you
**define** it — and you re-steer it by rewriting the target text, no model changes.

What makes langset different:

* 🎯 **Latent out, not a label.** You define the output space; the model answers in a *vector*. Retrieval,
  "find similar", ranking, clustering — classification is just one thing you can do downstream.
* 🧭 **You design the axis in words.** The target text defines the geometry. Point it at the signal you care
  about (and at something the input text can't trivially regenerate, or you're just distilling a text encoder).
* 🧠 **World knowledge does the work.** It's a generative LLM, so it generalizes from *hundreds* of examples,
  not millions — it *reads* the input rather than pattern-matching surface tokens.
* 🪞 **Your own embedding.** The latent lives in the model's own hidden space; the geometry comes from a
  self-contrastive objective against your target text — no external encoder in the loop.

## Install

```bash
pip install langset
```

## Usage

A langset dataset is rows of `input_text` → `target_text`. Pick an LLM backbone; langset trains the mapping.

```python
from langset import LangSetModel, Trainer, TrainingArguments

rows = [  # what you'll have at inference -> a description that DEFINES where it should land
    {"input_text": "an hour-long track of detuned riffs that never break stride, moving at the pace of continental drift",
     "target_text": "glacial detuned doom-metal, sludgy and hypnotic, buried roared vocals"},
    {"input_text": "chopped vocal ghosts drifting over vinyl crackle and the hiss of a city at 3am",
     "target_text": "crackly nocturnal UK garage, pitched vocal ghosts, wistful and restless"},
    # ...
]

model = LangSetModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")   # any HF causal LM
Trainer(model, TrainingArguments(), train_dataset=rows).train()

z = model.encode(["a wall of downtuned fuzz that buries the vocals under sheer volume"])
print(z.shape)   # (1, 576)  — a latent in the backbone's own space
```

See [`examples/sounds_like/`](examples/sounds_like/) for the full reference task (album review → "how it
sounds" latent).

## How it works

1. **Self-contrastive.** For each row, `emit(input_text)` is trained to match `emit(target_text)` — *both*
   emitted through the model into its own space — against in-batch negatives. The target text defines where
   each item lands; the negatives force different items apart (so the space can't collapse).
2. **Grounding aux.** A light reconstruction term makes the latent also *decode* the target text, tying it to
   the words. A light uniformity term keeps the space spread on the sphere.
3. **Collapse-aware selection.** langset early-stops on held-out input↔target retrieval and reconstruction,
   with a hard penalty on any collapse of the geometry — never on the training loss (which collapse can game).

## Dataset contract

| column | meaning |
|---|---|
| `input_text` | what you'll have at inference (a name, a query, a review) |
| `target_text` | a description of the same item that **defines** where it lands (the geometry) |

`Trainer` accepts a `datasets.Dataset` or `list[dict]`; use `column_mapping` to rename your columns.

## Using with SetFit

The name is the chain: **lang·set·fit** — a *language* model emits into the *set* geometry (langset, usable on
its own), which then *fit*s a classifier. `model.as_sentence_transformer()` is a drop-in
[SetFit](https://github.com/huggingface/setfit) `model_body`, so you can train a few-shot classifier directly
on your bespoke geometry.

The clean distinction: **SetFit answers with a *label*; langset answers with a *latent*.**

| | reach for **SetFit** | reach for **langset** |
|---|---|---|
| your answer is | a **label** (fixed classes) | a **point in a space** — retrieval, "find similar", ranking, clustering |
| you define the target by | enumerating classes | a **description** of the geometry ("how it sounds") |
| your input | text to classify | text *or an identifier* — leans on the LLM's world knowledge |

- **Use SetFit alone** for plain few-shot classification — you won't beat it by bolting on langset.
- **Use langset** when the answer is a *geometry, not a label* (you'll retrieve / rank / cluster in it).
- **Use langset → SetFit** when a task-shaped body helps the classifier.

```bash
pip install "langset[setfit]"      # pins the verified composition window (below)
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

**Dependency alignment.** SetFit's pins are loose, so versions matter:

| install | transformers / torch | Python | use |
|---|---|---|---|
| `langset` | latest (≥4.41) | 3.10+ | modern backbones incl. **Qwen3**; no SetFit |
| `langset[setfit]` | **4.46.x / <2.5** | **3.10–3.12** | verified SetFit composition |

SetFit imports `transformers.training_args.default_logdir` (removed after 4.46), and 4.46 + torch≥2.5 trips a
`torch.distributed.tensor` bug — hence the cap. Use the frozen-body `SetFitModel.fit`/`predict` path above; the
full `setfit.Trainer` (fine-tunes the body) is fragile in this window.

## Multi-latent — one input, a *set* of latents

Everything above emits **one** latent per input. Multi-latent emits a **variable-length set** — one latent per
distinct item in the input — autoregressively, with the model deciding *how many* via a learned STOP. Each
latent lands in the same bespoke geometry, so you decode it the same way you'd decode a single one (nearest
neighbor against a bank, a downstream head, whatever).

**Why:** a single vector is the wrong shape whenever one input contains an unknown number of things. Collapse
*"Apple and Microsoft partnered in California"* into one embedding and you've blended three items into one
blurry point. Multi-latent keeps them separate — three latents, each retrievable on its own — and, unlike a
fixed-slot head, it doesn't need you to know the count in advance.

Where it fits:

* **Multi-item extraction** — entities, keyphrases, skills, ingredients: read a document, emit a latent per
  item, retrieve each against a reference bank. ([`examples/ner-multi-latent/`](examples/ner-multi-latent/)
  does exactly this for named entities.)
* **Multi-vector retrieval** — represent a query or document as a *set* of latents (ColBERT-style late
  interaction) instead of one averaged vector, for finer-grained matching.
* **Multi-aspect / multi-label** — one latent per facet (a product's `{brand, category, material}`) or per
  applicable label, instead of one vector forced to mean several things at once.
* **Multi-intent parsing** — an utterance carrying several intents → a latent each.

```python
from langset import LangSetModel
import torch.nn.functional as F

m = LangSetModel.load("path/to/checkpoint", device="cpu")

# a reference bank you retrieve emitted latents against (any short label works)
bank = ["PER: Barack Obama", "LOC: Berlin", "PER: Angela Merkel", "ORG: Apple", "LOC: California"]
zb = F.normalize(m.emit(bank).float(), dim=-1)                       # [N, d]

# emit a VARIABLE-length set from one input — the count is decided by a learned STOP
lat = F.normalize(m.rollout("Barack Obama visited Berlin to meet Angela Merkel.").float(), dim=-1)
for v in lat:                                                        # -> one latent per entity
    print(bank[int((v @ zb.T).argmax())])                           # PER: Barack Obama / LOC: Berlin / PER: Angela Merkel
```

Under the hood each latent is finite-scalar-quantized (FSQ) into per-dimension digits the model predicts, an
EMA target twin keeps the set from collapsing, and every emitted latent is fed back into the stream so the next
one is conditioned on those already emitted.

## Status

v0.2 — the core engine, validated on a real task (album review → "how it sounds" latent) with a downstream
SetFit composition. Apache-2.0.
