# langset — read text, answer in a vector

**langset turns a language model into your own bespoke embedding space, few-shot.** Bolt a tiny vector head
onto a pretrained LLM, point it at a target geometry you define, and it learns to *read text and emit a latent*
into that space — using the LLM's world knowledge to do the work. From ~480 examples it builds a specialized
"how it sounds" geometry that **beats the encoder it was bootstrapped from on 4/5 held-out axes** — while the
*same architecture with a randomly-initialized backbone sits at chance*. That gap is the world knowledge, and
it's the whole point.

What makes langset different:

* 🎯 **Latent out, not a label.** You define the output space; the model answers in a *vector*, not a class.
  The geometry can be anything a description can seed — "how it sounds", "how it behaves", "what it's like".
* 🧠 **World knowledge does the work.** It's a generative LLM, so it generalizes from *hundreds* of examples,
  not millions. Swap in a random-init backbone and it barely moves off chance — the pretrained knowledge is
  the engine.
* 🌱 **Bootstrap, then specialize.** Seed the target geometry from *any* off-the-shelf encoder (no contrastive
  pairs, no labels), then an EMA self-target drifts it *off* that seed into your own task-shaped space — it
  stops being "text similarity" and becomes its own thing (measurably: it leaves the seed's cone).
* 🔀 **Input-agnostic & non-circular.** Many input views can map to the same point, and you point the target at
  a signal the input text can't regenerate — so it's not just distilling a text encoder.
* ✅ **Honest selection.** Optional per-row geometry labels are *eval-only* probes; langset early-stops on
  their held-out kNN-purity with a collapse guard — never on the training loss (which collapse can game).

## Install

```bash
pip install langset
```

## Usage

A langset dataset is rows of `input_text` → `target_text` (+ optional eval-only geometry labels). You pick the
LLM backbone and the bootstrap encoder; langset trains the mapping and specializes the geometry.

```python
from langset import LangSetModel, Trainer, TrainingArguments

rows = [  # a review SNIPPET (what you'll have at inference) -> a description that defines where it should land
    {"input_text": "an hour-long track of detuned riffs that never break stride, moving at the pace of continental drift",
     "target_text": "glacial detuned doom-metal, sludgy and hypnotic", "mood": "heavy"},
    {"input_text": "chopped vocal ghosts drifting over vinyl crackle and the hiss of a city at 3am",
     "target_text": "crackly nocturnal UK garage, pitched vocal ghosts", "mood": "calm"},
    # ...
]

model = LangSetModel.from_pretrained(
    llm_model="HuggingFaceTB/SmolLM2-135M",                    # any HF causal LM (this is what the examples validate on)
    bootstrap_model="sentence-transformers/all-MiniLM-L6-v2",  # seeds the target geometry
)
Trainer(model, TrainingArguments(), train_dataset=rows).train()  # 'mood' auto-detected as an eval-only label

z = model.encode(["a wall of downtuned fuzz that buries the vocals under sheer volume"])  # review snippet -> latent
print(z.shape)   # (1, 384)
```

See [`examples/sounds_like/`](examples/sounds_like/) for the full reference task (album review → "how it
sounds" latent, 481 albums): reproduced through this API at held-out kNN-purity **0.60**, beats-bootstrap
**4/5**.

## How it works

1. **Bootstrap.** Targets = the bootstrap encoder's embedding of `target_text`. No pairs, no labels.
2. **Contrastive fit.** InfoNCE (in-batch negatives) trains the LLM emitter to hit its own target and separate
   from others. A small cosine anchor optionally keeps it tethered to the seed.
3. **Specialize.** An EMA self-target drifts the geometry off the seed into the model's own arrangement (set
   `lam_anchor` low to let it go). Emergent structure falls out — never trained, never labeled.

## Validation / early-stop (the part that bites)

Two traps langset refuses to fall into:

- **Never select on training loss** — InfoNCE + EMA can minimize it by *collapsing* the geometry.
- **Never score retrieval against the frozen bootstrap targets** — the model specializes *away* from them.

So it selects on held-out geometry in the *current* space: input-view↔target-view retrieval + a **collapse
guard** by default; **held-out kNN-purity (+ beats-bootstrap)** when rows carry geometry labels. Early-stop =
patience + restore-best.

## Dataset contract

| column | meaning |
|---|---|
| `input_text` | what you'll have at inference (a name, query, review) |
| `target_text` | a description of the same item defining where it lands (seeds the geometry) |
| *anything else* | optional **eval-only** geometry labels (kNN-purity at validation; never trained on) |

`Trainer` accepts a `datasets.Dataset` or `list[dict]`; use `column_mapping` to rename your columns.

## Using with SetFit

The name is the chain: **lang·set·fit** — a *language* model emits into the *set* geometry (langset, usable on
its own), which then *fit*s a classifier. `model.as_sentence_transformer()` is a drop-in
[SetFit](https://github.com/huggingface/setfit) `model_body`, so you can train a few-shot classifier directly
on the specialized geometry. On the sounds-like example, a genre classifier on the **langset body beats one on
raw MiniLM, 0.240 vs 0.205**.

### When to reach for which

The clean distinction: **SetFit answers with a *label*; langset answers with a *latent*.** SetFit is a few-shot
*classifier*; langset is a few-shot *bespoke embedding space* — classification is just one thing you can do
downstream of a latent.

| | reach for **SetFit** | reach for **langset** |
|---|---|---|
| your answer is | a **label** (fixed classes) | a **point in a space** — retrieval, "find similar", ranking, clustering |
| you define the target by | enumerating classes | a **description** of the geometry ("how it sounds") |
| your input | text to classify | text *or an identifier* (a name) — leans on the LLM's world knowledge |
| plain few-shot classification | ✅ simpler, faster, proven | not its job |

- **Use SetFit alone** for plain few-shot classification — it directly optimizes class separation; you won't
  beat it by bolting on langset.
- **Use langset** when the answer is a *geometry, not a label* (you'll retrieve / rank / cluster in it).
- **Use langset → SetFit** when a task-shaped body helps the classifier (demo: genre 0.240 vs 0.205 — modest,
  so treat it as "plausibly helps", not a slam-dunk).

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
full `setfit.Trainer` (fine-tunes the body) is fragile in this window. Qwen3 + SetFit can't share one env until
SetFit drops that import.

## Status

v0.1 — the core engine, validated on a real task (album review → "how it sounds" latent), with a downstream
classifier composition (see above). No trust/hallucination layer yet (intentionally out of v1). Apache-2.0.
