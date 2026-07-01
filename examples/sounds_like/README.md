# Example: sounds-like

The reference langset task — read an album **review** and emit a latent into a "how it sounds" geometry, so
you can retrieve albums that *sound* alike. The demo this library was extracted from.

- `input_text`  = album review
- `target_text` = a sonic **fingerprint** (an LLM-written "how it sounds" description) that DEFINES the geometry

**No review text is shipped here.** `prepare.py` reconstructs the training rows from public data, joining on
artist+album:

| field | source |
|---|---|
| `input_text` (review) | [`mediumrarely/pitchfork-reviews`](https://huggingface.co/datasets/mediumrarely/pitchfork-reviews) — public review text (~99% coverage) |
| `target_text` (fingerprint) | [`Hanno-Labs/sounds-like-fingerprints`](https://huggingface.co/datasets/Hanno-Labs/sounds-like-fingerprints) — our sonic fingerprints |

Both are pulled with `datasets.load_dataset`, so the example is fully reproducible with no local data.

## Run

```bash
# quick CPU smoke (pipeline correctness)
uv run examples/sounds_like/train.py --limit 200 --epochs 5 --device cpu

# full run (GPU recommended)
uv run examples/sounds_like/train.py

# compose the trained body with SetFit on Modal
modal run --detach examples/sounds_like/setfit_compose.py::go
```

## What it demonstrates

1. **LLM understanding → bespoke cross-modal latent** — review text → a *sound* geometry, not text similarity.
   The review isn't a description of the sound; the model reads it and re-projects into the sound axis.
2. **The target text is the geometry.** The fingerprint describes how the album *sounds*, so the space clusters
   by sound. Rewrite the fingerprint to weight, say, vocals + emotion and the same pipeline re-clusters by that
   axis — you steer the geometry by editing the target text, not the model.
3. **Few-shot via borrowed knowledge** — a small set of examples generalizes because the LLM does the reading.
4. **SetFit composition** — `model.as_sentence_transformer()` is a drop-in `SetFitModel(model_body=...)`, so a
   few-shot classifier sits right on top of the bespoke geometry (`setfit_compose.py`).
