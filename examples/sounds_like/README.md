# Example: sounds-like

The reference langset task — read an album **review** and emit a latent into a "how it sounds" geometry, so
you can retrieve albums that *sound* alike. The validated demo this library was extracted from.

- `input_text`  = album review
- `target_text` = de-leaked sonic **fingerprint** (an LLM-written description; seeds the geometry)
- labels        = `lead_vocal_gender`, `acoustic_electronic`, `energy`, `tempo`, `distortion` — **eval-only**
  geometry probes (kNN-purity + beats-bootstrap at validation; never trained on)

`data/prepared.json` — 481 albums, farthest-point sampled across genres, fingerprints de-leaked so retrieval
is by sound, not identity.

## Run

```bash
# quick CPU smoke (pipeline correctness)
uv run examples/sounds_like/train.py --limit 200 --epochs 5 --device cpu

# full run (GPU recommended)
uv run examples/sounds_like/train.py

# full GPU validation on Modal: reproduce the geometry + compose with SetFit
modal run --detach examples/sounds_like/run_modal.py::go
```

## What it demonstrates

1. **LLM understanding → bespoke cross-modal latent** (review text → sound geometry, not text-similarity).
2. **Bootstrap → specialize** — seed the target from MiniLM, then EMA-drift off it into the model's own space
   (watch `collapse` fall while purity holds).
3. **Few-shot via borrowed knowledge** — 481 examples generalize; a random-init backbone sits at chance.
4. **Geometry selection without leakage** — the cared axes are eval-only; selecting on their held-out kNN
   purity (+ beats-bootstrap) is the early-stop signal.
5. **SetFit composition** — `model.as_sentence_transformer()` is a drop-in `SetFitModel(model_body=...)`, so a
   few-shot classifier sits right on top of the specialized geometry (`run_modal.py` compares it to a raw
   MiniLM body).
