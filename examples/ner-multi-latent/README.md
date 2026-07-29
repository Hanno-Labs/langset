# ner-multi-latent — langset's multi-latent training demo

Most langset examples emit **one** latent per input. This one emits a **variable-length sequence** of latents
per input and stops on its own: a sentence goes in, and the model autoregressively emits **one latent per named
entity** (in order of appearance), feeding each emitted latent back into its own hidden stream, terminating on
the natural-EOS `stop_logit`.

```
"Angela Merkel met Tim Cook at Apple headquarters in Cupertino ."
      │
      ▼  rollout_train (teacher-forced AR)  /  rollout (free, natural-EOS)
   [latent] [latent] [latent] [latent]  </s>
    PER      PER      ORG      LOC
```

### Why NER (and why it's not prediction)

NER is deliberately a **non-predictive** multi-latent task — every entity is already present in the input, so
there's no forecasting. That isolates the question we actually care about: *does multi-latent emission train
cleanly under langset's plain self-contrastive formula, with no EMA target?* Because the gold entity set is
complete, we can read quality as real entity **F1** — a ground-truth signal that a prediction task can't give
you (there, "collapsed to the plausible mean" and "correct" are indistinguishable).

### The training objective

langset's self-contrastive formula, applied per-entity and summed over the sequence:

| term | what it does |
|---|---|
| `InfoNCE(pred_j, ema_emit(entity_j))` | each AR step lands on that entity's target emission; **in-batch negatives = every entity in the batch** |
| `lam_recon · recon(pred_j → entity text)` | grounds each latent in its entity string |
| `lam_uniform · uniformity(pred)` | anti-collapse — spread the latents on the sphere |
| `lam_eos · BCE(stop_logit, …1 at last entity)` | learns *when* to stop (natural-EOS) |

**The target comes from a stop-grad EMA twin, and that's not optional for multi-latent.** We first ran it
*without* EMA (`target_j = emit(entity_j)` from the online model itself). On a lucky init it trained great
(f1 → 0.63, collapse → 0.14); on another init it **collapsed** (f1 stuck 0.05, collapse 0.85) and never
recovered — the classic representation collapse a stop-grad target exists to prevent. So multi-latent emission
turns EMA on **automatically** (single-latent langset core doesn't need it); there's deliberately no flag to
forget. `--ema_m` tunes the momentum.

### Run

```bash
uv run examples/ner-multi-latent/train.py                                  # full (GPU recommended)
uv run examples/ner-multi-latent/train.py --limit 300 --epochs 3 --device cpu --no-wandb   # quick smoke
```

Reports `f1 / precision / recall / collapse` per epoch (to stdout, and to wandb unless `--no-wandb`).
Data: CoNLL-2003 (PER/ORG/LOC/MISC), loaded from the HF Hub. Uses the ported `LangSetModel.rollout` /
`rollout_train` seams — the single-latent API (`encode`, SetFit composition) is untouched.
