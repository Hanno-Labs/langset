# maze-superposition — a langset world model that holds a *set* of futures

Most langset examples emit a latent that names **one** thing. This one trains a **world model** whose every
emitted latent holds a **superposition** — a calibrated *set* of possible next states — and shows that the
model knows *how many* possibilities there are.

The task is a maze search. A parallel breadth-first flood spreads out from `S`; at each tick the **frontier**
is the set of cells the wavefront currently occupies. langset emits **one latent per tick**, and that tick's
target describes the *whole frontier set*. So a single latent has to represent several cells at once — the
superposition — and the interesting question is whether it does so *calibratedly*: does the latent's own
uncertainty grow when the frontier is wide and shrink when it narrows to one cell?

```
Maze 6x6. #=wall S=start E=exit. Flood the search from S one step at a time; is E reachable?
#..E#.
#.##..
##....
.#..##
.S.#..
..###.
      │  one latent per tick (STOP-terminated)
      ▼
 tick 0: 1 cell  at r4c1
 tick 1: 3 cells at r4c0, r4c2, r5c1          ← the frontier is a SET → the latent is a superposition of 3
 tick 2: 3 cells at r3c0, r3c2, r5c0
 tick 3: 2 cells at r2c2, r3c3 | dead: r3c0, r5c0
 tick 4: 1 cell  at r2c3
 ...
 verdict: SOLVABLE E reached
```

### Why this is a *superposition* test (and not just multi-latent)

A world model in latent space is supposed to predict a **distribution over next states**, not a single guess —
that's the whole reason (per LeCun / JEPA) you'd predict in latent space at all. The maze frontier makes that
concrete and *directly supervised*: the target for tick *t* literally **is** the set of active cells, so "the
latent should hold a set of size *k*" is a ground-truth signal, not something we hope emerges.

That direct supervision is what makes it work. An earlier attempt supervised superposition *indirectly* (hope a
centroid emerges from K same-seed rows) and the discrete FSQ code came out high-entropy but **uncalibrated**.
With the frontier as the literal target, the FSQ code's entropy **tracks the frontier size** — the property
below.

### The langset pieces

| piece | why |
|---|---|
| `multi_latent=True` | variable-length latent set — one latent per tick, the model decides how many via a learned STOP |
| `selector=last_epoch_selector` | retrieval MRR rewards a **collapsed** one-cell-per-tick geometry — exactly the wrong signal here (it's *meant to fall* as the latent spreads over the set), so keep the last epoch instead of early-stopping on it |
| `rollout(..., return_soft=True)` | at eval, read the **expected** latent and its per-dim **entropy** — the model's native uncertainty — instead of the argmax |
| `target_source=SIGRegTarget` *(optional, `--sigreg`)* | EMA-free anti-collapse (LeJEPA) instead of the stop-grad twin — see [`langset/sigreg.py`](../../src/langset/sigreg.py) |

Nothing here is a flag on a monolith — each is a strategy injected into `TrainingArguments`. See
[`train.py`](train.py).

### Run it

```bash
pip install "langset" scipy scikit-learn        # eval uses scipy + sklearn probes

python gen_maze.py build 4000 maze.npz          # training corpus (mixed sizes, ~55% solvable)
python train.py --data maze.npz --out maze_model --wandb
python gen_maze.py build 800 maze_eval.npz 999  # DISJOINT eval corpus (different seed → no leakage)
python eval.py  --data maze_eval.npz --ckpt maze_model
```

`python gen_maze.py` with no args prints a few `(maze, per-tick frontier)` rows so you can see the corpus.
A tiny CPU smoke of the whole loop (seconds, no GPU):

```bash
python gen_maze.py build 40 maze.npz
python train.py --data maze.npz --out /tmp/m --device cpu --epochs 2 --bs 4 \
                --backbone hf-internal-testing/tiny-random-LlamaForCausalLM --fsq-dim 32 --max-fut 8
python gen_maze.py build 80 maze_eval.npz 999
python eval.py --data maze_eval.npz --ckpt /tmp/m --device cpu --max-steps 8
```

### What you should see

`eval.py` reports the headline calibration signal:

```json
"B_calibration": { "corr_entropy_nbranch": 0.35, "count_acc": ..., "count_mae": ... }
```

**`corr(entropy, nbranch) > 0`** is the result: the emitted latent's FSQ entropy rises with the frontier size,
so the single latent carries a *calibrated* superposition rather than one guess. On a real 135M run (SmolLM2,
30 epochs) this lands around **+0.34–0.39** (EMA twin / SIGReg), with frontier recall staying flat across branch
counts `k=1..5` — the opposite of discretization collapse. It's already visible in the CPU smoke above
(`corr ≈ 0.35` even from a tiny random backbone).

The second probe, `A_solvability` (can the emitted trajectory separate solvable from unsolvable mazes), needs
the real backbone and full training to clear its majority-class baseline — the tiny smoke leaves it at chance.
