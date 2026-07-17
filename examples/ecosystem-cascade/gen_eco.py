"""Trophic-cascade world -- the pretrain-WIN task (mirror of gen_maze for the ecosystem domain).

The point: a world model whose DYNAMICS are simple and learnable, but whose TRANSITIONS are gated on
real-entity knowledge that is NOT in the state. The rule (stated in every seed) is:

    each tick, a herbivore survives only if a plant is present, a carnivore only if a herbivore is
    present; plants always persist; everything else dies. Repeat until stable.

That rule is BFS-simple -- a random-init transformer learns it from a handful of examples. What it CANNOT
learn for a HELD-OUT species is that species' trophic ROLE ("is an osprey a carnivore?"): the role is never
shown, only the species name. Only pretraining supplies it. And it COMPOUNDS: predicting tick-2 survivors
requires having gotten tick-1 right. So the pretrained-vs-random gap should widen with rollout depth.

Ground truth is the hidden ROLE table below (plant / herbivore / carnivore), curated for unambiguous
real-world consensus. Train and eval corpora draw from DISJOINT species pools (held-out species) so a pass
on eval means the role knowledge GENERALIZED, not that it was memorized.

    python gen_eco.py peek
    python gen_eco.py build 6000 eco_train.npz            # train pool, seed 0
    python gen_eco.py build 1000 eco_eval.npz 999 heldout # HELD-OUT species pool, seed 999
"""
from __future__ import annotations

import json
import sys

import numpy as np

# ---- hidden ground-truth role table (never shown to the model) -----------------------------------------
# Curated for unambiguous role: obligate/clear cases only (no omnivores, scavengers, or debatable diets).
PLANT = [
    "oak", "maple", "fern", "moss", "algae", "kelp", "clover", "wheat", "grass", "bamboo",
    "cactus", "ivy", "dandelion", "seaweed", "pine", "birch", "willow", "reed", "lotus", "cattail",
    "sunflower", "nettle", "thistle", "heather", "sedge", "watercress", "duckweed", "milkweed", "goldenrod", "buttercup",
]
HERBIVORE = [
    "rabbit", "deer", "cow", "sheep", "goat", "zebra", "elephant", "giraffe", "grasshopper", "caterpillar",
    "snail", "aphid", "koala", "panda", "bison", "antelope", "gazelle", "hare", "squirrel", "vole",
    "beaver", "capybara", "tortoise", "manatee", "tadpole", "locust", "wildebeest", "hippopotamus", "impala", "chinchilla",
]
CARNIVORE = [
    "wolf", "lion", "tiger", "leopard", "hawk", "eagle", "owl", "shark", "snake", "cheetah",
    "jaguar", "crocodile", "falcon", "osprey", "heron", "spider", "scorpion", "mantis", "frog", "weasel",
    "orca", "barracuda", "piranha", "lynx", "cougar", "mongoose", "ferret", "stoat", "kingfisher", "pike",
]
ROLE = {**{s: "plant" for s in PLANT}, **{s: "herbivore" for s in HERBIVORE}, **{s: "carnivore" for s in CARNIVORE}}
NEEDS = {"plant": None, "herbivore": "plant", "carnivore": "herbivore"}   # what each role must see present


def _pools(heldout: bool) -> dict[str, list[str]]:
    """Disjoint train/held-out species pools per role (first ~70% train, last ~30% held out)."""
    out = {}
    for role, lst in (("plant", PLANT), ("herbivore", HERBIVORE), ("carnivore", CARNIVORE)):
        cut = int(round(len(lst) * 0.7))
        out[role] = lst[cut:] if heldout else lst[:cut]
    return out


def _cascade(present: set[str]) -> list[list[str]]:
    """Run the deterministic survival cascade. Returns the per-tick surviving set (sorted), stopping at a
    fixed point or empty. Monotone: a species survives iff its required prey/plant role is present THIS tick."""
    roles_present = lambda S: {ROLE[s] for s in S}
    ticks, cur = [], set(present)
    for _ in range(8):
        rp = roles_present(cur)
        nxt = {s for s in cur if NEEDS[ROLE[s]] is None or NEEDS[ROLE[s]] in rp}
        ticks.append(sorted(cur))
        if nxt == cur or not nxt:
            if not nxt:
                ticks.append([])
            break
        cur = nxt
    return ticks


def _seed_text(present: list[str]) -> str:
    return (
        "Ecosystem. Species present: " + ", ".join(present) + ".\n"
        "Rule: each tick, a herbivore survives only if a plant is present, a carnivore only if a "
        "herbivore is present; plants always persist; the rest die. List the survivors each tick "
        "until the ecosystem is stable."
    )


def _sample_start(pools: dict[str, list[str]], rng: np.random.Generator) -> list[str]:
    """Draw a starting community biased toward CASCADES: often drop a trophic layer so the chain collapses."""
    drop = rng.choice(["none", "plant", "herbivore", "both_top"], p=[0.30, 0.34, 0.24, 0.12])
    n = {"plant": 0 if drop in ("plant",) else int(rng.integers(1, 4)),
         "herbivore": 0 if drop in ("herbivore",) else int(rng.integers(1, 4)),
         "carnivore": 0 if drop == "both_top" else int(rng.integers(1, 4))}
    picks = []
    for role, k in n.items():
        if k and pools[role]:
            picks += list(rng.choice(pools[role], size=min(k, len(pools[role])), replace=False))
    rng.shuffle(picks)
    return list(picks)


def build(n: int, out: str, seed: int = 0, pool: str = "train") -> None:
    rng = np.random.default_rng(seed)
    pools = _pools(heldout=(pool == "heldout"))
    seeds_, futs_, roles_, depths_ = [], [], [], []
    tries = 0
    while len(seeds_) < n and tries < n * 20:
        tries += 1
        start = _sample_start(pools, rng)
        if len(start) < 3:
            continue
        ticks = _cascade(set(start))
        if len(ticks) < 2:                          # want at least one transition (a real rollout)
            continue
        fut = [f"tick {t}: " + (", ".join(cells) if cells else "(none)") for t, cells in enumerate(ticks)]
        seeds_.append(_seed_text(start))
        futs_.append(fut)
        roles_.append([ROLE[s] for s in start])
        depths_.append(len(ticks) - 1)              # number of transitions
    np.savez_compressed(out, seed=np.array(seeds_, dtype=object),
                        fut_text=np.array(futs_, dtype=object),
                        start_roles=np.array(roles_, dtype=object),
                        depth=np.array(depths_, dtype=np.int32))
    import collections
    dh = collections.Counter(depths_)
    print(f"-> {out}  ({len(seeds_)} rollouts, pool={pool}, seed={seed})")
    print("   depth histogram (transitions):", dict(sorted(dh.items())))


def peek(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    pools = _pools(heldout=False)
    print(f"train pool sizes: {{k: len(v) for k,v}} =", {k: len(v) for k, v in pools.items()})
    print(f"held-out pool sizes =", {k: len(v) for k, v in _pools(heldout=True).items()})
    shown = 0
    while shown < 4:
        start = _sample_start(pools, rng)
        if len(start) < 3:
            continue
        ticks = _cascade(set(start))
        if len(ticks) < 2:
            continue
        shown += 1
        print("=" * 90)
        print(_seed_text(start))
        print("  roles:", {s: ROLE[s] for s in start})
        for t, cells in enumerate(ticks):
            print(f"  tick {t}: {', '.join(cells) if cells else '(none)'}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "peek":
        peek(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif sys.argv[1] == "build":
        n = int(sys.argv[2]); out = sys.argv[3]
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        pool = sys.argv[5] if len(sys.argv) > 5 else "train"
        build(n, out, seed, pool)
    else:
        print(__doc__)
