"""Danger-corridor world -- the CLEANEST pretrain-WIN task: a mechanical walk gated on ONE known fact.

A 1xL corridor (cells 0..L-1). You start at cell 0, the exit is cell L-1, and a single animal waits at a
random interior cell k. You must walk the whole corridor one cell at a time, so you ALWAYS step onto the
animal (the path forces overlap). The rule, stated in every seed:

    a dangerous animal kills you; a harmless animal lets you pass.

Everything about the walk is mechanical -- a random-init transformer learns "advance one cell per tick" from
a handful of examples (maze already proved this). The ONLY thing it cannot learn for a HELD-OUT animal is that
animal's danger ("does a mamba kill you?"): danger is NEVER in the text, only the animal's NAME. Only
pretraining supplies it. So the crux tick -- the one where you stand on the animal -- is a binary decision
(die vs pass) gated purely on real-world knowledge. That makes the pretrained knowledge CAUSAL (it drives the
transition), not merely probeable.

Ground truth is the hidden DANGER table below. Train and eval corpora draw from DISJOINT animal pools, so a
pass on eval means the danger knowledge GENERALIZED to unseen species, not that it was memorized.

    python gen_danger.py peek
    python gen_danger.py build 6000 danger_train.npz            # train pool, seed 0
    python gen_danger.py build 1000 danger_eval.npz 999 heldout # HELD-OUT animal pool, seed 999
"""
from __future__ import annotations

import sys

import numpy as np

# ---- hidden ground-truth danger table (never shown to the model; only the animal NAME appears) -------------
# Curated for unambiguous real-world consensus: obligate killers vs plainly harmless. Single lowercase tokens.
DANGEROUS = [
    "lion", "tiger", "wolf", "bear", "shark", "crocodile", "cobra", "rattlesnake", "scorpion", "leopard",
    "jaguar", "hyena", "alligator", "viper", "hippopotamus", "rhinoceros", "grizzly", "cheetah", "panther", "boar",
    "piranha", "mamba", "anaconda", "cougar", "komodo", "wolverine", "barracuda", "stingray", "jellyfish", "python",
]
SAFE = [
    "rabbit", "deer", "sheep", "goat", "cow", "hamster", "gerbil", "turtle", "tortoise", "butterfly",
    "ladybug", "sparrow", "robin", "finch", "dove", "squirrel", "chipmunk", "hedgehog", "koala", "panda",
    "sloth", "chinchilla", "mouse", "duck", "lamb", "pony", "alpaca", "gecko", "snail", "otter",
]
DANGER = {**{a: True for a in DANGEROUS}, **{a: False for a in SAFE}}


def _pools(heldout: bool) -> dict[str, list[str]]:
    """Disjoint train/held-out animal pools per class (first 70% train, last 30% held out)."""
    out = {}
    for cls, lst in (("dangerous", DANGEROUS), ("safe", SAFE)):
        cut = int(round(len(lst) * 0.7))
        out[cls] = lst[cut:] if heldout else lst[:cut]
    return out


def _cont(k: int, animal: str) -> str:
    return f"at cell {k}: the {animal} is harmless, keep walking"


def _die(k: int, animal: str) -> str:
    return f"at cell {k}: the {animal} attacks, you die"


def _rollout(L: int, k: int, animal: str) -> list[str]:
    """Per-tick target texts. Dangerous -> walk to cell k then die (length k+1). Safe -> walk to the exit."""
    if DANGER[animal]:                                   # dies on the animal's cell
        return [f"at cell {c}" for c in range(k)] + [_die(k, animal)]
    ticks = []
    for c in range(L):
        if c == k:
            ticks.append(_cont(k, animal))
        elif c == L - 1:
            ticks.append(f"at cell {c}: reached the exit, survived")
        else:
            ticks.append(f"at cell {c}")
    return ticks


def _seed_text(L: int, k: int, animal: str) -> str:
    return (
        f"Corridor of {L} cells numbered 0 to {L - 1}. You start at cell 0 and the exit is cell {L - 1}. "
        f"A {animal} waits at cell {k}. Walk from the start to the exit one cell at a time; you must pass "
        f"through every cell. A dangerous animal kills you; a harmless animal lets you pass. Report your "
        f"position at each step until you exit or die."
    )


def build(n: int, out: str, seed: int = 0, pool: str = "train", L: int = 5) -> None:
    rng = np.random.default_rng(seed)
    pools = _pools(heldout=(pool == "heldout"))
    all_animals = pools["dangerous"] + pools["safe"]
    seeds_, futs_, animals_, dangers_, cruxes_, dies_, conts_ = [], [], [], [], [], [], []
    for _ in range(n):
        animal = str(rng.choice(all_animals))            # uniform over the pool -> ~50/50 danger (equal pool sizes)
        k = int(rng.integers(1, L - 1))                  # interior cell: you walk before meeting it, never on S/E
        seeds_.append(_seed_text(L, k, animal))
        futs_.append(_rollout(L, k, animal))
        animals_.append(animal)
        dangers_.append(int(DANGER[animal]))
        cruxes_.append(k)
        dies_.append(_die(k, animal))
        conts_.append(_cont(k, animal))
    np.savez_compressed(
        out, seed=np.array(seeds_, dtype=object), fut_text=np.array(futs_, dtype=object),
        animal=np.array(animals_, dtype=object), danger=np.array(dangers_, dtype=np.int32),
        crux_cell=np.array(cruxes_, dtype=np.int32), crux_die=np.array(dies_, dtype=object),
        crux_cont=np.array(conts_, dtype=object), corridor_len=np.array([L] * len(seeds_), dtype=np.int32))
    frac = float(np.mean(dangers_))
    print(f"-> {out}  ({len(seeds_)} rollouts, pool={pool}, seed={seed}, L={L}, frac_dangerous={frac:.3f}, "
          f"animals={len(all_animals)})")


def peek(seed: int = 0, L: int = 5) -> None:
    rng = np.random.default_rng(seed)
    pools = _pools(heldout=False)
    print("train pool sizes =", {k: len(v) for k, v in pools.items()})
    print("held-out pool sizes =", {k: len(v) for k, v in _pools(heldout=True).items()})
    all_animals = pools["dangerous"] + pools["safe"]
    for _ in range(4):
        animal = str(rng.choice(all_animals))
        k = int(rng.integers(1, L - 1))
        print("=" * 90)
        print(_seed_text(L, k, animal))
        print(f"  [hidden] {animal}: {'DANGEROUS' if DANGER[animal] else 'safe'}   crux cell={k}")
        for t, cell in enumerate(_rollout(L, k, animal)):
            print(f"  tick {t}: {cell}")


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
