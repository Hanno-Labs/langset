"""Maze search-frontier corpus for the superposition test.

The target is an accounting of a PARALLEL GREEDY SEARCH, not a per-cell snapshot. From S, branches
expand every step to all non-wall neighbours whose Manhattan distance to E strictly decreases
(monotone => no loops, finite). Two branches landing on the same cell MERGE; a branch with no
decreasing move DIES (dead-end pocket -> end token / drops off the frontier); a branch reaching E
SUCCEEDS. The FRONTIER at each tick = the set of active branch cells = the SUPERPOSITION the single
emitted latent must hold. The sequence of frontiers = the search unrolled in time (one latent/tick,
STOP when the frontier empties).

seed  = the maze ASCII (S/E/# marked).
fut   = per-tick frontier description (the superposition target for that tick).

This directly SUPERVISES the superposition (target literally is the branch set), unlike the TextWorld
corpus where a step-0 centroid only emerged indirectly from K same-seed rows.

Run:  python3 gen_maze.py            # prints a few example (seed, targets) rows
"""
from __future__ import annotations

import random
from typing import Optional


def _manhattan(a: tuple[int, int], e: tuple[int, int]) -> int:
    return abs(a[0] - e[0]) + abs(a[1] - e[1])


def _neighbors(rc: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = rc
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def gen_maze(rng: random.Random, h: int = 6, w: int = 6, wall_p: float = 0.28,
             rand_ends: bool = True, min_sep: Optional[int] = None
             ) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
    """S and E on ANY two open cells (never on a wall — they're placed after walls and override).
    rand_ends=False keeps the old corner-to-corner placement. min_sep = min Manhattan distance S<->E
    (defaults to half the grid diagonal) so endpoints aren't trivially adjacent."""
    grid = [["." for _ in range(w)] for _ in range(h)]
    cells = [(r, c) for r in range(h) for c in range(w)]
    if min_sep is None:
        min_sep = max(2, (h + w) // 2)
    if rand_ends:
        s, e = (0, 0), (h - 1, w - 1)
        for _ in range(200):
            a, b = rng.choice(cells), rng.choice(cells)
            if a != b and _manhattan(a, b) >= min_sep:
                s, e = a, b
                break
    else:
        s, e = (0, 0), (h - 1, w - 1)
    for r in range(h):
        for c in range(w):
            if (r, c) in (s, e):
                continue
            if rng.random() < wall_p:
                grid[r][c] = "#"
    grid[s[0]][s[1]] = "S"          # S/E placed last -> guaranteed open, never a wall
    grid[e[0]][e[1]] = "E"
    return grid, s, e


def frontier_sequence(grid: list[list[str]], s: tuple[int, int], e: tuple[int, int]
                      ) -> tuple[list[dict], bool]:
    """Parallel BFS reachability FLOOD from S (expands in ALL directions). Per-tick records:
    {active: wavefront cells first reached THIS tick, dead: wavefront cells that can't expand (no
    unvisited exit), reached_goal: bool, forks: int}. Runs until E is reached (SOLVABLE) or the flood
    exhausts every reachable cell without touching E (UNSOLVABLE). ALWAYS returns (seq, solvable) — no
    None; solvability is an emergent readout of running the search to completion, not a pre-filter."""
    h, w = len(grid), len(grid[0])

    def wall(rc):
        return grid[rc[0]][rc[1]] == "#"

    visited = {s}
    frontier = {s}
    reached = s == e
    seq: list[dict] = [{"active": [s], "dead": [], "reached_goal": reached, "forks": 0}]
    for _ in range(h * w):                      # BFS depth bounded by cell count
        if not frontier or reached:
            break
        nxt: set[tuple[int, int]] = set()
        dead: list[tuple[int, int]] = []
        forks = 0
        for cell in frontier:
            moves = [n for n in _neighbors(cell, h, w) if not wall(n) and n not in visited]
            if not moves:
                dead.append(cell)               # this wavefront cell can't advance (pocket / all-visited)
            else:
                if len(moves) > 1:
                    forks += 1                  # forks into >1 new cell
                nxt |= set(moves)               # set() => wavefronts meeting on a cell MERGE
        if not nxt:
            break                               # flood exhausted -> E unreachable
        visited |= nxt
        reached = e in nxt
        seq.append({"active": sorted(nxt), "dead": sorted(dead),
                    "reached_goal": reached, "forks": forks})
        frontier = nxt
    return seq, reached


def _cell(rc: tuple[int, int]) -> str:
    return f"r{rc[0]}c{rc[1]}"


def render_seed(grid: list[list[str]]) -> str:
    body = "\n".join("".join(row) for row in grid)
    h, w = len(grid), len(grid[0])
    return (f"Maze {h}x{w}. #=wall S=start E=exit. Flood the search from S one step at a time; "
            f"is E reachable?\n{body}")


def render_frontier(rec: dict, tick: int) -> str:
    active = rec["active"]
    n = len(active)
    cells = ", ".join(_cell(a) for a in active)
    tag = " GOAL" if rec["reached_goal"] else ""
    dead = f" | dead: {', '.join(_cell(d) for d in rec['dead'])}" if rec["dead"] else ""
    return f"tick {tick}: {n} cell{'s' if n != 1 else ''} at {cells}{dead}{tag}"


def render_verdict(solvable: bool, seq: list[dict]) -> str:
    """Terminal target the model emits AFTER the flood: the solvable/unsolvable readout."""
    flooded = len({c for rec in seq for c in rec["active"]})
    if solvable:
        return "verdict: SOLVABLE E reached"
    return f"verdict: UNSOLVABLE no route (flooded {flooded} cells, E unreachable)"


def build_row(rng: random.Random, h: int, w: int, wall_p: float,
              min_ticks: int = 3, need_fork: bool = True, rand_ends: bool = True) -> Optional[dict]:
    grid, s, e = gen_maze(rng, h, w, wall_p, rand_ends=rand_ends)
    seq, solvable = frontier_sequence(grid, s, e)
    if len(seq) < min_ticks:
        return None                             # too little search to be interesting (either class)
    max_branches = max(len(r["active"]) for r in seq)
    if need_fork and max_branches < 2:
        return None                             # want at least one real superposition
    fut = [render_frontier(rec, t) for t, rec in enumerate(seq)]
    fut.append(render_verdict(solvable, seq))   # TERMINAL verdict target = the model's solvable readout
    nbranch = [str(len(rec["active"])) for rec in seq] + ["0"]   # verdict tick has no wavefront cells
    solv = "yes" if solvable else "no"
    return {"seed": render_seed(grid), "fut_text": fut, "label_nbranch": nbranch,
            "label_solvable": [solv] * len(fut),               # per-tick solvable label (FSQ subspace-ready)
            "game": f"maze_{h}x{w}_{'S' if solvable else 'U'}_{rng.random():.9f}",  # unique id for holdout
            "solvable": solvable,
            "max_branches": max_branches, "n_ticks": len(seq),
            "total_forks": sum(r["forks"] for r in seq),
            "total_dead": sum(len(r["dead"]) for r in seq)}


def build_corpus(n_rows: int, out_path: str, seed: int = 0,
                 sizes: tuple[tuple[int, int, float], ...] = ((5, 5, 0.30), (6, 6, 0.32), (7, 7, 0.34),
                                                              (8, 8, 0.36), (9, 9, 0.38)),
                 min_ticks: int = 3, solv_frac: float = 0.55, rand_ends: bool = True) -> dict:
    """Generate n_rows fork-bearing mazes (random S/E, mixed sizes) as a BALANCED mix of SOLVABLE and
    UNSOLVABLE (default 55/45) -> npz for langset multi-latent. Columns: seed (maze ASCII), fut_text
    (per-tick BFS wavefront + terminal verdict), label_nbranch (per-tick wavefront count), label_solvable
    (per-tick yes/no), game (unique id for holdout). Returns coverage stats incl. solvable/unsolvable split."""
    import numpy as np
    rng = random.Random(seed)
    n_solv_target = int(round(n_rows * solv_frac))
    n_uns_target = n_rows - n_solv_target
    rows: list[dict] = []
    n_solv = n_uns = tries = 0
    dist_branches: dict[int, int] = {}
    while len(rows) < n_rows and tries < n_rows * 120:
        tries += 1
        h, w, wp = rng.choice(sizes)
        row = build_row(rng, h=h, w=w, wall_p=wp, min_ticks=min_ticks, rand_ends=rand_ends)
        if row is None:
            continue
        if row["solvable"] and n_solv >= n_solv_target:
            continue                                    # solvable bucket full -> keep hunting for unsolvable
        if not row["solvable"] and n_uns >= n_uns_target:
            continue
        rows.append(row)
        n_solv += int(row["solvable"]); n_uns += int(not row["solvable"])
        dist_branches[row["max_branches"]] = dist_branches.get(row["max_branches"], 0) + 1
    cols = {"seed": [r["seed"] for r in rows],
            "fut_text": [r["fut_text"] for r in rows],
            "label_nbranch": [r["label_nbranch"] for r in rows],
            "label_solvable": [r["label_solvable"] for r in rows],
            "game": [r["game"] for r in rows]}
    np.savez(out_path, **{k: np.array(v, dtype=object) for k, v in cols.items()})
    stats = {"n_rows": len(rows), "n_solvable": n_solv, "n_unsolvable": n_uns, "tries": tries,
             "mean_ticks": round(sum(r["n_ticks"] for r in rows) / max(len(rows), 1), 2),
             "mean_forks": round(sum(r["total_forks"] for r in rows) / max(len(rows), 1), 2),
             "frac_with_dead": round(sum(1 for r in rows if r["total_dead"] > 0) / max(len(rows), 1), 3),
             "max_branch_hist": dict(sorted(dist_branches.items()))}
    return stats


if __name__ == "__main__":
    import sys
    # build a corpus:  python gen_maze.py build <n_rows> <out.npz> [seed]
    # use a DIFFERENT seed for the eval corpus (e.g. 999) so it's disjoint from the training set.
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
        out = sys.argv[3] if len(sys.argv) > 3 else "maze_search.npz"
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        import json
        print(json.dumps(build_corpus(n, out, seed=seed), indent=2))
        print(f"-> {out}  (seed={seed})")
    else:
        rng = random.Random(7)
        shown = 0
        tries = 0
        while shown < 3 and tries < 5000:
            tries += 1
            row = build_row(rng, h=6, w=6, wall_p=0.28)
            if row is None:
                continue
            shown += 1
            print("=" * 72)
            print(f"[example {shown}] max_branches={row['max_branches']} ticks={row['n_ticks']} "
                  f"forks={row['total_forks']} dead={row['total_dead']}")
            print("--- SEED ---")
            print(row["seed"])
            print("--- TARGETS (one latent per tick = the frontier superposition) ---")
            for f in row["fut_text"]:
                print("  " + f)
        print("=" * 72)
        print(f"(generated {shown} example rows in {tries} tries)")
