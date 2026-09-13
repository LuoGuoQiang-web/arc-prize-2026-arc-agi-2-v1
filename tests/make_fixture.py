"""
Synthetic ARC-AGI-2 fixture builder for local end-to-end testing.

The transforms below are implemented independently of the engine's DSL on purpose:
if the engine solved them using its own primitives, a bug in a primitive would be
invisible. Here every expected output is computed by obvious, separate code.

Layout produced (mirrors the Kaggle competition files):
    <out>/arc-agi_evaluation_challenges.json
    <out>/arc-agi_evaluation_solutions.json
    <out>/arc-agi_test_challenges.json
    <out>/sample_submission.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def rand_grid(rng, h, w, colors=(0, 1, 2, 3)):
    return [[rng.choice(colors) for _ in range(w)] for _ in range(h)]


# ---- independent reference transforms -------------------------------------------------

def t_rot90(g):
    h, w = len(g), len(g[0])
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def t_cmap(g, mapping):
    return [[mapping.get(v, v) for v in row] for row in g]


def t_tile2x2(g):
    out = []
    for _ in range(2):
        for row in g:
            out.append(row + row)
    return out


def t_gravity_down(g, bg=0):
    h, w = len(g), len(g[0])
    out = [[bg] * w for _ in range(h)]
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c] != bg]
        for i, v in enumerate(col):
            out[h - len(col) + i][c] = v
    return out


def t_crop_nonzero(g, bg=0):
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def t_mirror_h(g, bg=0):
    h, w = len(g), len(g[0])
    out = [row[:] for row in g]
    for r in range(h):
        for c in range(w):
            if out[r][c] == bg and g[r][w - 1 - c] != bg:
                out[r][c] = g[r][w - 1 - c]
    return out


def t_scale_up2(g):
    out = []
    for row in g:
        big = [v for v in row for _ in range(2)]
        out.append(big[:])
        out.append(big[:])
    return out


def t_flipv(g):
    return [row[:] for row in g[::-1]]


# ---- fixture assembly -----------------------------------------------------------------

def build_fixture(out_dir: Path, seed: int = 7) -> dict:
    rng = random.Random(seed)
    tasks = {}

    def sq_grid():
        n = rng.choice([4, 5])
        return rand_grid(rng, n, n, (0, 1, 2, 3))

    tasks["t001rot90"] = {"fn": t_rot90, "gen": sq_grid}
    tasks["t002recolor"] = {"fn": lambda g: t_cmap(g, {0: 0, 1: 4, 2: 6, 3: 1}),
                            "gen": lambda: rand_grid(rng, 4, 6, (0, 1, 2, 3))}
    tasks["t003tile"] = {"fn": t_tile2x2, "gen": lambda: rand_grid(rng, 3, 4, (0, 1, 2))}
    tasks["t004gravity"] = {"fn": t_gravity_down, "gen": lambda: rand_grid(rng, 5, 5, (0, 0, 1, 2, 3))}
    tasks["t006const"] = {"fn": lambda g: [[7, 7, 7], [7, 0, 7], [7, 7, 7]],
                          "gen": lambda: rand_grid(rng, rng.choice([3, 4]), rng.choice([3, 4, 5]), (0, 1, 2, 3))}
    tasks["t008scaleup"] = {"fn": t_scale_up2, "gen": lambda: rand_grid(rng, 3, 3, (0, 1, 2, 3))}
    tasks["t009flipv_recolor"] = {"fn": lambda g: t_cmap(t_flipv(g), {0: 0, 1: 5, 2: 5, 3: 8}),
                                  "gen": lambda: rand_grid(rng, 4, 5, (0, 1, 2, 3))}
    tasks["t010crop_rot90"] = {"fn": lambda g: t_rot90(t_crop_nonzero(g)),
                               "gen": lambda: rand_grid(rng, 6, 6, (0, 0, 0, 1, 2, 3))}
    tasks["t005crop"] = {"fn": t_crop_nonzero, "gen": lambda: rand_grid(rng, 6, 7, (0, 0, 0, 1, 2, 3))}

    def mirrored_gen():
        # left half random, right half background -> mirror-fill is well defined
        h, w = rng.choice([4, 5]), rng.choice([5, 6])
        g = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w // 2):
                g[r][c] = rng.choice([1, 2, 3])
        return g

    tasks["t007mirror"] = {"fn": t_mirror_h, "gen": mirrored_gen}

    # unsolvable: independent random outputs (must fall back to priors and be scored as failures)
    tasks["t011noise"] = {"fn": None, "gen": lambda: rand_grid(rng, 4, 4, (0, 1, 2, 3))}
    tasks["t012noise2"] = {"fn": None, "gen": lambda: rand_grid(rng, 3, 5, (0, 1, 2, 3))}

    n_test_per_task = {"t003tile": 2, "t010crop_rot90": 3}

    challenges, solutions = {}, {}
    for tid, spec in tasks.items():
        train = []
        for _ in range(4):
            g = spec["gen"]()
            out = spec["fn"](g) if spec["fn"] else rand_grid(rng, rng.choice([2, 5]), rng.choice([2, 4]), (0, 1, 2, 3))
            train.append({"input": g, "output": out})
        n_test = n_test_per_task.get(tid, 1)
        test = []
        outs = []
        for _ in range(n_test):
            g = spec["gen"]()
            outs.append(spec["fn"](g) if spec["fn"] else rand_grid(rng, rng.choice([2, 5]), rng.choice([2, 4]), (0, 1, 2, 3)))
            test.append({"input": g})
        challenges[tid] = {"train": train, "test": test}
        solutions[tid] = outs

    sample = {tid: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(len(solutions[tid]))]
              for tid in sorted(solutions)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in [("arc-agi_evaluation_challenges.json", challenges),
                          ("arc-agi_evaluation_solutions.json", solutions),
                          ("arc-agi_test_challenges.json", challenges),
                          ("sample_submission.json", sample)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    return {"n_tasks": len(challenges), "dir": str(out_dir),
            "solvable": [t for t in tasks if tasks[t]["fn"] is not None]}


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixture_data")
    info = build_fixture(target)
    print(json.dumps(info, indent=2))
