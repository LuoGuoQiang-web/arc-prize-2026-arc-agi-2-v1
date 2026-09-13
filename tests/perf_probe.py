"""
Performance probe: how long does one realistic ARC-AGI-2 task take?

ARC-AGI-2 grids run up to 30x30 with 2-8 demonstration pairs (typically 3-4) and
1-3 test inputs. This measures the worst-case-ish shape so we can size the Kaggle
budget: total_time ~= n_tasks * per_task_time.
"""

from __future__ import annotations

import importlib.util
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402

spec = importlib.util.spec_from_file_location("arc_prize_v1", HERE.parent / "arc_prize_v1.py")
arc = importlib.util.module_from_spec(spec)
sys.modules["arc_prize_v1"] = arc
spec.loader.exec_module(arc)


def build_task(fn, rng, h, w, n_train, n_test, colors=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    train = []
    for _ in range(n_train):
        g = make_fixture.rand_grid(rng, h, w, colors)
        train.append({"input": g, "output": fn(g)})
    test = [{"input": make_fixture.rand_grid(rng, h, w, colors)} for _ in range(n_test)]
    return {"train": train, "test": test}


def main():
    cfg = arc.make_config({"enable_depth2": True})
    rng = random.Random(11)
    cases = [
        ("30x30 rot90, 4 demos, 3 tests",
         lambda g: make_fixture.t_rot90(g) if len(g) != len(g[0]) else make_fixture.t_rot90(g), 30, 30, 4, 3),
        ("20x25 recolor, 6 demos, 2 tests",
         lambda g: make_fixture.t_cmap(g, {0: 0, 1: 4, 2: 6, 3: 1, 4: 8, 5: 3, 6: 2, 7: 5, 8: 9, 9: 7}), 20, 25, 6, 2),
        ("30x30 tile2x2, 3 demos, 1 test",
         make_fixture.t_tile2x2, 30, 30, 3, 1),
        ("30x30 crop->rot90 (depth 2), 4 demos, 2 tests",
         lambda g: make_fixture.t_rot90(make_fixture.t_crop_nonzero(g)), 30, 30, 4, 2),
    ]
    total = 0.0
    print(f"{'case':46s} {'sec':>7s}  valid  program")
    for name, fn, h, w, ntr, nte in cases:
        task = build_task(fn, rng, h, w, ntr, nte)
        t0 = time.monotonic()
        res = arc.solve_task(task, cfg, time.monotonic() + 600)
        dt = time.monotonic() - t0
        total += dt
        print(f"{name:46s} {dt:7.2f}  {res['diagnostics']['n_valid']:5d}  {res['diagnostics']['top_program']}")
        assert res["diagnostics"]["source"] == "search", f"{name} fell back to priors"
    n_tasks = 240
    per_task = total / len(cases)
    print(f"\nmean per task: {per_task:.2f}s (worst {max(total/len(cases), 0):.2f}s)")
    print(f"projected for {n_tasks} Kaggle test tasks: {per_task * n_tasks / 60:.1f} min")
    print(f"projected for 120 evaluation tasks: {per_task * 120 / 60:.1f} min")


if __name__ == "__main__":
    main()
