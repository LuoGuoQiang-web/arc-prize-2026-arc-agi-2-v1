"""Measure how often the fallback pair is right on the public splits.

The fallback only matters for tasks the neural path fails to answer, but on a hidden
set that is expected to be a large fraction -- so its quality is worth a measurement
rather than an opinion. Compares the OLD fallback (mode-colour fill + all-zeros, which
collapsed to the same grid) against the NEW one (identity + fg/bg swap).

Pure CPU: no GPU and no model weights needed.
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, "arc26_solver.py")
COMP = os.path.join(os.path.dirname(HERE), "competition_data")

spec = importlib.util.spec_from_file_location("arc26_solver", SOLVER)
m = importlib.util.module_from_spec(spec)
sys.modules["arc26_solver"] = m
spec.loader.exec_module(m)


def old_fallback_pair(task, idx):
    """Reproduce the previous behaviour for comparison."""
    tests = task.get("test", [])
    shape = (1, 1)
    if 0 <= idx < len(tests):
        arr = m.validate_grid(tests[idx].get("input"))
        if arr is not None:
            shape = arr.shape
    colour = m.mode_colour(task)
    return np.full(shape, colour, dtype=int), np.zeros(shape, dtype=int)


def run(split_challenges, split_solutions, label):
    challenges = json.load(open(split_challenges, encoding="utf-8"))
    solutions = json.load(open(split_solutions, encoding="utf-8")) if split_solutions else None
    if solutions is None:
        print(f"{label}: no solutions, skipped")
        return None

    stats = {"old": {"a1": 0, "a2": 0, "either": 0, "identical": 0},
             "new": {"a1": 0, "a2": 0, "either": 0, "identical": 0}}
    n = 0
    for tid, task in challenges.items():
        truth = solutions.get(tid)
        if truth is None:
            continue
        for i, expected in enumerate(truth):
            exp = np.array(expected, dtype=int)
            for name, fn in (("old", old_fallback_pair), ("new", m.fallback_pair)):
                a1, a2 = fn(task, i)
                stats[name]["a1"] += int(np.array_equal(a1, exp))
                stats[name]["a2"] += int(np.array_equal(a2, exp))
                stats[name]["either"] += int(np.array_equal(a1, exp) or np.array_equal(a2, exp))
                stats[name]["identical"] += int(np.array_equal(a1, a2))
            n += 1

    print(f"\n=== {label}: {n} test inputs over {len(challenges)} tasks ===")
    print(f"{'variant':<6} {'a1 hits':>8} {'a2 hits':>8} {'either':>8} {'a1==a2':>8} {'either%':>9}")
    for name in ("old", "new"):
        s = stats[name]
        print(f"{name:<6} {s['a1']:>8} {s['a2']:>8} {s['either']:>8} {s['identical']:>8} "
              f"{100.0 * s['either'] / max(1, n):>8.2f}%")
    delta = stats["new"]["either"] - stats["old"]["either"]
    print(f"delta (new - old) on 'either': {delta:+d} inputs "
          f"({100.0 * delta / max(1, n):+.2f} pp)")
    return stats


def main():
    results = {}
    for label, ch, sol in (
        ("evaluation", "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
        ("training", "arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
    ):
        cp, sp = os.path.join(COMP, ch), os.path.join(COMP, sol)
        if os.path.isfile(cp) and os.path.isfile(sp):
            results[label] = run(cp, sp, label)
    print("\ninterpretation: 'a1==a2' is the bug being fixed -- the old pair wasted the "
          "second attempt slot by returning the same grid twice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
