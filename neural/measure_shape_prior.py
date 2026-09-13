"""How much would a demonstration-shape prior buy on candidate selection?

Motivation: coverage is already 100% and the binding constraint is precision. Inference
augmentation voting -- the reference's fix -- does not fit in a 12 h T4x2 session
(240 tasks x ~180 s already needs ~12 h), so the useful levers are the ones that cost no
extra generation time. One such lever is a shape prior: if every demonstration maps
inputs to outputs of the same shape, then for a shape-preserving task a candidate with a
different shape is almost certainly wrong, and it can be filtered for free.

This measures the ceiling of that prior on the public splits: how often the "all
demonstrations preserve shape" pattern holds, and how often it then correctly predicts
the test output's shape.

Pure CPU, no model weights needed.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(os.path.dirname(HERE), "competition_data")

SPLITS = [
    ("evaluation", "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    ("training", "arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
]


def shape(grid):
    return (len(grid), len(grid[0]))


def main() -> int:
    for label, ch_name, sol_name in SPLITS:
        ch_path = os.path.join(COMP, ch_name)
        sol_path = os.path.join(COMP, sol_name)
        if not (os.path.isfile(ch_path) and os.path.isfile(sol_path)):
            print(f"{label}: data missing, skipped")
            continue
        challenges = json.load(open(ch_path, encoding="utf-8"))
        solutions = json.load(open(sol_path, encoding="utf-8"))

        n_tasks = 0
        same_shape_tasks = 0
        # of the shape-preserving tasks, how often is the truth actually input-shaped?
        pred_ok = 0
        pred_total = 0
        # identity: how often is the answer exactly the input? (upper bound on the
        # trivial fallback, which we already know is zero -- included as a control)
        identity_hits = 0
        test_inputs = 0

        # how many distinct output shapes do the demos show, per task
        shape_hist = Counter()

        for tid, task in challenges.items():
            truth = solutions.get(tid)
            if truth is None:
                continue
            n_tasks += 1
            demos = task["train"]
            all_same = all(shape(d["input"]) == shape(d["output"]) for d in demos)
            out_shapes = {shape(d["output"]) for d in demos}
            shape_hist[len(out_shapes)] += 1
            if all_same:
                same_shape_tasks += 1
                for i, expected in enumerate(truth):
                    pred_total += 1
                    if i < len(task["test"]):
                        tin = task["test"][i]["input"]
                        if shape(tin) == shape(expected):
                            pred_ok += 1
            for i, expected in enumerate(truth):
                test_inputs += 1
                if i < len(task["test"]) and task["test"][i]["input"] == expected:
                    identity_hits += 1

        print(f"\n=== {label}: {n_tasks} tasks, {test_inputs} test inputs ===")
        pct = 100.0 * same_shape_tasks / max(1, n_tasks)
        print(f"  tasks where EVERY demonstration preserves shape : {same_shape_tasks}/{n_tasks}"
              f"  ({pct:.1f}%)")
        if pred_total:
            acc = 100.0 * pred_ok / pred_total
            print(f"  ...of those, test output really is input-shaped: {pred_ok}/{pred_total}"
                  f"  ({acc:.1f}%)")
            print(f"  => filtering non-input-shaped candidates on those tasks would remove "
                  f"wrong shapes; it would also be WRONG on {pred_total - pred_ok} input(s)")
        print(f"  distinct demo output shapes per task histogram: "
              f"{dict(sorted(shape_hist.items()))}")
        print(f"  [control] answer is exactly the input: {identity_hits}/{test_inputs} "
              f"({100.0 * identity_hits / max(1, test_inputs):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
