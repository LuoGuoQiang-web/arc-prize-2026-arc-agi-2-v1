"""Is "no demonstration output is uniform" enough to reject a uniform candidate?

Motivation, from the evaluation artifact: 10 of 32 attempt_1 grids (31%) are single-colour
uniform grids PRODUCED BY THE MODEL (recorded source = "neural"), and only one came from
the fallback layer. So the binding constraint is candidate GENERATION quality, not
coverage and not selection -- and more time per task cannot fix it.

The likely mechanism is the coverage guard that keeps the arg-max token when no child
clears the p>0.2 threshold: in an uncertain state the arg-max is often one repeated token,
i.e. a degenerate uniform grid, which then ranks first.

A uniform candidate can be rejected cheaply when every demonstration output is
non-uniform. As with the shape prior, the rule is only usable if it has a perfect record:
filtering deletes the correct answer whenever the rule is wrong.

Pure CPU.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(os.path.dirname(HERE), "competition_data")

SPLITS = [
    ("evaluation", "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    ("training", "arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
]


def uniform(grid):
    return len({cell for row in grid for cell in row}) == 1


def analyse(label, ch_name, sol_name):
    ch = json.load(open(os.path.join(COMP, ch_name), encoding="utf-8"))
    sol = json.load(open(os.path.join(COMP, sol_name), encoding="utf-8"))

    n_inputs = 0
    applicable = 0          # no demo output is uniform -> a uniform candidate is suspect
    uniform_truth = 0       # ...and yet the real answer IS uniform  -> counterexample
    uniform_truth_tasks = []
    demo_uniform_tasks = 0  # tasks where some demo output IS uniform (rule not applicable)

    for tid, task in ch.items():
        truth = sol.get(tid)
        if truth is None:
            continue
        demos = task["train"]
        any_demo_uniform = any(uniform(d["output"]) for d in demos)
        if any_demo_uniform:
            demo_uniform_tasks += 1
        for i, expected in enumerate(truth):
            n_inputs += 1
            if any_demo_uniform:
                continue
            applicable += 1
            if uniform(expected):
                uniform_truth += 1
                if len(uniform_truth_tasks) < 6:
                    uniform_truth_tasks.append((tid, i, len(expected), len(expected[0])))

    print(f"\n=== {label}: {n_inputs} test inputs ===")
    print(f"  tasks where some demonstration output IS uniform : {demo_uniform_tasks}"
          f"  (rule not applicable there)")
    print(f"  inputs where NO demonstration output is uniform  : {applicable}")
    print(f"    ...yet the true answer IS uniform             : {uniform_truth}"
          f"  ({100.0 * uniform_truth / max(1, applicable):.2f}%)")
    if uniform_truth:
        print(f"    COUNTEREXAMPLES (would be wrongly deleted): {uniform_truth_tasks}")
        print("    => the rule is UNSOUND as a hard filter.")
    else:
        print("    => zero counterexamples: rejecting uniform candidates on those tasks "
              "is SAFE as a hard filter.")


def main() -> int:
    for label, ch, sol in SPLITS:
        analyse(label, ch, sol)
    print("\nnote: 'safe' means it never deletes a correct answer on the public splits. "
          "It says nothing about how often it helps -- that needs a GPU run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
