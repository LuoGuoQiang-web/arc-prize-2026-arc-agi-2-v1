"""Two more candidate filters, measured for soundness before they are allowed to reject.

The methodology is the one that has been paying off: a filter may only be used when it has
a PERFECT record on the public splits, because filtering deletes the correct answer
whenever the rule is wrong. Shape (identity 836/836, uniform scale 56/56) and degeneracy
(1219/1219) both qualified; transpose and constant-output-shape both looked usable and
were rejected for having counterexamples.

Two rules are tested here:

R1 colour closure -- if every demonstration output uses only colours present in its own
   input, then the test output should too. (ARC tasks frequently introduce a colour, so
   this may well be unsound -- which is exactly what a measurement is for.)

R2 symmetry inheritance -- if every demonstration output is symmetric under some
   transform that its input is not, ... deliberately NOT tested here; see the note at the
   end. Only rules we can express faithfully are worth measuring.

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


def colours(grid):
    return {cell for row in grid for cell in row}


def closes_within_inputs(task) -> bool:
    """True when every demonstration output draws only on its input's colours."""
    demos = task["train"]
    if not demos:
        return False
    return all(colours(d["output"]) <= colours(d["input"]) for d in demos)


def analyse(label, ch_name, sol_name):
    ch = json.load(open(os.path.join(COMP, ch_name), encoding="utf-8"))
    sol = json.load(open(os.path.join(COMP, sol_name), encoding="utf-8"))

    n_inputs = 0
    applicable = 0
    counterexamples = []
    # how many colours a typical test output adds beyond the input, when the rule holds
    extra_hist = {}

    for tid, task in ch.items():
        truth = sol.get(tid)
        if truth is None:
            continue
        applies = closes_within_inputs(task)
        for i, expected in enumerate(truth):
            n_inputs += 1
            if not applies or i >= len(task["test"]):
                continue
            applicable += 1
            tin = task["test"][i]["input"]
            extra = colours(expected) - colours(tin)
            extra_hist[len(extra)] = extra_hist.get(len(extra), 0) + 1
            if extra and len(counterexamples) < 8:
                counterexamples.append((tid, i, sorted(extra)))

    print(f"\n=== {label}: {n_inputs} test inputs ===")
    print(f"  inputs where the colour-closure rule applies     : {applicable}")
    bad = sum(n for k, n in extra_hist.items() if k > 0)
    print(f"    ...true output introduces a NEW colour          : {bad}"
          f"  ({100.0 * bad / max(1, applicable):.2f}%)")
    print(f"  extra-colour-count histogram: {dict(sorted(extra_hist.items()))}")
    if counterexamples:
        print(f"  COUNTEREXAMPLES (first {len(counterexamples)}): {counterexamples}")
        print("  => UNSOUND: rejecting candidates that introduce a colour would delete "
              "correct answers.")
    else:
        print("  => zero counterexamples: the rule is SAFE as a hard filter.")


def main() -> int:
    for label, ch, sol in SPLITS:
        analyse(label, ch, sol)
    print()
    print("note: a rule can be SAFE and still useless -- it only helps if wrong candidates "
          "actually violate it. Safety is necessary, not sufficient, and the payoff still "
          "needs a GPU run to measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
