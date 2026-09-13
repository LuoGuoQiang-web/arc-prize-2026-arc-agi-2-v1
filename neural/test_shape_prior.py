"""Tests for the demonstration-shape prior, including a re-verification on real data."""
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

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def cand(grid, nll=1.0):
    return m.Candidate.make(np.array(grid, dtype=int), "neural", nll=nll)


preserving = {"train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
                        {"input": [[1]], "output": [[2]]}],
              "test": [{"input": [[1, 2], [3, 4]]}]}
growing = {"train": [{"input": [[1]], "output": [[1, 1], [1, 1]]}],
           "test": [{"input": [[1]]}]}

check("shape_preserving_task detects the preserving case",
      m.shape_preserving_task(preserving) is True)
check("shape_preserving_task detects the growing case",
      m.shape_preserving_task(growing) is False)
check("empty demos are not treated as preserving",
      m.shape_preserving_task({"train": [], "test": []}) is False)

# 1. mixed pool on a preserving task -> wrong shapes dropped
pool = [cand([[0, 0], [0, 0]]), cand([[0, 0, 0]]), cand([[0, 0], [0, 0], [0, 0]]),
        cand([[1, 1], [1, 1]])]
kept, dropped = m.apply_shape_prior(preserving, [[1, 2], [3, 4]], pool)
check("wrong shapes are dropped on a preserving task", dropped == 2 and len(kept) == 2,
      f"kept={len(kept)} dropped={dropped}")
check("the surviving candidates all match the input shape",
      all(c.grid.shape == (2, 2) for c in kept))

# 2. non-preserving task -> untouched
kept2, dropped2 = m.apply_shape_prior(growing, [[1]], pool)
check("a non-preserving task leaves the pool untouched", dropped2 == 0 and len(kept2) == 4)

# 3. coverage guard: every candidate has the wrong shape -> keep them all
wrong = [cand([[0, 0, 0]]), cand([[0, 0, 0], [0, 0, 0]])]
kept3, dropped3 = m.apply_shape_prior(preserving, [[1, 2], [3, 4]], wrong)
check("the filter never empties the pool (coverage outranks precision)",
      len(kept3) == 2 and dropped3 == 0, f"kept={len(kept3)}")

# 4. degenerate inputs
check("empty pool is a no-op", m.apply_shape_prior(preserving, [[1]], []) == ([], 0))
check("invalid test input is a no-op",
      len(m.apply_shape_prior(preserving, None, pool)[0]) == 4)

# 5. re-verify the measured claim through the real function on real data
ch_path = os.path.join(COMP, "arc-agi_evaluation_challenges.json")
sol_path = os.path.join(COMP, "arc-agi_evaluation_solutions.json")
if os.path.isfile(ch_path) and os.path.isfile(sol_path):
    challenges = json.load(open(ch_path, encoding="utf-8"))
    solutions = json.load(open(sol_path, encoding="utf-8"))
    n_preserving = 0
    n_predicted = 0
    n_counterexample = 0
    for tid, task in challenges.items():
        truth = solutions.get(tid)
        if truth is None or not m.shape_preserving_task(task):
            continue
        n_preserving += 1
        for i, expected in enumerate(truth):
            if i >= len(task["test"]):
                continue
            n_predicted += 1
            tin = task["test"][i]["input"]
            if len(tin) != len(expected) or len(tin[0]) != len(expected[0]):
                n_counterexample += 1
    check("evaluation: shape-preserving tasks found through the real function",
          n_preserving == 81, f"found {n_preserving} (measured 81)")
    check("evaluation: the prior had ZERO counterexamples",
          n_counterexample == 0 and n_predicted == 117,
          f"{n_predicted} predictions, {n_counterexample} counterexamples")
else:
    print("(skip: evaluation data not found)")

print()
if failures:
    print(f"FAILURES: {failures}")
    raise SystemExit(1)
print("all shape-prior checks passed")
