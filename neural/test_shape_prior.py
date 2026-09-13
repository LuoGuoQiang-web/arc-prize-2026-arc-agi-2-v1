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

# 2. a task whose shape relation no SOUND rule explains -> untouched.
#    NOTE: `growing` above is a uniform 2x upscale, which is a measured-sound rule and is
#    therefore covered by design (see #5 below) -- it is deliberately not used here.
nonrule = {"train": [{"input": [[1, 2, 3]], "output": [[7], [8]]}],   # 1x3 -> 2x1
           "test": [{"input": [[1, 2, 3]]}]}
kept2, dropped2 = m.apply_shape_prior(nonrule, [[1, 2, 3]], pool)
check("a task no sound shape rule explains leaves the pool untouched",
      dropped2 == 0 and len(kept2) == 4, f"kept={len(kept2)} dropped={dropped2}")

# 3. coverage guard: every candidate has the wrong shape -> keep them all
wrong = [cand([[0, 0, 0]]), cand([[0, 0, 0], [0, 0, 0]])]
kept3, dropped3 = m.apply_shape_prior(preserving, [[1, 2], [3, 4]], wrong)
check("the filter never empties the pool (coverage outranks precision)",
      len(kept3) == 2 and dropped3 == 0, f"kept={len(kept3)}")

# 4. degenerate inputs
check("empty pool is a no-op", m.apply_shape_prior(preserving, [[1]], []) == ([], 0))
check("invalid test input is a no-op",
      len(m.apply_shape_prior(preserving, None, pool)[0]) == 4)

# 5. general shape rules, and the ones deliberately EXCLUDED as unsound
upscale = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1] * 4] * 4}],   # 2x2 -> 4x4
           "test": [{"input": [[5, 6], [7, 8]]}]}
downscale = {"train": [{"input": [[1] * 4] * 4, "output": [[1, 2], [3, 4]]}],  # 4x4 -> 2x2
             "test": [{"input": [[1] * 6] * 6}]}

check("uniform 2x upscale is predicted",
      m.predicted_output_shape(upscale, [[5, 6], [7, 8]]) == (4, 4),
      f"-> {m.predicted_output_shape(upscale, [[5, 6], [7, 8]])}")
check("uniform 2x downscale is predicted",
      m.predicted_output_shape(downscale, [[1] * 6] * 6) == (3, 3),
      f"-> {m.predicted_output_shape(downscale, [[1] * 6] * 6)}")

pool_up = [cand([[0] * 4] * 4), cand([[0] * 2] * 2)]
kept_up, dropped_up = m.apply_shape_prior(upscale, [[5, 6], [7, 8]], pool_up)
check("the upscale rule filters the pool", dropped_up == 1 and kept_up[0].grid.shape == (4, 4))

# transpose fits these demonstrations perfectly but is measured UNSOUND, so it must not
# be allowed to filter: every transpose counterexample is a task whose answer is
# input-shaped, i.e. filtering would delete the correct candidate.
transpose_task = {"train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[1, 4], [2, 5], [3, 6]]}],
                  "test": [{"input": [[7, 8, 9], [1, 2, 3]]}]}
check("the transpose rule is NOT used even when it fits the demonstrations",
      m.predicted_output_shape(transpose_task, [[7, 8, 9], [1, 2, 3]]) is None)
kept_t, dropped_t = m.apply_shape_prior(transpose_task, [[7, 8, 9], [1, 2, 3]],
                                        [cand([[0, 0], [0, 0]]), cand([[0, 0, 0]] * 2)])
check("...so a transpose-shaped task's pool is left untouched",
      dropped_t == 0 and len(kept_t) == 2)

# constant output shape is likewise measured UNSOUND
const_task = {"train": [{"input": [[1, 2]], "output": [[9]]},
                        {"input": [[1, 2, 3]], "output": [[9]]}],
              "test": [{"input": [[1, 2, 3, 4]]}]}
check("the constant-output-shape rule is NOT used",
      m.predicted_output_shape(const_task, [[1, 2, 3, 4]]) is None)

# identity + a disagreeing scale rule -> ambiguous -> no filtering
ambiguous = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                       {"input": [[1]], "output": [[1]]}],
             "test": [{"input": [[5, 6], [7, 8]]}]}
check("identity reproduces this task",
      m.predicted_output_shape(ambiguous, [[5, 6], [7, 8]]) == (2, 2))

# a task where no sound rule fits at all
norule = {"train": [{"input": [[1, 2, 3]], "output": [[1], [2]]}],   # 1x3 -> 2x1
          "test": [{"input": [[1, 2, 3]]}]}
check("no sound rule fits -> no prediction, pool untouched",
      m.predicted_output_shape(norule, [[1, 2, 3]]) is None and
      m.apply_shape_prior(norule, [[1, 2, 3]], pool)[1] == 0)

# 6. re-verify the measured claim through the real function on real data
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
