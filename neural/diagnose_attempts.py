"""Diagnose the neural attempt pairs actually written by the evaluation run.

The report only records pool *sizes*; the submitted grids themselves say whether the
pool is actually useful. Two degeneracies would silently cap the score:

* ``attempt_1 == attempt_2`` -- the second slot carries no independent hypothesis, so
  pass@2 buys nothing (this is exactly the bug the old fallback had, at 70% on training).
* an attempt that is byte-identical to the *fallback* grid -- meaning the neural path
  produced nothing for that test input and the layer we measured at exactly 0 is what
  got submitted.

Pure CPU: analyses an artifact we already downloaded.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "eval_out", "submission.json")
COMP = os.path.join(os.path.dirname(HERE), "competition_data")


def grid_key(g):
    return (len(g), len(g[0]) if g else 0, tuple(tuple(r) for r in g))


def main() -> int:
    if not os.path.isfile(SUB):
        print(f"missing {SUB}")
        return 1
    sub = json.load(open(SUB, encoding="utf-8"))
    ch = json.load(open(os.path.join(COMP, "arc-agi_evaluation_challenges.json"),
                        encoding="utf-8"))

    n_inputs = 0
    identical = 0
    shapes = Counter()
    is_identity_fallback = 0
    is_constant_fallback = 0
    either_matches_fallback = 0
    shp_match_input = 0

    for tid, attempts in sub.items():
        task = ch.get(tid)
        if task is None:
            continue
        for i, entry in enumerate(attempts):
            n_inputs += 1
            a1, a2 = entry["attempt_1"], entry["attempt_2"]
            if grid_key(a1) == grid_key(a2):
                identical += 1
            for a in (a1, a2):
                shapes[(len(a), len(a[0]))] += 1
            tin = task["test"][i]["input"] if i < len(task["test"]) else None
            if tin is not None:
                if grid_key(a1) == grid_key(tin):
                    is_identity_fallback += 1
                if len(a1) == len(tin) and len(a1[0]) == len(tin[0]):
                    shp_match_input += 1
                # the constant-fill fallback: uniform grid of one colour
                flat = {c for row in a1 for c in row}
                if len(flat) == 1:
                    is_constant_fallback += 1
                if grid_key(a1) == grid_key(tin) or len(flat) == 1:
                    either_matches_fallback += 1

    print(f"test inputs analysed: {n_inputs}  (tasks: {len(sub)})")
    print()
    print(f"  attempt_1 == attempt_2                 : {identical}"
          f"  ({100.0 * identical / max(1, n_inputs):.1f}%)")
    print(f"  attempt_1 is exactly the test input    : {is_identity_fallback}"
          f"  ({100.0 * is_identity_fallback / max(1, n_inputs):.1f}%)")
    print(f"  attempt_1 is a uniform (one-colour) grid: {is_constant_fallback}"
          f"  ({100.0 * is_constant_fallback / max(1, n_inputs):.1f}%)")
    print(f"  attempt_1 looks like a fallback        : {either_matches_fallback}"
          f"  ({100.0 * either_matches_fallback / max(1, n_inputs):.1f}%)")
    print(f"  attempt_1 shape == test-input shape    : {shp_match_input}"
          f"  ({100.0 * shp_match_input / max(1, n_inputs):.1f}%)")
    print()
    print("  most common attempt_1 shapes:",
          ", ".join(f"{s}x{c}" for (s, c), n in shapes.most_common(8) for _ in [0]))

    print()
    if identical == 0:
        print("VERDICT: every second slot carries an independent hypothesis -- the "
              "distinctness rule holds in practice, not just in the unit test.")
    else:
        print(f"VERDICT: {identical} input(s) got a duplicated second attempt; the "
              f"distinctness rule is NOT holding and needs attention.")
    if either_matches_fallback:
        print(f"NOTE: {either_matches_fallback} input(s) fell back to the floor layer, "
              f"which is measured at exactly 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
