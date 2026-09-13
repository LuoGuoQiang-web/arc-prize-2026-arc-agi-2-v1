#!/usr/bin/env python3
"""End-to-end integration test: run the real ``main()`` and grade its artifact.

Every other test in this directory exercises a function in isolation. This one is the only
check that the whole entry point still works after a series of surgical edits (priors,
top-k forced branching, scheduler caps, flag threading) -- the kind of change that unit
tests happily keep passing while the assembled program breaks.

It runs the solver as a subprocess against the local competition data with no model
available (so the neural path degrades to the floor layer, which is the point: the
submission must still be complete and legal). Deliberately CPU-only and tiny.

Asserts the properties that the competition grader cares about:
  * the process exits 0 and does not raise
  * every challenge task id is present -- the file is pre-seeded, so a partial run must
    still emit a complete file (we lost one submission to a 120/240 row mismatch)
  * each task has one attempt entry per test input, both slots filled with legal grids
  * the report records the three cascade stages and no errors

Run with::
    python test_integration.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, "arc26_solver.py")
COMP = os.path.join(os.path.dirname(HERE), "competition_data")

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def legal_grid(grid) -> bool:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return False
    height, width = len(grid), len(grid[0])
    if not (1 <= height <= 30 and 1 <= width <= 30):
        return False
    for row in grid:
        if len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int) or not (0 <= cell <= 9):
                return False
    return True


def main() -> int:
    challenges_path = os.path.join(COMP, "arc-agi_training_challenges.json")
    if not os.path.isfile(challenges_path):
        print(f"skip: {challenges_path} not found")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "submission.json")
        report = os.path.join(tmp, "report.json")
        cmd = [
            sys.executable, SOLVER,
            "--split", "training", "--limit", "6",
            "--time-budget-seconds", "90", "--calibrate-tasks", "2",
            "--data-dir", COMP,
            "--out", out, "--report", report,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print("   |", line)

        check("solver exits 0", proc.returncode == 0,
              f"rc={proc.returncode} stderr={(proc.stderr or '')[-200:]}")
        check("submission.json written", os.path.isfile(out))
        check("report.json written", os.path.isfile(report))
        if not (os.path.isfile(out) and os.path.isfile(report)):
            print(f"\nFAILURES: {FAILURES}")
            return 1

        sub = json.load(open(out, encoding="utf-8"))
        rep = json.load(open(report, encoding="utf-8"))
        ch = json.load(open(challenges_path, encoding="utf-8"))
        expected_ids = set(sorted(ch)[:6])

        check("every requested task id is present (pre-seeding works)",
              set(sub) == expected_ids,
              f"got {len(sub)}, expected {len(expected_ids)}")

        bad_shape: list[str] = []
        bad_grid: list[str] = []
        blank: list[str] = []
        for tid, attempts in sub.items():
            want = len(ch[tid]["test"])
            if len(attempts) != want:
                bad_shape.append(f"{tid}:{len(attempts)}!={want}")
            for i, entry in enumerate(attempts):
                for slot in ("attempt_1", "attempt_2"):
                    grid = entry.get(slot)
                    if grid is None:
                        blank.append(f"{tid}[{i}].{slot}")
                    elif not legal_grid(grid):
                        bad_grid.append(f"{tid}[{i}].{slot}")

        check("each task has one attempt entry per test input", not bad_shape,
              str(bad_shape[:3]))
        check("no attempt is blank", not blank, str(blank[:3]))
        check("every attempt is a legal grid", not bad_grid, str(bad_grid[:3]))

        stages = [s.get("name") for s in rep.get("stages", [])]
        check("the report records all three cascade stages",
              stages == ["A_sweep", "B_ttt", "C_backfill"], str(stages))
        check("the report records no task errors", not rep.get("errors"),
              str(rep.get("errors", [])[:1]))
        check("the report carries the streaming submission-writing guarantee",
              "n_tasks_done" in rep, str(rep.get("n_tasks_done")))

    print()
    if FAILURES:
        print(f"FAILURES: {FAILURES}")
        return 1
    print("all integration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
