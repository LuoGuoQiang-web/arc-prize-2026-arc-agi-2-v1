"""Local rehearsal of engine_cpu_kernel.py's core logic.

Runs the exact API usage the Kaggle CPU kernel will use (importlib load of
arc_prize_v1.py -> make_config -> solve_task -> format validation) against the
local evaluation split, so the remote run is not the first time this code path
executes.

Usage:  python rehearse_engine_kernel.py [n_tasks]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "ARC", "Kaggle",
                      "arc_prize_v1", "arc_prize_v1.py")
if not os.path.isfile(ENGINE):
    ENGINE = r"C:\Users\Administrator\Desktop\Kaggle\ARC\Kaggle\arc_prize_v1\arc_prize_v1.py"
COMP = os.path.join(HERE, "competition_data")


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def validate(sub, challenges):
    problems = []
    if set(sub) != set(challenges):
        problems.append(f"key mismatch: {len(sub)} vs {len(challenges)}")
    for k, v in sub.items():
        exp = len(challenges[k]["test"])
        if len(v) != exp:
            problems.append(f"{k}: {len(v)} attempts vs {exp} test inputs")
            continue
        for i, at in enumerate(v):
            for slot in ("attempt_1", "attempt_2"):
                g = at.get(slot)
                if not isinstance(g, list) or not g or not isinstance(g[0], list):
                    problems.append(f"{k}[{i}].{slot}: not a 2D grid")
                    continue
                h, w = len(g), len(g[0])
                if not (1 <= h <= 30 and 1 <= w <= 30):
                    problems.append(f"{k}[{i}].{slot}: shape {h}x{w} out of range")
                if any(len(row) != w for row in g):
                    problems.append(f"{k}[{i}].{slot}: ragged rows")
                if any(not isinstance(c, int) or not (0 <= c <= 9)
                       for row in g for c in row):
                    problems.append(f"{k}[{i}].{slot}: bad colour value")
    return problems


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("engine :", ENGINE, os.path.isfile(ENGINE))
    print("comp   :", COMP, os.path.isdir(COMP))

    spec = importlib.util.spec_from_file_location("arc_prize_v1", ENGINE)
    eng = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eng)
    print("engine loaded; has solve_task:", hasattr(eng, "solve_task"))

    cfg = eng.make_config({})
    print("config:", {k: cfg[k] for k in ("max_chains", "per_task_seconds",
                                         "enable_depth2", "enable_object_family",
                                         "enable_context_family", "enable_shape_family")})

    ch = load_json(os.path.join(COMP, "arc-agi_evaluation_challenges.json"))
    sol = load_json(os.path.join(COMP, "arc-agi_evaluation_solutions.json"))
    keys = sorted(ch)[:n]
    print(f"\nsolving {len(keys)} evaluation tasks")

    submission = {}
    per_task = []
    n_correct = 0
    hits = {"attempt_1": 0, "attempt_2": 0, "either": 0}
    t0 = time.time()
    for k in keys:
        try:
            res = eng.solve_task(ch[k], cfg)
        except Exception:
            print(f"  !! {k} raised:\n{traceback.format_exc(limit=3)}")
            res = {"attempts": [{"attempt_1": [[0]], "attempt_2": [[0]]}
                                for _ in ch[k]["test"]], "diagnostics": {"source": "error"}}
        submission[k] = res["attempts"]
        diag = res.get("diagnostics", {})
        truth = sol.get(k)
        correct = None
        if truth is not None:
            correct = any(at["attempt_1"] == t or at["attempt_2"] == t
                          for at, t in zip(res["attempts"], truth))
            n_correct += int(correct)
            if len(truth) == 1:
                hits["attempt_1"] += int(res["attempts"][0]["attempt_1"] == truth[0])
                hits["attempt_2"] += int(res["attempts"][0]["attempt_2"] == truth[0])
                hits["either"] += int(correct)
        per_task.append({"task_id": k, "seconds": diag.get("elapsed_s"),
                         "source": diag.get("source"),
                         "top_program": diag.get("top_program"),
                         "second_program": diag.get("second_program"),
                         "n_valid": diag.get("n_valid"), "correct": correct})

    elapsed = time.time() - t0
    src = {}
    for r in per_task:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print(f"\nelapsed {elapsed:.2f}s  ({elapsed / len(keys):.3f}s/task)")
    print("sources:", src)
    print(f"correct: {n_correct}/{len(keys)} = {n_correct / len(keys):.3%}")
    print("single-test-input attempt hits:", hits)

    problems = validate(submission, {k: ch[k] for k in keys})
    print(f"format problems: {len(problems)}")
    for p in problems[:10]:
        print("   !", p)

    # never-blank guarantee
    blanks = [k for k, v in submission.items()
              if any(not at.get(s) for at in v for s in ("attempt_1", "attempt_2"))]
    print("tasks with a blank attempt:", len(blanks))

    out = os.path.join(HERE, "rehearsal_engine_local.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"n": len(keys), "elapsed_s": round(elapsed, 2), "sources": src,
                   "n_correct": n_correct, "accuracy": n_correct / len(keys),
                   "attempt_hits": hits, "format_problems": problems,
                   "per_task": per_task}, fh, indent=1)
    print("wrote", out)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
