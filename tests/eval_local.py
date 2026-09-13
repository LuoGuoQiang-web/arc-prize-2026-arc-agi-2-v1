"""
Local measurement harness — the "score-first" feedback loop.

Runs the engine over a real ARC-AGI-2 public evaluation set (or any task set with answers)
and prints an actionable report: accuracy, failure taxonomy, and *which tasks* to attack next.

Data layouts it accepts (auto-discovered by the engine):
  * Kaggle competition files: arc-agi_evaluation_challenges.json + ..._solutions.json
  * Official repo layout:     <anything>/data/evaluation/*.json   (public eval files carry answers)

Usage:
    python tests/eval_local.py                     # auto-discover, run evaluation set
    python tests/eval_local.py --dataset training  # 1000 training tasks (easy set, sanity check)
    python tests/eval_local.py --limit 20          # quick smoke run
    python tests/eval_local.py --report out.md     # also write a markdown report
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENGINE = ROOT / "arc_prize_v1.py"


def load_engine():
    spec = importlib.util.spec_from_file_location("arc_prize_v1", ENGINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arc_prize_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation", choices=["evaluation", "training", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only run the first N tasks (smoke test)")
    ap.add_argument("--budget", type=float, default=3600.0)
    ap.add_argument("--report", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arc = load_engine()
    cfg = arc.make_config({
        "mode": "dev",
        "dataset": args.dataset,
        "project_dir": str(ROOT / "arcprize_runtime"),
        "global_budget_seconds": args.budget,
        "force": args.force,
    })
    disc = arc.discover_data(cfg)
    print("discovered:")
    for k, v in sorted(disc["files"].items()):
        print(f"  file  {k:22s} {v}")
    for k, v in sorted(disc["task_dirs"].items()):
        n = len(list(Path(v).glob("*.json")))
        print(f"  dir   {k:22s} {v}  ({n} task files)")
    if not disc["files"] and not disc["task_dirs"]:
        print("\nNO DATA FOUND. Download the official ARC-AGI-2 repo ZIP and unzip it into the "
              "workspace so that ./ARC-AGI-2-main/data/evaluation/ exists.")
        return 2

    tasks, solutions, source = arc.load_dataset(cfg, disc, args.dataset)
    print(f"\nloaded {len(tasks)} tasks from {source}")
    n_test = sum(len(t["test"]) for t in tasks.values())
    n_ans = sum(1 for t in tasks.values() for p in t["test"] if "output" in p)
    print(f"test inputs: {n_test} | with ground truth: {n_ans}")
    if n_ans == 0:
        print("WARNING: no ground truth available -> this run cannot measure accuracy")

    if args.limit:
        keep = sorted(tasks)[:args.limit]
        tasks = {k: tasks[k] for k in keep}
        print(f"limit: running {len(tasks)} tasks")

    t0 = time.monotonic()
    summary = arc.main(**{**cfg, "dataset": args.dataset, "force": args.force})
    wall = time.monotonic() - t0

    print("\n" + "=" * 72)
    score = summary.get("dev_score") or {}
    print(f"accuracy          : {score.get('accuracy', 0.0) * 100:.2f}%  "
          f"({score.get('n_exact', 0)}/{score.get('n_scored_inputs', 0)} test inputs)")
    print(f"tasks fully solved: {score.get('n_tasks_solved', 0)}/{score.get('n_tasks_scored', 0)}"
          f"  (official metric is per test input)")
    print(f"wall clock        : {wall:.1f}s ({wall / max(1, len(tasks)):.2f}s per task)")
    print(f"solved by search  : {summary.get('solved_by_search')} | prior-only: {summary.get('prior_only')}")
    classes = summary.get("failure_classes") or {}
    print(f"failure classes   : {dict(classes)}")

    run_dir = Path(cfg["project_dir"]) / "runs" / summary.get("run_id", "")
    csv_path = run_dir / "diagnostics.csv"
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        by_class = {}
        for r in rows:
            if r["solved"] == "1":
                continue
            by_class.setdefault(r["failure_class"], []).append(r["task_id"])
        print("\nnext targets by failure class (these define v2 priorities):")
        for cls, tids in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
            print(f"  {cls:26s} {len(tids):4d}  e.g. {', '.join(sorted(set(tids))[:6])}")
        print(f"\nper-task CSV: {csv_path}")

        if args.report:
            lines = ["# ARC-AGI-2 local evaluation report", "",
                     f"- dataset: `{args.dataset}` ({len(tasks)} tasks, {n_test} test inputs)",
                     f"- accuracy: **{score.get('accuracy', 0.0) * 100:.2f}%** "
                     f"({score.get('n_exact', 0)}/{score.get('n_scored_inputs', 0)})",
                     f"- solved by validated program search: {summary.get('solved_by_search')}",
                     f"- prior-only: {summary.get('prior_only')}",
                     f"- wall clock: {wall:.1f}s", "",
                     "## Failure classes", ""]
            for cls, cnt in sorted(classes.items(), key=lambda kv: -kv[1]):
                lines.append(f"- `{cls}`: {cnt}")
            lines += ["", "## Task ids to attack next", ""]
            for cls, tids in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
                lines.append(f"### {cls} ({len(tids)})")
                lines.append("")
                lines.append(", ".join(f"`{t}`" for t in sorted(set(tids))))
                lines.append("")
            Path(args.report).write_text("\n".join(lines), encoding="utf-8")
            print(f"markdown report: {args.report}")
    print("=" * 72)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
