"""Merge per-shard submission.json files into one complete submission.

The full run is split across 2 concurrent Kaggle GPU sessions (Kaggle's ceiling),
so each session writes only its own interleaved half of the 240 tasks. This stitches
them back together and refuses to emit a file unless the union is complete and every
entry is format-valid -- a silently short submission is rejected outright by the
competition grader (we already lost one submission to exactly that failure mode).

Usage:
    python merge_shards.py --shards out/shard0/submission.json out/shard1/submission.json \
        --challenges competition_data/arc-agi_test_challenges.json \
        --out final/submission.json [--report out/*/report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate(submission, challenges):
    """Mirror the official submission rules; return a list of problems."""
    problems = []
    missing = sorted(set(challenges) - set(submission))
    extra = sorted(set(submission) - set(challenges))
    if missing:
        problems.append(f"{len(missing)} task(s) missing, e.g. {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected task id(s), e.g. {extra[:5]}")
    for tid, attempts in submission.items():
        if tid not in challenges:
            continue
        expected = len(challenges[tid]["test"])
        if not isinstance(attempts, list) or len(attempts) != expected:
            problems.append(f"{tid}: {len(attempts) if isinstance(attempts, list) else '?'} "
                            f"attempt entries, expected {expected}")
            continue
        for i, entry in enumerate(attempts):
            for slot in ("attempt_1", "attempt_2"):
                grid = entry.get(slot) if isinstance(entry, dict) else None
                if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                    problems.append(f"{tid}[{i}].{slot}: not a 2D grid")
                    continue
                h, w = len(grid), len(grid[0])
                if not (1 <= h <= 30 and 1 <= w <= 30):
                    problems.append(f"{tid}[{i}].{slot}: shape {h}x{w} out of 1..30")
                if any(len(row) != w for row in grid):
                    problems.append(f"{tid}[{i}].{slot}: ragged rows")
                if any(not isinstance(c, int) or not (0 <= c <= 9) for row in grid for c in row):
                    problems.append(f"{tid}[{i}].{slot}: colour outside 0..9")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--challenges", required=True,
                    help="the challenge file that defines the expected task ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reports", nargs="*", default=[])
    args = ap.parse_args()

    challenges = load(args.challenges)
    merged: dict[str, list] = {}
    provenance: Counter = Counter()
    overlaps: list[str] = []

    # expand any globs the shell did not
    shard_paths: list[str] = []
    for pattern in args.shards:
        shard_paths.extend(sorted(glob.glob(pattern)) or [pattern])

    for path in shard_paths:
        if not os.path.isfile(path):
            print(f"  ! missing shard file: {path}")
            continue
        sub = load(path)
        dupes = set(sub) & set(merged)
        if dupes:
            overlaps.extend(sorted(dupes))
        for tid, attempts in sub.items():
            merged[tid] = attempts
        provenance[os.path.basename(os.path.dirname(path)) or path] += len(sub)
        print(f"  {path}: {len(sub)} tasks")

    print(f"\nmerged tasks: {len(merged)} / expected {len(challenges)}")
    print("per-shard contribution:", dict(provenance))
    if overlaps:
        print(f"WARNING: {len(overlaps)} task(s) appeared in more than one shard "
              f"(last writer wins): {sorted(set(overlaps))[:5]}")

    problems = validate(merged, challenges)
    print(f"format problems: {len(problems)}")
    for problem in problems[:20]:
        print("   !", problem)

    if problems:
        print("\nREFUSING to write a submission that would be rejected by the grader.")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh)
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.3f} MB, "
          f"{len(merged)} tasks)")

    if args.reports:
        report_paths: list[str] = []
        for pattern in args.reports:
            report_paths.extend(sorted(glob.glob(pattern)) or [pattern])
        tot_done = 0
        sources: Counter = Counter()
        for path in report_paths:
            if not os.path.isfile(path):
                continue
            rep = load(path)
            tot_done += rep.get("n_tasks_done", 0)
            for rec in rep.get("per_task", []):
                for src in rec.get("attempts", []):
                    sources[src] += 1
        print(f"\nreports merged: tasks_done={tot_done}, attempt sources={dict(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
