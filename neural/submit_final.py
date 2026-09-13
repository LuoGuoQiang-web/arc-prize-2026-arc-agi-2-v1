"""Wait for the submittable full run, validate its artifact, then submit exactly once.

Why this is a script and not a couple of manual commands
-------------------------------------------------------
* `maxDailySubmissions=1` for this competition. A careless submit burns the day, so
  every gate below must pass before the submit call is made.
* The submission quota resets at 00:00 UTC, while the run finishes before that, so the
  script has to wait out the reset rather than retry blindly.
* We already lost one submission to a formatting bug (a dev-mode run overwrote the
  240-task file with 120 tasks). The validator here mirrors the official rules and
  refuses to submit if anything is off.

Usage:
    python submit_final.py                      # wait, validate, submit
    python submit_final.py --dry-run            # validate only, never submit
    python submit_final.py --deadline-min 900   # give up waiting after N minutes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = "luoguoqiang/arc26-submit-full"
COMPETITION = "arc-prize-2026-arc-agi-2"
OUT_DIR = os.path.join(HERE, "final_out")
CHALLENGES = os.path.join(HERE, "competition_data", "arc-agi_test_challenges.json")
EXPECTED_TASKS = 240


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def kernel_status() -> str:
    code, out = run([sys.executable, "-m", "kaggle", "kernels", "status", KERNEL], timeout=180)
    for token in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"):
        if token in out:
            return token
    return "UNKNOWN"


def validate(path: str) -> list[str]:
    """Mirror the official submission rules; return a list of problems."""
    problems: list[str] = []
    try:
        with open(path, encoding="utf-8") as fh:
            sub = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return [f"cannot read {path}: {type(exc).__name__}: {exc}"]

    try:
        with open(CHALLENGES, encoding="utf-8") as fh:
            challenges = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return [f"cannot read challenges: {type(exc).__name__}: {exc}"]

    if len(sub) != len(challenges):
        problems.append(f"task count {len(sub)} != {len(challenges)}")
    missing = sorted(set(challenges) - set(sub))
    if missing:
        problems.append(f"{len(missing)} task(s) missing, e.g. {missing[:5]}")
    extra = sorted(set(sub) - set(challenges))
    if extra:
        problems.append(f"{len(extra)} unexpected id(s), e.g. {extra[:5]}")

    for tid, attempts in sub.items():
        if tid not in challenges:
            continue
        expected = len(challenges[tid]["test"])
        if not isinstance(attempts, list) or len(attempts) != expected:
            problems.append(f"{tid}: {len(attempts) if isinstance(attempts, list) else '?'} "
                            f"entries, expected {expected}")
            continue
        for i, entry in enumerate(attempts):
            for slot in ("attempt_1", "attempt_2"):
                grid = entry.get(slot) if isinstance(entry, dict) else None
                if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                    problems.append(f"{tid}[{i}].{slot}: not a 2D grid")
                    continue
                h, w = len(grid), len(grid[0])
                if not (1 <= h <= 30 and 1 <= w <= 30):
                    problems.append(f"{tid}[{i}].{slot}: shape {h}x{w} outside 1..30")
                if any(len(row) != w for row in grid):
                    problems.append(f"{tid}[{i}].{slot}: ragged rows")
                if any(not isinstance(c, int) or not (0 <= c <= 9) for row in grid for c in row):
                    problems.append(f"{tid}[{i}].{slot}: colour outside 0..9")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validate only, never submit")
    ap.add_argument("--deadline-min", type=float, default=900.0,
                    help="give up waiting for the kernel after this many minutes")
    ap.add_argument("--reset-utc", default="2026-09-14T00:05:00",
                    help="wait until at least this UTC time before submitting (quota reset)")
    ap.add_argument("--message", default="v2: licence-clean TTT solver (Apache-2.0 Qwen3-4B "
                                        "SFT + own LoRA TTT + constrained DFS + cascade)")
    args = ap.parse_args()

    started = time.time()
    deadline = started + args.deadline_min * 60

    # ---- 1. wait for the kernel --------------------------------------------------
    print(f"[1/4] waiting for {KERNEL} ...", flush=True)
    status = "UNKNOWN"
    while time.time() < deadline:
        status = kernel_status()
        print(f"      {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z status={status}",
              flush=True)
        if status in ("COMPLETE", "ERROR", "CANCELLED"):
            break
        time.sleep(180)

    if status != "COMPLETE":
        print(f"ABORT: kernel status is {status!r}, not COMPLETE. Nothing submitted.")
        return 1

    # ---- 2. fetch the artifact ---------------------------------------------------
    print(f"[2/4] downloading output to {OUT_DIR}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    code, out = run([sys.executable, "-m", "kaggle", "kernels", "output", KERNEL,
                     "-p", OUT_DIR, "-q"])
    print("      " + out.strip().splitlines()[-1] if out.strip() else "      (no output)")

    sub_path = os.path.join(OUT_DIR, "submission.json")
    if not os.path.isfile(sub_path):
        print(f"ABORT: {sub_path} not found. Nothing submitted.")
        return 1
    size_mb = os.path.getsize(sub_path) / 1e6
    print(f"      submission.json {size_mb:.3f} MB")

    # ---- 3. validate -------------------------------------------------------------
    print("[3/4] validating against the official rules", flush=True)
    problems = validate(sub_path)
    if problems:
        print(f"ABORT: {len(problems)} problem(s); refusing to burn the daily submission.")
        for problem in problems[:25]:
            print("   !", problem)
        return 1
    with open(sub_path, encoding="utf-8") as fh:
        sub = json.load(fh)
    print(f"      OK: {len(sub)} tasks, all entries well-formed "
          f"(expected {EXPECTED_TASKS})")
    if len(sub) != EXPECTED_TASKS:
        print(f"ABORT: expected {EXPECTED_TASKS} tasks, got {len(sub)}.")
        return 1

    if args.dry_run:
        print("[4/4] --dry-run: validation passed, NOT submitting.")
        return 0

    # ---- 4. wait for the quota reset, then submit once ---------------------------
    reset_at = datetime.fromisoformat(args.reset_utc).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if now < reset_at:
        wait_s = (reset_at - now).total_seconds()
        print(f"[4/4] submission quota resets at {args.reset_utc}Z; "
              f"sleeping {wait_s / 60:.1f} min", flush=True)
        while datetime.now(timezone.utc) < reset_at:
            time.sleep(min(300, max(5, (reset_at - datetime.now(timezone.utc)).total_seconds())))

    print("      submitting ...", flush=True)
    code, out = run([sys.executable, "-m", "kaggle", "competitions", "submit",
                     "-c", COMPETITION, "-k", KERNEL, "-v", "1",
                     "-f", sub_path, "-m", args.message], timeout=900)
    print(out.strip()[-3000:])
    if code != 0 or "error" in out.lower() and "success" not in out.lower():
        print(f"\nsubmit returned code {code}; inspect the output above.")
    else:
        print("\nSUBMITTED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
