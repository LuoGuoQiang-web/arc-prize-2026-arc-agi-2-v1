"""Point the writeups at a VALID code submission.

Why this exists: the Paper Track / Solution Writeup rules require the writeup to be
linked to a code submission, and our cited ref (56199696) was **rejected** for a format
error -- Kaggle re-ran the notebook and got 120 rows instead of 240. The writeup must cite
the new, valid ref, and that edit happens minutes before posting, which is exactly when a
manual copy-paste of a number goes wrong.

This script verifies the ref actually appears in the competition's submission list before
writing it, so a typo or a stale id cannot reach the writeup.

Usage:
    python update_submission_ref.py --ref 12345678 [--dry-run]
    python update_submission_ref.py --latest      # newest COMPLETE submission with a score
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPETITION = "arc-prize-2026-arc-agi-2"
TARGETS = ["WRITEUP_PASTE_READY.md", "NEURAL_WRITEUP_PASTE_READY.md"]

REF_PATTERNS = [
    re.compile(r"(代码提交\s*ID[：:]\s*)(\d{6,})"),
    re.compile(r"(code submission (?:ref|ID)\s*[:：]?\s*)(\d{6,})", re.I),
    re.compile(r"(submission ref\s*[:：]\s*)(\d{6,})", re.I),
]


def list_submissions() -> list[tuple[str, str, str]]:
    """(ref, status, publicScore) for every submission, newest first, via the CLI."""
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "submissions", COMPETITION, "-v"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=300,
    )
    rows: list[tuple[str, str, str]] = []
    for line in (proc.stdout or "").splitlines():
        if not line or line.startswith("ref,"):
            continue
        fields = line.split(",")
        if not fields or not fields[0].strip().isdigit():
            continue
        ref = fields[0].strip()
        # The description field can itself contain commas, so fixed column positions are
        # unreliable -- locate the status token and read the score that follows it.
        status, public = "", ""
        for i, field in enumerate(fields):
            if field.strip().startswith("SubmissionStatus."):
                status = field.strip()
                public = fields[i + 1].strip() if i + 1 < len(fields) else ""
                break
        rows.append((ref, status, public))
    return rows


def latest_scored(rows: list[tuple[str, str, str]]) -> str | None:
    for ref, status, score in rows:
        if "COMPLETE" in status.upper() and score.strip():
            return ref
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="submission ref to cite")
    ap.add_argument("--latest", action="store_true",
                    help="pick the newest COMPLETE submission that has a score")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reason", default=None,
                    help="short note appended in the metadata line")
    args = ap.parse_args()

    rows = list_submissions()
    print(f"submissions found: {len(rows)}")
    for ref, status, score in rows:
        print(f"  ref={ref}  status={status}  publicScore={score or '(none)'}")

    ref = args.ref
    if args.latest and not ref:
        ref = latest_scored(rows)
        if not ref:
            print("\nABORT: no COMPLETE submission with a score yet. Re-run after the "
                  "graded run finishes.")
            return 1
        print(f"\n--latest selected ref {ref}")
    if not ref:
        print("\nABORT: pass --ref or --latest.")
        return 1

    known = {r for r, _s, _sc in rows}
    if ref not in known:
        print(f"\nABORT: ref {ref} is NOT in the submission list. Refusing to cite an id "
              f"that Kaggle does not recognise.")
        return 1
    status = next(s for r, s, _sc in rows if r == ref)
    score = next(sc for r, _s, sc in rows if r == ref)
    print(f"ref {ref} verified: status={status} publicScore={score or '(not yet shown)'}")

    changed = 0
    for name in TARGETS:
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            print(f"  {name}: missing, skipped")
            continue
        text = open(path, encoding="utf-8").read()
        new_text, n = text, 0
        for pattern in REF_PATTERNS:
            def repl(m, _ref=ref):
                return m.group(1) + _ref
            new_text, k = pattern.subn(repl, new_text)
            n += k
        if n == 0:
            print(f"  {name}: no submission-ref line found (nothing to update)")
            continue
        if new_text == text:
            # Same ref already cited. Report it rather than silently doing nothing --
            # a silent no-op here is indistinguishable from a broken regex.
            print(f"  {name}: already cites {ref} (no change)")
            continue
        if args.dry_run:
            print(f"  {name}: WOULD update {n} ref(s) -> {ref}")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new_text)
            print(f"  {name}: updated {n} ref(s) -> {ref}")
        changed += n

    print(f"\n{'would change' if args.dry_run else 'changed'} {changed} reference(s).")
    if changed == 0 and not args.dry_run:
        print("NOTE: nothing changed -- check that the writeup still carries a ref line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
