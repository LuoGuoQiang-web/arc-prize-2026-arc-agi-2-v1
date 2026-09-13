"""Verify the solution-writeup figure: valid PNG, sane geometry, and numbers that match.

Two separate risks with a hardcoded figure:
  1. the file is not a usable image (corrupt, absurd aspect, blank);
  2. the constants in the plotting script drift from the measurement they claim to show.

(2) is the dangerous one -- a figure that misstates data is worse than no figure. So this
re-derives every number from the artifacts and compares it against what the script drew.

No image library is used: the PNG header is parsed directly.
"""
from __future__ import annotations

import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "attempt_breakdown.png")
OUT = os.path.join(os.path.dirname(HERE), "eval_out")

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# ---- 1. the file is a real, sensibly shaped PNG -------------------------------------
check("figure exists", os.path.isfile(FIG))
with open(FIG, "rb") as fh:
    head = fh.read(33)
check("PNG signature", head[:8] == b"\x89PNG\r\n\x1a\n")
width, height = struct.unpack(">II", head[16:24])
check("dimensions are sane", 800 <= width <= 4000 and 400 <= height <= 3000,
      f"{width}x{height}")
check("aspect is landscape-ish (2-panel figure)", 1.6 <= width / height <= 3.2,
      f"{width / height:.2f}")
check("file is not suspiciously small", os.path.getsize(FIG) > 20_000,
      f"{os.path.getsize(FIG)} bytes")

# ---- 2. every drawn number is re-derived from the artifacts -------------------------
sub = json.load(open(os.path.join(OUT, "submission.json"), encoding="utf-8"))
rep = json.load(open(os.path.join(OUT, "report.json"), encoding="utf-8"))
per = {r["task_id"]: r for r in rep.get("per_task", [])}


def uniform(grid):
    return len({cell for row in grid for cell in row}) == 1


tally = {}
total = 0
identical = 0
for tid, attempts in sub.items():
    srcs = per.get(tid, {}).get("attempts") or []
    for i, entry in enumerate(attempts):
        total += 1
        src = srcs[i] if i < len(srcs) else "?"
        key = (src, "uniform" if uniform(entry["attempt_1"]) else "varied")
        tally[key] = tally.get(key, 0) + 1
        if entry["attempt_1"] == entry["attempt_2"]:
            identical += 1

neural_varied = tally.get(("neural", "varied"), 0)
neural_uniform = tally.get(("neural", "uniform"), 0)
fallback_uniform = sum(n for (s, k), n in tally.items() if s != "neural" and k == "uniform")

check("panel A bar 1 (neural, varies) matches the script constant 21",
      neural_varied == 21, f"measured {neural_varied}")
check("panel A bar 2 (neural, single colour) matches the script constant 10",
      neural_uniform == 10, f"measured {neural_uniform}")
check("panel A bar 3 (fallback, single colour) matches the script constant 1",
      fallback_uniform == 1, f"measured {fallback_uniform}")
check("panel A total is 32, as annotated", total == 32, f"measured {total}")
check("the '0 of 32' distinctness annotation is true", identical == 0,
      f"measured {identical}")
check("the '31%' claim in the figure is accurate",
      abs(100.0 * neural_uniform / total - 31.25) < 0.5,
      f"{100.0 * neural_uniform / total:.2f}%")

scored = rep.get("n_tasks_scored")
correct = rep.get("n_solved")
accuracy = 100.0 * (rep.get("accuracy") or 0.0)
check("panel B neural bar is 4.17%", abs(accuracy - 4.17) < 0.01, f"{accuracy:.4f}%")
check("panel B neural denominator is 1/24",
      (scored, correct) == (24, 1), f"{correct}/{scored}")
check("panel B coverage annotation (24 of 24) is the report's own number",
      rep.get("n_tasks_with_real_candidate") == 24,
      str(rep.get("n_tasks_with_real_candidate")))

print()
if failures:
    print(f"FAILURES: {failures}")
    sys.exit(1)
print("figure verified: valid PNG, and every drawn number re-derived from the artifacts")
