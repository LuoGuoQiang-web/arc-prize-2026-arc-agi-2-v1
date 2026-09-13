"""Exercise submit_final.validate() on a good and a bad artifact.

The submit step costs the single daily submission, so the gate must be proven to
reject bad input before it is ever pointed at the real thing.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("submit_final", os.path.join(HERE, "submit_final.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["submit_final"] = m
spec.loader.exec_module(m)

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


good = os.path.join(HERE, "_merge_test", "merged.json")
eval_sub = os.path.join(HERE, "eval_out", "submission.json")
smoke_sub = os.path.join(HERE, "smoke_out", "submission.json")
short = os.path.join(HERE, "_merge_test", "shard0.json")

if os.path.isfile(good):
    p = m.validate(good)
    check("240-task synthetic submission passes the gate", not p, f"problems={p[:3]}")
else:
    print("(skip: merged.json not present)")

if os.path.isfile(short):
    p = m.validate(short)
    check("120-task shard is REJECTED (this is the bug that burned a submission)",
          bool(p), f"first problem: {p[0] if p else None}")

if os.path.isfile(eval_sub):
    p = m.validate(eval_sub)
    check("24-task eval artifact is rejected for the test split", bool(p),
          f"first problem: {p[0] if p else None}")

if os.path.isfile(smoke_sub):
    p = m.validate(smoke_sub)
    check("smoke artifact is rejected", bool(p), f"first problem: {p[0] if p else None}")

# A structurally broken grid must also be caught.
bad = os.path.join(HERE, "_merge_test", "bad_grid.json")
import json  # noqa: E402

with open(good, encoding="utf-8") as fh:
    base = json.load(fh)
key = sorted(base)[0]
base[key][0]["attempt_1"] = [[0, 1], [2]]          # ragged
with open(bad, "w", encoding="utf-8") as fh:
    json.dump(base, fh)
p = m.validate(bad)
check("ragged grid is rejected", any("ragged" in x for x in p), f"problems={p[:2]}")

with open(good, encoding="utf-8") as fh:
    base = json.load(fh)
key = sorted(base)[0]
base[key][0]["attempt_1"] = [[0] * 31]             # 31 columns
with open(bad, "w", encoding="utf-8") as fh:
    json.dump(base, fh)
p = m.validate(bad)
check("out-of-range shape is rejected", any("outside 1..30" in x for x in p), f"problems={p[:2]}")

with open(good, encoding="utf-8") as fh:
    base = json.load(fh)
key = sorted(base)[0]
base[key][0]["attempt_2"] = [[99]]                 # illegal colour
with open(bad, "w", encoding="utf-8") as fh:
    json.dump(base, fh)
p = m.validate(bad)
check("illegal colour is rejected", any("colour" in x for x in p), f"problems={p[:2]}")

os.remove(bad)
print()
if failures:
    print(f"FAILURES: {failures}")
    raise SystemExit(1)
print("all submission-gate checks passed")
