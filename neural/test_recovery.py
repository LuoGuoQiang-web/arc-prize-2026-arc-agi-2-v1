"""Coverage-recovery tests: a beam cut off mid-row must still yield a candidate.

Why this matters: on the smoke run three of five tasks came back with an EMPTY
candidate pool because (a) turbo_dfs discarded its in-flight beams on timeout and
(b) parse_grid_string rejected any ragged row. Coverage is what scores, so both are
now recoverable -- but strict parsing must remain the default for validation.
"""
import importlib.util
import sys

import numpy as np

SOLVER = r"C:\Users\Administrator\Desktop\Kaggle\work\arc_w1\solver\arc26_solver.py"
spec = importlib.util.spec_from_file_location("arc26_solver", SOLVER)
m = importlib.util.module_from_spec(spec)
sys.modules["arc26_solver"] = m
spec.loader.exec_module(m)

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


ragged = "123\n45\n789"

# 1. strict default is unchanged
check("strict parse still rejects a ragged grid", m.parse_grid_string(ragged) is None)

# 2. recovery truncates at the first incomplete row
rec = m.parse_grid_string(ragged, recover_truncated=True)
check("recovery salvages the completed rows",
      rec is not None and np.array_equal(rec, np.array([[1, 2, 3]])),
      f"-> {None if rec is None else rec.tolist()}")

# 3. a beam stopped mid-final-row keeps every full row
full = "\n".join("".join(str((r + c) % 10) for c in range(5)) for r in range(29))
truncated = full + "\n12"          # 29 complete rows + a 2-cell partial row
rec2 = m.parse_grid_string(truncated, recover_truncated=True)
check("truncated 30x30 beam yields a rectangular grid",
      rec2 is not None and rec2.shape == (29, 5), f"shape={None if rec2 is None else rec2.shape}")
check("strict parse rejects that same truncated beam",
      m.parse_grid_string(truncated) is None)

# 4. tokens_to_array threads the flag through
toks = [int(ch) for ch in "123"] + [m.NEWLINE_TOKEN_ID] + [4, 5]
check("tokens_to_array strict rejects ragged",
      m.tokens_to_array(toks) is None)
rec3 = m.tokens_to_array(toks, recover_truncated=True)
check("tokens_to_array with recovery salvages",
      rec3 is not None and np.array_equal(rec3, np.array([[1, 2, 3]])),
      f"-> {None if rec3 is None else rec3.tolist()}")

# 5. a complete grid is unaffected by the flag
good = "\n".join("".join(str((r + c) % 10) for c in range(4)) for r in range(4))
check("complete grid parses identically with and without recovery",
      np.array_equal(m.parse_grid_string(good), m.parse_grid_string(good, recover_truncated=True)))

# 6. genuinely empty input is still rejected under recovery
check("blank input rejected even with recovery",
      m.parse_grid_string("\n\n", recover_truncated=True) is None)

print()
if failures:
    print(f"FAILURES: {failures}")
    raise SystemExit(1)
print("all recovery checks passed")
