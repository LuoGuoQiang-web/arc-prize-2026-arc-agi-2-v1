"""Import the solver module and exercise the data-dir auto-resolution."""
import importlib.util
import os
import sys
import traceback

SOLVER = r"C:\Users\Administrator\Desktop\Kaggle\work\arc_w1\solver\arc26_solver.py"
LOCAL_DATA = r"C:\Users\Administrator\Desktop\Kaggle\work\arc_w1\competition_data"

spec = importlib.util.spec_from_file_location("arc26_solver", SOLVER)
m = importlib.util.module_from_spec(spec)
# dataclasses resolves annotations via sys.modules[cls.__module__], so the module must
# be registered before exec_module. (In a notebook the code runs as __main__, which is
# already registered, so this is only needed for this import-by-path harness.)
sys.modules["arc26_solver"] = m
try:
    spec.loader.exec_module(m)
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("module imported OK")
print("DATA_DIR_CANDIDATES =", m.DATA_DIR_CANDIDATES)

# Point the candidate list at the local copy, then ask for a bogus --data-dir.
# This is exactly the code path that failed on Kaggle, so if it resolves here it
# will resolve there.
m.DATA_DIR_CANDIDATES = (LOCAL_DATA,) + tuple(m.DATA_DIR_CANDIDATES)
tasks, solutions, info = m.load_split("C:/definitely/not/a/real/path", "evaluation")
print("resolved data_dir :", info["data_dir"])
print("challenges        :", os.path.basename(info["challenges_path"]))
print("n_tasks           :", len(tasks))
print("solutions present :", solutions is not None and len(solutions))

# And confirm a valid --data-dir still short-circuits without the fallback.
tasks2, sol2, info2 = m.load_split(LOCAL_DATA, "training")
print("direct --data-dir :", info2["data_dir"], "n_tasks", len(tasks2))

# Splits without solutions must report solutions=None.
tasks3, sol3, info3 = m.load_split(LOCAL_DATA, "test")
print("test split        : n_tasks", len(tasks3), "solutions", sol3)
print("OK")
