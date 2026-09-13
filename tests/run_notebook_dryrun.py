"""
Dry-run the generated ARC_PRIZE_2026_v1.ipynb without Jupyter.

Jupyter is not installed in this environment, so this harness reproduces the two things a
notebook actually does for our cells: (1) `%%writefile` writes everything after the magic
line to a file, (2) every code cell executes in one shared namespace, in order.

Kaggle-specific adaptations (clearly marked, nothing else is altered):
  * "/kaggle/working" is rewritten to a scratch directory (the real C:\\kaggle is not writable here);
  * the submit-mode cell gets an explicit `submission_path=` kwarg, because the engine's default
    only targets /kaggle/working when that directory exists — this is the same override a user
    would pass on Kaggle.

Everything else — engine source, cell order, cell code — runs byte-for-byte as shipped.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTEBOOK = ROOT / "ARC_PRIZE_2026_v1.ipynb"
ENGINE = ROOT / "arc_prize_v1.py"
SCRATCH = ROOT / ".dryrun"
KAGGLE_WORKING = SCRATCH / "kaggle_working"
SUBMISSION = KAGGLE_WORKING / "submission.json"
# inject paths with forward slashes: they are valid on Windows and need no escaping when
# spliced into Python source (a raw backslash path would break the injected string literal)
KW_POSIX = KAGGLE_WORKING.as_posix()
SUB_POSIX = SUBMISSION.as_posix()

sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    # the notebook's cells rely on default input roots, which include "./data"
    make_fixture.build_fixture(SCRATCH / "data")
    import os
    os.chdir(SCRATCH)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"notebook: {len(nb['cells'])} cells ({len(code_cells)} code)")

    ns: dict = {"__name__": "__main__"}
    engine_written_md5 = None

    for i, cell in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        label = f"cell {i}"
        if source.startswith("%%writefile"):
            header, body = source.split("\n", 1)
            target = Path(header.split(None, 1)[1].strip().replace("/kaggle/working", KW_POSIX))
            target.parent.mkdir(parents=True, exist_ok=True)
            # newline="" emulates IPython writing on Linux; Windows text mode would turn
            # every \n into \r\n and make the byte comparison below meaningless
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(body)
            engine_written_md5 = target.read_bytes()
            print(f"  [{label}] %%writefile -> {target.name} ({len(engine_written_md5)} bytes)")
            continue

        source = source.replace("/kaggle/working", KW_POSIX)
        if "arc_prize_v1.main(" in source and 'mode="submit"' in source:
            source = source.replace('mode="submit",', f'mode="submit", submission_path=r"{SUB_POSIX}",', 1)
            print(f"  [{label}] (dry-run shim: explicit submission_path)")

        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(source, f"cell{i}", "exec"), ns)
        except Exception:
            print(f"  [FAIL] {label} raised:\n{traceback.format_exc()}")
            FAILURES.append(f"{label} raised")
            break
        tail = [l for l in buf.getvalue().strip().splitlines() if l.strip()][-6:]
        print(f"  [{label}] ok; last output lines:")
        for line in tail:
            print(f"        | {line}")

    print("\nverification:")
    check("engine written by %%writefile == arc_prize_v1.py",
          engine_written_md5 == ENGINE.read_bytes(),
          f"{len(engine_written_md5 or b'')} vs {ENGINE.stat().st_size} bytes")
    check("submission.json produced", SUBMISSION.exists(), str(SUBMISSION))
    check("submission.json non-trivial (>500 bytes)",
          SUBMISSION.exists() and SUBMISSION.stat().st_size > 500)

    spec = importlib.util.spec_from_file_location("arc_check", ENGINE)
    arc = importlib.util.module_from_spec(spec)
    sys.modules["arc_check"] = arc
    spec.loader.exec_module(arc)
    cfg = arc.make_config({"mode": "submit", "dataset": "test", "input_roots": [str(SCRATCH / "data")]})
    tasks, _, _ = arc.load_dataset(cfg, arc.discover_data(cfg), "test")
    ok, problems = arc.validate_submission(SUBMISSION, arc.expected_structure(tasks))
    check("produced submission passes the schema validator", ok, str(problems[:3]))

    # locally there is no /kaggle/working, so the engine falls back to ./arcprize_runtime
    project = SCRATCH / "arcprize_runtime"
    runs = sorted((project / "runs").glob("*"))
    check("run directory created", len(runs) >= 1, str([r.name for r in runs]))
    check("no empty/stray run directories left behind",
          all((r / "manifest.json").exists() for r in runs),
          str([r.name for r in runs if not (r / "manifest.json").exists()]))
    if runs:
        real = [r for r in runs if (r / "manifest.json").exists()]
        # diagnostics.csv is written only when ground truth is available, i.e. by the
        # dev-mode run; the submit-mode run has none. Run ids embed a content hash, so
        # their sort order is arbitrary -- check across all runs, not just the first.
        all_files = {p.name for r in real for p in r.iterdir()}
        check("manifest/checkpoint/diagnostics present",
              {"manifest.json", "checkpoint.jsonl", "diagnostics.csv"} <= all_files,
              str(sorted(all_files)))
    backups = sorted((project / "backups").glob("*.zip"))
    check("backup snapshot created", len(backups) >= 1, str([b.name for b in backups]))
    check("manual snapshot is not empty",
          any(b.stat().st_size > 1000 for b in backups),
          str([(b.name, b.stat().st_size) for b in backups]))

    print(f"\n{'DRY RUN OK' if not FAILURES else 'DRY RUN FAILED'}: "
          f"{len(FAILURES)} failure(s)")
    for f in FAILURES:
        print("  -", f)
    if "--keep" not in sys.argv:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
