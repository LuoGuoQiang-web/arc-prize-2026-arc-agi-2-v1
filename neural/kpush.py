"""kpush.py - build a single-cell Kaggle notebook from a .py source and push it.

Why this exists: `kaggle kernels push --accelerator X` is NOT honored by the server
(verified against Kaggle/kaggle-cli issue #821, comments 5304836116 / 5304712721, and
by our own probes). The only reliable channel is the `machine_shape` key inside
kernel-metadata.json. This helper always writes that key.

Usage:
    python kpush.py <source.py> --slug <owner/slug> --title "..." \
        [--shape NvidiaTeslaT4] [--competition arc-prize-2026-arc-agi-2] \
        [--model sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1] \
        [--dataset <owner/ds>] [--kernel <owner/kernel>] [--internet] [--no-gpu]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(r"C:\Users\Administrator\Desktop\Kaggle\work\arc_w1")
BUILD_ROOT = WORKSPACE / "kernels"


def build_notebook(source: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="path to the .py file to wrap")
    ap.add_argument("--slug", required=True, help="owner/kernel-slug")
    ap.add_argument("--title", required=True)
    ap.add_argument("--shape", default="NvidiaTeslaT4",
                    help="machine_shape; NvidiaTeslaT4 = GPU T4 x2 (verified). "
                         "L4 spellings are rejected and silently fall back to P100.")
    ap.add_argument("--competition", action="append", default=[])
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--kernel", action="append", default=[])
    ap.add_argument("--internet", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--timeout", type=int, default=None,
                    help="session timeout seconds (global max still applies)")
    ap.add_argument("--args", default="",
                    help="extra CLI argv injected at the TOP of the cell. Needed because a "
                         "Jupyter cell has __name__ == '__main__', so a trailing "
                         "if __name__ guard would already have fired with the kernel's argv.")
    args = ap.parse_args()

    src_path = Path(args.source)
    if not src_path.is_file():
        print(f"source not found: {src_path}", file=sys.stderr)
        return 2
    source = src_path.read_text(encoding="utf-8")

    if args.args:
        import shlex

        argv = [src_path.name] + shlex.split(args.args)
        # A successful Kaggle run publishes NO execution log; only failed runs do.
        # The only reliable channel for reading a successful run is /kaggle/working,
        # so tee every stream into a file there.
        prelude = (
            "# --- injected by kpush.py: pin sys.argv BEFORE any __main__ guard ---\n"
            "import sys as _sys, os as _os\n"
            f"_sys.argv = {argv!r}\n"
            "# A successful Kaggle run publishes NO execution log; only failed runs do.\n"
            "# The only reliable channel for reading a successful run is /kaggle/working,\n"
            "# so tee every stream into a file there.\n"
            "# __getattr__ delegates everything we do not define (isatty, fileno, encoding,\n"
            "# ...) to the original stream. This is load-bearing: transformers/tqdm call\n"
            "# sys.stdout.isatty(), and a tee missing it makes model loading fail with\n"
            "# AttributeError: '_Tee' object has no attribute 'isatty'.\n"
            "class _Tee:\n"
            "    def __init__(self, primary, secondary):\n"
            "        self._primary = primary\n"
            "        self._secondary = secondary\n"
            "    def write(self, s):\n"
            "        for _st in (self._primary, self._secondary):\n"
            "            try:\n"
            "                _st.write(s)\n"
            "            except Exception:\n"
            "                pass\n"
            "        return len(s)\n"
            "    def flush(self):\n"
            "        for _st in (self._primary, self._secondary):\n"
            "            try:\n"
            "                _st.flush()\n"
            "            except Exception:\n"
            "                pass\n"
            "    def __getattr__(self, name):\n"
            "        if name in ('_primary', '_secondary'):\n"
            "            raise AttributeError(name)\n"
            "        return getattr(self._primary, name)\n"
            "try:\n"
            "    _os.makedirs('/kaggle/working', exist_ok=True)\n"
            "    _fh = open('/kaggle/working/kernel_stdout.log', 'w', buffering=1, encoding='utf-8')\n"
            "    _sys.stdout = _Tee(_sys.stdout, _fh)\n"
            "    _sys.stderr = _Tee(_sys.stderr, _fh)\n"
            "    print('kpush: teeing stdout to /kaggle/working/kernel_stdout.log',\n"
            "          '| isatty =', _sys.stdout.isatty())\n"
            "except Exception as _e:\n"
            "    print('kpush: tee setup failed:', _e)\n"
        )
        source = prelude + source
        print(f"injected argv: {argv}")

    slug_name = args.slug.split("/")[-1]
    folder = BUILD_ROOT / slug_name
    folder.mkdir(parents=True, exist_ok=True)

    nb_path = folder / "kernel.ipynb"
    nb_path.write_text(json.dumps(build_notebook(source), indent=1), encoding="utf-8")

    meta = {
        "id": args.slug,
        "title": args.title,
        "code_file": "kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        "enable_internet": bool(args.internet),
        "competition_sources": args.competition,
        "dataset_sources": args.dataset,
        "kernel_sources": args.kernel,
        "model_sources": args.model,
    }
    if not args.no_gpu:
        # The whole point of this helper: machine_shape must be in the metadata.
        meta["machine_shape"] = args.shape
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(folder)]
    if args.timeout:
        cmd += ["-t", str(args.timeout)]
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in (proc.stdout or "").splitlines():
        print("  ", line)
    for line in (proc.stderr or "").splitlines():
        print("  !", line)
    print(f"machine_shape written: {meta.get('machine_shape')!r}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
