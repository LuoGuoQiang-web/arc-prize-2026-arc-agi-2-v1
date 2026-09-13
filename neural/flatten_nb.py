"""Flatten a .ipynb into plain Python/markdown text so it can be grepped and read.

Kaggle notebooks are stored as a single JSON line, which makes line-based search
useless. This writes one block per cell with a header.

Usage:
    python flatten_nb.py <notebook.ipynb> [-o out.py]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    nb_path = Path(args.notebook)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out_path = Path(args.out) if args.out else nb_path.with_suffix(".flat.py")

    # notebook-level kaggle metadata is informative (accelerator, docker, sources)
    kaggle_meta = nb.get("metadata", {}).get("kaggle")
    chunks: list[str] = []
    if kaggle_meta:
        chunks.append("# " + "=" * 76)
        chunks.append("# notebook kaggle metadata")
        chunks.append("# " + "=" * 76)
        chunks.append(json.dumps(kaggle_meta, indent=2, ensure_ascii=False))

    for i, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        kind = cell.get("cell_type", "?")
        chunks.append("")
        chunks.append(f"# {'=' * 76}")
        chunks.append(f"# CELL {i}  [{kind}]")
        chunks.append(f"# {'=' * 76}")
        if kind == "markdown":
            chunks.extend("# " + ln for ln in src.splitlines())
        else:
            chunks.append(src)

    text = "\n".join(chunks)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}  ({len(text)} chars, {text.count(chr(10)) + 1} lines, "
          f"{len(nb.get('cells', []))} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
