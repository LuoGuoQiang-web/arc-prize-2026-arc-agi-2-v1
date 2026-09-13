"""Decode a Kaggle kernel log (JSONL of stream entries) into plain text.

Kaggle logs are one JSON object per line:
    {"stream_name": "stdout", "time": 12.5, "data": "...\\n"}

Usage:
    python decode_klog.py <log> [--stream stdout|stderr|both] [--max-chars 220] [--head N]
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--stream", default="stdout", choices=["stdout", "stderr", "both"])
    ap.add_argument("--max-chars", type=int, default=220)
    ap.add_argument("--head", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    raw = open(args.log, encoding="utf-8", errors="replace").read()
    out: list[str] = []

    # The file is a JSON array-ish stream; entries may or may not be comma-separated.
    decoder = json.JSONDecoder()
    pos = 0
    n = len(raw)
    while pos < n:
        # skip separators
        while pos < n and raw[pos] in " \t\r\n,":
            pos += 1
        if pos >= n:
            break
        if raw[pos] == "[":
            pos += 1
            continue
        if raw[pos] == "]":
            pos += 1
            continue
        try:
            obj, end = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            # fall back to line-by-line
            line_end = raw.find("\n", pos)
            if line_end == -1:
                break
            chunk = raw[pos:line_end].lstrip(",")
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                pos = line_end + 1
                continue
            end = line_end
        pos = end

        if isinstance(obj, list):
            items = obj
        else:
            items = [obj]
        for it in items:
            if not isinstance(it, dict):
                continue
            stream = it.get("stream_name", "")
            if args.stream != "both" and stream != args.stream:
                continue
            data = it.get("data", "")
            for line in str(data).splitlines():
                line = line.rstrip()
                if not line:
                    continue
                if len(line) > args.max_chars:
                    line = line[: args.max_chars] + " …"
                out.append(f"[{stream[:3]}] {line}")

    if args.head:
        out = out[: args.head]
    print("\n".join(out))
    print(f"\n--- {len(out)} lines shown (stream={args.stream}) ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
