"""Cross-reference submitted grids against the recorded candidate source.

``diagnose_attempts.py`` found that ~34% of attempt_1 grids are single-colour and ~44%
"look like a fallback". That is ambiguous between two very different failures:

  (a) the neural path produced nothing and the FLOOR layer was submitted -- coverage is
      worse than the report's "24/24 tasks had a real candidate" suggests; or
  (b) the model itself emitted a degenerate uniform grid -- i.e. candidate *generation*
      quality is the bottleneck, not selection and not coverage.

The recorded per-attempt `sources` field distinguishes them, and the two call for
completely different fixes, so guessing is not acceptable.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "eval_out")


def uniform(grid):
    return len({cell for row in grid for cell in row}) == 1


def grid_key(g):
    return tuple(tuple(r) for r in g)


def main() -> int:
    sub = json.load(open(os.path.join(OUT, "submission.json"), encoding="utf-8"))
    rep = json.load(open(os.path.join(OUT, "report.json"), encoding="utf-8"))
    per = {r["task_id"]: r for r in rep.get("per_task", [])}

    # attempt-level tally
    tally = Counter()
    print(f"{'task':<10} {'#in':>4}  {'sources':<24} {'cand':>5}  attempt_1 kinds")
    print("-" * 78)
    for tid in sorted(sub):
        rec = per.get(tid, {})
        srcs = rec.get("attempts") or []
        kinds = []
        for i, entry in enumerate(sub[tid]):
            a1 = entry["attempt_1"]
            src = srcs[i] if i < len(srcs) else "?"
            kind = "UNIFORM" if uniform(a1) else "varied"
            if uniform(a1) and src == "neural":
                kind = "UNIFORM/neural"
            elif uniform(a1):
                kind = f"UNIFORM/{src}"
            tally[(src, "uniform" if uniform(a1) else "varied")] += 1
            kinds.append(kind)
        print(f"{tid:<10} {len(sub[tid]):>4}  {str(srcs):<24} "
              f"{rec.get('n_candidates'):>5}  {kinds}")

    print()
    print("attempt_1 breakdown by (recorded source, is-uniform):")
    for (src, kind), n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    source={src:<10} {kind:<8} n={n}")

    uni_neural = tally[("neural", "uniform")]
    uni_fallback = sum(n for (s, k), n in tally.items() if k == "uniform" and s != "neural")
    print()
    print(f"  uniform attempt_1 from the MODEL : {uni_neural}")
    print(f"  uniform attempt_1 from a FALLBACK: {uni_fallback}")
    print()
    if uni_neural >= uni_fallback:
        print("VERDICT: the uniform grids come from the model, so the bottleneck is "
              "candidate GENERATION quality, not coverage and not selection.")
        print("         Implication: more time per task will not fix it; the model needs "
              "a better decoding constraint or a rescorer that can rank a non-degenerate "
              "candidate above a degenerate one.")
    else:
        print("VERDICT: the uniform grids are mostly the floor layer, so coverage is "
              "worse than the report's task-level flag implies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
