"""
ARC Prize 2026 — ARC-AGI-2 v1 engine (offline, standard library only).
ARC Prize 2026 — ARC-AGI-2 第一版引擎（纯标准库、可离线运行）

Verified contract (2026-09-11, sources: arcprize.org/competitions/2026/arc-agi-2,
arcprize.org/competitions/2026/paper, official sample_submission.json):
  * Submission must be produced by a **Kaggle notebook** with **internet disabled**.
  * Predict **exactly 2 outputs for every test input**; a task scores 1 if ANY of the
    2 attempts matches the ground truth exactly; final score = mean over test outputs.
  * Submission file: `/kaggle/working/submission.json`
        { "<task_id>": [ {"attempt_1": grid, "attempt_2": grid}, ... ] }
    the list length equals the number of test inputs of that task; grid = rectangular
    list of lists of ints 0..9.
  * Closing the prizes requires open-sourcing the code (choose MIT / Apache-2.0 / CC0).

Design constraints taken from the project brief (8 GB local GPU, solo, ~100 h budget):
  * Symbolic program search + execution feedback is the main compute; no neural training.
  * Every candidate program must reproduce ALL demonstration pairs before it can vote.
  * Deterministic: same config -> same submission; nothing depends on wall-clock order.

What this engine provides (v1):
  1. Data discovery      : finds ARC-AGI-2 competition files under /kaggle/input (any layout).
  2. Solver              : DSL primitives -> depth-1 / depth-2 chains + fitted colour map.
  3. Two-attempt policy  : attempt_1 from the simplest validated program, attempt_2 from the
                           next distinct valid prediction, else from a ranked prior.
  4. Run management      : run dirs, manifest (code/​config/dataset fingerprints),
                           append-only checkpoint, resume, forced re-run, run pruning.
  5. Backup / update     : rotating zip snapshots, submission rebuild from checkpoint,
                           periodic submission flush, PROJECT_STATE.md, latest-backup restore.
  6. Self-scoring        : in dev mode scores against public evaluation solutions and emits
                           a failure taxonomy (feeds the paper's diagnostic section).
  7. Safety              : global deadline + per-task timeouts, guaranteed-valid submission,
                           schema validation before finishing.

Author: built for the ARC Prize 2026 campaign. Licence: MIT-0 (see LICENSE note in README).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

ENGINE_VERSION = "1.0.0"

# --------------------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------------------

DEFAULTS = {
    # locations
    "project_dir": None,            # None -> /kaggle/working/arcprize on Kaggle, else ./arcprize_runtime
    "input_roots": None,            # None -> auto-discovered candidate roots
    "submission_path": None,        # None -> /kaggle/working/submission.json if writable
    # what to run
    "mode": "auto",                 # auto | submit | dev
    "dataset": None,                # None -> auto (test in submit mode, evaluation in dev mode)
    # solver
    "enable_depth2": True,
    "enable_object_family": True,   # spend the 2nd attempt on a different hypothesis class
    "enable_context_family": True,  # prefer the context-conditioned class for that slot
    "enable_shape_family": True,    # shape-changing / counting family (ranks above priors)
    "max_chains": 4000,
    "per_task_seconds": 60.0,
    "global_budget_seconds": 7200.0,
    "reserve_seconds": 300.0,
    # run management
    "run_id": None,
    "resume": True,
    "force": False,
    "submission_flush_every": 25,
    "snapshot_on_finish": True,
    "backup_keep": 3,
    "runs_keep": 5,
    "log_every": 10,
    "seed": 0,
}

# keys that must NOT invalidate resume when changed (they do not change solved results)
NON_SEMANTIC_KEYS = {
    "force", "resume", "snapshot_on_finish", "backup_keep", "runs_keep",
    "log_every", "run_id", "project_dir", "submission_path", "input_roots",
}

DATA_PATTERNS = {
    "test_challenges": ["arc-agi_test_challenges.json", "test_challenges.json"],
    "training_challenges": ["arc-agi_training_challenges.json", "training_challenges.json"],
    "training_solutions": ["arc-agi_training_solutions.json", "training_solutions.json"],
    "evaluation_challenges": ["arc-agi_evaluation_challenges.json", "evaluation_challenges.json"],
    "evaluation_solutions": ["arc-agi_evaluation_solutions.json", "evaluation_solutions.json"],
    "sample_submission": ["sample_submission.json"],
}

DEFAULT_INPUT_ROOTS = [
    "/kaggle/input",
    "/kaggle/input/arc-prize-2026-arc-agi-2",
    "/kaggle/input/arc-prize-2026-agi-2",
    "/kaggle/input/arc-prize-2025",
    "./data",
    ".",
]


def make_config(overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        cfg.update(overrides)
    if cfg["project_dir"] is None:
        cfg["project_dir"] = "/kaggle/working/arcprize" if Path("/kaggle/working").is_dir() \
            else str(Path.cwd() / "arcprize_runtime")
    if cfg["input_roots"] is None:
        cfg["input_roots"] = list(DEFAULT_INPUT_ROOTS)
    random.seed(int(cfg["seed"]))
    return cfg


def config_hash(cfg: dict) -> str:
    semantic = {k: v for k, v in cfg.items() if k not in NON_SEMANTIC_KEYS}
    blob = json.dumps(semantic, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------------------
# 1. Small utilities
# --------------------------------------------------------------------------------------

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def file_fingerprint(path: Path, head_bytes: int = 1 << 20) -> dict:
    """Cheap, stable identity for a data file: size + hash of first head_bytes."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(head_bytes)
        return {"path": str(path), "size": size, "head_sha256": sha256_bytes(head)}
    except Exception as exc:  # pragma: no cover
        return {"path": str(path), "error": repr(exc)}


def write_json_atomic(path: Path, obj, indent: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Logger:
    """Prints to stdout and appends to the run log (survives notebook disconnects)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.t0 = time.time()
        self._dir_ready = False
        self._dir_ok = True

    def __call__(self, msg: str) -> None:
        elapsed = time.time() - self.t0
        line = f"[{time.strftime('%H:%M:%S')} +{elapsed:7.1f}s] {msg}"
        print(line, flush=True)
        if self.path and self._dir_ok:
            try:
                if not self._dir_ready:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._dir_ready = True
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                self._dir_ok = False


# --------------------------------------------------------------------------------------
# 2. Grid helpers
# --------------------------------------------------------------------------------------

def grid_shape(g) -> tuple:
    if not g or not isinstance(g, list):
        return (0, 0)
    return (len(g), len(g[0]) if isinstance(g[0], list) else 0)


def sanitize_grid(g) -> list:
    """Guarantee a legal submission grid: non-empty, rectangular, ints 0..9."""
    if not g or not isinstance(g, list) or not isinstance(g[0], list) or not g[0]:
        return [[0]]
    width = max(len(r) for r in g)
    out = []
    for row in g:
        row = list(row) + [0] * (width - len(row))
        out.append([max(0, min(9, int(v))) for v in row])
    return out or [[0]]


def grids_equal(a, b) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        if ra != rb:
            return False
    return True


def infer_bg(pairs) -> int:
    """Most frequent colour across demonstration inputs (ARC background prior)."""
    counter = Counter()
    for p in pairs:
        for row in p["input"]:
            counter.update(row)
    return counter.most_common(1)[0][0] if counter else 0


def flatten(g):
    for row in g:
        yield from row


def dihedral_variants(g):
    """The 8 elements of the dihedral group (used for diagnostics)."""
    t = [list(r) for r in zip(*g)]
    variants = [g, [r[::-1] for r in g], g[::-1], [r[::-1] for r in g[::-1]], t,
                [r[::-1] for r in t], t[::-1], [r[::-1] for r in t[::-1]]]
    return variants


# --------------------------------------------------------------------------------------
# 3. DSL primitives (grid -> grid | None); bg is bound per task
# --------------------------------------------------------------------------------------

def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def _rot180(g):
    return [r[::-1] for r in g[::-1]]


def _rot270(g):
    return [list(r) for r in zip(*g)][::-1]


def _fliph(g):
    return [r[::-1] for r in g]


def _flipv(g):
    return [r[:] for r in g[::-1]]


def _transpose(g):
    return [list(r) for r in zip(*g)]


def _antitrans(g):
    return _rot180(_transpose(g))


def _crop_nonbg(g, bg):
    cells = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg]
    if not cells:
        return None
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def _trim_edges(g, bg):
    """Strip uniform-bg rows/columns from all four edges."""
    out = [row[:] for row in g]
    while len(out) > 1 and all(v == bg for v in out[0]):
        out.pop(0)
    while len(out) > 1 and all(v == bg for v in out[-1]):
        out.pop()
    while out and len(out[0]) > 1 and all(row[0] == bg for row in out):
        out = [row[1:] for row in out]
    while out and len(out[0]) > 1 and all(row[-1] == bg for row in out):
        out = [row[:-1] for row in out]
    return out or None


def _gravity(g, direction, bg):
    h, w = grid_shape(g)
    out = [[bg] * w for _ in range(h)]
    if direction in ("down", "up"):
        for c in range(w):
            col = [g[r][c] for r in range(h) if g[r][c] != bg]
            if direction == "down":
                for i, v in enumerate(col):
                    out[h - len(col) + i][c] = v
            else:
                for i, v in enumerate(col):
                    out[i][c] = v
    else:
        for r in range(h):
            row = [v for v in g[r] if v != bg]
            if direction == "right":
                out[r][w - len(row):] = row
            else:
                out[r][:len(row)] = row
    return out


def _scale_up(g, k):
    out = []
    for row in g:
        big = [v for v in row for _ in range(k)]
        for _ in range(k):
            out.append(big[:])
    return out


def _scale_down(g, k):
    h, w = grid_shape(g)
    if h % k or w % k:
        return None
    return [[g[r * k][c * k] for c in range(w // k)] for r in range(h // k)]


def _tile(g, kr, kc):
    out = []
    for _ in range(kr):
        for row in g:
            out.append(row * kc)
    return out


def _mirror_fill(g, axis, bg):
    h, w = grid_shape(g)
    out = [row[:] for row in g]
    for r in range(h):
        for c in range(w):
            if out[r][c] != bg:
                continue
            if axis == "h":
                rr, cc = r, w - 1 - c
            elif axis == "v":
                rr, cc = h - 1 - r, c
            elif axis == "d":
                if h != w:
                    continue
                rr, cc = c, r
            else:  # anti-diagonal
                if h != w:
                    continue
                rr, cc = w - 1 - c, h - 1 - r
            if g[rr][cc] != bg:
                out[r][c] = g[rr][cc]
    return out


def _drop_uniform_rows(g):
    out = [row for row in g if len(set(row)) > 1]
    return out or None


def _drop_uniform_cols(g):
    h, w = grid_shape(g)
    keep = [c for c in range(w) if len({g[r][c] for r in range(h)}) > 1]
    if not keep:
        return None
    return [[row[c] for c in keep] for row in g]


def _components(g, bg):
    """4-connected components of non-background cells -> list of (cells, bbox)."""
    h, w = grid_shape(g)
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and g[nr][nc] != bg:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            rr = [x for x, _ in cells]
            cc_ = [y for _, y in cells]
            comps.append((cells, (min(rr), min(cc_), max(rr), max(cc_))))
    return comps


def _crop_component(g, bg, largest=True):
    comps = _components(g, bg)
    if not comps:
        return None
    comps.sort(key=lambda item: len(item[0]), reverse=largest)
    r0, c0, r1, c1 = comps[0][1]
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def build_primitives(bg: int):
    """Return list of (name, fn). fn(grid) -> grid | None."""
    return [
        ("id", lambda g: [row[:] for row in g]),
        ("rot90", _rot90),
        ("rot180", _rot180),
        ("rot270", _rot270),
        ("fliph", _fliph),
        ("flipv", _flipv),
        ("transpose", _transpose),
        ("antitrans", _antitrans),
        ("crop_nonbg", lambda g: _crop_nonbg(g, bg)),
        ("trim_edges", lambda g: _trim_edges(g, bg)),
        ("grav_down", lambda g: _gravity(g, "down", bg)),
        ("grav_up", lambda g: _gravity(g, "up", bg)),
        ("grav_left", lambda g: _gravity(g, "left", bg)),
        ("grav_right", lambda g: _gravity(g, "right", bg)),
        ("scale_up2", lambda g: _scale_up(g, 2)),
        ("scale_up3", lambda g: _scale_up(g, 3)),
        ("scale_down2", lambda g: _scale_down(g, 2)),
        ("scale_down3", lambda g: _scale_down(g, 3)),
        ("tile1x2", lambda g: _tile(g, 1, 2)),
        ("tile2x1", lambda g: _tile(g, 2, 1)),
        ("tile2x2", lambda g: _tile(g, 2, 2)),
        ("mirror_h", lambda g: _mirror_fill(g, "h", bg)),
        ("mirror_v", lambda g: _mirror_fill(g, "v", bg)),
        ("mirror_d", lambda g: _mirror_fill(g, "d", bg)),
        ("mirror_a", lambda g: _mirror_fill(g, "a", bg)),
        ("drop_uniform_rows", _drop_uniform_rows),
        ("drop_uniform_cols", _drop_uniform_cols),
        ("largest_cc_crop", lambda g: _crop_component(g, bg, True)),
        ("smallest_cc_crop", lambda g: _crop_component(g, bg, False)),
    ]


# depth-2 combinations are restricted to this subset (shape-changing / geometric primitives)
D2_NAMES = {
    "rot90", "rot180", "rot270", "fliph", "flipv", "transpose", "antitrans",
    "crop_nonbg", "trim_edges", "grav_down", "grav_up", "grav_left", "grav_right",
    "mirror_h", "mirror_v", "drop_uniform_rows", "drop_uniform_cols",
}


def fit_color_map(srcs, dsts):
    """Fit a colour mapping consistent with all demonstration pairs; None if impossible."""
    cmap = {}
    for s, d in zip(srcs, dsts):
        if s is None or grid_shape(s) != grid_shape(d):
            return None
        for rs, rd in zip(s, d):
            for a, b in zip(rs, rd):
                prev = cmap.get(a)
                if prev is None:
                    cmap[a] = b
                elif prev != b:
                    return None
    return cmap


def apply_color_map(g, cmap):
    return [[cmap.get(v, v) for v in row] for row in g]


def run_chain(chain, g):
    cur = g
    for _, fn in chain:
        cur = fn(cur)
        if cur is None or not cur or not cur[0]:
            return None
    return cur


# --------------------------------------------------------------------------------------
# 4. Candidate construction and solving
# --------------------------------------------------------------------------------------

def _chain_key(chain):
    return "->".join(name for name, _ in chain)


def build_search_candidates(task, bg, cfg):
    """
    Returns a list of dicts: {"rank": tuple, "name": str, "fn": callable, "prevalidated": bool}
    Every chain is pre-validated by construction (colour map is fitted to the demos).
    """
    prims = build_primitives(bg)
    by_name = dict(prims)
    tin = [p["input"] for p in task["train"]]
    tout = [p["output"] for p in task["train"]]

    chains = []
    for name, fn in prims:
        chains.append((0, [(name, fn)]))
    if cfg["enable_depth2"]:
        names = [n for n in D2_NAMES if n in by_name]
        for a in names:
            for b in names:
                chains.append((1, [(a, by_name[a]), (b, by_name[b])]))
    if len(chains) > int(cfg["max_chains"]):
        chains = chains[:int(cfg["max_chains"])]

    out = []
    for seq, (depth, chain) in enumerate(chains):
        transformed = []
        ok = True
        for g in tin:
            tg = run_chain(chain, g)
            if tg is None:
                ok = False
                break
            transformed.append(tg)
        if not ok:
            continue
        cmap = fit_color_map(transformed, tout)
        if cmap is None:
            continue
        n_changed = sum(1 for k, v in cmap.items() if k != v)
        cname = _chain_key(chain) + (f"+cmap{n_changed}" if n_changed else "")
        # tier 0 = validated program search; later fields break ties toward simplicity:
        # fewer composed ops -> fewer recoloured colours -> earlier registry position
        rank = (0, depth, n_changed, seq, cname)
        out.append({
            "rank": rank,
            "name": cname,
            "prevalidated": True,
            "fn": (lambda ch, cm: (lambda g: (lambda t: None if t is None else apply_color_map(t, cm))(run_chain(ch, g))))(chain, cmap),
        })

    # constant-output program (legal only if all demonstration outputs are identical)
    if tout and all(grids_equal(tout[0], t) for t in tout):
        const = [row[:] for row in tout[0]]
        out.append({
            "rank": (0, -1, 0, -1, "const"),
            "name": "const",
            "prevalidated": True,
            "fn": (lambda c: (lambda g: [row[:] for row in c]))(const),
        })
    return out


def _priority_candidates(task):
    """
    Shape/palette priors. They are tier 1: even when they reproduce the demonstrations
    exactly, a validated DSL program (tier 0) always outranks them, because a prior carries
    no structural explanation of the task.
    """
    tin = [p["input"] for p in task["train"]]
    tout = [p["output"] for p in task["train"]]
    out = []

    def add(prio: int, name: str, fn):
        out.append({"rank": (1, prio, 0, 0, name), "name": name, "prevalidated": False, "fn": fn})

    # 1. cell-wise majority (needs a common shape)
    shapes = {grid_shape(t) for t in tout}
    if len(shapes) == 1 and len({grid_shape(t) for t in tin}) == 1:
        h, w = grid_shape(tout[0])
        cells = [[Counter() for _ in range(w)] for _ in range(h)]
        for t in tout:
            for r in range(h):
                for c in range(w):
                    cells[r][c][t[r][c]] += 1
        majority = [[cells[r][c].most_common(1)[0][0] for c in range(w)] for r in range(h)]
        add(1, "cellwise_majority", (lambda m: (lambda g: [row[:] for row in m]))(majority))

    # 2. most frequent demonstration output reused for every test input
    counts = Counter(json.dumps(t) for t in tout)
    if counts:
        best = json.loads(counts.most_common(1)[0][0])
        add(2, "mode_output", (lambda m: (lambda g: [row[:] for row in m]))(best))

    # 3. constant canvas: most frequent output shape filled with the most frequent output colour
    if counts:
        h, w = grid_shape(tout[0])
        color_counts = Counter()
        for t in tout:
            for v in flatten(t):
                color_counts[v] += 1
        col = color_counts.most_common(1)[0][0]
        uni = [[col] * w for _ in range(h)]
        add(3, "uniform_canvas", (lambda m: (lambda g: [row[:] for row in m]))(uni))

    # 4. input tiled to the most frequent output shape
    def tiled_to_shape(g, target_h, target_w):
        h, w = grid_shape(g)
        if h == 0 or w == 0:
            return None
        return [[g[r % h][c % w] for c in range(target_w)] for r in range(target_h)]

    if counts:
        h, w = grid_shape(tout[0])
        add(4, "tile_to_shape", (lambda th, tw: (lambda g: tiled_to_shape(g, th, tw)))(h, w))

    # 5. identity
    add(5, "input_copy", lambda g: [row[:] for row in g])
    # 6. largest object crop
    bg = infer_bg(task["train"])
    add(6, "largest_cc_crop", lambda g: _crop_component(g, bg, True))
    return out


# --------------------------------------------------------------------------------------
# 3b. Object-level hypothesis family (v2, heterogeneous second attempt)
#
# Why: the ARC-AGI-2 public eval removes the transformation families a global-geometry
# DSL covers cheaply (per-cell recolouring, tiling, sub-grid extraction, counting all
# measure 0.0%), and 50.8% of eval tasks are object-level conditional transformations.
# Measured effect (training split, Kaggle metric): spending the second attempt on this
# family instead of a second guess from the geometry DSL moves 3.160% -> 4.182%
# (+1.02 pp), versus +0.19 pp for a within-family second guess. Every added hit lands
# in the uniform-object-recolouring family; no family regresses.
# Details and reproduction: work/arc_w1/v2/V2_RESULTS.md, work/arc_w1/diagnosis/.
#
# Same discipline as every other candidate source: parameters are FITTED on the
# demonstrations and a candidate is kept only if it reproduces every demonstration
# pair exactly. Nothing here is trained and nothing looks at test-time answers.
# --------------------------------------------------------------------------------------

OBJ_PREDICATES = {
    "is_largest": lambda objs, o: o["size"] == max(x["size"] for x in objs),
    "is_smallest": lambda objs, o: o["size"] == min(x["size"] for x in objs),
    "is_singleton": lambda objs, o: o["size"] == 1,
    "touches_border": lambda objs, o: o["touches_border"],
    "not_touches_border": lambda objs, o: not o["touches_border"],
    "is_square": lambda objs, o: o["square"],
    "has_hole": lambda objs, o: o["holes"] > 0,
    "no_hole": lambda objs, o: o["holes"] == 0,
    "most_common_color": lambda objs, o: o["color"] == Counter(x["color"] for x in objs).most_common(1)[0][0],
    "rare_color": lambda objs, o: Counter(x["color"] for x in objs)[o["color"]] == 1,
    "largest_bbox": lambda objs, o: o["h"] * o["w"] == max(x["h"] * x["w"] for x in objs),
    "smallest_bbox": lambda objs, o: o["h"] * o["w"] == min(x["h"] * x["w"] for x in objs),
}

OBJ_PROPERTIES = {
    "size": lambda objs, o: o["size"],
    "color": lambda objs, o: o["color"],
    "size_rank_desc": lambda objs, o: sorted(objs, key=lambda x: -x["size"]).index(o),
    "size_rank_asc": lambda objs, o: sorted(objs, key=lambda x: x["size"]).index(o),
    "bbox_area": lambda objs, o: o["h"] * o["w"],
    "height": lambda objs, o: o["h"],
    "width": lambda objs, o: o["w"],
    "touches_border": lambda objs, o: o["touches_border"],
    "square": lambda objs, o: o["square"],
    "holes": lambda objs, o: o["holes"],
}


def obj_components(grid, bg=0):
    """4-connected non-background components as lists of (row, col)."""
    h, w = grid_shape(grid)
    seen = [[False] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or grid[i][j] == bg:
                continue
            comp, queue = [], deque([(i, j)])
            seen[i][j] = True
            while queue:
                y, x = queue.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] != bg:
                        seen[ny][nx] = True
                        queue.append((ny, nx))
            comps.append(comp)
    return comps


def obj_holes_of(grid, comp, bg=0):
    """Number of background regions enclosed by the component (flood-fill from its bbox)."""
    h, w = grid_shape(grid)
    inside = set(comp)
    r0 = min(r for r, _ in comp)
    r1 = max(r for r, _ in comp)
    c0 = min(c for _, c in comp)
    c1 = max(c for _, c in comp)
    seen, queue = set(), deque()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if grid[r][c] == bg and (r, c) not in inside:
                queue.append((r, c))
                seen.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if grid[r][c] == bg and (r, c) not in inside:
                queue.append((r, c))
                seen.add((r, c))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if (r0 <= ny <= r1 and c0 <= nx <= c1 and grid[ny][nx] == bg
                    and (ny, nx) not in inside and (ny, nx) not in seen):
                seen.add((ny, nx))
                queue.append((ny, nx))
    n_holes, visited = 0, set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == bg and (r, c) not in inside and (r, c) not in seen and (r, c) not in visited:
                n_holes += 1
                qq = deque([(r, c)])
                visited.add((r, c))
                while qq:
                    y, x = qq.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (r0 <= ny <= r1 and c0 <= nx <= c1 and grid[ny][nx] == bg
                                and (ny, nx) not in inside and (ny, nx) not in visited):
                            visited.add((ny, nx))
                            qq.append((ny, nx))
    return n_holes


def obj_objects_of(grid, bg=0):
    """Objects with the properties the predicates and property-maps consume."""
    h, w = grid_shape(grid)
    objs = []
    for comp in obj_components(grid, bg):
        pix = set(comp)
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        colors_ = Counter(grid[r][c] for r, c in comp)
        objs.append({
            "pixels": pix,
            "size": len(comp),
            "color": colors_.most_common(1)[0][0],
            "bbox": (r0, c0, r1, c1),
            "h": r1 - r0 + 1,
            "w": c1 - c0 + 1,
            "touches_border": r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1,
            "square": (r1 - r0) == (c1 - c0),
            "holes": obj_holes_of(grid, comp, bg),
        })
    return objs


def obj_make_remove_candidate(pred_name, keep_matching, bg):
    """Erase objects that match (keep_matching=False) or that do not match (True)."""

    def fn(grid):
        objs = obj_objects_of(grid, bg)
        if not objs:
            return None
        out = [row[:] for row in grid]
        pred = OBJ_PREDICATES[pred_name]
        for o in objs:
            match = pred(objs, o)
            if match if keep_matching else not match:
                for r, c in o["pixels"]:
                    out[r][c] = bg
        return out

    return fn


def obj_collect_property_map(prop_name, pairs, bg):
    """Fit `property value -> output colour` from the demonstrations; None if inconsistent."""
    mapping = {}
    for pair in pairs:
        grid, target = pair["input"], pair["output"]
        if grid_shape(grid) != grid_shape(target):
            return None
        objs = obj_objects_of(grid, bg)
        if not objs:
            return None
        prop = OBJ_PROPERTIES[prop_name]
        for o in objs:
            outs = {target[r][c] for r, c in o["pixels"]}
            if len(outs) != 1:
                return None
            val, col = prop(objs, o), outs.pop()
            if mapping.get(val, col) != col:
                return None
            mapping[val] = col
    return mapping or None


def obj_make_recolor_fn(prop_name, mapping, bg):
    def fn(grid):
        objs = obj_objects_of(grid, bg)
        if not objs:
            return None
        out = [row[:] for row in grid]
        prop = OBJ_PROPERTIES[prop_name]
        for o in objs:
            col = mapping.get(prop(objs, o))
            if col is None:
                return None
            for r, c in o["pixels"]:
                out[r][c] = col
        return out

    return fn


def obj_fit_fill_holes(pairs, bg):
    """Colour that encloses every demonstration hole; None unless it is unique."""
    colors = set()
    for pair in pairs:
        grid, target = pair["input"], pair["output"]
        if grid_shape(grid) != grid_shape(target):
            return None
        for o in obj_objects_of(grid, bg):
            r0, c0, r1, c1 = o["bbox"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if grid[r][c] == bg and (r, c) not in o["pixels"] and target[r][c] != bg:
                        colors.add(target[r][c])
    return colors.pop() if len(colors) == 1 else None


def obj_make_fill_holes(color, bg):
    def fn(grid):
        out = [row[:] for row in grid]
        for o in obj_objects_of(grid, bg):
            if o["holes"] == 0:
                continue
            r0, c0, r1, c1 = o["bbox"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if out[r][c] == bg and (r, c) not in o["pixels"]:
                        out[r][c] = color
        return out

    return fn


def obj_fit_vanishing_colors(pairs, bg):
    """Colours that disappear entirely in every demonstration pair."""
    vanish = None
    for pair in pairs:
        grid, target = pair["input"], pair["output"]
        if grid_shape(grid) != grid_shape(target):
            return None
        gone = ({c for row in grid for c in row if c != bg}
                - {c for row in target for c in row if c != bg})
        vanish = gone if vanish is None else (vanish & gone)
    return vanish or None


def obj_make_drop_color(color, bg):
    return lambda grid: [[bg if v == color else v for v in row] for row in grid]


def obj_reproduces(fn, pairs):
    for pair in pairs:
        try:
            out = fn(pair["input"])
        except Exception:
            return False
        if out is None or out != pair["output"]:
            return False
    return True


def object_candidates(task, bg, max_candidates: int = 400):
    """Verified object-level candidates: [(name, fn)] with fn: grid -> grid."""
    pairs = task["train"]
    verified = []

    def consider(name, fn):
        if len(verified) >= max_candidates:
            return
        if obj_reproduces(fn, pairs):
            verified.append((name, fn))

    for pred_name in OBJ_PREDICATES:
        for keep_matching in (True, False):
            consider(f"obj_{'keep' if keep_matching else 'drop'}_{pred_name}",
                     obj_make_remove_candidate(pred_name, keep_matching, bg))

    for prop_name in OBJ_PROPERTIES:
        mapping = obj_collect_property_map(prop_name, pairs, bg)
        if mapping:
            consider(f"obj_recolor_by_{prop_name}", obj_make_recolor_fn(prop_name, mapping, bg))

    hole_color = obj_fit_fill_holes(pairs, bg)
    if hole_color is not None:
        consider(f"obj_fill_holes_{hole_color}", obj_make_fill_holes(hole_color, bg))

    for color in (obj_fit_vanishing_colors(pairs, bg) or set()):
        consider(f"obj_drop_color_{color}", obj_make_drop_color(color, bg))

    return verified


def _object_second_prediction(task, bg, test, first_outs):
    """Pick the object-family prediction for the second attempt, or None.

    Returns ({"name": ...}, outs) only when the object family proposes a prediction that
    differs from the first attempt's, so the two attempts stay two distinct hypotheses.
    """
    try:
        cands = object_candidates(task, bg)
    except Exception:
        return None
    if not cands:
        return None
    first_key = json.dumps(first_outs)
    for name, fn in cands:
        try:
            outs = [fn(t["input"]) for t in test]
        except Exception:
            continue
        if any(o is None for o in outs):
            continue
        outs = [sanitize_grid(o) for o in outs]
        if json.dumps(outs) == first_key:
            continue
        return ({"name": "obj:" + name}, outs)
    return None


# --------------------------------------------------------------------------------------
# 3c. Context-conditioned recolouring family (v2b, second-attempt candidate)
#
# Targets the dominant ARC-AGI-2 eval family (50.8% object-level conditional tasks) in
# its shape-preserving form: each cell's output colour is a function of *structural
# context* rather than of its own colour alone (which is all a per-cell colour map sees)
# or of a whole-object constant (which the object family requires).
#
# Measured on the training split: coverage 3.8% (38/1000) — higher than both the geometry
# DSL (2.7%) and the object family (1.9%) — standalone accuracy 1.673%, and as the second
# attempt it lifts the portfolio to 49/1076 = 4.554% (vs 45/1076 = 4.182% with the object
# family and 34/1076 = 3.160% within one class). On the public eval split all three
# vocabularies validate on exactly 0/120 tasks (see work/arc_w1/diagnosis/).
# --------------------------------------------------------------------------------------

CTX_FEATURE_SETS = [
    ("color",),
    ("color", "border"),
    ("color", "obj_size"),
    ("color", "obj_rank"),
    ("color", "touches"),
    ("color", "is_largest"),
    ("color", "rare"),
    ("color", "holes"),
    ("color", "square"),
    ("color", "grid_border"),
    ("obj_size",),
    ("obj_rank",),
    ("border",),
    ("touches",),
    ("holes",),
    ("is_largest",),
    ("color", "obj_size", "border"),
    ("color", "obj_rank", "border"),
    ("color", "obj_size", "touches"),
    ("color", "obj_rank", "is_largest"),
]


def ctx_cell_features(grid, bg):
    """Per-cell structural features; object-derived features are None for background."""
    h, w = grid_shape(grid)
    objs = obj_objects_of(grid, bg)
    owner = {}
    for idx, o in enumerate(objs):
        for rc in o["pixels"]:
            owner[rc] = idx
    sizes = [o["size"] for o in objs]
    order = sorted(range(len(objs)), key=lambda i: -sizes[i])
    rank_of = {idx: pos for pos, idx in enumerate(order)}
    color_count = Counter(o["color"] for o in objs)
    largest = order[0] if order else None

    feats = {}
    for r in range(h):
        for c in range(w):
            grid_border = (r == 0 or c == 0 or r == h - 1 or c == w - 1)
            idx = owner.get((r, c))
            if idx is None:
                feats[(r, c)] = {
                    "color": grid[r][c], "obj_size": None, "obj_rank": None, "border": None,
                    "touches": None, "holes": None, "square": None, "is_largest": None,
                    "rare": None, "grid_border": grid_border,
                }
                continue
            o = objs[idx]
            pix = o["pixels"]
            is_border = any((r + dr, c + dc) not in pix
                            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            feats[(r, c)] = {
                "color": grid[r][c],
                "obj_size": o["size"],
                "obj_rank": rank_of[idx],
                "border": is_border,
                "touches": o["touches_border"],
                "holes": o["holes"],
                "square": o["square"],
                "is_largest": (idx == largest),
                "rare": color_count[o["color"]] == 1,
                "grid_border": grid_border,
            }
    return feats


def ctx_fit_map(feature_names, pairs, bg):
    """Fit feature-tuple -> output colour on the demonstrations; None if inconsistent."""
    mapping = {}
    for pair in pairs:
        grid, target = pair["input"], pair["output"]
        if grid_shape(grid) != grid_shape(target):
            return None
        for (r, c), f in ctx_cell_features(grid, bg).items():
            key = tuple(f[name] for name in feature_names)
            val = target[r][c]
            if mapping.get(key, val) != val:
                return None
            mapping[key] = val
    return mapping or None


def ctx_make_fn(feature_names, mapping, bg):
    def fn(grid):
        out = [row[:] for row in grid]
        for (r, c), f in ctx_cell_features(grid, bg).items():
            val = mapping.get(tuple(f[name] for name in feature_names))
            if val is None:
                return None
            out[r][c] = val
        return out

    return fn


def context_candidates(task, bg, max_candidates: int = 40):
    """Verified context-conditioned candidates: [(name, fn)]."""
    pairs = task["train"]
    out = []
    for names in CTX_FEATURE_SETS:
        if len(out) >= max_candidates:
            break
        mapping = ctx_fit_map(names, pairs, bg)
        if not mapping:
            continue
        fn = ctx_make_fn(names, mapping, bg)
        if obj_reproduces(fn, pairs):
            out.append(("ctx_" + "+".join(names), fn))
    return out


# --------------------------------------------------------------------------------------
# 3d. Shape-changing / counting family (v3)
#
# Why: ~31% of eval tasks change the output shape, and every other family here is nearly
# shape-preserving (they fit per-cell colour maps or object recolouring, both needing matching
# shapes). On the real 240-task test set this family validates on 5 tasks, three of which no
# other family reaches, raising union coverage from 12/240 to 15/240.
#
# Precision is deliberately accounted for: on the training split this family's first proposal
# is correct only ~56% of the time (counting maps fit coincidentally), versus ~96-100% for the
# geometry/object/context families. Its candidates therefore rank BELOW every geometry chain
# and ABOVE the unvalidated priors, so they can only ever replace a prior guess.
# --------------------------------------------------------------------------------------

V3_MAX = 30


def _v3_sub(grid, bbox):
    r0, c0, r1, c1 = bbox
    out = [[grid[r][c] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    return out if len(out) <= V3_MAX and len(out[0]) <= V3_MAX else None


def _v3_bbox_nonbg(grid, bg):
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg]
    if not pts:
        return None
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _v3_scale(grid, k, up=True):
    h, w = grid_shape(grid)
    if up:
        nh, nw = h * k, w * k
        if nh > V3_MAX or nw > V3_MAX:
            return None
        return [[grid[r // k][c // k] for c in range(nw)] for r in range(nh)]
    if h % k or w % k:
        return None
    return [[grid[r * k][c * k] for c in range(w // k)] for r in range(h // k)]


def _v3_holes(grid, bg):
    """Number of background regions fully enclosed by non-background cells."""
    h, w = grid_shape(grid)
    seen = [[False] * w for _ in range(h)]
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == bg and not seen[r][c]:
                seen[r][c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == bg and not seen[r][c]:
                seen[r][c] = True
                queue.append((r, c))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == bg:
                seen[ny][nx] = True
                queue.append((ny, nx))
    holes = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not seen[r][c]:
                holes += 1
                seen[r][c] = True
                qq = deque([(r, c)])
                while qq:
                    y, x = qq.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == bg:
                            seen[ny][nx] = True
                            qq.append((ny, nx))
    return holes


def _v3_counts(grid, bg):
    objs = obj_objects_of(grid, bg)
    sizes = [o["size"] for o in objs]
    return {
        "n_objects": len(objs),
        "n_colors": len({v for row in grid for v in row}),
        "n_nonbg": sum(1 for row in grid for v in row if v != bg),
        "n_bg": sum(1 for row in grid for v in row if v == bg),
        "max_obj_size": max(sizes) if sizes else 0,
        "min_obj_size": min(sizes) if sizes else 0,
        "n_holes": _v3_holes(grid, bg),
    }


def _v3_trim(grid):
    keep_r = [i for i, row in enumerate(grid) if len(set(row)) > 1]
    keep_c = [j for j in range(len(grid[0])) if len({grid[i][j] for i in range(len(grid))}) > 1]
    if not keep_r or not keep_c:
        return [row[:] for row in grid]
    return [[grid[i][j] for j in keep_c] for i in keep_r]


def _v3_builders(bg):
    """Candidate generators for the shape-changing / counting family (bg is task-specific)."""
    out = []
    for pred in ("largest", "smallest", "unique_color"):
        def make(pred=pred):
            def fn(grid):
                objs = obj_objects_of(grid, bg)
                if not objs:
                    return None
                if pred == "largest":
                    pick = max(objs, key=lambda o: o["size"])
                elif pred == "smallest":
                    pick = min(objs, key=lambda o: o["size"])
                else:
                    cnt = Counter(o["color"] for o in objs)
                    uniq = [o for o in objs if cnt[o["color"]] == 1]
                    if not uniq:
                        return None
                    pick = uniq[0]
                return _v3_sub(grid, pick["bbox"])
            return fn
        out.append((f"v3crop_obj_{pred}", make()))

    for color in range(10):
        if color == bg:
            continue

        def make_rm(color=color):
            def fn(grid):
                erased = [[bg if v == color else v for v in row] for row in grid]
                b = _v3_bbox_nonbg(erased, bg)
                return None if b is None else _v3_sub(erased, b)
            return fn

        out.append((f"v3rmcolor{color}_crop", make_rm()))

    for k in (2, 3, 4):
        out.append((f"v3up{k}", (lambda kk: (lambda g: _v3_scale(g, kk, True)))(k)))
        out.append((f"v3down{k}", (lambda kk: (lambda g: _v3_scale(g, kk, False)))(k)))

    def trim_crop(grid):
        trimmed = _v3_trim(grid)
        b = _v3_bbox_nonbg(trimmed, bg)
        return None if b is None else _v3_sub(trimmed, b)

    out.append(("v3trim_crop", trim_crop))

    def ring(grid):
        h, w = grid_shape(grid)
        if h > V3_MAX or w > V3_MAX:
            return None
        return [[grid[r][c] if (r in (0, h - 1) or c in (0, w - 1)) else bg for c in range(w)]
                for r in range(h)]

    out.append(("v3ring", ring))
    return out


def shape_candidates(task, bg, max_candidates: int = 120):
    """Verified shape-changing / counting candidates: [(name, fn)] with fn: grid -> grid | None."""
    pairs = task["train"]
    good = []

    def reproduces(fn):
        for p in pairs:
            try:
                if fn(p["input"]) != p["output"]:
                    return False
            except Exception:
                return False
        return True

    for name, fn in _v3_builders(bg):
        if len(good) >= max_candidates:
            break
        if reproduces(fn):
            good.append((name, fn))

    # Counting family: a 1x1 output whose value is a fitted map of some input count.
    if pairs and all(grid_shape(p["output"]) == (1, 1) for p in pairs):
        for feat in ("n_objects", "n_colors", "n_nonbg", "n_bg", "max_obj_size",
                     "min_obj_size", "n_holes"):
            mapping = {}
            ok = True
            for p in pairs:
                v = _v3_counts(p["input"], bg)[feat]
                t = p["output"][0][0]
                if mapping.get(v, t) != t:
                    ok = False
                    break
                mapping[v] = t
            if not ok or not mapping:
                continue

            def fn(grid, feat=feat, mapping=mapping):
                v = _v3_counts(grid, bg)[feat]
                return None if v not in mapping else [[mapping[v]]]

            if len(good) < max_candidates:
                good.append((f"v3count_{feat}", fn))
    return good


def _context_second_prediction(task, bg, test, first_outs):
    """Pick the context-conditioned prediction for the second attempt, or None.

    Context-conditioned recolouring has the higher standalone accuracy of the two
    alternative classes (1.673% vs 1.115% on the development split), so it is offered
    the slot first; both are only ever measured on the training split.
    """
    try:
        cands = context_candidates(task, bg)
    except Exception:
        return None
    if not cands:
        return None
    first_key = json.dumps(first_outs)
    for name, fn in cands:
        try:
            outs = [fn(t["input"]) for t in test]
        except Exception:
            continue
        if any(o is None for o in outs):
            continue
        outs = [sanitize_grid(o) for o in outs]
        if json.dumps(outs) == first_key:
            continue
        return ({"name": "ctx:" + name}, outs)
    return None


def solve_task(task, cfg, task_deadline=None):
    """Solve one task. Returns a dict with predictions + diagnostics."""
    t_start = time.monotonic()
    bg = infer_bg(task["train"])
    test = task["test"]
    diagnostics = {
        "n_train": len(task["train"]),
        "n_test": len(test),
        "bg": bg,
        "timed_out": False,
        "n_candidates": 0,
        "n_valid": 0,
        "top_program": None,
        "second_program": None,
        "source": "prior",
        "elapsed_s": 0.0,
    }

    candidates = []
    try:
        candidates.extend(build_search_candidates(task, bg, cfg))
        # v3 (shape-changing / counting) candidates are verified by construction, but ranked
        # below every geometry chain (depth 3) and above the unvalidated priors (tier 1),
        # because their measured precision is ~56% versus ~96-100% for the other families.
        if cfg.get("enable_shape_family", True):
            for seq, (v3_name, v3_fn) in enumerate(shape_candidates(task, bg)):
                candidates.append({
                    "rank": (0, 3, 0, seq, v3_name),
                    "name": v3_name,
                    "prevalidated": True,
                    "fn": v3_fn,
                })
        candidates.extend(_priority_candidates(task))
    except Exception:
        diagnostics["build_error"] = traceback.format_exc(limit=3)
    diagnostics["n_candidates"] = len(candidates)

    tin = [p["input"] for p in task["train"]]
    tout = [p["output"] for p in task["train"]]

    valid = []
    for idx, cand in enumerate(candidates):
        if task_deadline and (idx & 15) == 0 and time.monotonic() > task_deadline:
            diagnostics["timed_out"] = True
            break
        fn = cand["fn"]
        try:
            if cand["prevalidated"]:
                ok = True
            else:
                ok = all(grids_equal(fn(g), t) for g, t in zip(tin, tout))
        except Exception:
            ok = False
        if ok:
            valid.append(cand)
    valid.sort(key=lambda c: c["rank"])
    diagnostics["n_valid"] = len(valid)

    # predictions, de-duplicated by the exact tuple of test outputs
    predictions = []
    seen = set()
    for cand in valid:
        if task_deadline and time.monotonic() > task_deadline + cfg["per_task_seconds"]:
            diagnostics["timed_out"] = True
            break
        try:
            outs = [cand["fn"](t["input"]) for t in test]
        except Exception:
            continue
        if any(o is None for o in outs):
            continue
        outs = [sanitize_grid(o) for o in outs]
        key = json.dumps(outs)
        if key in seen:
            continue
        seen.add(key)
        predictions.append((cand, outs))

    if predictions:
        diagnostics["source"] = "search"
        diagnostics["top_program"] = predictions[0][0]["name"]
    else:
        priors = _priority_candidates(task)
        for cand in priors:
            try:
                outs = [cand["fn"](t["input"]) for t in test]
            except Exception:
                continue
            if any(o is None for o in outs):
                continue
            outs = [sanitize_grid(o) for o in outs]
            key = json.dumps(outs)
            if key in seen:
                continue
            seen.add(key)
            predictions.append((cand, outs))
        if predictions:
            diagnostics["top_program"] = predictions[0][0]["name"]

    # guarantee: always exactly two attempts per test input
    if not predictions:
        predictions = [({"name": "fallback_zeros"}, [[[0]] for _ in test])]
    while len(predictions) < 2:
        predictions.append(({"name": predictions[0][0]["name"] + "#dup"}, predictions[0][1]))

    first, second = predictions[0], predictions[1]

    # Heterogeneous second attempt: offer the slot to a structurally different hypothesis
    # class when it proposes a prediction that differs from attempt_1, rather than spending
    # it on a second guess from the same solver. The context-conditioned family is offered
    # first (higher standalone accuracy on the development split); both are measured only
    # on the training split, never on eval.
    for enabled, picker in (
        (cfg.get("enable_context_family", True), _context_second_prediction),
        (cfg.get("enable_object_family", True), _object_second_prediction),
    ):
        if not enabled:
            continue
        alt = picker(task, bg, test, first[1])
        if alt is not None:
            second = alt
            diagnostics["heterogeneous_second"] = alt[0]["name"]
            break
    diagnostics["second_program"] = second[0]["name"]

    attempts = []
    for i in range(len(test)):
        attempts.append({
            "attempt_1": first[1][i],
            "attempt_2": second[1][i],
        })

    diagnostics["elapsed_s"] = round(time.monotonic() - t_start, 3)
    return {"attempts": attempts, "diagnostics": diagnostics}


def classify_failure(pred_attempts, truth):
    """Failure taxonomy for the paper's diagnostic section."""
    if any(grids_equal(a[k], truth) for a in pred_attempts for k in ("attempt_1", "attempt_2")):
        return "solved", 1.0
    shapes_pred = {grid_shape(a[k]) for a in pred_attempts for k in ("attempt_1", "attempt_2")}
    true_shape = grid_shape(truth)
    if true_shape not in shapes_pred:
        return "shape_mismatch", 0.0
    palette_true = set(flatten(truth))
    palettes_pred = [set(flatten(a[k])) for a in pred_attempts for k in ("attempt_1", "attempt_2")]
    best_px = 0.0
    for a in pred_attempts:
        for k in ("attempt_1", "attempt_2"):
            g = a[k]
            if grid_shape(g) == true_shape:
                total = true_shape[0] * true_shape[1]
                hits = sum(1 for x, y in zip(flatten(g), flatten(truth)) if x == y)
                best_px = max(best_px, hits / total if total else 0.0)
    if all(p != palette_true for p in palettes_pred):
        return "palette_mismatch", best_px
    if best_px >= 0.5:
        return "layout_mismatch_minor", best_px
    return "layout_mismatch_major", best_px


# --------------------------------------------------------------------------------------
# 5. Data discovery and loading
# --------------------------------------------------------------------------------------

def _iter_json_files(root: Path, max_depth: int = 3):
    root = Path(root)
    if not root.exists():
        return
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except Exception:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < max_depth and not e.name.startswith("."):
                        stack.append((Path(e.path), depth + 1))
                elif e.is_file(follow_symlinks=False) and e.name.endswith(".json"):
                    yield Path(e.path)
            except Exception:
                continue


def _looks_like_task_dir(d: Path, sample: int = 3) -> bool:
    """True if the directory holds one-task-per-file ARC JSONs (guards against name collisions)."""
    files = [f for f in sorted(Path(d).glob("*.json"))[:sample]]
    if not files:
        return False
    for f in files:
        try:
            body = read_json(f)
        except Exception:
            return False
        if not (isinstance(body, dict) and "train" in body and "test" in body):
            return False
    return True


def _find_task_dirs(roots, max_depth: int = 4) -> dict:
    """Locate evaluation/training/test task directories at any reasonable nesting depth.

    Handles the official repo layout after unzipping, e.g. ./ARC-AGI-2-main/data/evaluation/.
    """
    aliases = {
        "evaluation": ("evaluation", "eval", "public_eval"),
        "training": ("training", "train"),
        "test": ("test", "private", "semi_private"),
    }
    found = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            try:
                depth = len(Path(dirpath).relative_to(root).parts)
            except Exception:
                continue
            if depth > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if not filenames:
                continue
            name = Path(dirpath).name.lower()
            for key, names in aliases.items():
                if key in found or name not in names:
                    continue
                if _looks_like_task_dir(Path(dirpath)):
                    found[key] = Path(dirpath)
    return found


def discover_data(cfg) -> dict:
    """Locate competition files by name pattern, whatever the directory layout is."""
    found = {}
    for root in cfg["input_roots"]:
        root = Path(root)
        if not root.exists():
            continue
        for path in _iter_json_files(root):
            for key, names in DATA_PATTERNS.items():
                if path.name.lower() in names and key not in found:
                    found[key] = path
    return {"files": found, "task_dirs": _find_task_dirs(cfg["input_roots"])}


def load_task_file(path: Path) -> dict:
    data = read_json(path)
    tasks = {}
    for tid, body in data.items():
        if not isinstance(body, dict) or "train" not in body or "test" not in body:
            continue
        tasks[tid] = {"train": [{"input": p["input"], "output": p["output"]} for p in body["train"]],
                      "test": [{"input": p["input"]} for p in body["test"]]}
    return tasks


def load_task_dir(path: Path) -> dict:
    """Load a directory of one-task-per-file JSONs (the official ARC-AGI GitHub layout).

    Test outputs are kept when present: the public evaluation files ship their answers,
    which is what makes local self-scoring possible without a separate solutions file.
    """
    tasks = {}
    for f in sorted(Path(path).glob("*.json")):
        try:
            body = read_json(f)
        except Exception:
            continue
        if not (isinstance(body, dict) and "train" in body and "test" in body):
            continue
        test = []
        for p in body["test"]:
            entry = {"input": p["input"]}
            if "output" in p:
                entry["output"] = p["output"]
            test.append(entry)
        tasks[f.stem] = {
            "train": [{"input": p["input"], "output": p["output"]} for p in body["train"]],
            "test": test,
        }
    return tasks


def load_dataset(cfg, discovery, dataset: str):
    """Returns (tasks, solutions, source_description)."""
    files, task_dirs = discovery["files"], discovery["task_dirs"]
    challenges_key = f"{dataset}_challenges"
    solutions_key = f"{dataset}_solutions"
    tasks, source = {}, []

    if challenges_key in files:
        tasks = load_task_file(files[challenges_key])
        source.append(str(files[challenges_key]))
    elif dataset in task_dirs:
        tasks = load_task_dir(task_dirs[dataset])
        source.append(str(task_dirs[dataset]))

    solutions = {}
    if solutions_key in files and tasks:
        raw = read_json(files[solutions_key])
        for tid, outs in raw.items():
            if tid in tasks and isinstance(outs, list):
                solutions[tid] = outs
                for i, out in enumerate(outs):
                    if i < len(tasks[tid]["test"]):
                        tasks[tid]["test"][i]["output"] = out
        source.append(str(files[solutions_key]))
    return tasks, solutions, " + ".join(source) if source else "(none)"


def expected_structure(tasks: dict) -> list:
    return [(tid, len(tasks[tid]["test"])) for tid in sorted(tasks)]


# --------------------------------------------------------------------------------------
# 6. Submission IO and validation
# --------------------------------------------------------------------------------------

def write_submission(path: Path, results: dict, tasks: dict) -> None:
    """`results`: task_id -> list of {"attempt_1","attempt_2"}; always complete and legal."""
    payload = {}
    for tid in sorted(tasks):
        n = len(tasks[tid]["test"])
        attempts = results.get(tid) or []
        entries = []
        for i in range(n):
            src = attempts[i] if i < len(attempts) else {}
            entries.append({"attempt_1": sanitize_grid(src.get("attempt_1") or [[0]]),
                            "attempt_2": sanitize_grid(src.get("attempt_2") or [[0]])})
        payload[tid] = entries
    write_json_atomic(Path(path), payload)


def sample_expected_structure(sample) -> list:
    """Expected (task_id, n_test_inputs) taken from the official sample_submission.json."""
    out = []
    if isinstance(sample, dict):
        for tid, entries in sample.items():
            n = len(entries) if isinstance(entries, list) else 1
            out.append((tid, n))
    return out


def passthrough_from_sample(sample_path: Path, submission_path: Path, run_dir: Path | None = None):
    """
    Last-resort safety net: if challenge files cannot be located (renamed upstream, dataset
    not attached), mirror the official sample_submission structure so the notebook still
    produces a schema-valid `/kaggle/working/submission.json` instead of nothing.
    """
    sample = read_json(Path(sample_path))
    payload = {}
    for tid, entries in sample.items():
        if not isinstance(entries, list):
            entries = [entries]
        payload[tid] = [{"attempt_1": sanitize_grid(e.get("attempt_1") if isinstance(e, dict) else None),
                         "attempt_2": sanitize_grid(e.get("attempt_2") if isinstance(e, dict) else None)}
                        for e in entries]
    write_json_atomic(Path(submission_path), payload)
    if run_dir is not None:
        write_json_atomic(Path(run_dir) / "submission.json", payload)
    return payload


def validate_submission(path: Path, expected: list, sample_path: Path | None = None):
    """Returns (ok, problems list). `expected` = [(task_id, n_test_inputs), ...]."""
    problems = []
    try:
        payload = read_json(Path(path))
    except Exception as exc:
        return False, [f"cannot read submission: {exc!r}"]
    if not isinstance(payload, dict):
        return False, ["submission root is not a JSON object"]
    for tid, n in expected:
        if tid not in payload:
            problems.append(f"missing task {tid}")
            continue
        entries = payload[tid]
        if not isinstance(entries, list) or len(entries) != n:
            problems.append(f"task {tid}: expected {n} entries, got {len(entries) if isinstance(entries, list) else type(entries).__name__}")
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                problems.append(f"task {tid}[{i}]: keys must be exactly attempt_1/attempt_2")
                continue
            for k in ("attempt_1", "attempt_2"):
                g = entry[k]
                if not isinstance(g, list) or not g or not all(isinstance(r, list) and r for r in g):
                    problems.append(f"task {tid}[{i}].{k}: not a non-empty 2-D array")
                    continue
                w = len(g[0])
                if any(len(r) != w for r in g):
                    problems.append(f"task {tid}[{i}].{k}: ragged rows")
                if any((not isinstance(v, int)) or v < 0 or v > 9 for row in g for v in row):
                    problems.append(f"task {tid}[{i}].{k}: values must be ints 0..9")
    extra = set(payload) - {t for t, _ in expected}
    if extra:
        problems.append(f"{len(extra)} unexpected task ids (e.g. {sorted(extra)[:3]})")
    if sample_path and Path(sample_path).exists():
        try:
            sample = read_json(Path(sample_path))
            missing = set(sample) - set(payload)
            if missing:
                problems.append(f"sample_submission has {len(missing)} ids absent here")
        except Exception:
            pass
    return (not problems), problems


def score_submission(results: dict, tasks: dict, solutions: dict):
    """Exact-match scoring identical to the official rule (any attempt may match)."""
    rows = []
    n_inputs = 0
    n_exact = 0
    for tid in sorted(tasks):
        attempts = results.get(tid) or []
        for i, test_pair in enumerate(tasks[tid]["test"]):
            truth = test_pair.get("output")
            if truth is None and tid in solutions and i < len(solutions[tid]):
                truth = solutions[tid][i]
            if truth is None or i >= len(attempts):
                continue
            n_inputs += 1
            pred = attempts[i]
            label, px = classify_failure([pred], truth)
            if label == "solved":
                n_exact += 1
            rows.append({
                "task_id": tid,
                "test_index": i,
                "solved": int(label == "solved"),
                "top1_solved": int(grids_equal(pred["attempt_1"], truth)),
                "failure_class": label,
                "pixel_acc": round(px, 4),
                "pred_shape": f"{grid_shape(pred['attempt_1'])[0]}x{grid_shape(pred['attempt_1'])[1]}",
                "true_shape": f"{grid_shape(truth)[0]}x{grid_shape(truth)[1]}",
            })
    summary = {
        "n_scored_inputs": n_inputs,
        "n_exact": n_exact,
        "accuracy": round(n_exact / n_inputs, 4) if n_inputs else 0.0,
        "n_tasks_scored": len({r["task_id"] for r in rows}),
        "n_tasks_solved": len({r["task_id"] for r in rows if r["solved"]}),
    }
    return summary, rows


# --------------------------------------------------------------------------------------
# 7. Run management: manifest, checkpoint, resume, backup, state
# --------------------------------------------------------------------------------------

class Run:
    def __init__(self, cfg: dict, logger: Logger | None = None, create: bool = True):
        """`create=False` opens the project read-only (no directories are created), which is
        what inspection/backup helpers want — constructing a Run must never litter the disk."""
        self.cfg = cfg
        self._create = create
        self.project_dir = Path(cfg["project_dir"])
        self.runs_dir = self.project_dir / "runs"
        self.backups_dir = self.project_dir / "backups"
        self.state_dir = self.project_dir / "state"
        if create:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self.backups_dir.mkdir(parents=True, exist_ok=True)
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.confhash = config_hash(cfg)[:8]
        self._logger = logger
        self.bind_run_id(cfg["run_id"] or self._pick_run_id(), create=create)

    def bind_run_id(self, run_id: str, create: bool = True) -> None:
        """Point the run object at `run_id`, re-deriving every path (used by rebuild/restore)."""
        self.run_id = run_id
        self.dir = self.runs_dir / run_id
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self._logger or Logger(self.dir / "log.txt")
        self.log.path = self.dir / "log.txt"
        self.manifest_path = self.dir / "manifest.json"
        self.checkpoint_path = self.dir / "checkpoint.jsonl"
        self.submission_run_path = self.dir / "submission.json"
        self.diagnostics_path = self.dir / "diagnostics.json"
        self.diagnostics_csv = self.dir / "diagnostics.csv"
        self.manifest = {}

    # --- run id -------------------------------------------------------------------
    def _pick_run_id(self) -> str:
        if self.cfg["resume"] and not self.cfg["force"]:
            latest = self.latest_run_id()
            if latest:
                man = self.runs_dir / latest / "manifest.json"
                if man.exists():
                    try:
                        if read_json(man).get("config_hash") == config_hash(self.cfg):
                            return latest
                    except Exception:
                        pass
        # a fresh run id must never collide with an existing directory (same-second re-runs)
        base = f"{local_stamp()}-{self.confhash}"
        candidate, n = base, 2
        while (self.runs_dir / candidate).exists():
            candidate = f"{base}-{n}"
            n += 1
        return candidate

    def latest_run_id(self):
        latest_file = self.state_dir / "LATEST"
        if latest_file.exists():
            return latest_file.read_text(encoding="utf-8").strip() or None
        runs = sorted([p.name for p in self.runs_dir.glob("*") if p.is_dir()])
        return runs[-1] if runs else None

    # --- manifest -----------------------------------------------------------------
    def start_manifest(self, discovery, dataset, source, tasks, expected, engine_path: Path | None):
        self.manifest = {
            "engine_version": ENGINE_VERSION,
            "engine_sha256": file_fingerprint(engine_path or Path(__file__))["head_sha256"]
            if (engine_path or Path(__file__)).exists() else None,
            "config": {k: v for k, v in self.cfg.items()},
            "config_hash": config_hash(self.cfg),
            "run_id": self.run_id,
            "started_utc": utc_stamp(),
            "dataset": dataset,
            "dataset_source": source,
            "n_tasks": len(tasks),
            "n_test_inputs": sum(n for _, n in expected),
            "data_fingerprints": {k: file_fingerprint(v) for k, v in discovery["files"].items()},
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "kaggle_rerun": bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN")),
                "kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
                "cwd": os.getcwd(),
            },
            "status": "running",
        }
        write_json_atomic(self.manifest_path, self.manifest, indent=2)
        (self.state_dir / "LATEST").write_text(self.run_id, encoding="utf-8")
        self.log(f"run_id={self.run_id} config_hash={self.manifest['config_hash'][:12]} dataset={dataset}")

    def finish_manifest(self, status: str, extra: dict | None = None):
        self.manifest["status"] = status
        self.manifest["finished_utc"] = utc_stamp()
        if extra:
            self.manifest.update(extra)
        write_json_atomic(self.manifest_path, self.manifest, indent=2)

    # --- checkpoint ---------------------------------------------------------------
    def load_done(self) -> dict:
        """Read the append-only checkpoint; later records win (idempotent resume)."""
        done = {}
        if not self.checkpoint_path.exists():
            return done
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("status") == "ok" and rec.get("attempts"):
                    done[rec["task_id"]] = rec
        return done

    def append_checkpoint(self, record: dict):
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # --- backup -------------------------------------------------------------------
    def snapshot(self, label: str = "snapshot", keep: int | None = None) -> Path:
        keep = int(keep if keep is not None else self.cfg["backup_keep"])
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        target = self.backups_dir / f"{self.run_id}-{label}-{stamp}.zip"
        n = 2
        while target.exists():
            target = self.backups_dir / f"{self.run_id}-{label}-{stamp}-{n}.zip"
            n += 1
        members = ["manifest.json", "checkpoint.jsonl", "submission.json",
                   "diagnostics.json", "diagnostics.csv", "state.md", "log.txt"]
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in members:
                p = self.dir / name
                if p.exists():
                    zf.write(p, arcname=f"{self.run_id}/{name}")
        snapshots = sorted(self.backups_dir.glob(f"*-{label}-*.zip"),
                           key=lambda p: p.stat().st_mtime)
        for old in snapshots[:-keep] if keep > 0 else snapshots:
            try:
                old.unlink()
            except Exception:
                pass
        self.log(f"backup -> {target.name} ({target.stat().st_size} bytes)")
        return target

    @staticmethod
    def restore_backup(zip_path: Path, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        return dest_dir

    # --- housekeeping -------------------------------------------------------------
    def prune_runs(self, keep: int | None = None):
        keep = int(keep if keep is not None else self.cfg["runs_keep"])
        runs = sorted([p for p in self.runs_dir.glob("*") if p.is_dir()],
                      key=lambda p: p.stat().st_mtime)
        for old in runs[:-keep] if keep > 0 else runs:
            if old.name == self.run_id:
                continue
            try:
                shutil.rmtree(old)
                self.log(f"pruned old run {old.name}")
            except Exception:
                pass

    def write_state_file(self, lines: list):
        text = "\n".join(lines) + "\n"
        (self.dir / "state.md").write_text(text, encoding="utf-8")
        (self.state_dir / "PROJECT_STATE.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------------------
# 8. Main entry point
# --------------------------------------------------------------------------------------

def resolve_submission_path(cfg: dict, mode: str, run_dir: Path) -> Path:
    """Decide where this run writes its submission file.

    ONLY a submit-mode run may write the competition file. A dev/self-scoring run writes
    to its own path.

    Why this exists: on 2026-09-13 the notebook ran submit mode (240 tasks) and then dev
    mode (120 eval tasks); dev mode wrote to the same ``/kaggle/working/submission.json``
    and silently overwrote the real submission, so Kaggle scored a 120-task file and
    rejected it with "incorrect format: wrong number of rows". The bug is fixed here and
    guarded by a regression test in tests/run_notebook_dryrun.py.
    """
    if cfg["submission_path"]:
        return Path(cfg["submission_path"])
    if mode == "submit":
        return (Path("/kaggle/working/submission.json")
                if Path("/kaggle/working").is_dir() else run_dir / "submission.json")
    return Path(cfg["project_dir"]) / "dev_submission.json"


def decide_mode(cfg, discovery) -> str:
    if cfg["mode"] != "auto":
        return cfg["mode"]
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        return "submit"
    if discovery["files"].get("test_challenges"):
        return "submit"
    return "dev"


def decide_dataset(cfg, mode, discovery) -> str:
    if cfg["dataset"]:
        return cfg["dataset"]
    if mode == "submit":
        return "test"
    if discovery["files"].get("evaluation_challenges") or "evaluation" in discovery["task_dirs"]:
        return "evaluation"
    if discovery["files"].get("test_challenges"):
        return "test"
    return "training"


def apply_overrides_to_module_globals(cfg: dict):
    """Notebook convenience: `arc_prize_v1.CFG.update({...})` then main()."""
    return cfg


CFG = dict(DEFAULTS)


def main(**overrides) -> dict:
    """Run the pipeline. Returns a summary dict; safe to re-run (resumes by default)."""
    cfg = make_config({**CFG, **overrides}) if overrides else make_config(CFG)
    t_all = time.monotonic()
    discovery = discover_data(cfg)
    mode = decide_mode(cfg, discovery)
    dataset = decide_dataset(cfg, mode, discovery)
    tasks, solutions, source = load_dataset(cfg, discovery, dataset)

    run = Run(cfg)
    log = run.log
    submission_path = resolve_submission_path(cfg, mode, run.dir)
    log(f"engine v{ENGINE_VERSION} | mode={mode} dataset={dataset} | {len(tasks)} tasks")
    log(f"data source: {source or '(none)'}")

    if not tasks:
        sample_path = discovery["files"].get("sample_submission")
        if sample_path:
            log("WARNING: no challenge tasks found — emitting a sample-shaped passthrough submission")
            passthrough_from_sample(sample_path, submission_path, run.dir)
            sample = read_json(Path(sample_path))
            expected = sample_expected_structure(sample)
            ok, problems = validate_submission(submission_path, expected)
            run.finish_manifest("sample_passthrough", {"error": "no tasks", "submission_valid": ok})
            run.log(f"passthrough submission valid={ok}")
            return {"ok": ok, "mode": mode, "dataset": dataset, "run_id": run.run_id,
                    "n_tasks": 0, "sample_passthrough": True, "submission_path": str(submission_path),
                    "problems": problems,
                    "discovery": {k: str(v) for k, v in discovery["files"].items()}}
        log("FATAL: no tasks found and no sample_submission.json to fall back on.")
        run.finish_manifest("failed", {"error": "no tasks"})
        return {"ok": False, "error": "no tasks found",
                "discovery": {k: str(v) for k, v in discovery["files"].items()}}

    expected = expected_structure(tasks)
    run.start_manifest(discovery, dataset, source, tasks, expected, Path(__file__))

    done = run.load_done() if (cfg["resume"] and not cfg["force"]) else {}
    if done:
        log(f"resume: {len(done)}/{len(tasks)} tasks already checkpointed")
    if cfg["force"]:
        log("force=True: ignoring existing checkpoint")

    results = {}
    for tid, rec in done.items():
        if tid in tasks:
            results[tid] = rec["attempts"]

    deadline = t_all + float(cfg["global_budget_seconds"]) - float(cfg["reserve_seconds"])
    pending = [tid for tid, _ in expected if tid not in results]
    log(f"pending: {len(pending)} tasks | global deadline in {deadline - time.monotonic():.0f}s")

    # an always-complete (all-zero) submission is written up front, so a crash never
    # leaves the notebook without a schema-valid artifact
    write_submission(submission_path, results, tasks)

    diagnostic_rows = []
    solved = prior_only = timed_out = 0
    for idx, tid in enumerate(pending, start=1):
        if time.monotonic() > deadline:
            log(f"global deadline reached; filling remaining {len(pending) - idx + 1} tasks with priors")
            for rest in pending[idx - 1:]:
                try:
                    pred = solve_task(tasks[rest], {**cfg, "enable_depth2": False, "per_task_seconds": 1.0}, None)
                except Exception:
                    pred = {"attempts": [{"attempt_1": [[0]], "attempt_2": [[0]]}
                                         for _ in tasks[rest]["test"]], "diagnostics": {"source": "deadline"}}
                results[rest] = pred["attempts"]
                timed_out += 1
                run.append_checkpoint({"task_id": rest, "status": "ok", "attempts": pred["attempts"],
                                       "diagnostics": pred["diagnostics"], "partial": True,
                                       "ts_utc": utc_stamp()})
            break

        task_deadline = min(time.monotonic() + float(cfg["per_task_seconds"]), deadline)
        try:
            pred = solve_task(tasks[tid], cfg, task_deadline)
        except Exception:
            log(f"task {tid} crashed: {traceback.format_exc(limit=2)}")
            pred = {"attempts": [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[tid]["test"]],
                    "diagnostics": {"source": "crash"}}
        results[tid] = pred["attempts"]
        run.append_checkpoint({"task_id": tid, "status": "ok", "attempts": pred["attempts"],
                               "diagnostics": pred["diagnostics"], "ts_utc": utc_stamp()})

        if pred["diagnostics"].get("source") == "search":
            solved += 1
        else:
            prior_only += 1
        if pred["diagnostics"].get("timed_out"):
            timed_out += 1

        if idx % int(cfg["log_every"]) == 0 or idx == len(pending):
            elapsed = time.monotonic() - t_all
            rate = idx / elapsed if elapsed else 0
            eta = (len(pending) - idx) / rate if rate else 0
            log(f"[{idx}/{len(pending)}] search={solved} prior={prior_only} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s last={tid}:{pred['diagnostics'].get('top_program')}")

        if idx % int(cfg["submission_flush_every"]) == 0:
            write_submission(submission_path, results, tasks)

    # ---- finalise ----------------------------------------------------------------
    write_submission(submission_path, results, tasks)
    write_submission(run.submission_run_path, results, tasks)
    ok, problems = validate_submission(submission_path, expected,
                                       discovery["files"].get("sample_submission"))
    log(f"submission schema valid={ok}" + ("" if ok else f" problems={problems[:5]}"))

    summary = {"ok": ok, "mode": mode, "dataset": dataset, "run_id": run.run_id,
               "n_tasks": len(tasks), "n_test_inputs": sum(n for _, n in expected),
               "solved_by_search": solved, "prior_only": prior_only, "timed_out": timed_out,
               "submission_path": str(submission_path), "problems": problems}

    # ground truth may come from a solutions file OR be embedded in the task files themselves
    # (the official repo's public evaluation tasks ship their answers), so check both.
    has_truth = bool(solutions) or any("output" in p for t in tasks.values() for p in t["test"])
    if has_truth:
        score, rows = score_submission(results, tasks, solutions)
        summary["dev_score"] = score
        diagnostic_rows = rows
        with open(run.diagnostics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        failures = Counter(r["failure_class"] for r in rows)
        summary["failure_classes"] = dict(failures)
        log(f"dev score: {score['accuracy']*100:.2f}% ({score['n_exact']}/{score['n_scored_inputs']}) "
            f"| classes={dict(failures)}")

    # per-task diagnostics come from the checkpoint, so they survive crashes and resumes
    diag = {"summary": summary,
            "task_diagnostics": {rec["task_id"]: rec.get("diagnostics")
                                 for rec in run.load_done().values()},
            "diagnostic_rows": diagnostic_rows,
            "generated_utc": utc_stamp()}
    write_json_atomic(run.diagnostics_path, diag, indent=2)

    if cfg["snapshot_on_finish"]:
        run.snapshot("finish")
    run.prune_runs()

    run.write_state_file([
        "# ARC Prize 2026 — engine state",
        "",
        f"- engine: v{ENGINE_VERSION}",
        f"- run id: `{run.run_id}`",
        f"- mode/dataset: {mode} / {dataset}",
        f"- tasks: {len(tasks)} ({summary['n_test_inputs']} test inputs)",
        f"- solved by validated program search: {solved}",
        f"- prior-only (no validated program): {prior_only}",
        f"- timed out tasks: {timed_out}",
        f"- submission: `{submission_path}` (schema valid={ok})",
        f"- dev score: {summary.get('dev_score', 'n/a (no solutions attached)')}",
        f"- failure classes: {summary.get('failure_classes', {})}",
        f"- finished: {utc_stamp()}",
        f"- next action: inspect `diagnostics.csv`, then extend the DSL with the top failure class",
    ])

    run.finish_manifest("ok" if ok else "invalid_submission", {"summary": summary})
    log(f"done in {time.monotonic() - t_all:.1f}s | run dir {run.dir}")
    return summary


def latest_run(**overrides) -> "Run":
    """
    Open the most recent run *without creating anything* — use this for backups, inspection
    and rebuilds. `Run(cfg)` on its own would allocate a brand-new (empty) run directory.
    """
    cfg = make_config({**CFG, **overrides})
    run = Run(cfg, create=False)
    latest = run.latest_run_id()
    if latest:
        run.bind_run_id(latest, create=False)
    return run


def rebuild_submission(run_id: str | None = None, **overrides) -> dict:
    """Update path #1: re-create submission.json from an existing checkpoint (no re-solving).

    Use after changing only the submission writer / after a notebook crash.
    """
    cfg = make_config({**CFG, **overrides})
    discovery = discover_data(cfg)
    run = Run(cfg, create=False)
    target_run = run_id or run.latest_run_id()
    if not target_run or not (run.runs_dir / target_run).exists():
        run.log("rebuild: no run found to rebuild from")
        return {"ok": False, "error": "no run found"}
    run.bind_run_id(target_run, create=False)
    done = run.load_done()
    dataset = decide_dataset(cfg, decide_mode(cfg, discovery), discovery)
    tasks, solutions, _ = load_dataset(cfg, discovery, dataset)
    results = {tid: rec["attempts"] for tid, rec in done.items() if tid in tasks}
    submission_path = resolve_submission_path(cfg, decide_mode(cfg, discovery), run.dir)
    write_submission(submission_path, results, tasks)
    ok, problems = validate_submission(submission_path, expected_structure(tasks),
                                       discovery["files"].get("sample_submission"))
    run.log(f"rebuild from checkpoint: {len(results)}/{len(tasks)} tasks | valid={ok}")
    return {"ok": ok, "tasks_from_checkpoint": len(results), "n_tasks": len(tasks),
            "submission_path": str(submission_path), "problems": problems}


def list_runs(**overrides) -> list:
    cfg = make_config({**CFG, **overrides})
    runs_dir = Path(cfg["project_dir"]) / "runs"
    out = []
    for d in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        man = d / "manifest.json"
        info = {"run_id": d.name, "manifest": None, "has_checkpoint": (d / "checkpoint.jsonl").exists()}
        if man.exists():
            try:
                info["manifest"] = {k: read_json(man).get(k)
                                    for k in ("status", "dataset", "n_tasks", "started_utc", "finished_utc")}
            except Exception:
                pass
        out.append(info)
    return out


if __name__ == "__main__":
    started = time.monotonic()
    try:
        result = main()
    except Exception:
        traceback.print_exc()
        result = {"ok": False, "error": traceback.format_exc(limit=5)}
    print(json.dumps({k: v for k, v in result.items() if k != "failure_classes"},
                     ensure_ascii=False, indent=2, default=str)[:4000])
    print(f"wall clock: {time.monotonic() - started:.1f}s")
