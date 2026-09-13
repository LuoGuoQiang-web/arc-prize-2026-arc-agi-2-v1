"""对象算子**组合**搜索（v2c）：单步算子之外的最后一张牌。

动机：单步对象算子/上下文改色在评测集的覆盖率都是 0，而诊断显示主导家族是
「对象级**条件**变换」——天然的形态是"先按属性筛选/清理对象，再对结果改色/变换"。
因此搜索 depth-2 组合：
    第 1 步：参数无关、保形的对象算子（筛选/清理/填洞…），作用于示范输入
    第 2 步：在第 1 步的输出上**拟合**改色族（逐格颜色映射 或 上下文条件改色）
两步都必须由示范拟合，且整条链必须**逐对精确复现全部示范**才被接受。

恒等第 1 步（identity）作为内控：identity ∘ 上下文族必须复现 §7 记录的覆盖率
（训练集 38/1000、评测集 0/120），否则说明本脚本的复用有误。

用法：
    python compose_ops.py <data_dir> <split> [--limit N] [--verbose]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ENGINE = Path(r"C:\Users\Administrator\Desktop\Kaggle\ARC\Kaggle\arc_prize_v1")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENGINE))

from arc_object_v2 import objects_of, _erase  # noqa: E402
from context_recolor import FEATURE_SETS, fit_feature_map, make_fn  # noqa: E402

# 注意：本模块可被其它脚本导入，因此不在顶层读取 sys.argv
PRED_NAMES = [
    "is_largest", "is_smallest", "is_singleton", "touches_border", "not_touches_border",
    "is_square", "has_hole", "no_hole", "most_common_color", "rare_color",
]


def _pred(objs, o, name):
    if name == "is_largest":
        return o["size"] == max(x["size"] for x in objs)
    if name == "is_smallest":
        return o["size"] == min(x["size"] for x in objs)
    if name == "is_singleton":
        return o["size"] == 1
    if name == "touches_border":
        return o["touches_border"]
    if name == "not_touches_border":
        return not o["touches_border"]
    if name == "is_square":
        return o["square"]
    if name == "has_hole":
        return o["holes"] > 0
    if name == "no_hole":
        return o["holes"] == 0
    if name == "most_common_color":
        return o["color"] == Counter(x["color"] for x in objs).most_common(1)[0][0]
    if name == "rare_color":
        return Counter(x["color"] for x in objs)[o["color"]] == 1
    raise KeyError(name)


def stage1_ops():
    """参数无关的保形对象算子：{name: fn(grid)->grid}。"""
    ops = {"identity": lambda g, bg: [row[:] for row in g]}

    def make_filter(pred_name, keep_matching):
        def fn(grid, bg):
            objs = objects_of(grid, bg)
            if not objs:
                return None
            out = [row[:] for row in grid]
            for o in objs:
                match = _pred(objs, o, pred_name)
                if match if keep_matching else not match:
                    _erase(out, o["pixels"], bg)
            return out

        return fn

    for p in PRED_NAMES:
        ops[f"keep_{p}"] = make_filter(p, True)
        ops[f"drop_{p}"] = make_filter(p, False)

    def fill_holes(grid, bg):
        objs = objects_of(grid, bg)
        if not objs:
            return None
        out = [row[:] for row in grid]
        any_hole = False
        for o in objs:
            if o["holes"] == 0:
                continue
            any_hole = True
            r0, c0, r1, c1 = o["bbox"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if out[r][c] == bg and (r, c) not in o["pixels"]:
                        out[r][c] = o["color"]
        return out if any_hole else None

    ops["fill_holes_with_own_color"] = fill_holes

    def outline(grid, bg):
        objs = objects_of(grid, bg)
        if not objs:
            return None
        out = [row[:] for row in grid]
        changed = False
        h, w = len(grid), len(grid[0])
        for o in objs:
            for (r, c) in o["pixels"]:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in o["pixels"] and out[nr][nc] == bg:
                        out[nr][nc] = o["color"]
                        changed = True
        return out if changed else None

    ops["outline_objects"] = outline
    return ops


def fit_stage2(pairs2, bg):
    """在第 1 步输出上拟合第 2 步改色族；返回 [(name, fn)]（已验证复现全部示范）。"""
    out = []
    tout = [p["output"] for p in pairs2]
    tin = [p["input"] for p in pairs2]

    # a) 逐格颜色映射
    cmap = {}
    ok = True
    for g, t in zip(tin, tout):
        if len(g) != len(t) or len(g[0]) != len(t[0]):
            ok = False
            break
        for rg, rt in zip(g, t):
            for a, b in zip(rg, rt):
                if cmap.get(a, b) != b:
                    ok = False
                    break
                cmap[a] = b
            if not ok:
                break
        if not ok:
            break
    if ok and cmap:
        fn = lambda grid: [[cmap.get(v, v) for v in row] for row in grid]  # noqa: E731
        out.append(("cellmap", fn))

    # b) 上下文条件改色（20 个特征组合）
    for names in FEATURE_SETS:
        mapping = fit_feature_map(names, pairs2, bg)
        if not mapping:
            continue
        out.append(("ctx_" + "+".join(names), make_fn(names, mapping, bg)))

    # 只保留真的能复现全部示范的
    verified = []
    for name, fn in out:
        good = True
        for g, t in zip(tin, tout):
            try:
                if fn(g) != t:
                    good = False
                    break
            except Exception:
                good = False
                break
        if good:
            verified.append((name, fn))
    return verified


def composed_candidates(pairs, bg):
    """返回已验证的 depth-2 组合候选：[(name, fn)]。"""
    good = []
    for s1_name, s1 in stage1_ops().items():
        mapped = []
        ok = True
        for p in pairs:
            try:
                m = s1(p["input"], bg)
            except Exception:
                m = None
            if m is None:
                ok = False
                break
            mapped.append({"input": m, "output": p["output"]})
        if not ok:
            continue
        for s2_name, s2 in fit_stage2(mapped, bg):
            fn = (lambda a, b: (lambda g: (lambda m: None if m is None else b(m))(a(g, bg))))(s1, s2)

            def check(fn=fn, pairs=pairs, bg=bg):
                for p in pairs:
                    try:
                        if fn(p["input"]) != p["output"]:
                            return False
                    except Exception:
                        return False
                return True

            if check():
                good.append((f"{s1_name}>>{s2_name}", fn))
    return good


# ------------------------------------------------------------------ 主流程


def coverage(ddir, sp, lim=None, verbose_flag=False):
    """返回 (covered, total, identity_covered, elapsed_s, names_hit, depth_hist)。"""
    tdir = Path(ddir) / sp
    files = sorted(tdir.glob("*.json"))
    if lim:
        files = files[:lim]

    t0 = time.perf_counter()
    cov = ident = 0
    names_hit = Counter()
    depth_hist = Counter()
    for f in files:
        task = json.loads(f.read_text(encoding="utf-8"))
        bg = Counter(c for p in task["train"] for row in p["input"] for c in row).most_common(1)[0][0]
        cands = composed_candidates(task["train"], bg)
        if cands:
            cov += 1
            if any(n.startswith("identity>>") for n, _ in cands):
                ident += 1
            for n, _ in cands:
                names_hit[n.split(">>")[0]] += 1
            depth_hist[len(cands)] += 1
        if verbose_flag and cands:
            print(f"  {f.stem}: {len(cands)} 个组合候选，例 {cands[0][0]}")
    return cov, len(files), ident, time.perf_counter() - t0, names_hit, depth_hist


if __name__ == "__main__":
    data_dir = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "evaluation"
    limit = None
    verbose = "--verbose" in sys.argv
    for a in sys.argv[3:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])
    cov, total, ident, elapsed, names_hit, depth_hist = coverage(data_dir, split, limit, verbose)
    print(f"=== 组合搜索 {split}（{total} 题，{elapsed:.1f}s，{elapsed/max(total,1):.2f}s/题）===")
    print(f"  覆盖率（所有第 1 步算子）: {cov}/{total} = {100*cov/max(total,1):.1f}%")
    print(f"  内控：identity>>第2步 覆盖率: {ident}/{total} "
          f"= {100*ident/max(total,1):.1f}%（应等于独立测得的上下文族覆盖率）")
    if names_hit:
        print("  第 1 步算子命中分布（top 10）:", dict(names_hit.most_common(10)))
        print("  每题候选数分布:", dict(sorted(depth_hist.items())[:8]))
