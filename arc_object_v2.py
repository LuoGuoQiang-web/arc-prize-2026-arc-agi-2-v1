"""v2 对象级候选家族（针对 ARC-AGI-2 评测集主导家族）。

设计动机（W2 诊断硬证据，见 work/arc_w1/diagnosis/W2_DIAGNOSIS_1.md）：
  - 评测集 120 题中，v1 原语覆盖的"便宜家族"全部为 0：纯换色 0%、平铺 0%、裁剪 0%、计数 0%
  - 评测集主导家族是「形状不变 + 对象级条件变换」50.8%（训练集仅 28.9%）
  - 而 v1 的 29 个原语全是全局几何/尺寸变换，无法表达"按对象属性选择/改色"
→ 因此本模块新增**对象级算子**：连通块 + 属性谓词 + 参数由示范拟合。

范式与 v1 一致：每个候选先由示范数据**拟合参数**，再**逐对精确验证**，
只有能完全复现全部示范的候选才被接受（prevalidated）。这样候选是"被证据支持的假设"，
不是凭空猜测——这也是论文中"验证驱动"的核心主张。

用法：
    from arc_object_v2 import object_candidates
    cands = object_candidates(train_pairs, bg)   # -> [(name, fn), ...] 已验证
"""
from __future__ import annotations

from collections import Counter, deque

# ------------------------------------------------------------------------------
# 基础：连通块与对象属性
# ------------------------------------------------------------------------------


def grid_shape(g):
    return (len(g), len(g[0]))


def components(grid, bg=0):
    """4 连通、非背景的连通块；返回像素坐标列表的列表。"""
    h, w = grid_shape(grid)
    seen = [[False] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or grid[i][j] == bg:
                continue
            comp, q = [], deque([(i, j)])
            seen[i][j] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] != bg:
                        seen[ny][nx] = True
                        q.append((ny, nx))
            comps.append(comp)
    return comps


def holes_of(grid, comp, bg=0):
    """对象内部的洞：被该对象包围、且不连到网格边界的背景连通块数量。

    用"从边界灌水"确定外部背景，剩下的背景即为洞。
    """
    h, w = grid_shape(grid)
    inside = set(comp)
    r0 = min(r for r, _ in comp)
    r1 = max(r for r, _ in comp)
    c0 = min(c for _, c in comp)
    c1 = max(c for _, c in comp)
    seen = set()
    q = deque()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if grid[r][c] == bg and (r, c) not in inside:
                q.append((r, c))
                seen.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if grid[r][c] == bg and (r, c) not in inside:
                q.append((r, c))
                seen.add((r, c))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if r0 <= ny <= r1 and c0 <= nx <= c1 and grid[ny][nx] == bg and (ny, nx) not in inside and (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append((ny, nx))
    n_holes = 0
    visited = set()
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
                        if r0 <= ny <= r1 and c0 <= nx <= c1 and grid[ny][nx] == bg and (ny, nx) not in inside and (ny, nx) not in visited:
                            visited.add((ny, nx))
                            qq.append((ny, nx))
    return n_holes


def objects_of(grid, bg=0):
    """返回对象列表，每个对象含 pixels / 颜色 / 包围盒 / 属性。"""
    h, w = grid_shape(grid)
    objs = []
    for comp in components(grid, bg):
        pix = set(comp)
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        colors_ = Counter(grid[r][c] for r, c in comp)
        objs.append({
            "pixels": pix,
            "size": len(comp),
            "color": colors_.most_common(1)[0][0],
            "n_colors": len(colors_),
            "bbox": (r0, c0, r1, c1),
            "h": r1 - r0 + 1,
            "w": c1 - c0 + 1,
            "touches_border": r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1,
            "square": (r1 - r0) == (c1 - c0),
            "holes": holes_of(grid, comp, bg),
        })
    return objs


# ------------------------------------------------------------------------------
# 候选 A：按谓词删除 / 保留对象
# ------------------------------------------------------------------------------

PREDICATES = {
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


def _erase(grid, pixels, bg):
    for r, c in pixels:
        grid[r][c] = bg


def make_remove_candidate(pred_name, keep_matching: bool, bg: int):
    """keep_matching=True → 保留满足谓词的对象；False → 删除满足谓词的对象。"""

    def fn(grid):
        objs = objects_of(grid, bg)
        if not objs:
            return None
        out = [row[:] for row in grid]
        pred = PREDICATES[pred_name]
        for o in objs:
            match = pred(objs, o)
            drop = match if keep_matching else not match
            if drop:
                _erase(out, o["pixels"], bg)
        return out

    return fn


# ------------------------------------------------------------------------------
# 候选 B：按对象属性改色（调色板由示范拟合）
# ------------------------------------------------------------------------------

PROPERTIES = {
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


def make_recolor_by_property(prop_name: str, bg: int):
    """返回 (fn, fit_ok)。fn 用拟合好的映射给对象整体改色；映射在 fit 阶段确定。"""

    def collect(pairs):
        """从示范中收集 属性值 -> 输出颜色 的一致映射。"""
        mapping = {}
        for pair in pairs:
            grid, target = pair["input"], pair["output"]
            if grid_shape(grid) != grid_shape(target):
                return None
            objs = objects_of(grid, bg)
            if not objs:
                return None
            prop = PROPERTIES[prop_name]
            for o in objs:
                outs = {target[r][c] for r, c in o["pixels"]}
                if len(outs) != 1:
                    return None
                val, col = prop(objs, o), outs.pop()
                if mapping.get(val, col) != col:
                    return None
                mapping[val] = col
        return mapping or None

    return collect


# ------------------------------------------------------------------------------
# 候选 C：填洞（颜色由示范拟合）
# ------------------------------------------------------------------------------


def fill_holes_candidate(bg: int):
    """把对象内部的洞填成某色；颜色从示范中拟合，要求所有洞在示范里同色。"""

    def fn_factory(color):
        def fn(grid):
            out = [row[:] for row in grid]
            for o in objects_of(grid, bg):
                if o["holes"] == 0:
                    continue
                r0, c0, r1, c1 = o["bbox"]
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if out[r][c] == bg and (r, c) not in o["pixels"]:
                            out[r][c] = color
            return out

        return fn

    return fn_factory


def fit_fill_holes(pairs, bg):
    """从示范拟合填洞颜色：输入的洞位置在输出中必须同色且与输入该处不同。"""
    colors = set()
    for pair in pairs:
        grid, target = pair["input"], pair["output"]
        if grid_shape(grid) != grid_shape(target):
            return None
        for o in objects_of(grid, bg):
            r0, c0, r1, c1 = o["bbox"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if grid[r][c] == bg and (r, c) not in o["pixels"] and target[r][c] != bg:
                        colors.add(target[r][c])
    if len(colors) == 1:
        return colors.pop()
    return None


# ------------------------------------------------------------------------------
# 候选 D：删除某颜色的全部对象（颜色由示范拟合）
# ------------------------------------------------------------------------------


def fit_vanishing_colors(pairs, bg):
    """找出在所有示范中都"整色消失"的颜色集合。"""
    vanish = None
    for pair in pairs:
        grid, target = pair["input"], pair["output"]
        if grid_shape(grid) != grid_shape(target):
            return None
        in_colors = {c for row in grid for c in row if c != bg}
        out_colors = {c for row in target for c in row if c != bg}
        gone = in_colors - out_colors
        vanish = gone if vanish is None else (vanish & gone)
    return vanish or None


def make_drop_color(color, bg):
    def fn(grid):
        return [[bg if v == color else v for v in row] for row in grid]

    return fn


# ------------------------------------------------------------------------------
# 主入口：生成并验证
# ------------------------------------------------------------------------------


def _reproduces(fn, pairs):
    for pair in pairs:
        try:
            out = fn(pair["input"])
        except Exception:
            return False
        if out is None or out != pair["output"]:
            return False
    return True


def object_candidates(pairs, bg=0, max_candidates: int = 400):
    """返回验证过的对象级候选：[(name, fn), ...]（fn: grid -> grid）。

    每个候选都必须**逐对精确复现**全部示范，否则丢弃。
    """
    verified = []

    def consider(name, fn, builder=None):
        if len(verified) >= max_candidates:
            return
        if _reproduces(fn, pairs):
            verified.append((name, fn))

    # A. 删除/保留对象
    for pred_name in PREDICATES:
        for keep_matching in (True, False):
            fn = make_remove_candidate(pred_name, keep_matching, bg)
            tag = "keep" if keep_matching else "drop"
            consider(f"obj_{tag}_{pred_name}", fn)

    # B. 按属性改色（映射由示范拟合）
    for prop_name in PROPERTIES:
        collect = make_recolor_by_property(prop_name, bg)
        mapping = collect(pairs)
        if not mapping:
            continue

        def make_fn(prop_name=prop_name, mapping=mapping):
            def fn(grid):
                objs = objects_of(grid, bg)
                if not objs:
                    return None
                out = [row[:] for row in grid]
                prop = PROPERTIES[prop_name]
                for o in objs:
                    col = mapping.get(prop(objs, o))
                    if col is None:
                        return None
                    for r, c in o["pixels"]:
                        out[r][c] = col
                return out

            return fn

        consider(f"obj_recolor_by_{prop_name}", make_fn())

    # C. 填洞
    hole_color = fit_fill_holes(pairs, bg)
    if hole_color is not None:
        consider(f"obj_fill_holes_{hole_color}", fill_holes_candidate(bg)(hole_color))

    # D. 删除整色对象
    for color in (fit_vanishing_colors(pairs, bg) or set()):
        consider(f"obj_drop_color_{color}", make_drop_color(color, bg))

    return verified
