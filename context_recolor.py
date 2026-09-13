"""上下文条件改色族（context-conditioned recolouring）：瞄准评测集主导家族。

动机：诊断显示评测集 50.8% 是「形状不变 + 对象级条件变换」，而既有两类求解器都覆盖不到：
  - 逐格颜色映射（fit_color_map）：只看自身颜色 → 评测集该家族占 0%
  - 对象整体改色：要求"一个对象在输出中单色" → 训练集只验出 1.9%

本模块补上中间形态：**逐格按结构上下文决定颜色**。每个格子的特征由其所属对象的属性构成，
然后从示范中拟合「特征元组 → 输出颜色」的映射（与 fit_color_map 同一范式：拟合 + 逐对精确验证）。

候选特征组合（固定清单，不做 2^n 穷举，避免候选爆炸）：
  color                        自身颜色
  color+border                 是否位于所属对象的边界
  color+obj_size               所属对象像素数
  color+obj_rank               所属对象按大小的排名（降序）
  color+touches                所属对象是否接触网格边界
  color+is_largest             所属对象是否为最大对象
  color+rare                   所属对象的颜色是否在整图中唯一
  color+holes                  所属对象的洞数
  color+square                 所属对象的包围盒是否正方形
  color+grid_border            该格是否位于网格边界
  obj_size / obj_rank / border / touches / holes / is_largest   （不含自身颜色）
  color+obj_size+border
  color+obj_rank+border
  color+obj_size+touches
  color+obj_rank+is_largest

纪律与其它族一致：映射必须由示范拟合，且必须**逐对精确复现全部示范**才被接受。

用法：
    from context_recolor import context_candidates
    cands = context_candidates(train_pairs, bg)   # -> [(name, fn)]
"""
from __future__ import annotations

from collections import Counter, deque

FEATURE_SETS = [
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


def grid_shape(g):
    return (len(g), len(g[0]))


def _components(grid, bg):
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


def _holes(grid, comp, bg):
    """对象内部被包围的背景区数量（从对象包围盒边界灌水）。"""
    h, w = grid_shape(grid)
    inside = set(comp)
    r0 = min(r for r, _ in comp); r1 = max(r for r, _ in comp)
    c0 = min(c for _, c in comp); c1 = max(c for _, c in comp)
    seen, q = set(), deque()
    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if grid[r][c] == bg and (r, c) not in inside:
                q.append((r, c)); seen.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if grid[r][c] == bg and (r, c) not in inside:
                q.append((r, c)); seen.add((r, c))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if r0 <= ny <= r1 and c0 <= nx <= c1 and grid[ny][nx] == bg and (ny, nx) not in inside and (ny, nx) not in seen:
                seen.add((ny, nx)); q.append((ny, nx))
    n, vis = 0, set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == bg and (r, c) not in inside and (r, c) not in seen and (r, c) not in vis:
                n += 1
                qq = deque([(r, c)]); vis.add((r, c))
                while qq:
                    y, x = qq.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if r0 <= ny <= r1 and c0 <= nx <= c1 and grid[ny][nx] == bg and (ny, nx) not in inside and (ny, nx) not in vis:
                            vis.add((ny, nx)); qq.append((ny, nx))
    return n


def cell_features(grid, bg):
    """返回 {(r,c): {feature_name: value}}。背景格的对象类特征取 None。"""
    h, w = grid_shape(grid)
    comps = _components(grid, bg)
    sizes = [len(c) for c in comps]
    order = sorted(range(len(comps)), key=lambda i: -sizes[i])
    rank_of = {idx: pos for pos, idx in enumerate(order)}
    owner = {}
    obj_color = {}
    obj_holes = {}
    obj_square = {}
    obj_touches = {}
    for idx, comp in enumerate(comps):
        r0 = min(r for r, _ in comp); r1 = max(r for r, _ in comp)
        c0 = min(c for _, c in comp); c1 = max(c for _, c in comp)
        for rc in comp:
            owner[rc] = idx
        obj_color[idx] = Counter(grid[r][c] for r, c in comp).most_common(1)[0][0]
        obj_holes[idx] = _holes(grid, comp, bg)
        obj_square[idx] = (r1 - r0) == (c1 - c0)
        obj_touches[idx] = (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1)
    color_count = Counter(obj_color.values())
    largest = sizes.index(max(sizes)) if sizes else None

    feats = {}
    for r in range(h):
        for c in range(w):
            idx = owner.get((r, c))
            if idx is None:
                feats[(r, c)] = {
                    "color": grid[r][c], "obj_size": None, "obj_rank": None,
                    "border": None, "touches": None, "holes": None,
                    "square": None, "is_largest": None, "rare": None,
                    "grid_border": (r == 0 or c == 0 or r == h - 1 or c == w - 1),
                }
                continue
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            is_border = any(n not in owner or owner[n] != idx for n in neighbors)
            feats[(r, c)] = {
                "color": grid[r][c],
                "obj_size": sizes[idx],
                "obj_rank": rank_of[idx],
                "border": is_border,
                "touches": obj_touches[idx],
                "holes": obj_holes[idx],
                "square": obj_square[idx],
                "is_largest": (idx == largest),
                "rare": color_count[obj_color[idx]] == 1,
                "grid_border": (r == 0 or c == 0 or r == h - 1 or c == w - 1),
            }
    return feats


def fit_feature_map(feature_names, pairs, bg):
    """从示范拟合 特征元组 -> 输出颜色；不一致则返回 None。"""
    mapping = {}
    for pair in pairs:
        g, t = pair["input"], pair["output"]
        if grid_shape(g) != grid_shape(t):
            return None
        feats = cell_features(g, bg)
        for (r, c), f in feats.items():
            key = tuple(f[name] for name in feature_names)
            val = t[r][c]
            if mapping.get(key, val) != val:
                return None
            mapping[key] = val
    return mapping or None


def make_fn(feature_names, mapping, bg):
    def fn(grid):
        if grid_shape(grid) != grid_shape(grid):
            return None
        feats = cell_features(grid, bg)
        out = [row[:] for row in grid]
        for (r, c), f in feats.items():
            val = mapping.get(tuple(f[name] for name in feature_names))
            if val is None:
                return None
            out[r][c] = val
        return out

    return fn


def context_candidates(pairs, bg=0):
    """返回验证过的上下文改色候选：[(name, fn)]。"""
    out = []
    for names in FEATURE_SETS:
        mapping = fit_feature_map(names, pairs, bg)
        if not mapping:
            continue
        fn = make_fn(names, mapping, bg)
        # 逐对精确复现（拟合一致时通常自动成立，仍显式验证）
        ok = True
        for pair in pairs:
            try:
                if fn(pair["input"]) != pair["output"]:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            out.append(("ctx_" + "+".join(names), fn))
    return out
