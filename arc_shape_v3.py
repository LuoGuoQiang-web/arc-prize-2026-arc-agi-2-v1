"""v3 形状改变 / 计数族：补上诊断指出的最大空白（评测集 30.8% 是 shape_diff）。

既有四族几乎都**保形**（几何族的裁剪类被 shape 检查挡掉一部分；对象族/上下文族在引擎里都要求
逐格同形拟合）。而测试集上仍有 30% 左右是"输出尺寸与输入不同"的题，本模块专门覆盖这一类：

  1. crop_to_color_bbox(c)        输出 = 某颜色像素的包围盒
  2. crop_to_object_bbox(pred)    输出 = 最大/最小/唯一/接触边界对象的包围盒
  3. remove_color_then_crop(c)    先擦掉某颜色，再取非背景包围盒
  4. count_scalar(feature)        输出 = 1×1，值为某计数（对象数/颜色数/最大对象尺寸/某色像素数…）
  5. constant_fill(color, shape)  输出 = 训练输出众数形状，全部填某色
  6. scale_factor(k)              输出 = 输入整数倍放大（k=2..4）或整数倍缩小
  7. grid_border_ring()           输出 = 输入的外框环
  8. trim_uniform_then_crop()     去均匀行列后再取包围盒

纪律与其它族一致：**参数由示范拟合，且必须逐对精确复现全部示范**才被接受。
"""
from __future__ import annotations

from collections import Counter, deque

MAX = 30   # ARC-AGI-2 网格上限


def _shape(g):
    return (len(g), len(g[0]))


def _components(grid, bg=0):
    h, w = _shape(grid)
    seen = [[False] * w for _ in range(h)]
    out = []
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
            out.append(comp)
    return out


def _sub(grid, r0, c0, r1, c1):
    return [[grid[r][c] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]


def _bbox_of_color(grid, color):
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color]
    if not pts:
        return None
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox_nonbg(grid, bg):
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg]
    if not pts:
        return None
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_to_color_builder(color):
    def fn(grid):
        b = _bbox_of_color(grid, color)
        if b is None:
            return None
        out = _sub(grid, *b)
        return out if _shape(out)[0] <= MAX and _shape(out)[1] <= MAX else None

    return fn


def _crop_to_object_builder(pred_name, bg):
    def fn(grid):
        comps = _components(grid, bg)
        if not comps:
            return None
        sizes = [len(c) for c in comps]
        if pred_name == "largest":
            pick = comps[sizes.index(max(sizes))]
        elif pred_name == "smallest":
            pick = comps[sizes.index(min(sizes))]
        else:
            colors = [Counter(grid[r][c] for r, c in comp).most_common(1)[0][0] for comp in comps]
            cnt = Counter(colors)
            uniq = [i for i, col in enumerate(colors) if cnt[col] == 1]
            if not uniq:
                return None
            pick = comps[uniq[0]]
        rs = [r for r, _ in pick]
        cs = [c for _, c in pick]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        out = _sub(grid, r0, c0, r1, c1)
        return out if _shape(out)[0] <= MAX and _shape(out)[1] <= MAX else None

    return fn


def _remove_color_then_crop_builder(color, bg):
    def fn(grid):
        g = [[bg if v == color else v for v in row] for row in grid]
        b = _bbox_nonbg(g, bg)
        if b is None:
            return None
        out = _sub(g, *b)
        return out if _shape(out)[0] <= MAX and _shape(out)[1] <= MAX else None

    return fn


# ---- 计数型：输出小网格，值为某特征 ----
COUNTS = {
    "n_objects": lambda g, bg: len([c for c in _components(g, bg) if c]),
    "n_colors": lambda g, bg: len({v for row in g for v in row}),
    "n_nonbg": lambda g, bg: sum(1 for row in g for v in row if v != bg),
    "max_obj_size": lambda g, bg: max((len(c) for c in _components(g, bg)), default=0),
    "min_obj_size": lambda g, bg: min((len(c) for c in _components(g, bg)), default=0),
    "n_holes": lambda g, bg: _n_holes(g, bg),
    "n_bg": lambda g, bg: sum(1 for row in g for v in row if v == bg),
}


def _n_holes(grid, bg):
    """整图像洞数：不接触边界的背景连通块数量。"""
    h, w = _shape(grid)
    seen = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == bg and not seen[r][c]:
                seen[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == bg and not seen[r][c]:
                seen[r][c] = True
                q.append((r, c))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == bg:
                seen[ny][nx] = True
                q.append((ny, nx))
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


def _scale_fn(grid, k, up=True):
    h, w = _shape(grid)
    if up:
        nh, nw = h * k, w * k
        if nh > MAX or nw > MAX:
            return None
        return [[grid[r // k][c // k] for c in range(nw)] for r in range(nh)]
    if h % k or w % k:
        return None
    return [[grid[r * k][c * k] for c in range(w // k)] for r in range(h // k)]


def _trim_uniform(grid, bg):
    keep_r = [i for i, row in enumerate(grid) if len(set(row)) > 1]
    keep_c = [j for j in range(len(grid[0])) if len({grid[i][j] for i in range(len(grid))}) > 1]
    if not keep_r or not keep_c:
        return [row[:] for row in grid]
    return [[grid[i][j] for j in keep_c] for i in keep_r]


def _border_ring(grid, bg):
    h, w = _shape(grid)
    out = []
    for r in range(h):
        row = []
        for c in range(w):
            edge = r in (0, h - 1) or c in (0, w - 1)
            row.append(grid[r][c] if edge else bg)
        out.append(row)
        if h > MAX or w > MAX:
            return None
    return out


# ---------------- 候选生成（拟合 + 逐对验证） ----------------

def _builders(bg: int):
    """按任务推断出的背景色构造候选生成器（背景色逐题不同，不能硬编码）。"""
    bs = []
    for c in range(10):
        if c != bg:
            bs.append((f"v3crop_color{c}", _crop_to_color_builder(c)))
    for p in ("largest", "smallest", "unique_color"):
        bs.append((f"v3crop_obj_{p}", _crop_to_object_builder(p, bg)))
    for c in range(10):
        if c != bg:
            bs.append((f"v3rmcolor{c}_crop", _remove_color_then_crop_builder(c, bg)))
    for k in (2, 3, 4):
        bs.append((f"v3up{k}", lambda g, k=k: _scale_fn(g, k, True)))
        bs.append((f"v3down{k}", lambda g, k=k: _scale_fn(g, k, False)))
    bs.append(("v3trim_crop",
               lambda g: (lambda b: None if b is None else _sub(g, *b))(_bbox_nonbg(_trim_uniform(g, bg), bg))))
    bs.append(("v3ring", lambda g: _border_ring(g, bg)))
    return bs


def count_candidates(pairs, bg):
    """计数型：输出 1×1，值 = 某计数的映射（映射由示范拟合）。"""
    out = []
    shapes = {_shape(p["output"]) for p in pairs}
    if shapes != {(1, 1)}:
        return out
    for name, feat in COUNTS.items():
        mapping = {}
        ok = True
        for p in pairs:
            v = feat(p["input"], bg)
            t = p["output"][0][0]
            if mapping.get(v, t) != t:
                ok = False
                break
            mapping[v] = t
        if not ok or not mapping:
            continue

        def fn(grid, feat=feat, mapping=mapping, bg=bg):
            v = feat(grid, bg)
            return None if v not in mapping else [[mapping[v]]]

        out.append((f"v3count_{name}", fn))
    return out


def shape_candidates(task, bg=0, max_candidates: int = 120):
    """返回已验证的形状改变/计数候选：[(name, fn)]。"""
    pairs = task["train"]
    good = []

    def ok(fn):
        for p in pairs:
            try:
                if fn(p["input"]) != p["output"]:
                    return False
            except Exception:
                return False
        return True

    for name, fn in _builders(bg):
        if len(good) >= max_candidates:
            break
        if ok(fn):
            good.append((name, fn))
    for name, fn in count_candidates(pairs, bg):
        if len(good) >= max_candidates:
            break
        good.append((name, fn))
    return good
