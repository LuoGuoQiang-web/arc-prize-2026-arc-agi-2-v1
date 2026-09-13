"""神经路线专图：瓶颈是生成退化，不是覆盖（Solution Writeup 配图）。

数据来源（全部实测，2026-09-13）
--------------------------------
A 面板 —— 对评测运行导出的 submission.json 做读回分析，脚本 solver/crossref_attempts.py：
  32 个 attempt_1（24 题，部分题有 2 个 test input）构成为
    来源 neural  、内容多样        : 21
    来源 neural  、单色均匀网格    : 10   <-- 模型自己塌缩
    来源 fallback、单色均匀网格    :  1
  另外：attempt_1 == attempt_2 的比例为 0/32（双槽独立性成立）。
  运行：Kaggle 内核 arc26-eval-validate（lg 报告 report.json）。

B 面板 —— 三层各自的实测贡献：
  符号引擎  0/120 评测题 = 0.000%   （solver/rehearse_engine_kernel.py）
  保底层    0/172  评测输入命中     （solver/measure_fallback.py，新旧两版皆为 0）
  神经层    1/24   评测题   = 4.17% （内核 arc26-eval-validate，覆盖 24/24）

输出：work/arc_w1/figures/attempt_breakdown.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

BLUE, RED, GREY = "#4C78A8", "#E45756", "#9E9E9E"

# --- panel A: composition of the 32 top-ranked candidates ----------------------------
labels = ["neural,\ncontent varies", "neural,\nsingle colour", "fallback,\nsingle colour"]
counts = [21, 10, 1]
colors = [BLUE, RED, GREY]

# --- panel B: what each layer actually contributes -----------------------------------
layers = ["symbolic\nengine", "fallback\nlayer", "neural\nlayer"]
pct = [0.0, 0.0, 4.17]
denom = ["0 / 120 tasks", "0 / 172 inputs", "1 / 24 tasks"]
bar_colors = [GREY, GREY, BLUE]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=110)
fig.suptitle("Where the score can come from, and why it does not arrive",
             fontsize=15, fontweight="bold")

x = np.arange(len(labels))
ax1.bar(x, counts, 0.62, color=colors)
for xi, v in zip(x, counts):
    ax1.annotate(f"{v}", (xi, v + 0.4), ha="center", fontsize=12, fontweight="bold")
ax1.annotate("31% of top-ranked\ncandidates are the model\ncollapsing to one colour",
             (1, 10), textcoords="offset points", xytext=(30, 26), fontsize=10,
             color="#A02B2B",
             bbox=dict(boxstyle="round,pad=0.4", fc="#FDECEC", ec=RED),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("attempt_1 slots (of 32)")
ax1.set_ylim(0, 25)
ax1.set_title("A. Reading the submitted grids back: the top guess is often degenerate",
              fontsize=11.5)
ax1.grid(axis="y", alpha=0.3)
ax1.text(0.5, 21.5, "attempt_1 == attempt_2 in 0 of 32 cases\n(the second slot is always independent)",
         ha="center", fontsize=9, color="#2F5597",
         bbox=dict(boxstyle="round,pad=0.35", fc="#EAF1F8", ec=BLUE))

x2 = np.arange(len(layers))
ax2.bar(x2, pct, 0.55, color=bar_colors)
for xi, (v, d) in enumerate(zip(pct, denom)):
    ax2.annotate(f"{v:.2f}%", (xi, v + 0.14), ha="center", fontsize=12, fontweight="bold")
    ax2.annotate(d, (xi, 0.05), ha="center", fontsize=9, color="#444444", rotation=0)
ax2.annotate("coverage is complete on\nthe neural layer: 24 of 24\ntasks got a real candidate",
             (2, 4.17), textcoords="offset points", xytext=(-38, 40), fontsize=9.5,
             color="#2F5597",
             bbox=dict(boxstyle="round,pad=0.4", fc="#EAF1F8", ec=BLUE))
ax2.set_xticks(x2)
ax2.set_xticklabels(layers, fontsize=10.5)
ax2.set_ylabel("measured accuracy  [%]")
ax2.set_ylim(0, 6.4)
ax2.set_title("B. Three layers measured: two are exactly zero, so every point must come from the model",
              fontsize=11.5)
ax2.grid(axis="y", alpha=0.3)

fig.text(0.5, 0.015,
         "Every number is measured, not estimated. Data: official ARC-AGI-2 (Apache-2.0), "
         "Kaggle T4 x 2. Reproduce: solver/crossref_attempts.py, measure_fallback.py, "
         "rehearse_engine_kernel.py, kernel arc26-eval-validate",
         ha="center", fontsize=8.5, color="#555555")
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
out = OUT / "attempt_breakdown.png"
fig.savefig(out)
print("saved", out, out.stat().st_size, "bytes")
