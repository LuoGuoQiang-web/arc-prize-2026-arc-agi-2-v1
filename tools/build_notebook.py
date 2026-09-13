"""
Generate ARC_PRIZE_2026_v1.ipynb from arc_prize_v1.py.

Why a generator instead of hand-editing a notebook:
  * the notebook embeds the full engine, so the two can drift apart;
  * this script proves, at build time, that the `%%writefile` cell reproduces the engine
    byte-for-byte (see the final assertion) — otherwise Kaggle would silently run old code.

Usage:  python tools/build_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "arc_prize_v1.py"
OUT = ROOT / "ARC_PRIZE_2026_v1.ipynb"


def src(text: str) -> list:
    """Notebook source is a list of lines, each keeping its trailing newline."""
    lines = text.splitlines(keepends=True)
    return lines


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src(text)}


def engine_text() -> str:
    with ENGINE.open("r", encoding="utf-8", newline="") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


CELL_1_MD = """# ARC Prize 2026 — ARC-AGI-2 · v1（开箱即用）

**这个 Notebook 会做什么**：读竞赛数据 → 用符号搜索解每一题 → 写出 `/kaggle/working/submission.json`。

**你要做的只有两件事**：
1. 右边栏 **Add Input → Competitions → `ARC Prize 2026 - ARC-AGI-2`**（数据必须挂上）；
2. 顶部菜单 **Run → Run All**（或按顺序跑完每一格）。

**跑完的标志**：看到 `✅ 成功：提交文件已生成`。
之后右上角 **Save Version → Save & Run All** 推版本，就能在竞赛页 Submit 那个版本。

**建议设置（右侧 Session options）**：Internet **Off**、Accelerator **None（CPU）** —— 本方案不需要 GPU。

预计耗时：秒级到 1 分钟内（240 题，实测约 0.14 秒/题）。
"""

CELL_2_MD = """## 第 1 格：建目录
`%%writefile` 不会自动创建父目录，所以先建好。"""

CELL_2_CODE = '''import os
os.makedirs("/kaggle/working/arcprize", exist_ok=True)
print("目录就绪:", os.path.isdir("/kaggle/working/arcprize"))
'''

CELL_3_MD = """## 第 2 格：写入引擎
这格是 `%%writefile` 魔法命令，把下面整段代码**原样写进** `/kaggle/working/arcprize/arc_prize_v1.py`。

你不用读它，直接运行即可（运行后会显示 `Writing ...` ）。"""

CELL_4_MD = """## 第 3 格：检查数据挂上了没有
如果你看到 ❌，回到右边栏 **Add Input → Competitions → ARC Prize 2026 - ARC-AGI-2**，然后重新运行本格。"""

CELL_4_CODE = '''import importlib, sys
sys.path.insert(0, "/kaggle/working/arcprize")
import arc_prize_v1
importlib.reload(arc_prize_v1)

disc = arc_prize_v1.discover_data(arc_prize_v1.make_config())
names = {k: str(v) for k, v in disc["files"].items()}
print("找到的竞赛数据文件：")
for k, v in sorted(names.items()):
    print(f"  - {k}: {v}")
if not names:
    print("\\n❌ 没找到任何竞赛文件。请 Add Input → Competitions → 'ARC Prize 2026 - ARC-AGI-2'")
else:
    print("\\n✅ 数据已就绪，可以继续下一格")
'''

CELL_5_MD = """## 第 4 格：开始求解并生成提交（核心格）
默认按 **提交模式** 跑测试集。想省时间/网络断掉也没关系，超时会自动收尾并写出合法提交。"""

CELL_5_CODE = '''import importlib, json, sys
sys.path.insert(0, "/kaggle/working/arcprize")
import arc_prize_v1
importlib.reload(arc_prize_v1)          # 如果你改了引擎，重跑本格就生效

summary = arc_prize_v1.main(
    mode="submit",                  # submit = 用 test challenges 出提交
    dataset="test",
    global_budget_seconds=5400,     # 全局上限 90 分钟，到点也会写出合法提交
    per_task_seconds=60,            # 单题上限
    submission_flush_every=25,      # 每 25 题刷一次盘（防崩溃丢结果）
    snapshot_on_finish=True,        # 结束自动打包备份
)

print("\\n" + "=" * 64)
if summary.get("ok"):
    print("✅ 成功：提交文件已生成 ->", summary["submission_path"])
else:
    print("⚠️ 提交未通过结构校验，请看下面的 problems")
print("题目数:", summary.get("n_tasks"), "| test 输入总数:", summary.get("n_test_inputs"))
print("搜索解出:", summary.get("solved_by_search"), "| 先验兜底:", summary.get("prior_only"))
print("problems:", summary.get("problems"))
print("=" * 64)
'''

CELL_6_MD = """## 第 5 格：确认提交文件（提交前必看）
必须看到 `VALID: True`，否则先别推版本。"""

CELL_6_CODE = '''from pathlib import Path
import arc_prize_v1
importlib.reload(arc_prize_v1)

sub = Path("/kaggle/working/submission.json")
print("文件存在:", sub.exists(), "| 大小(字节):", sub.stat().st_size if sub.exists() else 0)

cfg = arc_prize_v1.make_config({"mode": "submit", "dataset": "test"})
disc = arc_prize_v1.discover_data(cfg)
tasks, _, source = arc_prize_v1.load_dataset(cfg, disc, "test")
ok, problems = arc_prize_v1.validate_submission(
    sub, arc_prize_v1.expected_structure(tasks), disc["files"].get("sample_submission")
)
print("VALID:", ok, "| 题目数:", len(tasks), "| 数据来源:", source)
print("problems:", problems[:5])

if ok:
    payload = arc_prize_v1.read_json(sub)
    first = sorted(payload)[0]
    print("\\n示例题", first, "->", "test 输入个数 =", len(payload[first]))
else:
    print("\\n❌ 校验失败：可以跑下一格用 rebuild 从断点重建，或整格重跑第 4 格")
'''

CELL_7_MD = """## 第 6 格（可选）：备份 / 续跑 / 重建
- **续跑**：直接再跑一次第 4 格，会复用同一个 run，只补没做完的题（不会重算）。
- **重建**：只凭已有的断点文件重写 `submission.json`，不重跑求解。
- **备份**：打包一份快照，默认每个 run 保留 3 份。
"""

CELL_7_CODE = '''import arc_prize_v1
importlib.reload(arc_prize_v1)

# 续跑（同配置复用同一 run，只补未完成的题）
# arc_prize_v1.main(mode="submit", dataset="test")

# 从断点重建提交（不重跑求解）
print(arc_prize_v1.rebuild_submission(mode="submit", dataset="test"))

# 手动打快照；latest_run() 只打开最近的 run，不会新建空目录
r = arc_prize_v1.latest_run(mode="submit", dataset="test")
print("快照:", r.snapshot("manual", keep=3))

# 查看历史运行
for info in arc_prize_v1.list_runs(mode="submit", dataset="test"):
    print(info)
'''

CELL_8_MD = """## 第 7 格（可选）：本地自评 + 失败分类
用竞赛自带的公开评估集（有答案）给自己打分，输出每题失败原因。这是论文「先诊断」的原料，不影响提交。

跑完后在右侧 **Output** 里能看到 `arcprize/runs/<run_id>/diagnostics.csv`。"""

CELL_8_CODE = '''import arc_prize_v1
importlib.reload(arc_prize_v1)

summary_dev = arc_prize_v1.main(mode="dev", dataset="evaluation", force=True)
print("自评准确率:", summary_dev.get("dev_score"))
print("失败分类:", summary_dev.get("failure_classes"))
'''

CELL_9_MD = """## 提交前最后检查清单

1. 第 5 格打印 `VALID: True`；
2. 右上角 **Save Version → Save & Run All**；
3. 等版本跑完，在竞赛页 **Submit** 面板选这个版本；
4. 确认面板显示的不是 `Submission Scoring Error`（第一次提交就是验证这一条）。

如果出错，把第 4 格打印的 `summary` 整段发回给我们即可。"""


def build() -> dict:
    text = engine_text()
    writefile_cell = "%%writefile /kaggle/working/arcprize/arc_prize_v1.py\n" + text

    cells = [
        md(CELL_1_MD),
        md(CELL_2_MD), code(CELL_2_CODE),
        md(CELL_3_MD), code(writefile_cell),
        md(CELL_4_MD), code(CELL_4_CODE),
        md(CELL_5_MD), code(CELL_5_CODE),
        md(CELL_6_MD), code(CELL_6_CODE),
        md(CELL_7_MD), code(CELL_7_CODE),
        md(CELL_8_MD), code(CELL_8_CODE),
        md(CELL_9_MD),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # ---- verify ------------------------------------------------------------------
    embedded_cells = [c for c in cells if c["cell_type"] == "code"
                      and "".join(c["source"]).startswith("%%writefile")]
    if len(embedded_cells) != 1:
        raise SystemExit(f"FATAL: expected exactly one %%writefile cell, found {len(embedded_cells)}")
    embedded = "".join(embedded_cells[0]["source"])
    body = embedded.split("\n", 1)[1] if "\n" in embedded else ""
    if body != text:
        raise SystemExit("FATAL: embedded engine does not match arc_prize_v1.py")
    json.dumps(nb)  # must be JSON-serialisable
    return nb


def main() -> int:
    nb = build()
    rendered = json.dumps(nb, indent=1, ensure_ascii=False)
    check_only = "--check" in sys.argv

    if check_only:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != rendered:
            print(f"STALE: {OUT.name} does not match {ENGINE.name} — run: python tools/build_notebook.py")
            return 1
        print(f"UP TO DATE: {OUT.name} matches {ENGINE.name} ({len(rendered)} bytes)")
        return 0

    OUT.write_text(rendered, encoding="utf-8")
    reload_check = json.loads(OUT.read_text(encoding="utf-8"))
    n_code = sum(1 for c in reload_check["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"cells: {len(reload_check['cells'])} ({n_code} code)")
    print(f"json round-trip: OK")
    print(f"embedded engine == arc_prize_v1.py: True ({ENGINE.stat().st_size} bytes source)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
