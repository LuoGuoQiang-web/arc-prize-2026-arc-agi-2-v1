# Kaggle 操作手册（v1）

> 想跳过所有技术细节：**只看方法 A 的 6 步**。
> 本目录所有内容都在本地真实跑过：引擎自测 **36/36** 通过，生成的 Notebook 也已整份执行通过（见 `README.md` §6 与 §12）。
>
> **提交通路已完整彩排**：用真实的竞赛格式测试文件（100 题、**无真值**，含 1/2/3 个 test input 的题目）跑完整 submit 模式，**7/7 检查通过**——题号集合与官方 `sample_submission` 完全一致、每题 attempts 列表长度正确、attempt_1/2 齐备、网格合法。脚本：`../work/arc_w1/rehearsal_submit.py`。→ 上传后除 Kaggle 环境差异外，管线侧无未知风险。

---

## 方法 A（推荐，零复制）：上传现成 Notebook

### 第 1 步：找到这个文件

`arc_prize_v1/ARC_PRIZE_2026_v1.ipynb` —— 这就是**做好的 Notebook**，引擎已经嵌在里面，你一个字都不用粘。

### 第 2 步：把它传进 Kaggle

1. 浏览器打开 <https://www.kaggle.com/code> → 右上 **+ New Notebook**（新建一个空 Notebook）；
2. 在编辑器顶部菜单点 **File → Import Notebook**；
3. 选 **Upload**，把这个 `.ipynb` 文件拖进去 → 确认。页面会变成我们写好的 16 格 Notebook。

> 传完后可以顺手把标题改成 `ARC Prize 2026 v1`（右上角 `File → ...` 或标题处重命名）。

### 第 3 步：挂上竞赛数据（不做这步会报"没找到数据"）

右侧栏 **Add Input** → 选 **Competitions** 标签 → 搜 `ARC Prize 2026` → 添加 **ARC Prize 2026 - ARC-AGI-2**。

### 第 4 步：设置运行环境

右上角 **Session options**（或右侧栏设置）：
- **Internet**：`Off`（官方规则：评测期无网络）
- **Accelerator**：`None`（本方案 CPU 跑，**不消耗你的 GPU 额度**）

### 第 5 步：点 Run All

顶部菜单 **Run → Run All**，等待跑完。看到这一行就算成功：

```
✅ 成功：提交文件已生成 -> /kaggle/working/submission.json
```

预计耗时：**几秒到 1 分钟**（240 题，实测约 0.14 秒/题）。

### 第 6 步：推版本 + 提交

1. 确认倒数第二格打印 `VALID: True`；
2. 右上角 **Save Version** → 选 **Save & Run All (Commit)** → 保存；
3. 等版本跑完（几分钟），到竞赛页 **Submit** 面板，选这个版本提交；
4. 确认面板没有出现 `Submission Scoring Error`（第一次提交主要就是验证这一条）。

---

### 以后改了代码怎么办

Notebook 里的引擎是从 `arc_prize_v1.py` 生成的。改了引擎后，在本地跑一次：

```bash
cd arc_prize_v1
python tools/build_notebook.py     # 重新生成 ARC_PRIZE_2026_v1.ipynb
```

脚本会在生成时**校验**内嵌引擎与源文件逐字节一致（不一致会直接报错退出），所以不会出现"Kaggle 上跑的是旧代码"。

---

## 方法 B（备选）：手动粘贴

如果你不想传文件，也可以用「粘贴」的方式，在空 Notebook 里按下面 3 格做：

**第 0 格**（建目录）：

```python
import os
os.makedirs("/kaggle/working/arcprize", exist_ok=True)
print(os.path.isdir("/kaggle/working/arcprize"))
```

**第 1 格**：第一行写 `%%writefile /kaggle/working/arcprize/arc_prize_v1.py`，
**紧接着**把 `arc_prize_v1.py` 的全部内容贴在下面（从文件第一个字符开始，别漏开头三引号）。
运行后应显示 `Writing /kaggle/working/arcprize/arc_prize_v1.py`。

**第 2 格**：跑求解

```python
import importlib, sys
sys.path.insert(0, "/kaggle/working/arcprize")
import arc_prize_v1
importlib.reload(arc_prize_v1)

summary = arc_prize_v1.main(
    mode="submit", dataset="test",
    global_budget_seconds=5400, per_task_seconds=60,
    submission_flush_every=25, snapshot_on_finish=True,
)
print(summary)
```

**第 3 格**：校验

```python
from pathlib import Path
import arc_prize_v1
cfg = arc_prize_v1.make_config({"mode": "submit", "dataset": "test"})
disc = arc_prize_v1.discover_data(cfg)
tasks, _, _ = arc_prize_v1.load_dataset(cfg, disc, "test")
print(arc_prize_v1.validate_submission(
    Path("/kaggle/working/submission.json"),
    arc_prize_v1.expected_structure(tasks),
    disc["files"].get("sample_submission")))
```

---
## 常用操作速查（Notebook 里另开一格运行）

| 想做什么 | 代码 |
|---|---|
| 续跑（只补没做完的题） | `arc_prize_v1.main(mode="submit", dataset="test")` |
| 全部重算 | `arc_prize_v1.main(mode="submit", dataset="test", force=True)` |
| 只凭断点重建提交 | `arc_prize_v1.rebuild_submission(mode="submit", dataset="test")` |
| 打一份备份 | `arc_prize_v1.latest_run(mode="submit", dataset="test").snapshot("manual", keep=3)` |
| 看历史运行 | `arc_prize_v1.list_runs(mode="submit", dataset="test")` |
| 自己给自己的评估集打分 | `arc_prize_v1.main(mode="dev", dataset="evaluation", force=True)` |

## 运行产物在哪

```
/kaggle/working/
├── submission.json                 ← Kaggle 评分读这个
└── arcprize/
    ├── arc_prize_v1.py             ← 引擎
    ├── backups/*.zip               ← 轮转快照（默认每 run 保留 3 份）
    ├── state/PROJECT_STATE.md      ← 当前状态摘要（人话版）
    └── runs/<run_id>/
        ├── manifest.json           ← 引擎/配置/数据 指纹 + 环境
        ├── checkpoint.jsonl        ← 断点（每题一行，续跑靠它）
        ├── diagnostics.csv         ← 每题失败原因（论文素材）
        ├── submission.json / state.md / log.txt
```

## 出问题怎么办

| 现象 | 处理 |
|---|---|
| 提示没找到数据 | 回到第 3 步，**Add Input → Competitions → ARC Prize 2026 - ARC-AGI-2** |
| 提交里全是 0 | 说明走了保命路径（只找到 sample_submission）。看 `state/PROJECT_STATE.md` 里的 `status=sample_passthrough` |
| 跑太久 | `global_budget_seconds` 到点会自动收尾并写出合法提交；看 `log.txt` 的 ETA 行 |
| 报 `VALID: False` | 跑一格 `arc_prize_v1.rebuild_submission(...)`；还不行就跑 `main(..., force=True)` |
| Notebook 跑崩了 | 重新 Run All 即可自动续跑（不会重算已完成的题） |

## 提交前检查清单

1. 校验格打印 `VALID: True`；
2. 题目数与竞赛 `arc-agi_test_challenges.json` 一致（2026 官方规模：240 题）；
3. 在竞赛 Submit 面板确认**今日剩余提交次数**（提交是 Notebook-only）；
4. 若要计入奖牌：确认代码已按 CC0/MIT-0 开源。

## 请回传给我的信息（用于 W1 报告）

1. 求解格打印的 `summary` 全文；
2. 自评格的 `dev_score` 与 `failure_classes`；
3. Session options 里剩余 GPU 小时数（即便 v1 不用 GPU，后面对齐手册 §2.4 仍需要）。
