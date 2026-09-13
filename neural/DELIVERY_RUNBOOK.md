# 交付手册（ARC Prize 2026 · Option C）

> 这份文件的用途：**当我不在时，交付也能完成。**
> 每条命令都已在本机实际验证过形式（`--help` 或真实运行）。

---

## 0. 当前状态（2026-09-13 20:1x）

| 项 | 状态 |
|---|---|
| `arc26-submit-full`（可提交内核） | **RUNNING**，240 题，预算 39000 s，`-t 43200` |
| `arc26-submit-shard0`（测量用，**不可提交**） | RUNNING，真实隐藏集 120 题 |
| 提交任务 `pwsh-6` | 守候中：等内核完成 → 下载 → 严格校验 → 等配额重置 → 提交一次 |
| 配额 | 约 4.3h / 30h 已用；**2026-09-19 刷新** |
| 公开仓库 | https://github.com/LuoGuoQiang-web/arc-prize-2026-arc-agi-2-v1 （MIT-0） |

预计时序：两个内核约 **04:40–05:20（北京）** 完成；每日提交配额在
**UTC 00:00 = 北京 08:00** 重置；提交任务约 08:05 自动提交。

---

## 1. 会自动完成的事（无需干预）

`pwsh-6` 会依次做：

1. 每 3 分钟轮询 `arc26-submit-full`，直到状态为 COMPLETE；
2. 下载 `/kaggle/working/submission.json`（**带 5 次退避重试**，瞬时网络故障不会
   丢掉一个跑完的运行）；
3. **严格校验**：任务数 == 240、key 与挑战文件一致、每题 attempt 条数 == test input
   数、两个槽都是 1–30 的合法网格、颜色 0–9、无空答；
4. 等到 UTC 00:05（配额重置后），提交一次。

**校验不通过就放弃提交**——`maxDailySubmissions=1`，宁可今天不提交，也不能烧掉配额
换来一次注定被拒的提交。

---

## 2. 需要你手动做的事（按顺序）

### 2.1 确认提交成功

```powershell
$env:KAGGLE_API_TOKEN='<你的 KGAT_ token>'
python -m kaggle competitions submissions arc-prize-2026-arc-agi-2
```

期望：出现一条**新的 ref**，`status` 为 `SubmissionStatus.COMPLETE`。
若只有旧的 `56199696`，说明提交没发生——看 `pwsh-6` 的输出判断卡在哪一步。

### 2.2 把 Writeup 指向新的提交号（**粘贴前必须做**）

现在 Writeup 里引用的是 `56199696`，那是**被拒**的一次（Kaggle 重跑得到 120 行而非
240 行，无分）。按规则 Writeup 必须挂在**有效**的代码提交上。

```powershell
# 自动挑最新一条有分的提交
python work/arc_w1/writeup/update_submission_ref.py --latest

# 或显式指定 ref（会先核对它确实存在于提交列表，不存在则拒绝）
python work/arc_w1/writeup/update_submission_ref.py --ref <新的ref> --dry-run
python work/arc_w1/writeup/update_submission_ref.py --ref <新的ref>
```

注意：Kaggle 的公开分可能延迟显示。若 `--latest` 报「没有带分的 COMPLETE 提交」，
就等分出来，或用 `--ref` 显式传。

### 2.3 粘贴两份 Writeup（Kaggle 无 Writeup API，只能网页操作）

| 赛道 | 粘贴文件 | 上限 |
|---|---|---|
| **Paper Track** | `work/arc_w1/writeup/WRITEUP_PASTE_READY.md` | 1500 词（正文 1492） |
| **ARC-AGI-2 Solution Writeup** | `work/arc_w1/writeup/NEURAL_WRITEUP_PASTE_READY.md` | 1500 词（正文 1499） |

两份文件开头都有元数据块：标题、副标题、赛道、必需的 Project Links、需要上传的图、
以及提交前清单。

**配图**（Paper Track 必须有封面）：

- `work/arc_w1/figures/cover.png` —— 封面（必需）
- `work/arc_w1/figures/coverage_curve.png` —— 覆盖率曲线

### 2.4 确认代码提交与公开仓库链接都在 Writeup 里

规则要求 Writeup 关联一个**公开 notebook** 和一个**有效的代码提交**。两份粘贴版里
都已含链接，替换 ref 后请扫一眼确认。

---

## 3. 手动补跑（若自动提交失败）

```powershell
$env:KAGGLE_API_TOKEN='<你的 KGAT_ token>'
python work/arc_w1/submit_final.py --deadline-min 60
```

它会重新走完「等内核 → 下载（带重试）→ 校验 → 提交」。若内核早已完成，第一步会立刻通过。

**注意 `-f` 的口径**：`kaggle competitions submit` 的 `-f` 既可以是完整本地路径，
也可以是内核产出的文件名。脚本传的是**完整本地路径**——这样校验的就是真正上传的那个
文件，而不是内核里那份未经检查的产物。

---

## 4. 若想再跑一次更好的版本

当前在跑的**两个内核用的都是修复前的代码**，所以：

- `arc26-submit-full` **不含 TTT**（Stage A 吃光了预算，Stage B 从未进入）；
- 不含形状先验、不含退化过滤、不含 top-k 兜底。

修复版已在仓库里，但**尚未经 GPU 验证**。下一步应当是 I.1 的**受控对比实验**
（见 `EXPERIMENTS.md`）：用**完全相同**的 24 题评测抽样，只改代码，读日志判定
（Stage A 上限是否生效、Stage B 是否真的跑了 TTT、单色首选槽是否消失）。

配额 2026-09-19 刷新；赛程到 **2026-11-02**，时间充裕。每日可提交一次，所以
「先提交当前版本、下周提交改进版」是安全的顺序。

---

## 5. 别踩的坑（都是实测踩过的）

| 坑 | 后果 | 规避 |
|---|---|---|
| metadata 只写 `enable_gpu` 不给 `machine_shape` | 落到 **P100 死核**，一算就 `no kernel image` | 只用 `kpush.py` 推核 |
| 写 `NvidiaL4` 等 L4 拼写 | **静默**归一化成 P100，push 仍返回成功 | 只用 `NvidiaTeslaT4` |
| 分片内核拿去提交 | Kaggle 重跑只产出 120 行 → **被拒** | 只有未分片 240 题内核可提交 |
| 假设成功运行的执行日志可取 | 成功运行**不发布**执行日志 | 一切写 `/kaggle/working` |
| 在 notebook 里写 `raise SystemExit(main())` | cell 判失败，整次运行变 ERROR | 只写 `main()` |
| 手动抄提交号 | 可能引用被拒的 ref | 用 `update_submission_ref.py` |
| `git push` 到 github.com | 间歇性连接重置 | 走 `push_github_api.mjs`（`gh auth token`） |

---

## 6. 诚实结论（不应被遗漏）

**4.17% 拿不到任何奖金档。** 进度奖需 top-8（约 34%+），参照实现在同一基准上是
33.89，用 4×L4 和每任务约 4 倍算力。以 1×T4×2 的算力，本方案天花板在 4–6% 量级。

**奖金路径只能是写作。** 代码提交的作用是取得参赛资格——Writeup 必须挂在有效提交上。

三项未达成，已写入 `EXPERIMENTS.md` 的 J 节：混合候选池是**惰性组件**（符号引擎未加载
且其命中率为 0）；当前提交物**不含 TTT 贡献**；Paper Track 的引用 ref 在替换前是无效的。
