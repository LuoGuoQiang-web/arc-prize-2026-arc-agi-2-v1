# ARC Prize 2026 — 神经路线状态（Option C）

> 更新时间：2026-09-13。本文件是路线索引，细节在各自专门文档里。
> 标「实测」的结论都来自本机对 Kaggle 的真实运行，不是文档转述。

---

## 0. 一句话现状

**神经路线已经端到端跑通了。** 在 Kaggle T4x2 上：模型加载 → LoRA TTT →
约束 DFS 生成 → NLL 重排 → 级联调度 → 写出格式合法的 `submission.json`，
全链路验证通过；5 题训练冒烟 **4/5 正确**。剩下的工作是测真实准确率并跑全量提交。

---

## 1. 已验证的硬事实

### 1.1 GPU（详见 `KAGGLE_GPU_RECIPE.md`）

| 写法 | 实际拿到 | 可用 |
|---|---|---|
| `machine_shape: "NvidiaTeslaT4"` | **2 × Tesla T4, sm_75, 14.6 GiB** | ✅ |
| `NvidiaL4` / `NvidiaL4x4` / `NvidiaL4X4` | Tesla P100, sm_60 | ❌ |
| 只写 `enable_gpu: true` | Tesla P100, sm_60 | ❌ |

- P100 是**死核**：`cuda.is_available()==True` 但一算就 `no kernel image is available`。
- CLI 的 `--accelerator` flag **不被服务器采纳**，只能写 metadata 的 `machine_shape`。
- 无效拼写被**静默归一化**成 `Gpu`，push 照样返回成功。
- **并发 GPU 会话上限 2**；单会话 12h；周配额 30h。

### 1.2 模型 I/O（详见 `ARC26_SOLVER_SPEC.md` §1）

`sorokin/qwen3_4b_grids15_sft139`（Apache-2.0），Qwen3ForCausalLM，hidden 2560，
36 层，**3.634B 参数**，**词表只有 16 个 token**：

| id | 含义 |
|---|---|
| 0–9 | ARC 的 10 种颜色（**一个格子 = 一个 token**） |
| 10 | `Ċ` = 换行（行分隔） |
| 11 | `<\|im_start\|>user\n` |
| 12 | `<\|im_start\|>assistant\n` |
| 13 | `<\|endoftext\|>` = PAD |
| 15 | `<\|im_end\|>` = EOS |

约束解码只用 **12 个 token** = `[0..10] + [15]`。最大生成长度 **930**。
加载耗时实测 **68–169 s**（T4，多次运行波动较大）。

### 1.3 环境

```
torch 2.10.0+cu128   transformers 5.0.0   accelerate 1.13.0   datasets 5.0.0
unsloth MISSING   trl MISSING   bitsandbytes MISSING
peft 0.19.1 存在但【不可用】—— 见 §2.1
```

### 1.4 数据挂载（踩过的坑）

CLI 推送的内核挂载路径与直觉不同，实测：

```
/kaggle/input/competitions/arc-prize-2026-arc-agi-2/   ← 竞赛数据（多一层 competitions/）
/kaggle/input/models/<owner>/<model>/...              ← 模型（无额外层级）
```

本地 `arc-agi_test_challenges.json` 是**占位文件**（240/240 是训练题副本），
只有 Kaggle 运行时挂载的才是真实隐藏测试集。
本地可用的真值：训练 1000 题、评测 120 题，**都带答案**。

---

## 2. 这一轮修掉的 6 个 bug（全部是实测暴露的）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | 模型加载 `AttributeError: '_Tee' object has no attribute 'isatty'` | 我加的 stdout tee 没实现 `isatty`，`transformers` 会查它 | `__getattr__` 兜底委托给原始流 |
| 2 | `--data-dir does not exist` | 挂载路径多了 `competitions/` 一层 | 运行时自动解析 + 回退到 `/kaggle/input` 递归查找 |
| 3 | TTT 每步 `mat2 is on cpu, different from cuda:0` | LoRA 参数是加载**之后**新建的，`device_map` 不会搬运 | 创建时就放到基座权重的 device |
| 4 | `peft` 导入即抛 `torchao 0.10.0` 版本不兼容 | 镜像里 torchao 太旧 | **自研 LoRA**（`LoRALinear`，~40 行，零依赖） |
| 5 | 难任务候选池为空 `pool sizes [0]` | `turbo_dfs` 超时**直接丢弃**在飞 beam | 超时时冲刷当前 beam 进结果 |
| 6 | 截断网格仍被丢弃 | `parse_grid_string` 对任何不齐行返回 None | 新增 `recover_truncated`，截断到第一个不齐行（默认仍严格） |

**#5 和 #6 是覆盖率问题的关键**：修复后同一批 5 题从 2/5 → **4/5**。

---

## 3. 质量保证

全部本地可跑（**不需要 GPU，也不需要下载权重**）：

| 测试 | 内容 | 结果 |
|---|---|---|
| `test_format.py` | 序列化/prompt/增强往返、常量 | 全过 |
| `test_dataloc.py` | 数据路径自动解析（含错误路径回退） | 全过 |
| `test_lora.py` | **20 项** LoRA 机制：冻结、no-op、梯度流、设备、卸载还原 | 全过 |
| `test_recovery.py` | 截断恢复 vs 严格拒绝 | 全过 |
| `merge_shards.py` | 分片合并 + 不足时**拒绝写出** | 全过 |

其中 `test_lora.py` 特意跑在 CUDA 上——本机有 CUDA，所以这类设备 bug 以后能在本地抓住，
不必再烧 GPU 运行。

`test_lora.py` 里最有价值的三条断言（都印证了实现的正确性）：
- 零初始化时适配器是**精确 no-op**（`max|delta| = 0`）
- 第一步 `dL/dA` **恰为 0**、`dL/dB` 非零（零初始化 B 的必然结果）
- 第二步 `dL/dA` 变为非零 → **A 真的在训练**

---

## 4. 实测成本（决定全量预算）

| 项 | 实测 |
|---|---|
| 模型加载 | 68–169 s |
| Stage A（零样本生成）单题 | 4–61 s（简单题几秒；难题吃满时间片） |
| TTT 单步 | 3–60 s（取决于序列长度，最长 8192） |
| Stage B 单题（含 TTT） | 7–359 s |
| 5 题全流程 | 1079 s ≈ **216 s/题** |

**推论**：240 题 × 216 s ≈ **14.4 h**，超过单会话 12 h 上限 → **必须 2 分片**。
2 个并发会话各跑 120 题 → 约 7.2 h/会话，落在 12 h 内。

配额：**已用 0.95h / 剩 29.05h**（墙钟口径，2026-09-19 刷新）。

---

## 5. 符号引擎与保底层的定位（已精确界定）
### 5.1 符号引擎

实测（ARC-AGI-2 评测集全 120 题）：

```
sources: {'prior': 120}          ← 没有任何程序通过验证
correct: 0/120 = 0.000%
format problems: 0
tasks with a blank attempt: 0
0.034 s/task
```

### 5.2 神经网络保底层（新测量）

保底只对「神经路径完全失败」的任务生效。实测新旧两版在公开切分上的命中：

| 变体 | 评测集 172 输入 | 训练集 1076 输入 | `attempt_1 == attempt_2` |
|---|---|---|---|
| 旧（众数色填充 + 全零） | **0** | **0** | 60/172（35%）、752/1076（70%） |
| 新（恒等 + 前景/背景互换） | **0** | **0** | 0 |

两个结论：

1. 新实现对旧实现是**结构性改进**——旧版因为 `mode_colour` 把示范输入输出一起统计、
   背景 0 占绝对多数，两个 attempt 常返回**同一张图**，第二个槽纯属浪费。
2. 但**保底答案在任何一题上都从未正确（0%）**。恒等与前景/背景互换在 ARC-AGI-2 里
   都不是答案。

**因此「绝不空答」的价值是保证提交格式合法（格式非法的提交会被直接拒掉），
不是得分。全部分数只能来自神经候选。** 这直接把唯一杠杆锁定为
「在预算内让尽可能多的题拿到真实神经候选」。

### 5.3 附带的路线否决

把引擎做成 Kaggle 数据集的路子**走不通**：`www.googleapis.com:443`（GCS）在本机
网络不可达，dataset 文件上传必然超时。内核推送走 `api.kaggle.com` 所以正常。
鉴于引擎得分贡献为 0，这条路的损失可忽略。

---

## 5.4 真实留出准确率（唯一可信的分数证据）

运行：`arc26-eval-validate`，ARC-AGI-2 公开**评测集**（120 题全有答案）中
**均匀间隔抽取的 24 题**（`--num-shards 5 --shard-index 0`，避免字母序偏差），
T4x2 单会话，预算 5400 s。

| 指标 | 实测 |
|---|---|
| **准确率** | **1/24 = 4.17%** |
| 拿到真实神经候选的题数 | **24 / 24（覆盖率 100%）** |
| 错误数 | **0** |
| 峰值显存 | **14.09 GiB**（上限 14.6，未 OOM） |
| 单题耗时 | 202 s（校准值 169.2 s） |
| 模型加载 | 140.3 s |
| 求解总耗时 | 4844 s |

**怎么读这个数字**：4.17% 不高，但**覆盖率是 100%**——失败在**精度**，不在**覆盖**。
这与 §5.1/§5.2 的测量完全一致：符号层和保底层都恰好为 0，所以「有真实神经候选」
是唯一可能得分的题，而这次每一题都属于这一类。

### 5.5 ⚠ 重要更正：那次 4.17% 完全不含 TTT

评测日志末尾是：

```
[stage B] stopping: reserve reached (remaining=556s)
```

**Stage A（零样本扫描）吃掉了全部预算，Stage B（TTT 精修）从未进入。** 根因：
Stage A 把每题的时长算成「剩余**全部**预算的公平份额」：

```python
remaining_after_reserve = (remaining - reserve - stage_c_reserve) / tasks_left
slice_a = min(cheap_slice(index), remaining_after_reserve)
```

于是它必然吃光预算，Stage B 永远拿不到份额。**所以 4.17% 是「SFT 模型 + 约束 DFS +
NLL 重排 + 级联」的成绩，测试时训练贡献为 0。**

这是一处必须报告的缺陷，不是细节。已修复：新增 `STAGE_A_BUDGET_SHARE = 0.45`
与 `--stage-a-share`，把扫描限制在可用预算的 45%，其余留给 Stage B。

### 5.6 同一次日志暴露的第二个缺陷：重排 OOM

3 个任务上出现：

```
rescoring batch failed: CUDA out of memory. Tried to allocate 3.75 GiB
```

14.6 GiB 卡上整批重排失败 → **该批所有候选都没被打分，排序直接退化**。
已修复：把单批打分抽成递归函数，OOM 时二分重试直到单条候选。

### 5.7 第三个缺陷：时间片单位

时间片按**每个 test input** 计，所以有 2 个 test input 的题实际跑了约 270s，
而调度器以为只有 150s——这正是扫描能吃掉整个会话的原因。

**这三个缺陷 + §5.4 的 100% 覆盖率，构成了「精度 vs 覆盖」之外的第三条证据线：
子系统静默失效（stage 未执行、批次被丢弃、单位算错）不会被任何分数变化暴露，
只能靠读日志发现。**

参照：2025 冠军血统在同一基准上是 **33.89**，用 4×L4、每任务约 4 倍算力，
并且包含我们没有复现的合成数据 SFT 阶段（我们直接用公开的 Apache-2.0 权重）。

**推论：以这个算力预算，本方案的分数落在 4–6% 量级，拿不到任何奖金档。**
奖金路径始终是写作（Paper Track / Solution Writeup），代码提交的作用是**取得参赛资格**
（Writeup 必须挂在有效的代码提交上）。

---

## 6. 与基线的三点差异（用户要求「在现有方法上进一步提升」）

1. **绝不空答**：`attempt_1/attempt_2` 永远填满。三级保底：神经候选 →
   符号引擎（若加载）→ 启发式（训练示范众数颜色按测试输入形状填充）。
   并且做了两处针对性增强：DFS 超时**冲刷在飞 beam**、解析器**截断恢复**。
   这两处正是把 5 题冒烟从 2/5 拉到 4/5 的原因。
2. **级联调度**：先用零样本廉价扫全量（覆盖优先），剩余预算只花在低置信度题上，
   最后符号回填；每阶段按校准出的实测单题耗时动态分片，替代静态分区。
3. **混合候选池**：神经 DFS 候选 + 符号候选 + 增强投票候选合并，统一用
   teacher-forced NLL 重排，再取两个**互不相同**的答案。

---

## 7. 文档与工具地图

| 文件 | 内容 |
|---|---|
| `KAGGLE_GPU_RECIPE.md` | GPU 配方、实测矩阵、配额与并发、环境版本、复现命令 |
| `ARC26_SOLVER_SPEC.md` | 求解器完整规格（字母表/序列化/prompt/管线/超参/CLI/report schema） |
| `NEURAL_ROUTE_STATUS.md` | 本文件 |
| `solver/arc26_solver.py` | **求解器实现**（约 2500 行，自包含单文件） |
| `solver/test_*.py` | 4 套本地测试 |
| `kpush.py` | 推核工具（写 `machine_shape` + 注入 argv + stdout tee） |
| `decode_klog.py` | 解 Kaggle 的 JSONL 内核日志 |
| `flatten_nb.py` | 把 ipynb 拍平成可检索文本 |
| `merge_shards.py` | 分片合并（不足时拒绝写出） |
| `rehearse_engine_kernel.py` | 本地彩排符号引擎调用与格式校验 |
| `probe_accel_matrix.py` / `mount_probe.py` | GPU 与挂载结构探测 |

---

## 8. ⚠ 提交机制的关键约束（分片不可提交）

Kaggle 代码竞赛的评分方式是**对提交的 notebook 重新运行**，而不是拿我们本地产出的
`submission.json` 去算分。这一点第一次提交被拒时已经证实：当时 Kaggle 重跑内核，
dev 模式覆盖写出的 120 行版本被拿去评分，于是报
`wrong number of rows`。

**推论：分片内核永远不可提交。** 分片 0（`--num-shards 2 --shard-index 0`）单次只产出
120 题，被 Kaggle 重跑时会因为行数不符被直接拒掉。

因此最终提交物必须是**单个未分片、跑完全部 240 题的内核**。

这一条之所以可行，是因为求解器在**开始时就用保底答案预填全部 task_ids 并立即写盘**
（`for tid in task_ids: submission[tid] = fallback_attempts(...)` + `flush_submission`）。
所以即使 240 题的预算不够用，写出的 `submission.json` 仍然是
**240 题齐全、格式合法**的——只是超预算的那些题落到保底层。

预算换算：240 题 × 162 s = 10.8 h，落在 12 h 会话上限内。
单个未分片运行因此是本项目的**唯一**可提交形态，分片只用于测量。

---

## 9. 下一步

1. **评测集准确率**（进行中）：`arc26-eval-validate`，评测集均匀间隔 24 题，
   预算 5400 s。这是唯一的真实留出测量，Writeup 必需。
2. **可提交的全量运行**（关键）：**单个未分片**会话，
   `--split test --time-budget-seconds 39000`，240 题。等评测验证空出槽位后启动。
   这一个内核才是真正拿去提交的形态（见 §8）。
3. **分片 0**（在跑，不可提交）：以每任务 325 s 的宽裕预算覆盖真实隐藏测试集的一半，
   用于给出质量上限与逐题数据。CLI 无取消命令，故让它跑完。
4. **合并与提交**：取回全量运行的 `submission.json` → 本地校验 →
   `kaggle competitions submit -c arc-prize-2026-arc-agi-2 -k <kernel> -v <version> -f submission.json`
   （注意每日 1 次配额）。
5. **配额分配**：28.5h 可用 → 评测 1.5h + 分片0 约 11h + 全量约 11h ≈ 23.5h，
   余约 5h 用于补测。
6. 把神经路线结果补进 master brief 与 Writeup（含 `arc_prize_v1.py` 的
   0/120 诚实结论）。
