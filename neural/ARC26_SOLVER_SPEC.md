# ARC-AGI-2 求解器规格（Option C：许可干净重实现）

> 本文件是实现依据。**所有事实均已在本机验证**，来源标注在每节末尾。
> 实现者请严格按本规格写代码，不要引入未标注来源的外部代码。

---

## 0. 目标与硬约束

**目标**：在 Kaggle 上跑通一个测试时训练（TTT）求解器，对竞赛
`arc-prize-2026-arc-agi-2` 的 240 个隐藏测试任务产出 `submission.json`，
每个 task 的每个 test input 给 2 个 attempt。

**硬约束（实测）**

| 约束 | 值 | 后果 |
|---|---|---|
| GPU 型号 | 只能拿到 **T4 x2**（`machine_shape: "NvidiaTeslaT4"`） | L4 不可得，P100 是死核 |
| 显存 | 2 × 14.6 GiB，**单卡 14.6 GiB 是真实上限** | 3.63B bf16 权重占 7.3 GB，必须梯度检查点 |
| 单会话墙钟 | 12 h（`maxGpuRuntimeMinutes=720`） | 必读 `--time-budget-seconds` |
| 周配额 | 30 h（当前剩 29.9 h） | 只够 2 个满会话 + 6 h |
| 并发会话 | **最多 2 个** | 可 2 分片并行 |
| 每日提交 | 1 次 | 调试绝不消耗提交 |
| 网络 | 竞赛 notebook 无网 | 不能用 `pip install`，只能用镜像内的包 |

**可用包**：`torch 2.10.0+cu128`、`transformers 5.0.0`、`peft 0.19.1`、
`accelerate 1.13.0`、`datasets 5.0.0`。
**不可用**：`unsloth`、`trl`、`bitsandbytes`。

**许可合规（用户明确要求）**：只允许使用
- Apache-2.0 权重 `sorokin/qwen3_4b_grids15_sft139`
- 我们自己写的代码
**禁止**复制 `1ytic/NVARC`（无 license）或任何 fork 的实现代码。
允许阅读格式与超参事实，实现必须原创。

---

## 1. 模型与 I/O 格式（关键，全部实测）

### 1.1 权重位置与规格

```
/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1
```

- 架构 `Qwen3ForCausalLM`：hidden 2560，36 层，32 attn heads / 8 KV heads，
  head_dim 128，intermediate 9728，bf16，`max_position_embeddings` 262144
- **参数量 3.634 B**（词表被缩到 16，省掉 embedding 的大头）
- 加载耗时实测 **62.8 s**（T4）
- 文件：`model-00001-of-00002.safetensors` (4.997 GB) + `model-00002-of-00002.safetensors` (2.270 GB)
- 无 `chat_template`

*来源：Stage 0 探针内核 `luoguoqiang/arc26-stage0-model-introspect` 实测输出。*

### 1.2 词表（16 个 token）

| id | 含义 | 常量 |
|---|---|---|
| 0–9 | ARC 的 10 种颜色，**一个格子 = 一个 token** | — |
| 10 | `Ċ` = 换行符 `\n`（行分隔） | — |
| 11 | `<\|im_start\|>user\n`（user 轮标记，**单 token**） | `USER_TOKEN_ID = 11` |
| 12 | `<\|im_start\|>assistant\n`（assistant 轮标记，**单 token**） | `ASSISTANT_TOKEN_ID = 12` |
| 13 | `<\|endoftext\|>` = **PAD** | `PAD_ID = 13` |
| 14 | `<\|im_start\|>`（未使用） | — |
| 15 | `<\|im_end\|>` = **EOS** | `EOS_ID = 15` |

**约束解码用的 12 个 token** = ids `[0,1,2,3,4,5,6,7,8,9,10,15]`
（10 色 + 换行 + EOS）。

*来源：`ARC_VOCAB` / `PAD_ID` / `EOS_ID` 常量定义，以及 Stage 0 的
`tokenizer_config.json` + `added_tokens_decoder` 实测。*

### 1.3 网格序列化（必须逐字节一致）

```python
def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))      # 每格一个数字字符
        text += "\n"                    # 每行一个换行
    return text.strip()                 # 去掉末尾换行
```

### 1.4 Prompt 格式

```python
def fmt_query(test_input_grid) -> str:      # 用户轮 + 助手提示符
    return USER + convert_grid_to_string(test_input_grid) + EOS + ASSISTANT

def fmt_reply(output_grid) -> str:          # 助手答案
    return convert_grid_to_string(output_grid) + EOS

# 完整 prompt = 所有示范轮 + 最后一轮 user（测试输入）+ assistant 提示符
def fmt_train(demos, test_input_grid) -> str:
    text = ""
    for d in demos:
        text += USER + convert_grid_to_string(d["input"]) + EOS
        text += ASSISTANT + convert_grid_to_string(d["output"]) + EOS
    text += fmt_query(test_input_grid)
    return text
```
其中 `USER`/`ASSISTANT` 是上表的单 token（编码整串时 tokenizer 会产出 11/12）。

### 1.5 解码解析

```python
def tokens_to_array(tokens, limit_rows=30):
    # 丢掉最后一个 token（EOS），再 decode
    text = tokenizer.decode(tokens[:-1])
    lines = text.strip().split("\n")
    rows = [[int(ch) for ch in line if ch.isdigit()] for line in lines]
    rows = [r for r in rows if r]
    if len(rows) > limit_rows: rows = rows[:limit_rows]
    arr = np.array(rows, dtype=int)
    if arr.ndim == 2 and arr.size and all(1 <= s <= 30 for s in arr.shape):
        return arr
    return None
```

**最大生成长度**：
```python
max_new_tokens = len(tokenizer.encode(fmt_reply(np.zeros([30,30],dtype=int)))) + 1
```
≈ 930（30×30=900 个数字 + 29 个换行 + 1 个 EOS）。

*来源：`convert_grid_to_string` / `QwenFormatter` / `convert_tokens_to_array` /
`max_new_tokens` 的实现语义，已逐行核对。*

---

## 2. 算法管线

### 2.1 每个 task 的流程

```
对每个 puzzle（task）：
  1. 取 train 示范 + test 输入列表（1~2 个）
  2. 构造 TTT 数据集：对示范做增强（几何 + 颜色置换），n=16 起步
     - 截断到 max_seq_length = 8192
  3. 挂 LoRA，在增强后的示范上做 TTT（1 epoch，lr 5e-5）
  4. 对每个 test 输入：
     a. 用约束 DFS 生成候选（见 2.3）
     b. 多增强投票：对若干增强分别生成候选
     c. 用 teacher-forced NLL 重排（见 2.4）
     d. 取 NLL 最低的两个**互不相同**的候选 → attempt_1 / attempt_2
  5. 卸载 LoRA，写结果
  6. 时间检查：超预算立刻停止并固化已算出的结果
```

### 2.2 TTT 超参（工作解的既有取值，作为起点）

| 项 | 值 |
|---|---|
| LoRA r / alpha / dropout | 16 / 32 / 0.0 |
| `use_rslora` | True |
| target modules | 全部 attention + MLP 投影（`q,k,v,o,gate,up,down_proj`） |
| lr | 5e-5 |
| epochs | 1 |
| per-device batch | 1 |
| max_grad_norm | 1.0 |
| precision | bf16 |
| max_seq_length | 8192 |
| 训练增强数 n | 16（`shfl_keys=True`） |
| **梯度检查点** | **必须开**（14.6 GiB 装不下 8192 长度的全激活） |

### 2.3 约束 DFS 候选生成 `turbo_dfs`

递归 + 批处理的 beam DFS：

```python
def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time):
    # 只看 12 个 ARC token 的 logits：先 log_softmax，然后取 ARC_TOKENS 上的 12 个值
    # 对每个 beam：把 logprob > max_score 的 token 保留为新分支
    #   score = 父累计 logprob + 本 token logprob
    # 若 max_new_tokens > 1 则递归（用 KV cache）
    # 若无存活分支则回退（用 PAD 占位，score=1000 表示死）
    # 每次递归前检查 time.time() > end_time，超时立即返回
    # 返回 {batch_id: [(score, tokens), ...]}
```

**关键实现要点（性能）**：full-vocabulary logits 的归一化留在 GPU，
只把 12 个 token 的结果传回 CPU。这是避免 CPU↔GPU 传输成为瓶颈的核心。

**阈值**：`max_score = -np.log(0.2)`，即逐 token 概率 > 0.2 才保留。

### 2.4 重排 `calc_scores`

teacher-forced NLL：

```python
def calc_scores(queries, answers, tokenizer, model):
    # 对每个 (query, answer)：拼接，右边 pad 到 batch 内最大长度（pad 用 PAD_ID）
    # 一次前向，取 logits.float()
    # log_norm = logsumexp(logits, dim=-1)         # 全词表归一化
    # nll = -(logits[target_tokens] - log_norm) 的和，只对 answer 位置
    # 返回每个候选的平均 NLL（越低越好）
```

### 2.5 增强（`augment`）

对 task 的所有网格**一致地**施加一个由随机 key 描述的组合变换，生成新示范；
推理时生成候选后再施加**逆变换**得到答案。可用的原子操作：
`rot90`、水平/垂直翻转、转置、以及 10 色的随机置换（`permute` + 数字描述符）。

### 2.6 提交格式与「绝不空答」

```python
submission = {
  task_id: [ {"attempt_1": grid, "attempt_2": grid} for each test input ]
}
```
**初始化时必须填占位网格**（例如 `[[0]]` 或训练示范的众数颜色全图），
而不是留空 —— 覆盖率为 0 的任务必须仍有一个合法网格。
这是从社区教训里得到的：过度保守的阈值会让第二 attempt 直接放弃。

---

## 3. 我们要超越基线的三点（用户明确要求）

### 改进 1：绝不空答 + 保底候选池
每个 test input 永远填满 `attempt_1` / `attempt_2`。
保底顺序：① 神经候选 → ② **自有符号引擎**（`arc_prize_v1.py`，已验证程序
96.3–100% 精确）→ ③ 启发式（训练示范众数颜色填充成 test 输入的形状）。
基线在阈值保守时会退化成 `[[0]]`；我们把保底换成符号引擎输出。

### 改进 2：动态调度，替代静态分区
基线把 12 h 静态切分给不同阶段，实测浪费了约 12 h。
本实现按**剩余任务数 × 单任务实测耗时**动态分配：
- 先跑一个**校准阶段**（前 ~5 个任务），测出真实的「每任务秒数」
- 用 `剩余时间 / 每任务秒数` 决定后续每个任务的时间片
- 时间片随剩余预算收缩；超预算的任务立即用保底答案固化
- 绝不出现「预算耗尽但任务没写」的情况

### 改进 3：混合候选池
`turbo_dfs` 的神经候选 + 符号引擎候选 + 增强投票候选合并进**同一个池**，
统一用 teacher-forced NLL 重排，再取两个互不相同的答案。
基线的 attempt_2 常是同一候选的噪声变体；混合池能显著提高 pass@2 的独立性。

---

## 4. 交付物与接口

**单一自包含文件**：`work/arc_w1/solver/arc26_solver.py`

必须支持的命令行参数：
```
--limit N                 只跑前 N 个 task（用于冒烟测试，必须支持；0 = 全部）
--split NAME              数据切分：test | training | evaluation，默认 test
                          - test       -> arc-agi_test_challenges.json（无答案，正式提交用）
                          - training   -> arc-agi_training_challenges.json + _solutions.json
                          - evaluation -> arc-agi_evaluation_challenges.json + _solutions.json
                          有答案时必须在 report.json 里逐题给出是否正确，作为链路校验
--time-budget-seconds S   总时间预算，默认 39000（10h50m，留足 12h 余量）
--calibrate-tasks K       校准阶段任务数，默认 5
--aug-train N             TTT 增强数，默认 16
--aug-infer N             推理增强数，默认 4
--out PATH                submission.json 输出路径，默认 /kaggle/working/submission.json
--report PATH             report.json 输出路径，默认 /kaggle/working/report.json
--data-dir PATH           竞赛数据目录，默认 /kaggle/input/arc-prize-2026-arc-agi-2
--engine PATH             自有符号引擎路径（可选；缺失时自动跳过）
```

**⚠ `SystemExit` 陷阱**：在 Jupyter cell 里 `__name__` **就是** `"__main__"`，
所以在文件底部必须写成
```python
if __name__ == "__main__":
    main()
```
**绝对不要写 `raise SystemExit(main())`** —— 在 notebook 里抛 `SystemExit`
会让 cell 被判定为失败，整个 Kaggle 运行变成 ERROR（即使结果已经正确写出）。
`main()` 内部用 `return` 表达退出码即可，不要让异常逃逸出 cell。


**必须产出 `/kaggle/working/report.json`**，含：
```json
{
  "started_at": ..., "finished_at": ..., "elapsed_seconds": ...,
  "n_tasks_total": 240, "n_tasks_done": ...,
  "calibration": {"tasks": 5, "seconds_per_task": ...},
  "per_task": [{"task_id":..., "seconds":..., "n_candidates":...,
                "attempts":["neural","neural"], "source":"neural|symbolic|fallback"}],
  "gpu": "...", "torch": "...", "peak_mem_gb": ...
}
```
报告写到 `/kaggle/working` 是**唯一可靠的取回通道**（成功运行的执行日志不会发布）。

**健壮性要求**：
- 任何单个 task 抛异常 → 记录到 report 并用保底答案继续，**绝不整体崩溃**
- 全局 `try/finally` 保证 `submission.json` 一定被写出
- 定期（每完成 10 个 task）增量写 `submission.json`，防止超时丢结果
- 时间预算到 → 立即停止并写出当前结果

---

## 5. 冒烟测试要求（必须先通过再上全量）

1. **本地**：`python -m py_compile arc26_solver.py` 必须通过。
2. **Kaggle 单次小跑**：`--limit 5 --time-budget-seconds 1800 --aug-train 4 --aug-infer 1`
   - 必须在 report.json 里给出 `seconds_per_task`
   - 必须能对至少 1 个**训练集已知答案**的任务给出正确结果（用于验证格式链路）
   - 必须打印每个 task 的 `attempt_1` 与真值是否相等
3. 通过后再跑全量 `--limit 0`（= 全部 240）。

---

## 6. 环境事实速查

```
torch 2.10.0+cu128   transformers 5.0.0   peft 0.19.1   accelerate 1.13.0
GPU: 2 x Tesla T4 (sm_75), 14.6 GiB each
unsloth / trl / bitsandbytes: MISSING
```

推送内核用 `work/arc_w1/kpush.py`（务必走 metadata 的 `machine_shape`）。
取回结果用 `python -m kaggle kernels output <slug> -p <dir>`。

---

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 8192 长度 + 3.63B 在 14.6 GiB 上 OOM | 开梯度检查点；必要时把 max_seq_length 降到 4096 并只喂部分示范 |
| DFS 候选爆炸导致单任务超时 | 阈值 `-log(0.2)` + 每任务硬时间片 + 超时取当前最优 |
| T4 上 bf16 慢 | 若实测慢，切 fp16（Turing 有原生 fp16 张量核） |
| 单会话 12h 跑不完 240 题 | 用第 2 个并发会话跑第 2 分片（120+120） |
| 生成格式错误导致 0 分 | 已用 `is_valid_solution` 校验；每任务落盘前校验形状 1–30 |
| 拿错 GPU（P100 死核） | 只在 metadata 写 `machine_shape`，跑完立刻核对日志里的 GPU 名 |
