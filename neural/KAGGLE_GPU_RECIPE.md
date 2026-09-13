# Kaggle GPU 选择：实测结论与配方

> 全部结论来自本机对 Kaggle API 的真实探测（2026-09-13，账号 `luoguoqiang`），
> 不是文档转述。探针内核日志保存在 `work/arc_w1/accel_matrix/out/`。

## 一句话配方

在 `kernel-metadata.json` 里写：

```json
{
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4"
}
```

然后 `kaggle kernels push -p <folder>` → 稳定拿到 **GPU T4 x2**（2 × Tesla T4, sm_75, 14.6 GiB each）。

## 三个必须知道的坑

### 1. `enable_gpu: true` 单独用 = 死核

默认落到 **Tesla P100-PCIE-16GB（sm_60）**。当前 Kaggle 镜像里的
`torch 2.10.0+cu128` 只编译了 sm_70/75/80/86/90/100/120，于是任何 CUDA 运算直接抛：

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

即 `torch.cuda.is_available()` 返回 `True`、`get_device_properties` 也能读出名字，
**但一算就炸**。这是一个会静默吞掉整次提交的陷阱。

### 2. CLI 的 `--accelerator` / `--acc` flag 不被服务器采纳

CLI 源码里它只是被塞进 `machine_shape`：

```python
# kaggle/api/kaggle_api_extended.py:6450
request.machine_shape = acc if acc else self.get_or_default(meta_data, "machine_shape", None)
```

但实测 `--accelerator NvidiaTeslaT4` 推上去仍然跑在 P100。
Kaggle 官方 issue #821 的评论 5304836116 由第三方独立复现并明确纠正：

> `--accelerator` / `--acc` did NOT work for us — `machine_shape` in metadata did (verified Aug 2026)

**结论：只在 metadata 里写 `machine_shape`，不要依赖 flag。**

### 3. 无效的 `machine_shape` 会被静默归一化，不报错

服务器把无法识别的值归一化成 `Gpu`（= 默认 GPU = P100），push 本身**返回成功**。
判断方法：`kaggle kernels pull <slug> -m`，读回来的服务端 `machine_shape` 就是真实生效值。

实测矩阵：

| 写入的 `machine_shape` | push 结果 | 服务端存回 | 实际 GPU | 可用 |
|---|---|---|---|---|
| `NvidiaTeslaT4` | 成功 | `NvidiaTeslaT4` | 2 × Tesla T4, sm_75 | **是** |
| `NvidiaL4` | 成功 | `Gpu` | Tesla P100, sm_60 | 否 |
| `NvidiaL4x4` | 成功 | （同 slug v1） | Tesla P100, sm_60 | 否 |
| `NvidiaL4X4` | 成功 | `Gpu` | Tesla P100, sm_60 | 否 |
| `nvidiaL4X4` | 成功 | `Gpu` | Tesla P100, sm_60 | 否 |
| `enable_gpu` only | 成功 | `Gpu` | Tesla P100, sm_60 | 否 |

**L4 的 API 拼写无任何公开先例**：Kaggle/cli issue #821 全部 21 条评论中没有一条提到 L4；
`NvidiaL4`（按 `nvidiaTeslaT4` ↔ `NvidiaTeslaT4` 的大小写映射规律最应当正确的那个）
实测被拒。推测是 L4x4 需要账号级 entitlement，本账号没有。
**不再在 L4 上花额度。**

## 配额与并发

```
$ python -m kaggle quota
resource  used   remaining  total   refreshAt
GPU       0.04h  29.96h     30.00h  2026-09-19T00:00:00
TPU       0.00h  20.00h     20.00h  2026-09-19T00:00:00
```

- **并发上限 2 个 GPU 会话**。实测第 3 个 push 直接失败：
  `Kernel push error: Maximum batch GPU session count of 2 reached.`
- 单会话墙钟上限 12h（竞赛 `maxGpuRuntimeMinutes=720`）。
- 30h ÷ 12h = **2 个满会话 + 6h 余量**。
- 计时口径：跑 3 次约 1 分钟的内核后 `used` 仅 0.04h ≈ 2.4 min，接近墙钟而非
  GPU·小时，且 T4x2 的那次未见翻倍。**按墙钟规划**（T4x2 一次 12h 会话 ≈ 24 T4·小时）。

## 算力对照（用于给方案定预算）

| 设备 | fp16 tensor | 相对 1×L4 |
|---|---|---|
| 1 × L4 (sm_89) | 121 TFLOPS | 1.00 |
| **2 × T4 (sm_75)** | 65 × 2 = 130 TFLOPS | **1.07** |
| 4 × L4（2025 冠军配置） | 484 TFLOPS | 4.00 |

即 **T4x2 ≈ 1 张 L4**，是冠军 4×L4 的约 1/4 算力。
但权重已是 SFT 成品（省掉冠军 4×H100 跑 27h 的合成数据+SFT 训练），
且自有符号引擎承担一部分任务，所以预算仍然可用。

## 环境包版本（默认 GPU 镜像实测）

```
torch 2.10.0+cu128     (sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120)
transformers 5.0.0
peft 0.19.1            <-- LoRA TTT 够用
unsloth   MISSING
trl       MISSING
bitsandbytes MISSING
```

因为 `peft` 在、`unsloth`/`trl`/`bitsandbytes` 不在，per-task LoRA TTT 走
**`transformers` + `peft`**，不做 4-bit 量化（无 bitsandbytes），改用 bf16/fp16。
bf16 matmul 在 T4 上实测可跑（`bf16 matmul ok`），但 Turing 无原生 bf16 张量核，
实战用 **fp16** 更快。

## 复现方式

```powershell
# 推任意脚本为单 cell notebook，并强制写入 machine_shape
python work/arc_w1/kpush.py <source.py> --slug <owner/slug> --title "..." `
  --model <model-source> --competition arc-prize-2026-arc-agi-2

# 读回服务端真实生效的 machine_shape
python -m kaggle kernels pull <owner/slug> -m -p <dir>

# 读日志
python -m kaggle kernels output <owner/slug> -p <dir>
```

探测脚本：`work/arc_w1/probe_accel_matrix.py`、`work/arc_w1/find_accel.py`、`work/arc_w1/accel_probe.py`
