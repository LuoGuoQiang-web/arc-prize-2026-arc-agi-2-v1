"""Exercise the self-implemented LoRA path without needing the 7 GB SFT weights.

Checks the properties that actually matter for TTT correctness:
  1. wrapping freezes every base parameter
  2. the only trainable parameters are the LoRA factors
  3. with B initialised to zero the adapter is an exact no-op
  4. one optimizer step changes the output (i.e. gradients really flow)
  5. detach restores the original nn.Linear objects and drops the adapters
  6. it survives the bf16 autocast context that run_ttt uses
  7. a module with no gradient-checkpointing API still wraps (with a warning)
"""
import importlib.util
import sys

import torch

SOLVER = r"C:\Users\Administrator\Desktop\Kaggle\work\arc_w1\solver\arc26_solver.py"
spec = importlib.util.spec_from_file_location("arc26_solver", SOLVER)
m = importlib.util.module_from_spec(spec)
sys.modules["arc26_solver"] = m
spec.loader.exec_module(m)

D = 64
failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


class MiniBlock(torch.nn.Module):
    """Mimics the Qwen3 projection names the solver targets."""

    def __init__(self, d=D):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d)
        self.k_proj = torch.nn.Linear(d, d)
        self.v_proj = torch.nn.Linear(d, d)
        self.o_proj = torch.nn.Linear(d, d)
        self.gate_proj = torch.nn.Linear(d, d)
        self.up_proj = torch.nn.Linear(d, d)
        self.down_proj = torch.nn.Linear(d, d)
        self.not_targeted = torch.nn.Linear(d, d)

    def forward(self, x):
        # Use every wrapped projection so each adapter actually receives a gradient.
        return (self.q_proj(x) + self.k_proj(x) + self.v_proj(x) + self.o_proj(x)
                + self.gate_proj(x) + self.up_proj(x) + self.down_proj(x))


class MiniModel(torch.nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.block = MiniBlock(d)
        self.gc_enabled = False

    def forward(self, x):
        return self.block(x)

    # transformers-style API
    def gradient_checkpointing_enable(self, **kw):
        self.gc_enabled = True

    def gradient_checkpointing_disable(self):
        self.gc_enabled = False

    def enable_input_require_grads(self):
        pass


torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEV}")

model = MiniModel().to(DEV)
base_originals = {n: mod for n, mod in model.named_modules()
                  if isinstance(mod, torch.nn.Linear)}
base_snapshot = {n: mod.weight.detach().clone() for n, mod in base_originals.items()}

x = torch.randn(2, 5, D, device=DEV)
with torch.no_grad():
    before = model(x).clone()

returned = m.attach_lora(model)
check("attach_lora returns the same object", returned is model)
check("gradient checkpointing was enabled", model.gc_enabled)

# Regression guard for the Kaggle failure:
#   Expected all tensors to be on the same device, but got mat2 is on cpu
# where freshly created LoRA params stayed on CPU while the base lived on cuda.
lora_params = [p for p in model.parameters() if p.requires_grad]
check("every LoRA factor is on the base model's device",
      all(p.device.type == torch.device(DEV).type for p in lora_params),
      f"devices={sorted({str(p.device) for p in lora_params})}")

frozen = [n for n, p in model.named_parameters() if p.requires_grad]
check("all trainable params are LoRA factors",
      all(n.endswith(("lora_A", "lora_B")) for n in frozen), f"{len(frozen)} tensors")
n_wrapped = sum(1 for _n, mod in model.named_modules() if isinstance(mod, m.LoRALinear))
check("expected number of modules wrapped", n_wrapped == 7, f"wrapped={n_wrapped}")
check("non-targeted linear left alone",
      not isinstance(model.block.not_targeted, m.LoRALinear))

with torch.no_grad():
    after_attach = model(x)
check("adapter is an exact no-op at init (B == 0)",
      torch.allclose(before, after_attach, atol=0, rtol=0),
      f"max|delta|={(before - after_attach).abs().max().item():.3e}")

# One real optimizer step under the same autocast context run_ttt uses.
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-5)
model.train()
with torch.autocast(DEV, dtype=torch.bfloat16):
    out = model(x)
    loss = out.float().pow(2).mean()
loss.backward()
grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
check("every LoRA factor received a finite gradient", grads_ok)

# Property of zero-initialised B: on the very first step dL/dA is exactly zero because
# it is scaled by B. This is expected, not a bug -- but it must become non-zero once B
# has moved, otherwise TTT would never train A at all.
a_grads_step1 = [p.grad for p in params if p.shape[0] == m.LORA_R]
b_grads_step1 = [p.grad for p in params if p.shape[1] == m.LORA_R]
check("step 1: dL/dA is exactly zero (B starts at 0)",
      all(g is not None and torch.count_nonzero(g) == 0 for g in a_grads_step1),
      f"{len(a_grads_step1)} A-factors")
check("step 1: dL/dB is non-zero",
      all(g is not None and torch.count_nonzero(g) > 0 for g in b_grads_step1),
      f"{len(b_grads_step1)} B-factors")

torch.nn.utils.clip_grad_norm_(params, 1.0)
opt.step()
opt.zero_grad(set_to_none=True)

# Second step: now that B != 0, A must receive real gradient.
with torch.autocast(DEV, dtype=torch.bfloat16):
    model(x).float().pow(2).mean().backward()
check("step 2: dL/dA becomes non-zero once B has moved",
      all(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in params
          if p.shape[0] == m.LORA_R))
opt.zero_grad(set_to_none=True)
model.eval()

with torch.no_grad():
    after_step = model(x)
check("one step changes the output", not torch.allclose(before, after_step, atol=1e-6),
      f"max|delta|={(before - after_step).abs().max().item():.3e}")

check("frozen base weights were not modified",
      all(torch.equal(base_snapshot[n], mod.weight)
          for n, mod in base_originals.items()))

m.detach_lora(model)
n_left = sum(1 for _n, mod in model.named_modules() if isinstance(mod, m.LoRALinear))
check("detach removed every adapter", n_left == 0, f"left={n_left}")
check("detach restored the original nn.Linear objects",
      all(model.get_submodule(n) is mod for n, mod in base_originals.items()))
check("gradient checkpointing disabled again", not model.gc_enabled)
with torch.no_grad():
    after_detach = model(x)
check("detached model again matches the pre-attach output",
      torch.allclose(before, after_detach, atol=0, rtol=0))


# A module with no gradient-checkpointing API must still wrap (warning path).
class Bare(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(D, D)

    def forward(self, x):
        return self.q_proj(x)


bare = Bare().to(DEV)
m.attach_lora(bare)
check("bare module (no checkpointing API) still wraps",
      isinstance(bare.q_proj, m.LoRALinear))
check("bare module has exactly 2 trainable factors",
      sum(1 for _n, p in bare.named_parameters() if p.requires_grad) == 2)
check("bare module adapters on the right device",
      all(p.device.type == torch.device(DEV).type
          for p in bare.parameters() if p.requires_grad))

print()
if failures:
    print(f"FAILURES: {failures}")
    raise SystemExit(1)
print("all LoRA checks passed")
