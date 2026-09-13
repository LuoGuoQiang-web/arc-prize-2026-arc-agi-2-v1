# A Licence-Clean Test-Time-Training Solver for ARC-AGI-2, and Why the Symbolic Floor Is Exactly Zero

**Track:** ARC-AGI-2 Solution Writeup (companion to the Paper Track diagnostic
*"When Cheap Transformations Vanish"*).

**Code (MIT-0):** https://github.com/LuoGuoQiang-web/arc-prize-2026-arc-agi-2-v1
**Notebook:** https://www.kaggle.com/code/luoguoqiang/arc26-submit-full

**Status of this draft:** every number below is measured. The one placeholder is marked
`[[EVAL_ACCURACY]]` and is filled from the held-out evaluation run.

---

## 1. Two measured facts that set the whole design

We began from a symbolic solver — program search over a fixed vocabulary of grid
transformations — and measured it honestly before building anything on top of it.

**(a) The symbolic floor is zero, not small.** On the full 120-task ARC-AGI-2 public
evaluation split the engine solved **0/120 (0.000%)**. Not "few": zero validated
programs; every one of the 120 tasks fell through to an unvalidated prior. The same
engine scores 4.833% on the training split, so the collapse is a distribution shift,
not an implementation defect. This corroborates, on the 2026 data, the community
finding that BARC-style search-and-learn solves ~0.8% of ARC-AGI-2 private tasks
against ~22% on ARC-AGI-1.

**(b) The cheap fallback is also zero.** Only a task the neural path fails to answer
ever reaches the fallback layer, so its quality matters at scale. We measured two
generations of it on the public splits:

| fallback variant | eval hits (172 test inputs) | train hits (1076) | `attempt_1 == attempt_2` |
|---|---|---|---|
| mode-colour fill + all-zeros | 0 | 0 | 35% / 70% |
| identity + fg/bg swap (final) | 0 | 0 | 0% |

The second version fixes a real structural bug — the first returned the *same* grid in
both slots on 70% of training inputs, wasting the second attempt — but neither version
ever produces a correct answer. Constant fills, the identity, and the foreground /
background swap are simply never the answer on this benchmark.

**Consequence.** Every point must come from the model. Coverage of genuine neural
candidates is the only lever, and "never leave an attempt blank" is a *format-validity*
property (a malformed submission is rejected outright), not a scoring strategy.

---

## 2. Method

### 2.1 Base model and its 16-token alphabet

We use `sorokin/qwen3_4b_grids15_sft139` (**Apache-2.0**), a Qwen3-4B fine-tune whose
vocabulary has been reduced to **16 tokens**: ids 0–9 are the ten ARC colours, id 10 is
a newline, ids 11/12 are the user/assistant turn markers, 13 is padding and 15 is EOS.
A grid is serialised cell-by-cell, one digit token per cell and one newline per row.
Constrained decoding is therefore over 12 tokens (ten colours + newline + EOS), and the
longest possible answer is 930 tokens.

This is a much easier object to search than a full-vocabulary LLM: the model can only
ever emit legal grid symbols, so no output can be syntactically invalid.

### 2.2 Our own LoRA, not `peft`

The Kaggle image ships `torchao 0.10.0`, which makes `peft 0.19.1` raise on import:

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

Competition notebooks have no internet, so this is not fixable by installing anything.
We implement LoRA directly: a `LoRALinear` wrapper computes `Wx + scale·B(Ax)` with `A`
kaiming-initialised and `B` zero, so the adapter starts as an exact no-op and the SFT
behaviour is preserved until training moves it. `scale` follows rsLoRA
(`alpha/√r = 32/4 = 8`).

This is ~40 lines and no dependency, which also makes the whole submission
licence-clean: an Apache-2.0 checkpoint plus code we wrote ourselves.

Two properties matter for correctness and are covered by local tests: on the first step
`∂L/∂A` is *exactly* zero because `B = 0` (so a single step would silently never train
`A`), and it becomes non-zero on the second step. Adapters are created on the base
weight's device — allocating them afterwards leaves them on CPU while
`device_map="cuda:0"` has already placed the model, and every step then dies with a
device-mismatch error.

### 2.3 Per-task test-time training

For each task we build augmented demonstrations (dihedral transforms plus colour
permutations), rotating which demonstration is the target so that every demonstration
is predicted under a different augmentation. The model is trained for one epoch on
those sequences with LoRA only, bf16 autocast, lr 5e-5, batch 1, grad-norm 1.0 and
gradient checkpointing (mandatory: a 3.63B model at sequence length 8192 does not fit
14.6 GiB otherwise). Training stops between optimiser steps once the task's time slice
expires.

### 2.4 Constrained beam DFS with two coverage guards

Candidates are produced by a batched beam DFS over the 12 allowed tokens, keeping a
token when its per-token probability exceeds 0.2 and always keeping the arg-max when
nothing clears the bar. A teacher-forced NLL re-scores every candidate, including ones
from other sources.

Two guards exist purely for coverage, and both came from measurement:

- **Timeout flush.** The first implementation discarded its in-flight beams on timeout.
  Hard tasks — exactly the ones that need more than their slice to emit a 930-token
  grid — therefore returned an *empty* candidate pool. Flushing the live beams instead
  turned three empty pools into pools of 2, 1 and 8.
- **Truncation recovery.** A beam cut off mid-row parses as ragged, and the strict
  parser rejected it, discarding the flushed candidates. Recovery truncates at the
  first incomplete row. Strict parsing remains the default for validation.

Together these moved a five-task smoke run from **2/5 to 4/5 correct**, with no change
to the model or the training loop.

### 2.5 Cascade scheduling over a fixed 12-hour session

A Kaggle GPU session is capped at 12 hours and the weekly budget at 30 hours, with at
most two concurrent GPU sessions. Rather than partition the budget statically — a
documented way to waste most of a session — we run three stages: a cheap no-TTT sweep
over every task to establish coverage; TTT plus regeneration for the tasks with the
weakest confidence signal, until the budget is spent; and a symbolic/heuristic backfill.
Per-task cost is calibrated from the first few tasks and re-allocated as the run
progresses. `submission.json` is pre-seeded with a legal fallback for every task and
flushed immediately, so the run always emits a complete, format-valid 240-task file even
if it is killed at the wall clock.

---

## 3. Results

**Held-out accuracy: 4.17% (1/24)** on a uniformly spaced 24-task sample of the
ARC-AGI-2 public evaluation split, in a single 8-hour-class T4 x 2 session
(4844 s of solving after a 140 s model load, 202 s per task, peak 14.09 GiB).

The number is modest and we do not dress it up. The more informative result is the one
next to it: **24 of 24 tasks received a genuine neural candidate**. Coverage is
complete; the failure is in *precision*, not in reach. That is exactly the split the
symbolic and fallback measurements predicted — since both of those layers are worth
exactly zero, a task with a real candidate is the only kind of task that can ever score,
and here every task is that kind.

For scale: the same benchmark was scored at 33.89 by the 2025-winning lineage, which
used 4 x L4 and roughly four times our compute per task, plus a synthetic-data SFT
stage we did not reproduce (we use the published Apache-2.0 checkpoint instead).

Smoke run, five training tasks, same session shape: **4/5 correct**, ~216 s per task —
but those are the first five tasks in key order and are markedly easier than the
evaluation sample, so 4/5 should not be read as an accuracy estimate.

---

## 4. What this says

The honest summary is that the symbolic route is closed on ARC-AGI-2 — measured at
exactly zero on 120 public evaluation tasks — and that the cheap fallback layer is also
exactly zero, so no amount of calibration at the edges moves the score. The only
working lever is whether a genuine neural candidate exists for a task.

Our measurements separate those two questions cleanly. Coverage engineering — timeout
flushing, truncation recovery, cascade scheduling, never emitting a malformed file —
took the pipeline from 2/5 to 4/5 on smoke tasks and to 24/24 candidate coverage on the
evaluation sample, with no change to the model or the training loop. It does not, on its
own, produce accuracy: the remaining gap is precision, and closing it needs either
substantially more compute per task or a better selection signal than the candidate
agreement we currently use.

We therefore claim a reproducible, licence-clean pipeline with its failure modes
measured rather than assumed, two specific coverage bugs found by measurement and fixed,
and an honest floor for what this class of solver achieves at this compute budget — not
competitiveness.
