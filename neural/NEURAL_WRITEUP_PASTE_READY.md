# Paste-ready: ARC-AGI-2 Solution Writeup

> 直接整段粘贴到 Kaggle Writeup 编辑器即可，正文无需改动。
>
> **Title**
> A Licence-Clean Test-Time-Training Solver for ARC-AGI-2, and Why the Symbolic Floor Is Exactly Zero
>
> **Subtitle**
> Coverage, not search, is the binding constraint: we measure the symbolic floor at exactly 0/120, the fallback layer at exactly 0, and a licence-clean LoRA test-time-training pipeline at 4.17% with 100% candidate coverage.
>
> **Track**
> ARC-AGI-2 Solution Writeup (companion to the Paper Track diagnostic
> *"When Cheap Transformations Vanish"*). Word count of the body below: 1390 (limit 1500).
>
> **Project Links — required by the rules**
> - Public notebook (this is the code submission and the notebook Kaggle re-runs):
>   https://www.kaggle.com/code/luoguoqiang/arc26-submit-full
> - Version 1 of that notebook is the submitted version.
> - Open-source repository (MIT-0):
>   https://github.com/LuoGuoQiang-web/arc-prize-2026-arc-agi-2-v1
> - The neural pipeline lives in `neural/` of that repository (solver, 4 test suites,
>   the Kaggle GPU recipe, and the submission gate).
>
> **Media Gallery — upload both images**
> 1. `cover.png` — cover image (required)
> 2. `coverage_curve.png` — coverage-curve figure
>
> **Checklist before pressing Submit**
> - [ ] Pasted the body below (everything from the first `#` heading onwards)
> - [ ] Cover image uploaded
> - [ ] Public notebook linked
> - [ ] Code submission linked (ref for the current submission is recorded in
>       `work/arc_w1/NEURAL_ROUTE_STATUS.md`)
> - [ ] Repository link present

---

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

(a) The symbolic floor is zero, not small. On the full 120-task ARC-AGI-2 public
evaluation split the engine solved **0/120 (0.000%)** — not "few": zero validated
programs; all 120 tasks fell through to an unvalidated prior. The same engine scores
4.833% on the training split, so the collapse is a distribution shift, not an
implementation defect. This corroborates the community finding that BARC-style
search-and-learn solves ~0.8% of ARC-AGI-2 private tasks against ~22% on ARC-AGI-1.

**(b) The cheap fallback is also zero.** Only a task the neural path fails to answer
ever reaches the fallback layer, so its quality matters at scale. We measured two
generations of it on the public splits:

| fallback variant | eval hits (172 test inputs) | train hits (1076) | `attempt_1 == attempt_2` |
|---|---|---|---|
| mode-colour fill + all-zeros | 0 | 0 | 35% / 70% |
| identity + fg/bg swap (final) | 0 | 0 | 0% |

The second version fixes a real structural bug — the first returned the *same* grid in
both slots on 70% of training inputs, wasting the second attempt — but neither ever
produces a correct answer: constant fills, the identity and the fg/bg swap are never the
answer here.

**Consequence.** Every point must come from the model. "Never leave an attempt blank" is
a *format-validity* property, not a scoring strategy; coverage of genuine neural
candidates is the only lever.

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

It is ~40 dependency-free lines, which also keeps the submission licence-clean: an
Apache-2.0 checkpoint plus our own code.

Two correctness properties are covered by local tests: on the first step `∂L/∂A` is
*exactly* zero because `B = 0`, becoming non-zero on the second; and the adapters must be
created on the base weight's device, since `device_map` does not relocate parameters
allocated afterwards and every step then dies with a device mismatch.

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

Two guards exist for coverage, and both came from measurement: the first implementation
discarded its in-flight beams on timeout, so hard tasks — the ones needing more than
their slice to emit a 930-token grid — returned an *empty* pool; and a beam cut off
mid-row parses as ragged, so the strict parser then threw those flushed candidates away.
Flushing the live beams and recovering at the first incomplete row turned three empty
pools into pools of 2, 1 and 8 and took a five-task smoke run from **2/5 to 4/5
correct**, with no change to the model or the training loop.

Selection uses a demonstration-shape prior — the cheapest precision lever, since it costs
no extra generation. Only rules with a *perfect* record may filter, because filtering
deletes the correct answer whenever the rule is wrong: identity is 117/117 and 719/719
across the public splits, and a uniform integer scale rule is 56/56, so other shapes are
dropped. Transpose (95.1%/97.8%) and constant-output-shape (66.1%/95.3%) look usable and
are deliberately excluded — every counterexample for both has an input-shaped answer. The
filter is skipped if it would empty the pool; we verified soundness, not score effect.

### 2.5 Cascade scheduling over a fixed 12-hour session

A Kaggle GPU session is capped at 12 hours, the weekly budget at 30 hours and
concurrency at two sessions. Rather than partition the budget statically — a documented
way to waste most of a session — we run three stages: a cheap no-TTT sweep for coverage;
TTT plus regeneration for the weakest-confidence tasks; then a symbolic/heuristic
backfill. Per-task cost is calibrated from the first few tasks and re-allocated as the
run progresses. `submission.json` is pre-seeded with a legal fallback for every task and
flushed immediately, so the run always emits a complete, format-valid 240-task file even
if it is killed at the wall clock.

---

## 3. Results

**Held-out accuracy: 4.17% (1/24)** on a uniformly spaced 24-task sample of the
ARC-AGI-2 public evaluation split, in a single T4 x 2 session (4844 s of solving after a
140 s model load, 202 s per task, peak 14.09 GiB).

**This number contains no test-time training.** The run's log ends with `[stage B]
stopping: reserve reached`: the no-TTT sweep sized each slice as a share of the *whole*
remaining budget, so it necessarily consumed it and Stage B was never entered. A
scheduled stage that never executes is a defect, not a detail; it is fixed by capping the
sweep at 45% of the budget.

Two further measured defects came out of the same log and are fixed:

- **Rescoring OOM.** Three tasks hit `CUDA out of memory. Tried to allocate 3.75 GiB` on
  a 14.6 GiB card; the whole micro-batch was abandoned, leaving those candidates
  unscored and ranked last. The scorer now halves the batch down to one candidate.
- **Sweep slice units.** Slices applied per *test input* meant two-input tasks ran
  ~270 s against a slice the scheduler read as 150 s, which let the sweep eat the session.

The more informative result sits next to the accuracy: **24 of 24 tasks received a
genuine neural candidate**. Coverage is complete; the failure is in *precision*, not in
reach. That is exactly what the symbolic and fallback measurements predicted — since
both of those layers are worth exactly zero, a task with a real candidate is the only
kind of task that can ever score, and here every task is that kind.

For scale, the 2025-winning lineage scored 33.89 using 4 x L4 and roughly four times our
compute per task, plus a synthetic-data SFT stage we did not reproduce. This hardware
also bounds the obvious fix: at ~180 s per task, 240 tasks already need ~12 h, so the
reference's 16 inference augmentations do not fit in one session here.

Smoke run, five easiest training tasks: **4/5** — not an accuracy estimate.

---

## 4. What this says

The symbolic route is closed on ARC-AGI-2 — exactly zero on 120 evaluation tasks — and
the fallback layer is exactly zero too, so no edge calibration moves the score. The only
lever is whether a genuine neural candidate exists, and on the evaluation sample it did.

Coverage engineering (timeout flushing, truncation recovery, a budget-capped cascade,
never emitting a malformed file) took smoke tasks from 2/5 to 4/5 and reached 24/24
candidate coverage without touching the model. It does not by itself produce accuracy:
the remaining gap is precision, and closing it needs either far more compute per task or
a better selection signal than candidate agreement.

We claim a reproducible, licence-clean pipeline whose failure modes are measured rather
than assumed, four specific defects found by measurement and fixed, and an honest floor
for this class of solver at this compute budget — not competitiveness.
