#!/usr/bin/env python3
"""Regression test for the NLL scorer's CUDA-OOM recovery.

Measured defect (evaluation split, T4 x2): three tasks logged

    rescoring batch failed: CUDA out of memory. Tried to allocate 3.75 GiB

and the entire micro-batch was abandoned, so every candidate in it stayed unscored
(``inf``) and ranked last -- a silent precision loss that no score change would reveal.

The fix halves the micro-batch and retries down to a single candidate. This test drives
``calc_scores`` with a fake model that raises a CUDA-OOM error for any batch larger than
one, and asserts that every candidate still ends up with a finite score. A second fake
that fails at every size asserts graceful degradation rather than an exception.

CPU only: no GPU and no weights needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc26_solver as S  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


class TinyTokenizer:
    """Deterministic stand-in: one id per character, mapped into the 16-token alphabet.

    Implements ``__call__`` because the production ``_encode_text`` uses the HuggingFace
    calling convention ``tokenizer(text, add_special_tokens=False)``.
    """

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        return {"input_ids": [ord(ch) % 16 for ch in text]}

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(ch) % 16 for ch in text]

    def decode(self, ids):
        return "".join(str(int(i) % 10) for i in ids)


class OOMAboveOne:
    """Returns proper logits for batch == 1 and raises a CUDA OOM for anything larger."""

    def __init__(self):
        self.calls = 0
        self.peak_batch = 0
        self.param = torch.zeros(1)

    def parameters(self):
        yield self.param

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False,  # noqa: ARG002
                 return_dict=True):
        self.calls += 1
        batch, length = input_ids.shape
        self.peak_batch = max(self.peak_batch, int(batch))
        if batch > 1:
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total "
                "capacity of 14.56 GiB of which 2.98 GiB is free."
            )
        logits = torch.zeros(batch, length, 16)
        # put a little mass on the ARC tokens so the NLL is finite and ordered
        logits[..., :12] = 1.0
        return type("Out", (), {"logits": logits})()


class AlwaysOOM(OOMAboveOne):
    def __call__(self, input_ids=None, attention_mask=None, use_cache=False,  # noqa: ARG002
                 return_dict=True):
        self.calls += 1
        self.peak_batch = max(self.peak_batch, int(input_ids.shape[0]))
        raise RuntimeError("CUDA out of memory. Tried to allocate 3.75 GiB.")


def main() -> int:
    tok = TinyTokenizer()
    queries = ["0123"] * 6
    answers = ["45", "45", "6789", "1", "0123456789", "22"]

    # ---- 1. batch > 1 OOMs, smaller batches succeed --------------------------------
    model = OOMAboveOne()
    logs: list[str] = []
    scores = S.calc_scores(queries, answers, tok, model, log_fn=logs.append)

    check("one score per candidate", len(scores) == len(queries), f"{len(scores)}")
    check("EVERY candidate got a finite score after OOM recovery",
          all(s != float("inf") and s == s for s in scores),
          f"scores={[round(s, 3) for s in scores]}")
    check("the failure was logged, not swallowed silently",
          any("splitting" in m for m in logs), f"logs={logs[:3]}")
    check("the scorer actually fell back to single-candidate batches",
          any("batch of" in m for m in logs))
    check("model was called more than once (retries happened)", model.calls > 1,
          f"calls={model.calls}")

    # ---- 2. failure at every size must degrade, not raise --------------------------
    model2 = AlwaysOOM()
    logs2: list[str] = []
    try:
        scores2 = S.calc_scores(queries[:3], answers[:3], tok, model2, log_fn=logs2.append)
        raised = False
    except Exception as exc:  # noqa: BLE001
        raised = True
        scores2 = []
        print(f"  raised: {type(exc).__name__}: {exc}")

    check("total failure does not raise out of calc_scores", not raised)
    check("unscorable candidates stay inf (they rank last) rather than crashing",
          len(scores2) == 3 and all(s == float("inf") for s in scores2),
          f"scores={scores2}")
    check("single-candidate failures are reported",
          any("single candidate failed" in m for m in logs2), f"logs={logs2[:2]}")

    # ---- 3. the happy path is unchanged --------------------------------------------
    class Fine(OOMAboveOne):
        def __call__(self, input_ids=None, attention_mask=None, use_cache=False,  # noqa: ARG002
                     return_dict=True):
            self.calls += 1
            batch, length = input_ids.shape
            self.peak_batch = max(self.peak_batch, int(batch))
            logits = torch.zeros(batch, length, 16)
            logits[..., :12] = 1.0
            return type("Out", (), {"logits": logits})()

    model3 = Fine()
    logs3: list[str] = []
    scores3 = S.calc_scores(queries, answers, tok, model3, log_fn=logs3.append)
    check("a healthy model scores everything with no splitting",
          all(s != float("inf") for s in scores3) and not logs3,
          f"logs={logs3}")

    print()
    if FAILURES:
        print(f"FAILURES: {FAILURES}")
        return 1
    print("all rescore-OOM checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
