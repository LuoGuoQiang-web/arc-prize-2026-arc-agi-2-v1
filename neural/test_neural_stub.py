#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU stub-model test for the *neural* core of ``arc26_solver.py``.

There is no GPU/weights locally, so this exercises the parts that would otherwise
only ever run on Kaggle:

* ``constrained_generate`` -> ``turbo_dfs`` (recursion, branching, the p > 0.2
  threshold, forced survival, terminal EOS handling, candidate de-duplication),
* batched KV-cache surgery (``_cache_select`` on a ``layers[i].keys/values`` layout:
  index_select along the batch dimension while beams branch),
* the greedy safety net that runs when ``turbo_dfs`` fails (cache API change, OOM),
* ``calc_scores`` teacher-forced NLL (position alignment, full-vocab logsumexp,
  mean over the answer tokens),
* the DFS -> grid decode -> inverse-augmentation chain.

The stub model is deterministic -- a scripted per-position probability table -- so the
expected continuations are known exactly.  Note the convention the stub encodes: the
logits at the prompt's LAST position predict the first generated token, i.e. a token
emitted at continuation index i is drawn from the script entry for absolute position
``prompt_len - 1 + i``.

Run with::

    python test_neural_stub.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc26_solver as S  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------------------
# A deterministic stub causal LM with a batch-indexable KV cache
# --------------------------------------------------------------------------------------

class StubLayer:
    def __init__(self) -> None:
        self.keys = torch.zeros(1, 1, 0, 4)
        self.values = torch.zeros(1, 1, 0, 4)


class StubCache:
    """Mimics the modern DynamicCache: a list of layers with keys/values tensors."""

    def __init__(self) -> None:
        self.layers = [StubLayer() for _ in range(2)]

    def get_seq_length(self) -> int:
        return int(self.layers[0].keys.shape[-2])


class StubModel:
    """Next-token distribution scripted by ABSOLUTE position (see `script`)."""

    def __init__(self, script, vocab: int = 16):
        self.script = script          # {predicting_position: {token: prob}}
        self.config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=1,
                                      head_dim=4, vocab_size=vocab)
        self._params = [torch.zeros(2)]
        self.calls = 0

    # -- nn.Module-ish surface ---------------------------------------------------------
    def parameters(self):
        return iter(self._params)

    def eval(self):
        return self

    def _dist(self, position: int) -> torch.Tensor:
        row = torch.full((16,), -1e9)
        probs = dict(self.script.get(position, {}))
        rest = 1.0 - sum(probs.values())
        if rest > 1e-9:
            probs[S.PAD_ID] = probs.get(S.PAD_ID, 0.0) + rest
        for tok, prob in probs.items():
            row[tok] = math.log(max(prob, 1e-12))
        return row

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None,
                 cache_position=None, use_cache=True, return_dict=True, **kwargs):
        self.calls += 1
        ids = input_ids
        batch, length = ids.shape
        if cache_position is not None and length == 1:
            start = int(cache_position.reshape(-1)[0].item())
        elif past_key_values is not None:
            start = past_key_values.get_seq_length()
        else:
            start = 0
        cache = past_key_values if past_key_values is not None else StubCache()
        for layer in cache.layers:  # grow every layer by `length` positions
            layer.keys = torch.cat(
                [layer.keys, torch.zeros(batch, 1, length, 4)], dim=2)
            layer.values = torch.cat(
                [layer.values, torch.zeros(batch, 1, length, 4)], dim=2)
        logits = torch.stack([self._dist(start + j) for j in range(length)], dim=0)
        logits = logits.unsqueeze(0).repeat(batch, 1, 1)  # [B, L, 16]
        return SimpleNamespace(logits=logits, past_key_values=cache)


PROMPT = [S.USER_TOKEN_ID, 1, 2, S.ASSISTANT_TOKEN_ID]
FIRST = len(PROMPT) - 1  # position whose logits predict the first generated token


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_turbo_dfs_branches() -> None:
    model = StubModel({
        FIRST: {3: 0.6, 4: 0.3},          # both above p > 0.2 -> a real branch
        FIRST + 1: {5: 1.0},
        FIRST + 2: {S.EOS_ID: 1.0},
    })
    tokens_list, stats = S.constrained_generate(
        model, PROMPT, max_new_tokens=10, end_time=float("inf"),
        cache_budget_bytes=64 * 1024 * 1024,
    )
    check("turbo_dfs path was used (no fallback)", stats.get("path") == "turbo_dfs",
          str(stats.get("path")))
    check("DFS explored both branches", len(tokens_list) == 2, str(tokens_list))
    check("branch A = 3,5,EOS", [3, 5, S.EOS_ID] in tokens_list, str(tokens_list))
    check("branch B = 4,5,EOS", [4, 5, S.EOS_ID] in tokens_list, str(tokens_list))
    check("highest-probability branch ranks first",
          tokens_list[0] == [3, 5, S.EOS_ID], str(tokens_list))
    grids = [S.tokens_to_array(t) for t in tokens_list]
    check("DFS tokens decode to grids", all(g is not None for g in grids),
          str([None if g is None else g.tolist() for g in grids]))
    check("decoded grids are the two branches",
          grids[0].tolist() == [[3, 5]] and grids[1].tolist() == [[4, 5]],
          str([g.tolist() for g in grids]))
    check("cache surgery ran (nodes visited > 0)", stats.get("nodes", 0) > 0,
          str(stats.get("nodes")))
    check("batch-axis cache growth happened (multi-row forward)",
          model.calls >= 3, f"model calls={model.calls}")


def test_threshold_forces_survival() -> None:
    """Every branch below 0.2 must still yield one candidate (never empty)."""
    prompt = [S.USER_TOKEN_ID, 1]
    first = len(prompt) - 1
    model = StubModel({first: {7: 0.05}, first + 1: {S.EOS_ID: 1.0}})
    tokens_list, stats = S.constrained_generate(
        model, prompt, max_new_tokens=5, end_time=float("inf"),
        cache_budget_bytes=64 * 1024 * 1024,
    )
    check("sub-threshold level still yields a candidate", len(tokens_list) == 1,
          str(tokens_list))
    check("the forced branch is the arg-max ARC token",
          tokens_list and tokens_list[0][0] == 7, str(tokens_list))
    check("forced-branch counter incremented", stats.get("forced_branches", 0) >= 1,
          str(stats.get("forced_branches")))


def test_greedy_fallback_wiring() -> None:
    """If turbo_dfs blows up, constrained_generate must fall back to greedy."""
    prompt = [S.USER_TOKEN_ID, 1]
    first = len(prompt) - 1
    script = {first: {8: 0.9}, first + 1: {S.EOS_ID: 1.0}}
    original = S.turbo_dfs

    def boom(*_args, **_kwargs):
        raise RuntimeError("unsupported KV cache layout")

    S.turbo_dfs = boom
    try:
        model = StubModel(script)
        tokens_list, stats = S.constrained_generate(
            model, prompt, max_new_tokens=5, end_time=float("inf"),
            cache_budget_bytes=64 * 1024 * 1024,
        )
    finally:
        S.turbo_dfs = original
    check("greedy fallback engaged after a turbo_dfs failure",
          stats.get("path") == "greedy_fallback", str(stats.get("path")))
    check("fallback error recorded", bool(stats.get("error")), str(stats.get("error")))
    check("greedy fallback still returns a candidate", len(tokens_list) >= 1,
          str(tokens_list))
    check("greedy fallback picked the 0.9 token",
          tokens_list and tokens_list[0][0] == 8, str(tokens_list))


class DigitTok:
    """Minimal tokenizer: one id per digit / newline character of the text."""

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            if ch.isdigit():
                ids.append(int(ch))
            elif ch == "\n":
                ids.append(S.NEWLINE_TOKEN_ID)
        return {"input_ids": ids}

    def decode(self, ids):
        return S.decode_arc_tokens(ids)


class BiasModel:
    """Always puts a huge logit on one token id (parameter-free)."""

    def __init__(self, bias_token: int):
        self.bias_token = bias_token
        self._params = [torch.zeros(1)]

    def parameters(self):
        return iter(self._params)

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, use_cache=False,
                 return_dict=True, **kw):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 16)
        logits[:, :, self.bias_token] = 5.0
        return SimpleNamespace(logits=logits)


def test_calc_scores_ordering() -> None:
    tok = DigitTok()
    model = BiasModel(bias_token=4)
    # a realistic prompt: a demonstration pair plus a test input
    query = S.fmt_train([{"input": [[1, 2]], "output": [[3, 4]]}], [[5, 6]])
    answers = [S.fmt_reply([[4, 4]]), S.fmt_reply([[1, 2]])]
    problems = []
    scores = S.calc_scores([query, query], answers, tok, model,
                           log_fn=problems.append)
    check("calc_scores reported no internal failure", not problems, str(problems))
    check("calc_scores returns one finite score per candidate",
          len(scores) == 2 and all(math.isfinite(s) for s in scores), str(scores))
    check("the biased-token answer scores strictly better (lower NLL)",
          len(scores) == 2 and scores[0] < scores[1], str(scores))
    check("query tokenisation is non-empty (position alignment is meaningful)",
          len(S._encode_text(tok, query)) > 0)


def test_augmented_decode_chain() -> None:
    """DFS under an augmentation, inverted, must reproduce the original grid."""
    prompt = [S.USER_TOKEN_ID, 1]
    first = len(prompt) - 1
    seq = [1, 2, S.NEWLINE_TOKEN_ID, 3, 0, S.EOS_ID]   # grid [[1,2],[3,0]]
    script = {first + i: {tok: 1.0} for i, tok in enumerate(seq)}
    model = StubModel(script)
    tokens_list, stats = S.constrained_generate(
        model, prompt, max_new_tokens=12, end_time=float("inf"),
        cache_budget_bytes=64 * 1024 * 1024,
    )
    check("multi-row generation produced a candidate", bool(tokens_list),
          str(tokens_list))
    grid = S.tokens_to_array(tokens_list[0])
    check("multi-row generation decodes to a 2x2 grid",
          grid is not None and grid.shape == (2, 2),
          "None" if grid is None else str(grid.shape))
    check("decoded content matches the script",
          grid is not None and grid.tolist() == [[1, 2], [3, 0]],
          "None" if grid is None else str(grid.tolist()))
    key = S.AugKey(rot=1, flip=True, transpose=True,
                   perm=(0, 2, 1, 3, 4, 5, 6, 7, 8, 9))
    back = S.invert_augment(S.apply_augment(grid, key), key)
    check("augment -> DFS -> invert round-trips the grid",
          np.array_equal(back, grid), str(back.tolist()))


def test_rescore_then_select() -> None:
    """The pooled selection must prefer the better NLL and keep the two distinct."""
    best = S.Candidate.make(np.array([[1, 1]]), "neural", nll=0.1)
    worse = S.Candidate.make(np.array([[2, 2]]), "neural", nll=0.9)
    verified = S.Candidate.make(np.array([[3, 3]]), "symbolic", nll=0.0)
    prior = S.Candidate.make(np.array([[4, 4]]), "prior", nll=S.UNSCORED_PRIOR_NLL)
    pool = S.dedupe_pool([worse, best, verified, prior])
    a1, a2, s1, s2 = S.select_attempts(pool, (np.array([[9]]), np.array([[8]])))
    check("validated symbolic program wins attempt_1",
          a1.tolist() == [[3, 3]] and s1 == "symbolic", f"{a1.tolist()} {s1}")
    check("attempt_2 is the best distinct remaining candidate",
          a2.tolist() == [[1, 1]] and s2 == "neural", f"{a2.tolist()} {s2}")
    only = S.dedupe_pool([best])
    a1b, a2b, _s1, s2b = S.select_attempts(only, (np.array([[9]]), np.array([[8]])))
    check("a single-candidate pool still fills both slots distinctly",
          a1b.tolist() == [[1, 1]] and a2b.tolist() == [[8]] and s2b == "fallback",
          f"{a1b.tolist()} {a2b.tolist()}")
    dup = S.dedupe_pool([best, S.Candidate.make(np.array([[1, 1]]), "neural", nll=5.0)])
    check("duplicate grids collapse to the better-ranked instance",
          len(dup) == 1 and dup[0].nll == 0.1, str([c.nll for c in dup]))


def main() -> int:
    print(f"torch {torch.__version__} (device: cpu)")
    test_turbo_dfs_branches()
    test_threshold_forces_survival()
    test_greedy_fallback_wiring()
    test_calc_scores_ordering()
    test_augmented_decode_chain()
    test_rescore_then_select()
    print("-" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all neural-core checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
