#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the PURE helpers of ``arc26_solver.py``.

Runs anywhere (no GPU, no torch, no transformers): it only imports the format
helpers, which is exactly what makes the prompt/serialisation chain testable
offline.  Run with::

    python test_format.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc26_solver as S  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def test_3x3_roundtrip() -> None:
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    text = S.convert_grid_to_string(grid)
    check("3x3 serialisation is exact", text == "123\n456\n789", repr(text))
    back = S.parse_grid_string(text)
    check("3x3 round-trips to the identical array",
          back is not None and np.array_equal(back, np.array(grid)))
    check("3x3 shape preserved", back is not None and back.shape == (3, 3))

    # the same grid, this time through the token-id path (pure decoder, no tokenizer)
    ids = []
    for row in grid:
        ids.extend(int(c) for c in row)
        ids.append(S.NEWLINE_TOKEN_ID)
    roundtrip = S.tokens_to_array(ids + [S.EOS_ID])
    check("3x3 round-trips through token ids",
          roundtrip is not None and np.array_equal(roundtrip, np.array(grid)))


def test_30x30_roundtrip() -> None:
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 10, size=(30, 30)).tolist()
    text = S.convert_grid_to_string(grid)
    check("30x30 serialisation has 30 rows", text.count("\n") == 29)
    check("30x30 serialisation has 900 digits",
          sum(ch.isdigit() for ch in text) == 900)
    back = S.parse_grid_string(text)
    check("30x30 round-trips to the identical array",
          back is not None and np.array_equal(back, np.array(grid)))
    check("30x30 shape preserved", back is not None and back.shape == (30, 30))

    ids = []
    for row in grid:
        ids.extend(int(c) for c in row)
        ids.append(S.NEWLINE_TOKEN_ID)
    roundtrip = S.tokens_to_array(ids + [S.EOS_ID])
    check("30x30 round-trips through token ids",
          roundtrip is not None and np.array_equal(roundtrip, np.array(grid)))


def test_ragged_row_rejected() -> None:
    # a short row is the classic malformed generation: row 2 has 2 cells, not 3
    ragged = "123\n45\n789"
    check("ragged row string rejected by parse_grid_string",
          S.parse_grid_string(ragged) is None)
    check("ragged row rejected by validate_grid",
          S.validate_grid([[1, 2, 3], [4, 5], [7, 8, 9]]) is None)

    # ... and the same thing coming through the token path must not silently
    # produce an object array or a wrong shape
    ids = [1, 2, 3, S.NEWLINE_TOKEN_ID, 4, 5, S.NEWLINE_TOKEN_ID,
           7, 8, 9, S.EOS_ID]
    check("ragged row rejected by tokens_to_array",
          S.tokens_to_array(ids) is None)

    # other rejects the validator must catch
    check("empty grid rejected", S.validate_grid([]) is None)
    check("oversized grid rejected",
          S.validate_grid([[0] * 31 for _ in range(2)]) is None)
    check("out-of-range colour rejected",
          S.validate_grid([[0, 10], [1, 2]]) is None)
    check("None rejected", S.validate_grid(None) is None)
    check("blank text rejected", S.parse_grid_string("\n\n") is None)


def test_prompt_and_augment_helpers() -> None:
    demos = [{"input": [[1, 2], [3, 4]], "output": [[5, 5], [5, 5]]}]
    prompt = S.fmt_train(demos, [[6, 7], [8, 9]])
    check("prompt starts with the user marker", prompt.startswith(S.USER))
    check("prompt contains the assistant prompt",
          prompt.endswith(S.ASSISTANT))
    check("prompt carries the test input", "67\n89" in prompt)
    check("reply ends with EOS",
          S.fmt_reply([[1, 2], [3, 4]]).endswith(S.ENDOFTEXT))

    grid = np.array([[1, 2, 3], [4, 5, 6]])
    for key in (
        S.AugKey.identity(),
        S.AugKey(rot=1, flip=True, transpose=False, perm=None),
        S.AugKey(rot=3, flip=False, transpose=True, perm=tuple(range(9, -1, -1))),
        S.AugKey(rot=2, flip=True, transpose=True, perm=(0, 2, 1, 3, 4, 5, 6, 7, 8, 9)),
    ):
        aug = S.apply_augment(grid, key)
        inv = S.invert_augment(aug, key)
        check(f"augment/invert is exact for {key.to_json()}", np.array_equal(inv, grid))

    colour = S.AugKey(perm=tuple([1, 0] + list(range(2, 10))))
    check("colour permutation is applied cell-wise",
          np.array_equal(S.apply_augment(np.array([[0, 1]]), colour),
                         np.array([[1, 0]])))


def test_grid_to_json_and_max_tokens() -> None:
    check("grid_to_json is plain int lists",
          S.grid_to_json(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]])
    check("grid_to_json falls back for an invalid grid",
          S.grid_to_json(None) == [[0]])
    n = S.max_new_tokens_for(None)
    check("max_new_tokens constant is 930 (spec 1.5)", n == 930, str(n))
    check("ARC token set is the documented 12 ids",
          tuple(S.ARC_TOKEN_IDS) == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15))
    check("PAD/EOS ids match the spec", S.PAD_ID == 13 and S.EOS_ID == 15)


def main() -> int:
    print(f"arc26_solver module: {S.__file__}")
    print(f"torch importable: {S.torch is not None}")
    test_3x3_roundtrip()
    test_30x30_roundtrip()
    test_ragged_row_rejected()
    test_prompt_and_augment_helpers()
    test_grid_to_json_and_max_tokens()
    print("-" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
