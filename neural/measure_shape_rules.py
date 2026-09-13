"""Extend the shape prior from identity to general demonstration shape rules.

The identity prior (output shape == input shape) already covers 67.5% of evaluation
tasks with zero counterexamples, and lets us drop any candidate of the wrong shape for
free. But it leaves a third of tasks unfiltered. This asks whether a *deterministic
shape rule* fitted on the demonstrations -- constant output shape, transpose, integer
upscale/downscale -- predicts the test output shape as reliably, and how much more of
the benchmark it covers.

A rule is only usable when every consistent rule AGREES on the test shape; if two rules
fit the demonstrations but disagree, filtering on either would be unsound, so the task is
counted as uncovered rather than guessed at.

Pure CPU: no GPU and no model weights needed.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(os.path.dirname(HERE), "competition_data")

SPLITS = [
    ("evaluation", "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    ("training", "arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
]


def shape(grid):
    return (len(grid), len(grid[0]))


# --- rule definitions: each returns the predicted output shape, or None ---------------

def rule_identity(inp, _demos):
    return inp


def rule_transpose(inp, _demos):
    return (inp[1], inp[0])


def rule_constant(inp, demos):  # noqa: ARG001
    outs = {shape(d["output"]) for d in demos}
    return outs.pop() if len(outs) == 1 else None


def rule_upscale(inp, demos):
    ks = set()
    for d in demos:
        i, o = shape(d["input"]), shape(d["output"])
        if i[0] == 0 or i[1] == 0:
            return None
        if o[0] % i[0] or o[1] % i[1]:
            return None
        kr, kc = o[0] // i[0], o[1] // i[1]
        if kr != kc or kr < 2:
            return None
        ks.add(kr)
    if len(ks) != 1:
        return None
    k = ks.pop()
    return (inp[0] * k, inp[1] * k)


def rule_downscale(inp, demos):
    ks = set()
    for d in demos:
        i, o = shape(d["input"]), shape(d["output"])
        if o[0] == 0 or o[1] == 0:
            return None
        if i[0] % o[0] or i[1] % o[1]:
            return None
        kr, kc = i[0] // o[0], i[1] // o[1]
        if kr != kc or kr < 2:
            return None
        ks.add(kr)
    if len(ks) != 1:
        return None
    k = ks.pop()
    if inp[0] % k or inp[1] % k:
        return None
    return (inp[0] // k, inp[1] // k)


RULES = [
    ("identity", rule_identity),
    ("transpose", rule_transpose),
    ("constant", rule_constant),
    ("upscale_k", rule_upscale),
    ("downscale_k", rule_downscale),
]


def analyse(label, ch_name, sol_name):
    ch_path, sol_path = os.path.join(COMP, ch_name), os.path.join(COMP, sol_name)
    if not (os.path.isfile(ch_path) and os.path.isfile(sol_path)):
        print(f"{label}: data missing, skipped")
        return
    challenges = json.load(open(ch_path, encoding="utf-8"))
    solutions = json.load(open(sol_path, encoding="utf-8"))

    # per-rule soundness: over the inputs where THE RULE ALONE reproduces every
    # demonstration, how often does it get the test shape right? A rule with any
    # counterexample must not be used to filter, because filtering removes the correct
    # candidate whenever the rule is wrong.
    per_rule = {name: {"n": 0, "ok": 0, "examples": []} for name, _fn in RULES}

    n_tasks = 0
    covered_inputs = 0
    correct_inputs = 0
    ambiguous_inputs = 0
    uncovered_inputs = 0
    rule_hits = Counter()
    counterexamples = []
    identity_only_inputs = 0

    for tid, task in challenges.items():
        truth = solutions.get(tid)
        if truth is None:
            continue
        n_tasks += 1
        demos = task["train"]
        for i, expected in enumerate(truth):
            if i >= len(task["test"]):
                continue
            tin = shape(task["test"][i]["input"])
            tout = shape(expected)

            if all(shape(d["input"]) == shape(d["output"]) for d in demos):
                identity_only_inputs += 1

            consistent = []
            for name, fn in RULES:
                ok = True
                for d in demos:
                    pred = fn(shape(d["input"]), demos)
                    if pred is None or pred != shape(d["output"]):
                        ok = False
                        break
                pred_test = None
                if ok:
                    pred_test = fn(tin, demos)
                    if pred_test is None:
                        ok = False
                if ok:
                    consistent.append((name, pred_test))
                    per_rule[name]["n"] += 1
                    if pred_test == tout:
                        per_rule[name]["ok"] += 1
                    elif len(per_rule[name]["examples"]) < 3:
                        per_rule[name]["examples"].append((tid, i, tin, pred_test, tout))

            if not consistent:
                uncovered_inputs += 1
                continue
            predicted = {p for _n, p in consistent}
            if len(predicted) > 1:
                ambiguous_inputs += 1
                continue
            (pred,) = predicted
            covered_inputs += 1
            if pred == tout:
                correct_inputs += 1
                for name, _p in consistent:
                    rule_hits[name] += 1
            else:
                counterexamples.append((tid, i, tin, pred, tout,
                                        [n for n, _p in consistent]))

    total = covered_inputs + ambiguous_inputs + uncovered_inputs
    print(f"\n=== {label}: {n_tasks} tasks, {total} test inputs ===")
    print(f"  identity-prior inputs                     : {identity_only_inputs}"
          f"  ({100.0 * identity_only_inputs / max(1, total):.1f}%)")
    print(f"  a consistent shape rule predicts the shape: {covered_inputs}"
          f"  ({100.0 * covered_inputs / max(1, total):.1f}%)")
    print(f"    ...of which correct                      : {correct_inputs}/{covered_inputs}"
          f"  ({100.0 * correct_inputs / max(1, covered_inputs):.2f}%)")
    print(f"  ambiguous (rules fit but disagree)         : {ambiguous_inputs}")
    print(f"  no rule fits                              : {uncovered_inputs}")
    print()
    print("  PER-RULE soundness (a rule with ANY counterexample must not filter):")
    for name, _fn in RULES:
        s = per_rule[name]
        if not s["n"]:
            print(f"    {name:<12} n=0")
            continue
        acc = 100.0 * s["ok"] / s["n"]
        verdict = "SAFE" if s["ok"] == s["n"] else f"UNSOUND ({s['n'] - s['ok']} bad)"
        print(f"    {name:<12} n={s['n']:<5} correct={s['ok']:<5} ({acc:6.2f}%)  {verdict}")
        for tid, i, tin, pred, tout in s["examples"]:
            print(f"        e.g. {tid}[{i}] in={tin} predicted={pred} actual={tout}")



def main() -> int:
    for label, ch, sol in SPLITS:
        analyse(label, ch, sol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
