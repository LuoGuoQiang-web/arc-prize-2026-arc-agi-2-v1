"""
End-to-end self test for arc_prize_v1.py — runs entirely locally, no Kaggle, no network.

Checks, in order:
  1. fixture build (12 synthetic tasks: 10 solvable, 2 not)
  2. dev-mode run: submission schema, dev score, diagnostics
  3. resume: a second identical run must reuse the run id and re-solve nothing
  4. rebuild_submission(): submission regenerated from the checkpoint alone
  5. force=True: fresh run directory, everything re-solved
  6. submit-mode run on test challenges (no solutions attached)
  7. backup snapshot exists and can be restored
  8. negative test: a corrupted submission must be rejected by the validator
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE.parent / "arc_prize_v1.py"
sys.path.insert(0, str(HERE))

import make_fixture  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def load_engine():
    spec = importlib.util.spec_from_file_location("arc_prize_v1", ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arc_prize_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    # NOTE: keep the sandbox inside the workspace; the OS temp area is not writable here.
    tmp = HERE.parent / ".selftest_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    data_dir = tmp / "input" / "arc-prize-2026-arc-agi-2"
    runtime = tmp / "runtime"
    info = make_fixture.build_fixture(data_dir)
    print(f"fixture: {info['n_tasks']} tasks -> {info['dir']}")

    arc = load_engine()
    base_cfg = dict(
        input_roots=[str(tmp / "input")],
        project_dir=str(runtime),
        mode="dev",
        dataset="evaluation",
        log_every=4,
        submission_flush_every=3,
        global_budget_seconds=600,
        per_task_seconds=30,
        backup_keep=2,
        runs_keep=3,
    )

    print("\n[1] dev-mode run")
    summary = arc.main(**base_cfg)
    check("run reports ok", summary.get("ok") is True, json.dumps(summary.get("problems", []))[:200])
    check("all tasks processed", summary["n_tasks"] == info["n_tasks"], str(summary["n_tasks"]))
    acc = summary.get("dev_score", {}).get("accuracy", 0.0)
    check("dev accuracy >= 0.80 on solvable fixture", acc >= 0.80, f"accuracy={acc}")
    check("most tasks solved by validated search", summary["solved_by_search"] >= 8,
          f"search={summary['solved_by_search']} prior={summary['prior_only']}")
    check("unsolvable tasks fell back to priors", summary["prior_only"] >= 1,
          f"prior_only={summary['prior_only']}")

    sub_path = Path(summary["submission_path"])
    payload = json.loads(sub_path.read_text(encoding="utf-8"))
    check("submission has every task id", set(payload) == set(json.loads(
        (data_dir / "arc-agi_evaluation_solutions.json").read_text(encoding="utf-8"))))
    check("t003tile has 2 attempts entries", len(payload["t003tile"]) == 2)
    check("t010crop_rot90 has 3 attempts entries", len(payload["t010crop_rot90"]) == 3)
    check("attempt keys exact", all(set(e) == {"attempt_1", "attempt_2"}
                                    for v in payload.values() for e in v))
    check("failure taxonomy produced", "failure_classes" in summary,
          json.dumps(summary.get("failure_classes", {})))

    run_id_1 = summary["run_id"]
    run_dir_1 = runtime / "runs" / run_id_1
    check("checkpoint written", (run_dir_1 / "checkpoint.jsonl").exists())
    check("diagnostics.csv written", (run_dir_1 / "diagnostics.csv").exists())
    check("manifest written with dataset fingerprint",
          "data_fingerprints" in json.loads((run_dir_1 / "manifest.json").read_text(encoding="utf-8")))
    check("state file written", (runtime / "state" / "PROJECT_STATE.md").exists())

    print("\n[2] resume: identical config must re-solve nothing")
    summary2 = arc.main(**base_cfg)
    check("same run id reused", summary2["run_id"] == run_id_1,
          f"{run_id_1} -> {summary2['run_id']}")
    check("nothing re-solved", summary2["solved_by_search"] == 0,
          f"search={summary2['solved_by_search']}")
    check("dev score identical after resume",
          summary2["dev_score"]["accuracy"] == acc,
          f"{acc} -> {summary2['dev_score']['accuracy']}")

    print("\n[3] rebuild_submission() from checkpoint only")
    rebuilt = arc.rebuild_submission(run_id=run_id_1, **base_cfg)
    check("rebuild ok", rebuilt["ok"] is True, json.dumps(rebuilt["problems"])[:200])
    check("rebuild used all checkpointed tasks", rebuilt["tasks_from_checkpoint"] == info["n_tasks"],
          str(rebuilt["tasks_from_checkpoint"]))
    # regression guard: bind_run_id() must re-derive every path, not just .dir
    probe = arc.Run(arc.make_config(base_cfg))
    probe.bind_run_id(run_id_1)
    check("run paths follow an explicitly selected run id",
          probe.checkpoint_path == probe.dir / "checkpoint.jsonl"
          and probe.checkpoint_path.parent.name == run_id_1,
          str(probe.checkpoint_path))

    print("\n[4] force=True re-runs into a fresh run dir")
    summary3 = arc.main(**{**base_cfg, "force": True})
    check("new run id", summary3["run_id"] != run_id_1, summary3["run_id"])
    check("re-solved by search again", summary3["solved_by_search"] >= 8,
          str(summary3["solved_by_search"]))

    print("\n[5] submit-mode run on test challenges")
    summary4 = arc.main(**{**base_cfg, "mode": "submit", "dataset": "test", "force": True})
    check("submit mode ok", summary4["ok"] is True, json.dumps(summary4["problems"])[:200])
    check("no dev score without solutions", "dev_score" not in summary4)

    print("\n[6] backup snapshot + restore")
    snapshots = sorted((runtime / "backups").glob("*.zip"))
    check("snapshots exist", len(snapshots) >= 1, f"{len(snapshots)} files")
    check("snapshot rotation keeps <= 2 per label",
          len([s for s in snapshots if "-finish-" in s.name]) <= 2)
    restore_dir = tmp / "restored"
    if snapshots:
        arc.Run.restore_backup(snapshots[-1], restore_dir)
        restored_files = [p.name for p in restore_dir.rglob("*") if p.is_file()]
        check("restore produced submission.json", any(f == "submission.json" for f in restored_files),
              str(restored_files))

    print("\n[7] negative test: corrupted submission must be rejected")
    bad_dir = tmp / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    bad = json.loads(sub_path.read_text(encoding="utf-8"))
    bad.pop(sorted(bad)[0])
    bad_path = bad_dir / "submission.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    expected = [(tid, len(e)) for tid, e in payload.items()]
    ok, problems = arc.validate_submission(bad_path, expected)
    check("validator rejects a missing task", ok is False and any("missing task" in p for p in problems),
          str(problems[:2]))

    bad2 = dict(payload)
    first = sorted(bad2)[0]
    bad2[first] = [{"attempt_1": [[0, 1], [2]], "attempt_2": [[0]]}]
    bad2_path = bad_dir / "submission2.json"
    bad2_path.write_text(json.dumps(bad2), encoding="utf-8")
    ok2, problems2 = arc.validate_submission(bad2_path, expected)
    check("validator rejects ragged rows", ok2 is False and any("ragged" in p for p in problems2),
          str(problems2[:2]))

    print("\n[8] safety net: only sample_submission.json present (no challenge files)")
    only_sample = tmp / "input_only_sample" / "arc-prize-2026-arc-agi-2"
    only_sample.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_dir / "sample_submission.json", only_sample / "sample_submission.json")
    summary5 = arc.main(**{**base_cfg, "input_roots": [str(tmp / "input_only_sample")],
                           "project_dir": str(tmp / "runtime2"), "force": True})
    check("passthrough run reports ok", summary5.get("ok") is True
          and summary5.get("sample_passthrough") is True, json.dumps(summary5)[:220])
    pass_payload = json.loads(Path(summary5["submission_path"]).read_text(encoding="utf-8"))
    check("passthrough mirrors sample task ids", set(pass_payload) == set(payload))
    check("passthrough entries are schema-valid",
          all(len(v) == len(payload[k]) and all(set(e) == {"attempt_1", "attempt_2"} for e in v)
              for k, v in pass_payload.items()))

    print("\n[9] official repo layout (data/evaluation/*.json) is discovered and self-scored")
    repo_base = tmp / "repo_only"
    repo_eval = repo_base / "ARC-AGI-2-main" / "data" / "evaluation"
    repo_eval.mkdir(parents=True, exist_ok=True)
    ch = json.loads((data_dir / "arc-agi_evaluation_challenges.json").read_text(encoding="utf-8"))
    so = json.loads((data_dir / "arc-agi_evaluation_solutions.json").read_text(encoding="utf-8"))
    for tid, body in ch.items():
        payload = {"train": body["train"],
                   "test": [{"input": p["input"], "output": out}
                            for p, out in zip(body["test"], so[tid])]}
        (repo_eval / f"{tid}.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg9 = arc.make_config({"mode": "dev", "dataset": "evaluation",
                            "input_roots": [str(repo_base)],
                            "project_dir": str(tmp / "runtime3"), "force": True})
    disc9 = arc.discover_data(cfg9)
    check("repo-style evaluation dir discovered",
          str(disc9["task_dirs"].get("evaluation", "")) == str(repo_eval),
          str(disc9["task_dirs"]))
    tasks9, _, src9 = arc.load_dataset(cfg9, disc9, "evaluation")
    check("per-file tasks loaded", len(tasks9) == info["n_tasks"], f"{len(tasks9)} tasks")
    check("test ground truth preserved from task files",
          all("output" in p for t in tasks9.values() for p in t["test"]), src9)
    s9 = arc.main(mode="dev", dataset="evaluation", input_roots=[str(repo_base)],
                  project_dir=str(tmp / "runtime3"), force=True, global_budget_seconds=600)
    check("self-scored from repo layout", s9.get("dev_score", {}).get("accuracy", 0) >= 0.80,
          json.dumps(s9.get("dev_score")))

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print("  -", f)
    keep = "--keep" in sys.argv
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"artifacts kept at {tmp}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 2
    sys.exit(code)
