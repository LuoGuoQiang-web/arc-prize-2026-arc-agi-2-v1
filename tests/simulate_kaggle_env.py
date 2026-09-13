"""Kaggle 环境模拟验证：在存在 /kaggle/working 的条件下跑「先 submit 后 dev」序列。

为什么需要它：2026-09-13 的真实提交被 Kaggle 判为格式错误——dev 自评把
/kaggle/working/submission.json 覆盖成了 120 题的评测提交。本地干跑没抓到，因为本机
没有 /kaggle/working 目录，旧默认逻辑让 dev 写进了自己的 run 目录。

本脚本在 Windows 上创建 C:\\kaggle\\working（Python 的 Path("/kaggle/working") 即指向它），
复现 Kaggle 的分支条件，然后断言：
  1) submit 模式把 100 题的正式提交写到 /kaggle/working/submission.json
  2) dev 模式**不碰**该文件，改写 dev_submission.json
  3) 序列结束后正式提交仍是 100 题

用法：
    python simulate_kaggle_env.py
"""
import json
import shutil
import sys
from pathlib import Path

ENGINE = Path(r"C:\Users\Administrator\Desktop\Kaggle\ARC\Kaggle\arc_prize_v1")
WORK = Path(r"C:\Users\Administrator\Desktop\Kaggle\work\arc_w1")
sys.path.insert(0, str(ENGINE))
import arc_prize_v1 as E  # noqa: E402

KAGGLE_WORKING = Path("/kaggle/working")          # -> C:\kaggle\working on Windows
KAGGLE_PROJECT = KAGGLE_WORKING / "arcprize"
SUBMIT_INPUT = WORK / "rehearsal" / "input"       # 100 题、无真值
DEV_INPUT = WORK / "data" / "ARC-AGI-2-main" / "data"  # 含 evaluation（有真值）

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


print("准备 Kaggle 式目录:", KAGGLE_WORKING)
shutil.rmtree(KAGGLE_WORKING, ignore_errors=True)
KAGGLE_PROJECT.mkdir(parents=True, exist_ok=True)
check("Path('/kaggle/working').is_dir() 为真（Kaggle 分支已激活）", KAGGLE_WORKING.is_dir())

print("\n=== 1) submit 模式（100 题、无真值）===")
s1 = E.main(mode="submit", dataset="test", input_roots=[str(SUBMIT_INPUT)],
            project_dir=str(KAGGLE_PROJECT), force=True)
top = KAGGLE_WORKING / "submission.json"
n1 = len(json.loads(top.read_text(encoding="utf-8")))
check("正式提交写到 /kaggle/working/submission.json", top.exists(), str(top))
check("题数 = 100", n1 == 100, f"实际 {n1}")

print("\n=== 2) dev 模式（120 题评测集，有真值）===")
s2 = E.main(mode="dev", dataset="evaluation", input_roots=[str(DEV_INPUT)],
            project_dir=str(KAGGLE_PROJECT), force=True)
n2 = len(json.loads(top.read_text(encoding="utf-8")))
dev_file = KAGGLE_PROJECT / "dev_submission.json"
check("dev 运行产出了独立的 dev_submission.json", dev_file.exists(), str(dev_file))
if dev_file.exists():
    nd = len(json.loads(dev_file.read_text(encoding="utf-8")))
    check("dev_submission.json 是 120 题（与正式提交区分）", nd == 120, f"实际 {nd}")
check("★ dev 运行没有覆盖正式提交（仍为 100 题）", n2 == 100, f"实际 {n2}")
check("★ dev 的 dev score 正常", s2.get("dev_score", {}).get("n_scored_inputs", 0) > 0,
      str(s2.get("dev_score")))

print("\n=== 3) 结论 ===")
if failures:
    print("模拟验证失败:", failures)
    sys.exit(1)
print("✅ 在 Kaggle 式环境下，dev 自评不再覆盖正式提交（2026-09-13 的线上 bug 已修复）")

# 清理模拟目录，避免影响后续本地运行
shutil.rmtree(KAGGLE_WORKING, ignore_errors=True)
print("已清理模拟目录", KAGGLE_WORKING)
