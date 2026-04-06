#!/usr/bin/env python3
"""check_env.py — PyPTO 环境快速预检 + autodev 运行时目录初始化。

退出码: 0=环境正常, 1=部分功能不可用(可降级), 2=PyPTO不可用

用法:
    python check_env.py
    python check_env.py --csv autodev/scan_results.csv --work-dir autodev/custom
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 确保 PyPTO 项目路径优先，避免系统包冲突
_PYPTO_PYTHON = os.path.join(os.path.dirname(__file__), "..", "..", "python")
if os.path.isdir(_PYPTO_PYTHON):
    _PYPTO_PYTHON = os.path.abspath(_PYPTO_PYTHON)


def check(name, cmd, critical=False):
    env = os.environ.copy()
    if _PYPTO_PYTHON and os.path.isdir(_PYPTO_PYTHON):
        env["PYTHONPATH"] = _PYPTO_PYTHON + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True, timeout=10, env=env,
        )
        return {
            "name": name, "ok": result.returncode == 0, "critical": critical,
            "error": result.stderr.decode()[:200] if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {"name": name, "ok": False, "critical": critical, "error": "timeout"}
    except Exception as e:
        return {"name": name, "ok": False, "critical": critical, "error": str(e)[:200]}


def init_autodev_dirs(csv_path: str | None, work_dir: str | None) -> dict:
    """初始化 autodev 运行时目录和文件。返回初始化动作记录。"""
    actions = {}
    if csv_path:
        csv_p = Path(csv_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        if not csv_p.exists():
            # 导入 FIELDS 以创建 CSV header
            sys.path.insert(0, str(Path(__file__).parent))
            from csv_ops import FIELDS
            with open(csv_p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
            actions["csv_initialized"] = str(csv_p)

    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        actions["work_dir_created"] = work_dir

    # known-limitations 种子复制
    if csv_path:
        autodev_root = Path(csv_path).parent
        kl_runtime = autodev_root / "known-limitations.md"
        if not kl_runtime.exists():
            seed = Path(__file__).resolve().parent.parent / "references" / "known-limitations-seed.md"
            if seed.exists():
                shutil.copy2(seed, kl_runtime)
                actions["known_limitations_seeded"] = str(kl_runtime)

    return actions


def main():
    parser = argparse.ArgumentParser(description="PyPTO 环境预检 + 初始化")
    parser.add_argument("--csv", default=None, help="CSV 文件路径（可选，传入时执行初始化）")
    parser.add_argument("--work-dir", default=None, help="算子工件根目录（可选）")
    args = parser.parse_args()

    # 初始化
    init_actions = init_autodev_dirs(args.csv, args.work_dir)

    # 环境检查
    results = [
        check("pypto_import", "import pypto", critical=True),
        check("torch_import", "import torch", critical=True),
        check("npu_available",
              "import subprocess; r=subprocess.run(['npu-smi','info'],"
              "capture_output=True); exit(r.returncode)"),
        check("torch_npu", "import torch_npu"),
    ]
    critical_fails = [r for r in results if not r["ok"] and r["critical"]]
    warnings = [r for r in results if not r["ok"] and not r["critical"]]
    output = {
        "all_passed": len(critical_fails) == 0 and len(warnings) == 0,
        "can_proceed": len(critical_fails) == 0,
        "results": results,
        "init_actions": init_actions,
    }
    if critical_fails:
        output["recommendation"] = "PyPTO 或 torch 不可用，请先运行 pypto-environment-setup。"
    elif warnings:
        output["recommendation"] = "NPU 或 torch_npu 不可用，将以降级模式执行。"
    print(json.dumps(output, indent=2, ensure_ascii=False))
    sys.exit(2 if critical_fails else (1 if warnings else 0))


if __name__ == "__main__":
    main()
