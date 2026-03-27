#!/usr/bin/env python3
"""check_env.py — PyPTO 环境快速预检。

退出码: 0=环境正常, 1=部分功能不可用(可降级), 2=PyPTO不可用
"""
import json
import os
import subprocess
import sys

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


def main():
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
    }
    if critical_fails:
        output["recommendation"] = "PyPTO 或 torch 不可用，请先运行 pypto-environment-setup。"
    elif warnings:
        output["recommendation"] = "NPU 或 torch_npu 不可用，将以降级模式执行。"
    print(json.dumps(output, indent=2, ensure_ascii=False))
    sys.exit(2 if critical_fails else (1 if warnings else 0))


if __name__ == "__main__":
    main()
