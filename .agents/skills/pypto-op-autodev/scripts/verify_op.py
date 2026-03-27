#!/usr/bin/env python3
"""verify_op.py — 独立验证算子开发结果。

检查产物完整性，可选执行测试，更新 CSV 中的 verify_status。

用法:
    python verify_op.py --csv autodev/scan_results.csv --op gelu
    python verify_op.py --csv autodev/scan_results.csv --op gelu --run-test
    python verify_op.py --csv autodev/scan_results.csv --verify-all

退出码:
    0: 验证通过
    1: 验证失败（产物不完整或测试未通过）
    2: 参数错误或算子不存在
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

if __package__:
    from .data.csv_ops import read_csv, upsert_row
else:
    module_path = Path(__file__).parent / "data" / "csv_ops.py"
    spec = importlib.util.spec_from_file_location("pypto_op_autodev_csv_ops", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load csv_ops from {module_path}")
    csv_ops = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, csv_ops)
    spec.loader.exec_module(csv_ops)
    read_csv = csv_ops.read_csv
    upsert_row = csv_ops.upsert_row

REQUIRED_ARTIFACTS = [
    "spec.md",
    "design.md",
    "{op}_golden.py",
    "{op}_impl.py",
    "test_{op}.py",
]

OPTIONAL_ARTIFACTS = [
    "api_report.md",
    "README.md",
]


def check_artifacts(op_name, op_dir):
    missing = []
    present = []
    optional_present = []
    for artifact in REQUIRED_ARTIFACTS:
        name = artifact.format(op=op_name)
        if (op_dir / name).exists():
            present.append(name)
        else:
            missing.append(name)
    for artifact in OPTIONAL_ARTIFACTS:
        name = artifact.format(op=op_name)
        if (op_dir / name).exists():
            optional_present.append(name)
    return missing, present, optional_present


def run_test(op_name, op_dir):
    test_file = op_dir / f"test_{op_name}.py"
    if not test_file.exists():
        return {"passed": False, "error": "test file not found", "precision_marker": None}
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True, text=True, timeout=300, cwd=str(op_dir),
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "test timeout (300s)", "precision_marker": None}
    output = result.stdout + result.stderr
    precision_marker = None
    if "[PRECISION_PASS]" in output:
        precision_marker = "PASS"
    elif "[PRECISION_FAIL]" in output:
        precision_marker = "FAIL"
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "precision_marker": precision_marker,
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def verify_single(csv_path, op_name, custom_dir, do_run_test=False):
    op_dir = custom_dir / op_name
    if not op_dir.exists():
        return {
            "op_name": op_name,
            "verify_status": "NO_DIR",
            "artifacts_complete": False,
            "missing": ["(directory not found)"],
        }
    missing, present, optional = check_artifacts(op_name, op_dir)
    artifacts_complete = len(missing) == 0
    result = {
        "op_name": op_name,
        "artifacts_complete": artifacts_complete,
        "present": present,
        "optional": optional,
        "missing": missing,
    }
    if not artifacts_complete:
        result["verify_status"] = "INCOMPLETE"
    elif do_run_test:
        test_result = run_test(op_name, op_dir)
        result["test_result"] = test_result
        if test_result["passed"] and test_result.get("precision_marker") == "PASS":
            result["verify_status"] = "VERIFIED"
        elif test_result["passed"]:
            result["verify_status"] = "TEST_PASS_NO_PRECISION"
        else:
            result["verify_status"] = "TEST_FAIL"
    else:
        result["verify_status"] = "ARTIFACTS_OK"
    if csv_path:
        upsert_row(csv_path, op_name, verify_status=result["verify_status"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--op", help="Operator name to verify")
    parser.add_argument("--run-test", action="store_true", help="Execute test file")
    parser.add_argument("--verify-all", action="store_true", help="Verify all completed operators")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    custom_dir = csv_path.parent / "custom"

    if args.verify_all:
        rows = read_csv(args.csv, status="completed")
        results = []
        for row in rows:
            r = verify_single(args.csv, row["op_name"], custom_dir, args.run_test)
            results.append(r)
        summary = {
            "total": len(results),
            "verified": sum(1 for r in results if r["verify_status"] in ("VERIFIED", "ARTIFACTS_OK", "TEST_PASS_NO_PRECISION")),
            "failed": sum(1 for r in results if r["verify_status"] in ("INCOMPLETE", "TEST_FAIL", "NO_DIR")),
            "results": results,
        }
        print(json.dumps(summary, ensure_ascii=False))
        has_failure = any(r["verify_status"] in ("INCOMPLETE", "TEST_FAIL", "NO_DIR") for r in results)
        sys.exit(1 if has_failure else 0)
    elif args.op:
        existing = read_csv(args.csv, op_name=args.op)
        if not existing:
            print(json.dumps({"error": f"op '{args.op}' not found in CSV"}), file=sys.stderr)
            sys.exit(2)
        result = verify_single(args.csv, args.op, custom_dir, args.run_test)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result["verify_status"] not in ("INCOMPLETE", "TEST_FAIL", "NO_DIR") else 1)
    else:
        parser.error("必须指定 --op 或 --verify-all")


if __name__ == "__main__":
    main()
