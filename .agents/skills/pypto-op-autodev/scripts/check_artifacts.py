#!/usr/bin/env python3
"""check_artifacts.py — 算子产物一致性校验。

用法:
    python check_artifacts.py --op-dir autodev/custom/gelu --op gelu
    python check_artifacts.py --operators-dir autodev/custom --check-all

退出码:
    0: 所有检查通过
    1: 存在不一致
    2: 参数错误
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path


def extract_function_names(filepath):
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except (SyntaxError, UnicodeDecodeError):
        return []


def extract_function_args(filepath, func_name):
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return [arg.arg for arg in node.args.args]
        return None
    except (SyntaxError, UnicodeDecodeError):
        return None


def extract_dtypes_from_spec(spec_path):
    if not spec_path.exists():
        return []
    text = spec_path.read_text(encoding="utf-8")
    dtypes = set()
    dtype_map = {
        r"(?:float16|fp16|half)": "float16",
        r"(?:float32|fp32)": "float32",
        r"(?:bfloat16|bf16)": "bfloat16",
    }
    for pattern, normalized in dtype_map.items():
        if re.search(pattern, text, re.IGNORECASE):
            dtypes.add(normalized)
    return sorted(dtypes)


def extract_dtypes_from_test(test_path):
    if not test_path.exists():
        return []
    text = test_path.read_text(encoding="utf-8")
    dtypes = set()
    for pattern, dtype in [
        (r"torch\.float16", "float16"), (r"torch\.half", "float16"),
        (r"torch\.float32", "float32"), (r"torch\.float", "float32"),
        (r"torch\.bfloat16", "bfloat16"),
    ]:
        if re.search(pattern, text):
            dtypes.add(dtype)
    return sorted(dtypes)


def check_single(op_name, op_dir):
    issues = []
    checks_passed = []
    golden_path = op_dir / f"{op_name}_golden.py"
    impl_path = op_dir / f"{op_name}_impl.py"
    test_path = op_dir / f"test_{op_name}.py"
    spec_path = op_dir / "spec.md"

    if impl_path.exists():
        funcs = extract_function_names(impl_path)
        if any(op_name in f for f in funcs):
            checks_passed.append("impl_has_entry_function")
        else:
            issues.append({"check": "impl_entry_function", "detail": f"No function containing '{op_name}' in {impl_path.name}. Found: {funcs}"})
    else:
        issues.append({"check": "impl_exists", "detail": f"{impl_path.name} not found"})

    if test_path.exists():
        test_text = test_path.read_text(encoding="utf-8")
        if f"{op_name}_golden" in test_text:
            checks_passed.append("test_imports_golden")
        else:
            issues.append({"check": "test_imports_golden", "detail": f"test file does not reference '{op_name}_golden'"})
        if op_name in test_text and "impl" in test_text:
            checks_passed.append("test_imports_impl")
        else:
            issues.append({"check": "test_imports_impl", "detail": "test file does not reference impl module"})
    else:
        issues.append({"check": "test_exists", "detail": f"{test_path.name} not found"})

    if spec_path.exists() and test_path.exists():
        spec_dtypes = extract_dtypes_from_spec(spec_path)
        test_dtypes = extract_dtypes_from_test(test_path)
        if spec_dtypes:
            missing_dtypes = [d for d in spec_dtypes if d not in test_dtypes]
            if missing_dtypes:
                issues.append({"check": "dtype_coverage", "detail": f"spec declares {spec_dtypes}, test covers {test_dtypes}, missing: {missing_dtypes}"})
            else:
                checks_passed.append("dtype_coverage")

    if golden_path.exists() and impl_path.exists():
        golden_func = f"{op_name}_golden"
        impl_funcs = extract_function_names(impl_path)
        impl_func = next((f for f in impl_funcs if op_name in f), None)
        if impl_func:
            golden_args = extract_function_args(golden_path, golden_func)
            impl_args = extract_function_args(impl_path, impl_func)
            if golden_args is not None and impl_args is not None:
                g = [a for a in golden_args if a != "self"]
                i = [a for a in impl_args if a not in ("self", "out")]
                if len(g) == len(i):
                    checks_passed.append("arg_count_match")
                else:
                    issues.append({"check": "arg_count_match", "detail": f"golden({golden_func}) has {len(g)} args {g}, impl({impl_func}) has {len(i)} args {i}"})

    return {"op_name": op_name, "consistent": len(issues) == 0, "checks_passed": checks_passed, "issues": issues}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dir", help="Single operator directory")
    parser.add_argument("--op", help="Operator name")
    parser.add_argument("--operators-dir", help="Parent directory containing operator dirs")
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()

    if args.check_all and args.operators_dir:
        operators_dir = Path(args.operators_dir)
        results = []
        for d in sorted(operators_dir.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "sources":
                r = check_single(d.name, d)
                results.append(r)
        summary = {"total": len(results), "consistent": sum(1 for r in results if r["consistent"]), "inconsistent": sum(1 for r in results if not r["consistent"]), "results": results}
        print(json.dumps(summary, ensure_ascii=False))
        sys.exit(1 if any(not r["consistent"] for r in results) else 0)
    elif args.op_dir and args.op:
        result = check_single(args.op, Path(args.op_dir))
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result["consistent"] else 1)
    else:
        parser.error("必须指定 --op-dir --op 或 --operators-dir --check-all")


if __name__ == "__main__":
    main()
