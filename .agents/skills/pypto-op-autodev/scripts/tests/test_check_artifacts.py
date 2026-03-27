import json
import os
import sys
import subprocess
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "check_artifacts.py")

def run(args):
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True)

@pytest.fixture
def complete_op(tmp_path):
    op_dir = tmp_path / "myop"
    op_dir.mkdir()
    (op_dir / "spec.md").write_text("支持 float16 和 float32")
    (op_dir / "myop_golden.py").write_text("def myop_golden(x, alpha=1.0): return x * alpha")
    (op_dir / "myop_impl.py").write_text("def myop_impl(x, alpha=1.0): return x * alpha")
    (op_dir / "test_myop.py").write_text("from myop_golden import myop_golden\nfrom myop_impl import myop_impl\nimport torch\nx = torch.randn(4, dtype=torch.float16)\ny = torch.randn(4, dtype=torch.float32)\n")
    return str(op_dir)

def test_consistent_op(complete_op):
    r = run(["--op-dir", complete_op, "--op", "myop"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["consistent"] is True

def test_missing_impl(tmp_path):
    op_dir = tmp_path / "badop"
    op_dir.mkdir()
    (op_dir / "test_badop.py").write_text("pass")
    r = run(["--op-dir", str(op_dir), "--op", "badop"])
    assert r.returncode == 1
    assert any(i["check"] == "impl_exists" for i in json.loads(r.stdout)["issues"])

def test_missing_golden_reference(tmp_path):
    op_dir = tmp_path / "nogold"
    op_dir.mkdir()
    (op_dir / "nogold_impl.py").write_text("def nogold_impl(): pass")
    (op_dir / "test_nogold.py").write_text("from nogold_impl import nogold_impl\nimpl = True")
    r = run(["--op-dir", str(op_dir), "--op", "nogold"])
    assert r.returncode == 1
    assert any(i["check"] == "test_imports_golden" for i in json.loads(r.stdout)["issues"])

def test_dtype_coverage_mismatch(tmp_path):
    op_dir = tmp_path / "dtypeop"
    op_dir.mkdir()
    (op_dir / "spec.md").write_text("支持 float16, float32, bfloat16")
    (op_dir / "dtypeop_impl.py").write_text("def dtypeop_impl(): pass")
    (op_dir / "dtypeop_golden.py").write_text("def dtypeop_golden(): pass")
    (op_dir / "test_dtypeop.py").write_text("from dtypeop_golden import dtypeop_golden\nfrom dtypeop_impl import dtypeop_impl\nimport torch\nx = torch.randn(4, dtype=torch.float32)\n")
    r = run(["--op-dir", str(op_dir), "--op", "dtypeop"])
    assert any(i["check"] == "dtype_coverage" for i in json.loads(r.stdout)["issues"])

def test_arg_count_mismatch(tmp_path):
    op_dir = tmp_path / "argop"
    op_dir.mkdir()
    (op_dir / "argop_golden.py").write_text("def argop_golden(x): return x")
    (op_dir / "argop_impl.py").write_text("def argop_impl(x, y, z): return x + y + z")
    (op_dir / "test_argop.py").write_text("from argop_golden import argop_golden\nfrom argop_impl import argop_impl\n")
    r = run(["--op-dir", str(op_dir), "--op", "argop"])
    assert any(i["check"] == "arg_count_match" for i in json.loads(r.stdout)["issues"])

def test_check_all(tmp_path):
    ops_dir = tmp_path / "operators"
    ops_dir.mkdir()
    for op in ["op1", "op2"]:
        d = ops_dir / op
        d.mkdir()
        (d / f"{op}_impl.py").write_text(f"def {op}_impl(): pass")
        (d / f"{op}_golden.py").write_text(f"def {op}_golden(): pass")
        (d / f"test_{op}.py").write_text(f"from {op}_golden import {op}_golden\nfrom {op}_impl import {op}_impl\n")
    r = run(["--operators-dir", str(ops_dir), "--check-all"])
    data = json.loads(r.stdout)
    assert data["consistent"] == 2

def test_check_all_skips_hidden_dirs(tmp_path):
    ops_dir = tmp_path / "operators"
    ops_dir.mkdir()
    (ops_dir / ".hidden").mkdir()
    (ops_dir / "sources").mkdir()
    r = run(["--operators-dir", str(ops_dir), "--check-all"])
    assert json.loads(r.stdout)["total"] == 0
