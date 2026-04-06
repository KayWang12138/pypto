import json
import os
import sys
import subprocess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from csv_ops import FIELDS

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "verify_op.py")
HEADER = ",".join(FIELDS) + "\n"

def make_csv_row(op, status="completed"):
    row = {f: "" for f in FIELDS}
    row.update({"op_name": op, "source": "manual", "status": status,
                "complexity": "easy", "category": "elementwise",
                "dev_result": "SUCCESS", "fail_count": "0",
                "fps_total": "0", "fps_confirmed": "0",
                "create_time": "2026-03-26T09:00:00"})
    return ",".join(row.get(f, "") for f in FIELDS) + "\n"

def run(args):
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True)

@pytest.fixture
def setup_op(tmp_path):
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER + make_csv_row("myop"))
    op_dir = tmp_path / "custom" / "myop"
    op_dir.mkdir(parents=True)
    (op_dir / "spec.md").write_text("# spec")
    (op_dir / "design.md").write_text("# design")
    (op_dir / "myop_golden.py").write_text("def myop_golden(x): return x")
    (op_dir / "myop_impl.py").write_text("def myop_impl(x): return x")
    (op_dir / "test_myop.py").write_text("import sys\nprint('[PRECISION_PASS]')\nsys.exit(0)\n")
    return str(csv_path), str(op_dir)

def test_verify_artifacts_ok(setup_op):
    csv_path, _ = setup_op
    r = run(["--csv", csv_path, "--op", "myop"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["verify_status"] == "ARTIFACTS_OK"

def test_verify_missing_artifact(tmp_path):
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER + make_csv_row("myop"))
    op_dir = tmp_path / "custom" / "myop"
    op_dir.mkdir(parents=True)
    (op_dir / "spec.md").write_text("spec")
    r = run(["--csv", str(csv_path), "--op", "myop"])
    assert r.returncode == 1
    data = json.loads(r.stdout)
    assert data["verify_status"] == "INCOMPLETE"

def test_verify_no_dir(tmp_path):
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER + make_csv_row("ghost_op"))
    r = run(["--csv", str(csv_path), "--op", "ghost_op"])
    assert r.returncode == 1
    assert json.loads(r.stdout)["verify_status"] == "NO_DIR"

def test_verify_run_test_pass(setup_op):
    csv_path, _ = setup_op
    r = run(["--csv", csv_path, "--op", "myop", "--run-test"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["verify_status"] == "VERIFIED"

def test_verify_run_test_fail(tmp_path):
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER + make_csv_row("failop"))
    op_dir = tmp_path / "custom" / "failop"
    op_dir.mkdir(parents=True)
    for f in ["spec.md", "design.md"]: (op_dir / f).write_text("x")
    (op_dir / "failop_golden.py").write_text("def failop_golden(): pass")
    (op_dir / "failop_impl.py").write_text("def failop_impl(): pass")
    (op_dir / "test_failop.py").write_text("import sys; sys.exit(1)")
    r = run(["--csv", str(csv_path), "--op", "failop", "--run-test"])
    assert r.returncode == 1
    assert json.loads(r.stdout)["verify_status"] == "TEST_FAIL"

def test_verify_updates_csv(setup_op):
    csv_path, _ = setup_op
    run(["--csv", csv_path, "--op", "myop"])
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["verify_status"] == "ARTIFACTS_OK"

def test_verify_all(tmp_path):
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER + make_csv_row("op1") + make_csv_row("op2"))
    for op in ["op1", "op2"]:
        op_dir = tmp_path / "custom" / op
        op_dir.mkdir(parents=True)
        (op_dir / "spec.md").write_text("spec")
        (op_dir / "design.md").write_text("design")
        (op_dir / f"{op}_golden.py").write_text(f"def {op}_golden(): pass")
        (op_dir / f"{op}_impl.py").write_text(f"def {op}_impl(): pass")
        (op_dir / f"test_{op}.py").write_text("pass")
    r = run(["--csv", str(csv_path), "--verify-all"])
    assert r.returncode == 0
    assert json.loads(r.stdout)["verified"] == 2

def test_verify_op_not_in_csv(tmp_path):
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER)
    r = run(["--csv", str(csv_path), "--op", "nonexistent"])
    assert r.returncode == 2

def test_verify_accepts_compat_cli_args(setup_op):
    csv_path, op_dir = setup_op
    r = run(["--csv", csv_path, "--op", "myop", "--op-dir", op_dir, "--timeout", "10"])
    assert r.returncode == 0
    assert json.loads(r.stdout)["verify_status"] == "ARTIFACTS_OK"
