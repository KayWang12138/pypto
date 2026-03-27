import json
import os
import sys
import subprocess
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "add_op.py")
HEADER = "op_name,source,status,complexity,category,dev_result,fail_count,fps_total,fps_confirmed,create_time,start_time,end_time,depends_on,note\n"

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER)
    return str(p)

@pytest.fixture
def csv_with_existing(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER +
        "softmax,manual,pending,medium,normalization,,0,0,0,2026-03-26T09:00:00,,,,\n"
    )
    return str(p)

def run_add_op(csv, op, source="manual", complexity="easy", category="elementwise"):
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--csv", csv, "--op", op,
         "--source", source, "--complexity", complexity, "--category", category],
        capture_output=True, text=True
    )
    return result

def test_add_new_op_success(csv_path):
    r = run_add_op(csv_path, "relu")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["success"] is True
    assert data["op_name"] == "relu"
    assert data["is_new"] is True

def test_add_new_op_defaults(csv_path):
    run_add_op(csv_path, "relu")
    import csv as csv_module
    with open(csv_path) as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["status"] == "pending"
    assert rows[0]["fail_count"] == "0"
    assert rows[0]["fps_total"] == "0"
    assert rows[0]["fps_confirmed"] == "0"
    assert rows[0]["create_time"] != ""

def test_add_duplicate_returns_not_new(csv_with_existing):
    r = run_add_op(csv_with_existing, "softmax")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["is_new"] is False
    import csv as csv_module
    with open(csv_with_existing) as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == 1

def test_add_op_source_auto_discovered(csv_path):
    r = run_add_op(csv_path, "gelu", source="auto_discovered", complexity="medium", category="activation")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["is_new"] is True

def test_add_existing_complete_artifacts_returns_not_new(tmp_path):
    op_dir = tmp_path / "custom" / "gelu"
    op_dir.mkdir(parents=True)
    (op_dir / "gelu_impl.py").write_text("impl")
    (op_dir / "test_gelu.py").write_text("test")
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER)
    r = run_add_op(str(csv_path), "gelu")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["is_new"] is False
    assert data["reason"] == "complete_artifacts_exist"

def test_add_partial_dir_still_adds(tmp_path):
    op_dir = tmp_path / "custom" / "gelu"
    op_dir.mkdir(parents=True)
    (op_dir / "spec.md").write_text("spec only")
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER)
    r = run_add_op(str(csv_path), "gelu")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["is_new"] is True
