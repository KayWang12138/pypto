import json
import os
import sys
import subprocess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from csv_ops import FIELDS

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "add_op.py")
HEADER = ",".join(FIELDS) + "\n"

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER)
    return str(p)

@pytest.fixture
def csv_with_existing(tmp_path):
    import csv as csv_module
    p = tmp_path / "scan_results.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerow({"op_name": "softmax", "source": "manual", "status": "pending",
                         "complexity": "medium", "category": "normalization",
                         "description": "归一化指数函数",
                         "fail_count": "0", "fps_total": "0", "fps_confirmed": "0",
                         "create_time": "2026-03-26T09:00:00"})
    return str(p)

def run_add_op(csv, op, source="manual", complexity="easy", category="elementwise",
               description="test op"):
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--csv", csv, "--op", op,
         "--source", source, "--complexity", complexity, "--category", category,
         "--description", description],
        capture_output=True, text=True
    )
    return result

def test_add_new_op_created(csv_path):
    r = run_add_op(csv_path, "relu")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["action"] == "created"
    assert data["op_name"] == "relu"

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
    assert rows[0]["description"] == "test op"

def test_add_duplicate_returns_skipped(csv_with_existing):
    r = run_add_op(csv_with_existing, "softmax")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["action"] == "skipped"
    assert data["reason"] == "already_in_csv"
    import csv as csv_module
    with open(csv_with_existing) as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == 1

def test_add_op_source_auto_discovered(csv_path):
    r = run_add_op(csv_path, "gelu", source="auto_discovered", complexity="medium", category="activation")
    assert r.returncode == 0
    assert json.loads(r.stdout)["action"] == "created"

def test_complete_artifacts_returns_skipped(tmp_path):
    op_dir = tmp_path / "custom" / "gelu"
    op_dir.mkdir(parents=True)
    (op_dir / "gelu_impl.py").write_text("impl")
    (op_dir / "test_gelu.py").write_text("test")
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER)
    r = run_add_op(str(csv_path), "gelu")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["action"] == "skipped"
    assert data["reason"] == "complete_artifacts_exist"

def test_add_op_with_requirement_fields(csv_path):
    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--csv", csv_path, "--op", "gated_delta_net",
         "--source", "manual", "--complexity", "hard", "--category", "attention",
         "--requirement", "sources/gated_delta_net.md",
         "--reference", "transformers/models/qwen3_5/modeling_qwen3_5.py::GatedDeltaNet",
         "--description", "线性注意力变体，门控 Delta 规则"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert json.loads(r.stdout)["action"] == "created"
    import csv as csv_module
    with open(csv_path) as f:
        rows = list(csv_module.DictReader(f))
    row = rows[0]
    assert row["requirement"] == "sources/gated_delta_net.md"
    assert row["reference"] == "transformers/models/qwen3_5/modeling_qwen3_5.py::GatedDeltaNet"
    assert row["description"] == "线性注意力变体，门控 Delta 规则"

def test_partial_dir_still_creates(tmp_path):
    op_dir = tmp_path / "custom" / "gelu"
    op_dir.mkdir(parents=True)
    (op_dir / "spec.md").write_text("spec only")
    csv_path = tmp_path / "scan_results.csv"
    csv_path.write_text(HEADER)
    r = run_add_op(str(csv_path), "gelu")
    assert r.returncode == 0
    assert json.loads(r.stdout)["action"] == "created"

def test_description_required(csv_path):
    r = subprocess.run(
        [sys.executable, SCRIPT,
         "--csv", csv_path, "--op", "relu",
         "--source", "manual", "--complexity", "easy", "--category", "elementwise"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
