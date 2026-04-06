import csv
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from csv_ops import read_csv, upsert_row, FIELDS

HEADER = ",".join(FIELDS)

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + "\n")
    return str(p)

@pytest.fixture
def csv_with_data(tmp_path):
    p = tmp_path / "scan_results.csv"
    row1_vals = {"op_name": "add_basic", "source": "manual", "status": "completed",
                 "complexity": "easy", "category": "elementwise", "dev_result": "SUCCESS",
                 "fail_count": "0", "fps_total": "0", "fps_confirmed": "0",
                 "create_time": "2026-03-26T09:00:00", "start_time": "2026-03-26T10:00:00",
                 "end_time": "2026-03-26T10:15:00"}
    row2_vals = {"op_name": "softmax", "source": "auto_discovered", "status": "pending",
                 "complexity": "medium", "category": "normalization",
                 "fail_count": "0", "create_time": "2026-03-26T09:30:00"}
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerow(row1_vals)
        writer.writerow(row2_vals)
    return str(p)

def test_read_csv_all_rows(csv_with_data):
    rows = read_csv(csv_with_data)
    assert len(rows) == 2

def test_read_csv_filter_status(csv_with_data):
    rows = read_csv(csv_with_data, status="pending")
    assert len(rows) == 1
    assert rows[0]["op_name"] == "softmax"

def test_read_csv_filter_no_match(csv_with_data):
    rows = read_csv(csv_with_data, status="failed")
    assert rows == []

def test_read_csv_empty_file(csv_path):
    rows = read_csv(csv_path)
    assert rows == []

def test_read_csv_multiple_filters(csv_with_data):
    rows = read_csv(csv_with_data, status="completed", source="manual")
    assert len(rows) == 1
    assert rows[0]["op_name"] == "add_basic"

def test_upsert_row_insert_new(csv_path):
    result = upsert_row(csv_path, "relu", status="pending", source="manual",
                        complexity="easy", category="activation",
                        fail_count="0", fps_total="0", fps_confirmed="0",
                        create_time="2026-03-26T10:00:00")
    assert result["op_name"] == "relu"
    rows = read_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"

def test_upsert_row_update_existing(csv_with_data):
    upsert_row(csv_with_data, "add_basic", status="failed", dev_result="PRECISION")
    rows = read_csv(csv_with_data, op_name="add_basic")
    assert rows[0]["status"] == "failed"
    assert rows[0]["dev_result"] == "PRECISION"

def test_upsert_row_creates_backup(csv_with_data):
    upsert_row(csv_with_data, "add_basic", status="failed")
    assert os.path.exists(csv_with_data + ".bak")

def test_upsert_row_preserves_other_rows(csv_with_data):
    upsert_row(csv_with_data, "add_basic", status="failed")
    rows = read_csv(csv_with_data)
    assert len(rows) == 2

def test_upsert_row_partial_update_preserves_fields(csv_with_data):
    upsert_row(csv_with_data, "add_basic", status="failed")
    rows = read_csv(csv_with_data, op_name="add_basic")
    assert rows[0]["source"] == "manual"
    assert rows[0]["complexity"] == "easy"

def test_upsert_row_depends_on(csv_path):
    upsert_row(csv_path, "child_op", status="pending", depends_on="parent1|parent2")
    rows = read_csv(csv_path, op_name="child_op")
    assert rows[0]["depends_on"] == "parent1|parent2"

def test_fields_include_depends_on():
    assert "depends_on" in FIELDS

def test_upsert_handles_old_csv_format(tmp_path):
    old_header = "op_name,source,status,complexity,category,dev_result,fail_count,fps_total,fps_confirmed,create_time,start_time,end_time,note"
    p = tmp_path / "scan_results.csv"
    p.write_text(old_header + "\nrelu,manual,pending,easy,elementwise,,0,0,0,2026-03-26T09:00:00,,,\n")
    upsert_row(str(p), "relu", status="in_progress")
    rows = read_csv(str(p))
    assert rows[0]["status"] == "in_progress"
    assert "depends_on" in rows[0]
