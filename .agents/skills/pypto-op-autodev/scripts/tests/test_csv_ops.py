import csv
import os
import sys
import tempfile
import pytest

# 让 Python 能找到 data 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.csv_ops import read_csv, upsert_row

HEADER = "op_name,source,status,complexity,category,dev_result,fail_count,fps_total,fps_confirmed,create_time,start_time,end_time,note"

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + "\n")
    return str(p)

@pytest.fixture
def csv_with_data(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER + "\n"
        "add_basic,manual,completed,easy,elementwise,SUCCESS,0,0,0,"
        "2026-03-26T09:00:00,2026-03-26T10:00:00,2026-03-26T10:15:00,\n"
        "softmax,auto_discovered,pending,medium,normalization,,0,,,2026-03-26T09:30:00,,,\n"
    )
    return str(p)

# --- read_csv ---

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

# --- upsert_row ---

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
    # 只更新 status，其他字段保持原值
    upsert_row(csv_with_data, "add_basic", status="failed")
    rows = read_csv(csv_with_data, op_name="add_basic")
    assert rows[0]["source"] == "manual"       # 原值保留
    assert rows[0]["complexity"] == "easy"      # 原值保留
