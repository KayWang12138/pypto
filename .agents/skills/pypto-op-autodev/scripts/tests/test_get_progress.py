import json
import os
import sys
import subprocess
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "get_progress.py")
HEADER = "op_name,source,status,complexity,category,dev_result,fail_count,fps_total,fps_confirmed,create_time,start_time,end_time,note\n"

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER +
        "add,manual,completed,easy,elementwise,SUCCESS,0,0,0,2026-03-26T09:00:00,,,\n"
        "mul,manual,completed,easy,elementwise,SUCCESS,0,1,1,2026-03-26T09:00:00,,,\n"
        "softmax,manual,failed,medium,normalization,BLOCKED_API,1,2,1,2026-03-26T09:00:00,,,\n"
        "gelu,auto_discovered,pending,medium,activation,,0,0,0,2026-03-26T09:00:00,,,\n"
    )
    return str(p)

def run(csv):
    return subprocess.run([sys.executable, SCRIPT, "--csv", csv], capture_output=True, text=True)

def test_total_count(csv_path):
    data = json.loads(run(csv_path).stdout)
    assert data["total"] == 4

def test_by_status(csv_path):
    data = json.loads(run(csv_path).stdout)
    assert data["by_status"]["completed"] == 2
    assert data["by_status"]["failed"] == 1
    assert data["by_status"]["pending"] == 1

def test_by_category(csv_path):
    data = json.loads(run(csv_path).stdout)
    assert data["by_category"]["elementwise"] == 2

def test_success_rate(csv_path):
    data = json.loads(run(csv_path).stdout)
    assert abs(data["success_rate"] - 2/3) < 0.01

def test_fps_totals(csv_path):
    data = json.loads(run(csv_path).stdout)
    assert data["fps_total"] == 3
    assert data["fps_confirmed"] == 2

def test_empty_csv(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER)
    data = json.loads(run(str(p)).stdout)
    assert data["total"] == 0
    assert data["success_rate"] == 0
