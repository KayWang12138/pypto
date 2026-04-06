import csv as csv_module
import json
import os
import sys
import subprocess
import pytest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from csv_ops import FIELDS

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "update_op.py")
HEADER = ",".join(FIELDS) + "\n"

def make_row(op, status="pending", fail_count=0, start_time="", end_time="", dev_result=""):
    now = "2026-03-26T09:00:00"
    row = {f: "" for f in FIELDS}
    row.update({"op_name": op, "source": "manual", "status": status,
                "complexity": "easy", "category": "elementwise",
                "dev_result": dev_result, "fail_count": str(fail_count),
                "fps_total": "0", "fps_confirmed": "0",
                "create_time": now, "start_time": start_time, "end_time": end_time})
    return ",".join(row.get(f, "") for f in FIELDS) + "\n"

@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("softmax", status="pending"))
    return str(p)

@pytest.fixture
def csv_in_progress(tmp_path):
    p = tmp_path / "scan_results.csv"
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%S")
    active_time = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(
        HEADER +
        make_row("stale_op", status="in_progress", start_time=stale_time) +
        make_row("active_op", status="in_progress", start_time=active_time)
    )
    return str(p)

def run(args):
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True)

def read(path):
    with open(path) as f:
        return list(csv_module.DictReader(f))

def test_mode1_set_in_progress(csv_path):
    r = run(["--csv", csv_path, "--op", "softmax", "--status", "in_progress"])
    assert r.returncode == 0
    rows = read(csv_path)
    assert rows[0]["status"] == "in_progress"
    assert rows[0]["start_time"] != ""

def test_mode1_set_completed(csv_path):
    r = run(["--csv", csv_path, "--op", "softmax", "--status", "completed",
             "--dev-result", "SUCCESS", "--fps-total", "0", "--fps-confirmed", "0"])
    assert r.returncode == 0
    rows = read(csv_path)
    assert rows[0]["status"] == "completed"
    assert rows[0]["dev_result"] == "SUCCESS"
    assert rows[0]["end_time"] != ""

def test_mode1_set_failed_increments_fail_count(csv_path):
    run(["--csv", csv_path, "--op", "softmax", "--status", "failed",
         "--dev-result", "BLOCKED_API"])
    rows = read(csv_path)
    assert rows[0]["fail_count"] == "1"
    assert rows[0]["end_time"] != ""

def test_mode1_set_failed_twice_increments_twice(csv_path):
    run(["--csv", csv_path, "--op", "softmax", "--status", "failed", "--dev-result", "PRECISION"])
    run(["--csv", csv_path, "--op", "softmax", "--status", "failed", "--dev-result", "TIMEOUT"])
    rows = read(csv_path)
    assert rows[0]["fail_count"] == "2"

def test_mode1_output_is_json(csv_path):
    r = run(["--csv", csv_path, "--op", "softmax", "--status", "in_progress"])
    data = json.loads(r.stdout)
    assert data["op_name"] == "softmax"

def test_mode1_blocked_stage(csv_path):
    r = run(["--csv", csv_path, "--op", "softmax", "--status", "failed",
             "--dev-result", "PRECISION", "--blocked-stage", "6"])
    rows = read(csv_path)
    assert rows[0]["blocked_stage"] == "6"

def test_mode1_last_strategy(csv_path):
    r = run(["--csv", csv_path, "--op", "softmax", "--status", "completed",
             "--dev-result", "SUCCESS", "--last-strategy", "workflow"])
    assert r.returncode == 0
    rows = read(csv_path)
    assert rows[0]["last_strategy"] == "workflow"

def test_mode2_reset_clears_dev_result(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("mul", status="failed", fail_count=2, dev_result="PRECISION",
                                    end_time="2026-03-26T10:00:00"))
    r = run(["--csv", str(p), "--op", "mul", "--reset"])
    assert r.returncode == 0
    rows = read(str(p))
    assert rows[0]["status"] == "pending"
    assert rows[0]["dev_result"] == ""
    assert rows[0]["start_time"] == ""
    assert rows[0]["end_time"] == ""

def test_mode2_reset_preserves_fail_count(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("mul", status="failed", fail_count=2))
    run(["--csv", str(p), "--op", "mul", "--reset"])
    rows = read(str(p))
    assert rows[0]["fail_count"] == "2"

def test_mode3_resets_stale_op(csv_in_progress):
    r = run(["--csv", csv_in_progress, "--reset-stale", "--timeout-hours", "6"])
    data = json.loads(r.stdout)
    assert "stale_op" in data["reset_ops"]

def test_mode3_active_op_returns_exit1(csv_in_progress):
    r = run(["--csv", csv_in_progress, "--reset-stale", "--timeout-hours", "6"])
    assert r.returncode == 1
    data = json.loads(r.stdout)
    assert data["has_active"] is True
    assert data["active_op"] == "active_op"

def test_mode3_no_in_progress_returns_exit0(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("add", status="pending"))
    r = run(["--csv", str(p), "--reset-stale", "--timeout-hours", "6"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["reset_ops"] == []
    assert data["has_active"] is False
