import csv as csv_module
import json
import os
import sys
import subprocess
import pytest
from datetime import datetime, timezone, timedelta

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "select_next_op.py")
HEADER = "op_name,source,status,complexity,category,dev_result,fail_count,fps_total,fps_confirmed,create_time,start_time,end_time,note\n"

def ts(delta_days=0, delta_hours=0):
    dt = datetime.now(timezone.utc) - timedelta(days=delta_days, hours=delta_hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def make_row(op, status="pending", complexity="easy", category="elementwise",
             fail_count=0, dev_result="", create_days_ago=0, end_hours_ago=0):
    create = ts(delta_days=create_days_ago)
    end = ts(delta_hours=end_hours_ago) if end_hours_ago else ""
    return (f"{op},manual,{status},{complexity},{category},{dev_result},"
            f"{fail_count},0,0,{create},,{end},\n")

@pytest.fixture
def csv_with_single_easy(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("relu", status="pending", complexity="easy"))
    return str(p)

@pytest.fixture
def csv_empty(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER)
    return str(p)

def run(csv):
    return subprocess.run([sys.executable, SCRIPT, "--csv", csv], capture_output=True, text=True)

def test_selects_single_candidate(csv_with_single_easy):
    r = run(csv_with_single_easy)
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["action"] == "select_from_csv"
    assert data["selected"]["op_name"] == "relu"

def test_no_candidates_returns_exit1(csv_empty):
    r = run(csv_empty)
    assert r.returncode == 1

def test_excludes_in_progress(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("busy_op", status="in_progress"))
    r = run(str(p))
    assert r.returncode == 1

def test_excludes_completed(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("done_op", status="completed"))
    r = run(str(p))
    assert r.returncode == 1

def test_excludes_not_implementable(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("bad_op", status="failed", dev_result="NOT_IMPLEMENTABLE"))
    r = run(str(p))
    assert r.returncode == 1

def test_easy_beats_hard(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER +
        make_row("easy_op", complexity="easy") +
        make_row("hard_op", complexity="hard")
    )
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "easy_op"

def test_fail_penalty_lowers_score(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER +
        make_row("clean_op", fail_count=0) +
        make_row("failed_op", status="failed", fail_count=3, end_hours_ago=50)
    )
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "clean_op"

def test_age_bonus_increases_score(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER +
        make_row("old_failed", status="failed", fail_count=1,
                 create_days_ago=10, end_hours_ago=50) +
        make_row("new_clean", fail_count=0, create_days_ago=0)
    )
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["score"] >= 25

def test_score_below_threshold_returns_exit1(tmp_path):
    # hard + fail=3 + recent_fail: 100-30-60-25 = -15，低于 25
    p = tmp_path / "scan_results.csv"
    p.write_text(
        HEADER +
        make_row("low_score", status="failed", complexity="hard",
                 fail_count=3, end_hours_ago=5)
    )
    r = run(str(p))
    assert r.returncode == 1

def test_dry_run_does_not_modify_csv(csv_with_single_easy):
    import hashlib
    def md5(path):
        return hashlib.md5(open(path, "rb").read()).hexdigest()
    before = md5(csv_with_single_easy)
    subprocess.run([sys.executable, SCRIPT, "--csv", csv_with_single_easy, "--dry-run"],
                   capture_output=True)
    assert md5(csv_with_single_easy) == before

def test_failed_op_with_existing_custom_dir_can_still_retry(tmp_path):
    autodev_dir = tmp_path / "autodev"
    (autodev_dir / "custom" / "relu").mkdir(parents=True)
    csv_path = autodev_dir / "scan_results.csv"
    csv_path.write_text(
        HEADER +
        make_row("relu", status="failed", fail_count=1, end_hours_ago=50)
    )
    r = run(str(csv_path))
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "relu"
