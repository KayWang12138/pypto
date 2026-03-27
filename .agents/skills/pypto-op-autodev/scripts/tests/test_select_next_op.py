import csv as csv_module
import json
import os
import sys
import subprocess
import pytest
from datetime import datetime, timezone, timedelta

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "select_next_op.py")
HEADER = "op_name,source,status,complexity,category,dev_result,fail_count,fps_total,fps_confirmed,create_time,start_time,end_time,depends_on,note\n"

def ts(delta_days=0, delta_hours=0):
    dt = datetime.now(timezone.utc) - timedelta(days=delta_days, hours=delta_hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def make_row(op, status="pending", complexity="easy", category="elementwise",
             fail_count=0, dev_result="", create_days_ago=0, end_hours_ago=0,
             depends_on=""):
    create = ts(delta_days=create_days_ago)
    end = ts(delta_hours=end_hours_ago) if end_hours_ago else ""
    return (f"{op},manual,{status},{complexity},{category},{dev_result},"
            f"{fail_count},0,0,{create},,{end},{depends_on},\n")

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
    p.write_text(HEADER + make_row("easy_op", complexity="easy") + make_row("hard_op", complexity="hard"))
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "easy_op"

def test_fail_penalty_lowers_score(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("clean_op", fail_count=0) + make_row("failed_op", status="failed", fail_count=3, end_hours_ago=50))
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "clean_op"

def test_age_bonus_increases_score(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("old_failed", status="failed", fail_count=1, create_days_ago=10, end_hours_ago=50) + make_row("new_clean", fail_count=0, create_days_ago=0))
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["score"] >= 25

def test_score_below_threshold_returns_exit1(tmp_path):
    p = tmp_path / "scan_results.csv"
    rows = HEADER
    rows += make_row("low_score", status="failed", complexity="hard", fail_count=3, end_hours_ago=5)
    for i in range(6):
        rows += make_row(f"done_{i}", status="completed", complexity="easy", end_hours_ago=i + 1)
    p.write_text(rows)
    r = run(str(p))
    assert r.returncode == 1

def test_complexity_escalation_prefers_medium_after_many_easy(tmp_path):
    """After 5+ easy ops completed, medium should score higher than new easy."""
    p = tmp_path / "scan_results.csv"
    rows = HEADER
    # 6 completed easy ops
    for i in range(6):
        rows += make_row(f"easy_done_{i}", status="completed", complexity="easy",
                         category=f"cat{i}", end_hours_ago=i + 1)
    # 1 pending easy, 1 pending medium
    rows += make_row("new_easy", complexity="easy", category="newcat1")
    rows += make_row("new_medium", complexity="medium", category="newcat2")
    p.write_text(rows)
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "new_medium"

def test_complexity_escalation_prefers_hard_after_easy_and_medium(tmp_path):
    """After 3+ easy and 2+ medium completed, hard should get a bonus."""
    p = tmp_path / "scan_results.csv"
    rows = HEADER
    for i in range(4):
        rows += make_row(f"easy_{i}", status="completed", complexity="easy",
                         category=f"cat{i}", end_hours_ago=i + 10)
    for i in range(3):
        rows += make_row(f"med_{i}", status="completed", complexity="medium",
                         category=f"mcat{i}", end_hours_ago=i + 10)
    rows += make_row("new_easy", complexity="easy", category="xcat1")
    rows += make_row("new_hard", complexity="hard", category="xcat2")
    p.write_text(rows)
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "new_hard"

def test_dry_run_does_not_modify_csv(csv_with_single_easy):
    import hashlib
    def md5(path):
        return hashlib.md5(open(path, "rb").read()).hexdigest()
    before = md5(csv_with_single_easy)
    subprocess.run([sys.executable, SCRIPT, "--csv", csv_with_single_easy, "--dry-run"], capture_output=True)
    assert md5(csv_with_single_easy) == before

def test_resume_from_stage_1_no_artifacts(csv_with_single_easy):
    r = run(csv_with_single_easy)
    data = json.loads(r.stdout)
    assert data["selected"]["resume_from_stage"] == 1
    assert data["selected"]["existing_artifacts"] == []

def test_resume_detects_existing_artifacts(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("myop", status="failed", fail_count=1, end_hours_ago=50))
    op_dir = tmp_path / "custom" / "myop"
    op_dir.mkdir(parents=True)
    (op_dir / "spec.md").write_text("# spec")
    (op_dir / "api_report.md").write_text("# api")
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["resume_from_stage"] == 3
    assert "spec.md" in data["selected"]["existing_artifacts"]

def test_resume_with_golden(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("myop", status="failed", fail_count=1, end_hours_ago=50))
    op_dir = tmp_path / "custom" / "myop"
    op_dir.mkdir(parents=True)
    for f in ["spec.md", "api_report.md", "design.md"]: (op_dir / f).write_text("x")
    (op_dir / "myop_golden.py").write_text("def myop_golden(): pass")
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["resume_from_stage"] == 5

def test_dependency_blocks_selection(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("child_op", depends_on="parent_op") + make_row("parent_op", status="pending"))
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "parent_op"

def test_dependency_met_allows_selection(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("child_op", depends_on="parent_op") + make_row("parent_op", status="completed"))
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "child_op"

def test_dep_blocked_reported_in_discovery(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("child_op", depends_on="missing_parent"))
    r = run(str(p))
    assert r.returncode == 1
    data = json.loads(r.stdout)
    assert len(data["dep_blocked"]) == 1

def test_retry_strategy_for_precision_failure(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("prec_op", status="failed", dev_result="PRECISION", fail_count=1, end_hours_ago=50))
    op_dir = tmp_path / "custom" / "prec_op"
    op_dir.mkdir(parents=True)
    for f in ["spec.md", "api_report.md", "design.md"]: (op_dir / f).write_text("x")
    (op_dir / "prec_op_golden.py").write_text("def prec_op_golden(): pass")
    (op_dir / "prec_op_impl.py").write_text("def prec_op_impl(): pass")
    (op_dir / "test_prec_op.py").write_text("# test")
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["retry_strategy"]["action"] == "retry_from_stage6"
    assert data["selected"]["resume_from_stage"] == 6

def test_blocked_api_gets_heavy_penalty(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("blocked_op", status="failed", dev_result="BLOCKED_API", fail_count=1, end_hours_ago=50) + make_row("clean_op", status="pending"))
    r = run(str(p))
    data = json.loads(r.stdout)
    assert data["selected"]["op_name"] == "clean_op"

def test_excludes_op_with_complete_artifacts(tmp_path):
    p = tmp_path / "scan_results.csv"
    p.write_text(HEADER + make_row("done_op", status="pending"))
    op_dir = tmp_path / "custom" / "done_op"
    op_dir.mkdir(parents=True)
    (op_dir / "done_op_impl.py").write_text("impl")
    (op_dir / "test_done_op.py").write_text("test")
    r = run(str(p))
    assert r.returncode == 1
