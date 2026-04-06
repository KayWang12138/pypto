"""detect_dev_success 回归测试。

覆盖场景：
- 合法 SUCCESS → 返回成功
- 合法 FAILED / BLOCKED → 返回失败
- 非法 PRECISION_PASS → 返回失败（不能被误判为成功）
- 文件不存在 → 返回 TIMEOUT
- JSON 格式错误 → 返回 TIMEOUT
"""

import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scheduler import detect_dev_success


@pytest.fixture
def op_dir(tmp_path):
    return tmp_path


def _write_result(op_dir, data):
    (op_dir / ".dev_result.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def test_success(op_dir):
    _write_result(op_dir, {"status": "SUCCESS", "notes": ""})
    ok, detail, stage = detect_dev_success(op_dir, logging.getLogger())
    assert ok is True
    assert detail == "SUCCESS"


def test_failed(op_dir):
    _write_result(op_dir, {"status": "FAILED", "notes": "compile error"})
    ok, detail, stage = detect_dev_success(op_dir, logging.getLogger())
    assert ok is False


def test_blocked(op_dir):
    _write_result(op_dir, {
        "status": "BLOCKED",
        "blocked_reason": "BLOCKED_API",
    })
    ok, detail, stage = detect_dev_success(op_dir, logging.getLogger())
    assert ok is False
    assert "BLOCKED_API" in detail


def test_illegal_precision_pass_is_not_success(op_dir):
    """非法 status PRECISION_PASS 不能被判为成功。"""
    _write_result(op_dir, {"status": "PRECISION_PASS", "notes": ""})
    ok, detail, stage = detect_dev_success(op_dir, logging.getLogger())
    assert ok is False, "PRECISION_PASS 不应被判为 SUCCESS"


def test_missing_file(op_dir):
    ok, detail, stage = detect_dev_success(op_dir, logging.getLogger())
    assert ok is False
    assert detail == "TIMEOUT"


def test_invalid_json(op_dir):
    (op_dir / ".dev_result.json").write_text("not json", encoding="utf-8")
    ok, detail, stage = detect_dev_success(op_dir, logging.getLogger())
    assert ok is False
    assert detail == "TIMEOUT"
