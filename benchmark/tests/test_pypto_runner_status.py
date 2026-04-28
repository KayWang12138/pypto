#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for PyPTO workflow completion classification."""

from __future__ import annotations

from benchmark.pypto_runner import (
    PyptoRunResult,
    PyptoRunStatus,
    _attempt_log_file,
    _attempt_logs,
    _state_all_stages_completed,
    _state_has_failed_stage,
    _state_incomplete_without_failure,
    render_prompt,
)


def test_state_stage_completion_helpers() -> None:
    assert _state_has_failed_stage({"stage_status": {"1": "completed", "2": "failed"}})
    assert not _state_all_stages_completed({"stage_status": {"1": "completed", "2": "failed"}})
    assert not _state_all_stages_completed({"stage_status": {"1": "completed", "2": "completed"}})
    assert _state_all_stages_completed(
        {"stage_status": {str(i): "completed" for i in range(1, 8)}}
    )
    assert _state_incomplete_without_failure(
        {"stage_status": {"1": "completed", "2": "in_progress"}}
    )
    assert not _state_incomplete_without_failure(
        {"stage_status": {"1": "completed", "2": "failed"}}
    )
    assert _state_incomplete_without_failure(None)


def test_attempt_log_file_suffix(tmp_path) -> None:
    log_file = tmp_path / "pypto_run.log"
    assert _attempt_log_file(log_file, 1) == log_file
    assert _attempt_log_file(log_file, 2) == tmp_path / "pypto_run.attempt2.log"
    assert _attempt_logs(log_file) == [log_file]
    assert _attempt_logs(None) == []


def test_run_result_serializes_retry_metadata(tmp_path) -> None:
    log_file = tmp_path / "pypto_run.attempt2.log"
    result = PyptoRunResult(
        op_name="Foo",
        status=PyptoRunStatus.SUCCESS,
        workdir=tmp_path / "Foo",
        log_file=log_file,
        attempt_log_files=[tmp_path / "pypto_run.log", log_file],
        retry_count=1,
    )
    data = result.to_dict()
    assert data["retry_count"] == 1
    assert data["attempt_log_files"] == [
        str(tmp_path / "pypto_run.log"),
        str(log_file),
    ]


def test_modelnew_prompt_requires_nn_module_and_to() -> None:
    prompt = render_prompt(
        "Foo",
        "custom/level1/Foo",
        task_desc_rel="custom/level1/Foo/task_desc.py",
        init_args_repr="[]",
        model_init_source="def __init__(self): pass",
        forward_source="def forward(self, x): return x",
    )

    assert "ModelNew` 必须继承 `torch.nn.Module`" in prompt
    assert "assert isinstance(model, nn.Module)" in prompt
    assert "assert hasattr(model, \"to\")" in prompt
    assert "torch.Size([])" in prompt
    assert "torch.Size([1])" in prompt
    assert "no_marker" in prompt
    assert "state_transition" in prompt
    assert "failed" in prompt


if __name__ == "__main__":
    test_state_stage_completion_helpers()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        test_attempt_log_file_suffix(Path(tmp))
        test_run_result_serializes_retry_metadata(Path(tmp))
    test_modelnew_prompt_requires_nn_module_and_to()
