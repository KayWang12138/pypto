#!/usr/bin/env python3

from pathlib import Path

import pytest
import torch

from pypto_gpu_vk.compile import compile_op


def test_compile_op_produces_real_spirv():
    compiled = compile_op("mul", torch.randn(32), torch.randn(32))
    assert compiled.supports_real_vulkan is True
    assert compiled.artifact.is_real_spirv is True
    assert compiled.artifact.spirv[0] == 0x07230203
    assert compiled.artifact.source_hash
    assert Path(compiled.artifact.glsl_path).is_file()
    assert Path(compiled.artifact.spirv_path).is_file()
    assert len(compiled.dump_spirv()) >= 20


def test_compile_where_op_produces_real_spirv():
    compiled = compile_op("where", torch.randn(32), torch.randn(32), torch.randn(32))
    assert compiled.artifact.is_real_spirv is True
    assert compiled.artifact.spirv[0] == 0x07230203


def test_compile_op_reports_missing_validator(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="TOOL_NOT_FOUND"):
        compile_op(
            "mul",
            torch.randn(8),
            torch.randn(8),
            options={"glslang_validator_path": "/tmp/does-not-exist/glslangValidator"},
        )
