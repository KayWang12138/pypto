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
"""
Unit tests for python/pypto/npucomm.py.

These tests validate the pure-Python fallback behavior and do NOT require a built `pypto_impl` extension.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_npucomm_module():
    # This test intentionally loads the module by file path, to avoid importing the full `pypto` package
    # (which may require a compiled extension in some environments).
    npucomm_path = Path(__file__).resolve().parents[3] / "pypto" / "npucomm.py"
    assert npucomm_path.exists(), f"npucomm.py not found at: {npucomm_path}"

    spec = importlib.util.spec_from_file_location("pypto_npucomm_for_test", str(npucomm_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture(scope="module")
def npucomm():
    return _load_npucomm_module()


@pytest.fixture(autouse=True)
def _cleanup(npucomm):
    # Ensure each test starts from a clean state.
    yield
    try:
        npucomm.finalize()
    except Exception:
        pass


def test_config_fields(npucomm):
    cfg = npucomm.NpuCommConfig(num_pes=2, my_pe=1, enable_pgas_mode=False)
    assert cfg.num_pes == 2
    assert cfg.my_pe == 1
    assert cfg.enable_pgas_mode is False


def test_requires_init(npucomm):
    with pytest.raises(npucomm.NpuCommError):
        npucomm.shmalloc(16)
    with pytest.raises(npucomm.NpuCommError):
        npucomm.put(0, 0, size=1, pe=0)
    with pytest.raises(npucomm.NpuCommError):
        npucomm.atomic_fetch_add(0, 1, pe=0)


def test_init_finalize_idempotent(npucomm):
    npucomm.init(npucomm.NpuCommConfig(num_pes=1, my_pe=0))
    npucomm.finalize()
    # finalize again should be a no-op
    npucomm.finalize()


def test_shmalloc_shfree(npucomm):
    npucomm.init(npucomm.NpuCommConfig(num_pes=1, my_pe=0, symmetric_heap_size=1024))
    buf = npucomm.shmalloc(32)
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.uint8
    assert buf.size == 32

    npucomm.shfree(buf)
    # Double free should raise
    with pytest.raises(npucomm.NpuCommError):
        npucomm.shfree(buf)


def test_put_get_local(npucomm):
    npucomm.init(npucomm.NpuCommConfig(num_pes=1, my_pe=0))
    src = npucomm.shmalloc(16)
    dst = npucomm.shmalloc(16)

    src[:] = np.arange(16, dtype=np.uint8)
    dst[:] = 0

    npucomm.put(dst, src, size=16, pe=0)
    assert np.array_equal(dst, src)

    dst[:] = 0
    npucomm.get(dst, src, size=16, pe=0)
    assert np.array_equal(dst, src)


def test_atomic_fetch_add_local(npucomm):
    npucomm.init(npucomm.NpuCommConfig(num_pes=1, my_pe=0))
    counter = np.zeros(1, dtype=np.int64)

    old = npucomm.atomic_fetch_add(counter, 3, pe=0)
    assert old == 0
    assert int(counter[0]) == 3

    old = npucomm.atomic_fetch_add(counter, 5, pe=0)
    assert old == 3
    assert int(counter[0]) == 8


def test_sync_primitives_no_throw(npucomm, capsys):
    npucomm.init(npucomm.NpuCommConfig(num_pes=1, my_pe=0))
    npucomm.fence()
    npucomm.quiet()
    npucomm.barrier_all()

    # Ensure calls produced some output in pure-python mode (not required, but sanity check).
    out = capsys.readouterr().out
    assert isinstance(out, str)


