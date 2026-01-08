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
ST test for NPU environment + nvshmem python binding.

- Always validates that the host sees Ascend NPUs via `npu-smi info`.
- If `pypto` + `pypto_impl.nvshmem` are available, also runs a basic init/barrier/finalize flow.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


_RUN_ID = time.strftime("%Y%m%d_%H%M%S")


def _get_log_path() -> Path:
    log_dir = Path(os.environ.get("PYPTO_ST_LOG_DIR", "/tmp/pypto_st_logs"))
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", "NA") or "NA"
    pid = os.getpid()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{_RUN_ID}_npucomm_nvshmem_{worker}_dev{device_id}_pid{pid}.log"


def _st_print(msg: str) -> None:
    """
    Print helper that is friendlier under pytest-xdist:
    - prefixes timestamp
    - flushes immediately (avoid buffered output getting lost/hidden)
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}\n"
    # 1) Best-effort to show in single-process pytest runs
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except Exception:
        pass
    # 2) Always persist to a per-worker log file for xdist (worker stdout may not be forwarded on success)
    try:
        _get_log_path().open("a", encoding="utf-8").write(line)
    except Exception:
        pass


def _print_st_context(tag: str) -> None:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    wid = os.environ.get("PYTEST_XDIST_WORKER_COUNT", "")
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", "")
    pid = os.getpid()
    _st_print(f"[ST][{tag}] worker={worker} worker_count={wid} pid={pid} TILE_FWK_DEVICE_ID={device_id}")


def test_npu_smi_info_visible():
    _print_st_context("npu-smi")
    proc = subprocess.run(
        ["npu-smi", "info"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = (proc.stdout or "").strip()
    # Helpful context when this test fails (device not visible, driver issue, permission issue, etc.)
    _st_print(f"[ST][npu-smi] returncode={proc.returncode}")
    if out:
        _st_print(f"[ST][npu-smi] stdout(head): {out[:1200]}")
        if len(out) > 1200:
            _st_print(f"[ST][npu-smi] stdout(tail): {out[-1200:]}")
    assert proc.returncode == 0, f"npu-smi failed: {proc.stdout}"

    # Keep checks lightweight and robust across versions.
    assert "Ascend" in out, f"Unexpected npu-smi output (no 'Ascend'): {out[:5000]}"
    assert "OK" in out, f"Unexpected npu-smi output (no 'OK'): {out[:5000]}"


def test_nvshmem_basic_flow():
    _print_st_context("nvshmem")
    try:
        import pypto
    except Exception as e:
        pytest.skip(f"pypto is not importable in this environment: {e}")

    _st_print(f"[ST][nvshmem] pypto={getattr(pypto, '__file__', None)}")
    if not hasattr(pypto, "pypto_impl"):
        pytest.skip("pypto_impl extension not available")

    ext = pypto.pypto_impl
    _st_print(f"[ST][nvshmem] pypto_impl={getattr(ext, '__file__', None)}")
    if not hasattr(ext, "nvshmem"):
        pytest.skip("pypto_impl.nvshmem bindings not available")

    nv = ext.nvshmem
    _st_print(f"[ST][nvshmem] nvshmem module={nv}")

    # Best-effort: initialize and finalize even if some calls are no-ops in stub implementations.
    nv.nvshmem_init()
    try:
        pe = nv.nvshmem_my_pe()
        npes = nv.nvshmem_n_pes()
        _st_print(f"[ST][nvshmem] my_pe={pe} n_pes={npes}")
        assert isinstance(pe, int)
        assert isinstance(npes, int)
        assert pe >= 0
        assert npes >= 1

        # Optional metadata functions (may vary by implementation)
        if hasattr(nv, "nvshmem_info_get_name"):
            name = nv.nvshmem_info_get_name()
            _st_print(f"[ST][nvshmem] name={name}")
            assert name is not None
        if hasattr(nv, "nvshmem_info_get_version"):
            ver = nv.nvshmem_info_get_version()
            _st_print(f"[ST][nvshmem] version={ver}")
            assert ver is not None

        if hasattr(nv, "nvshmem_barrier_all"):
            _st_print("[ST][nvshmem] barrier_all() ...")
            nv.nvshmem_barrier_all()
    finally:
        nv.nvshmem_finalize()


