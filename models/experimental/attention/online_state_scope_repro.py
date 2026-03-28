#!/usr/bin/env python3
# coding: utf-8
"""Minimal repro for parser scope issue: variable assigned only in loop-begin branch."""

import os

import pypto
import torch


def _get_device_id() -> int:
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        raise RuntimeError("Please set TILE_FWK_DEVICE_ID")
    return int(os.environ["TILE_FWK_DEVICE_ID"])


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def state_scope_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 32], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, 32], pypto.DT_FP32),
):
    n = x.shape[0]
    loop_n = n // 16
    for idx in pypto.loop(loop_n):
        x_tile = x[idx * 16:(idx + 1) * 16, :]
        if pypto.is_loop_begin(idx):
            m_acc = x_tile
        else:
            # Expected parser complaint in some dynamic control-flow patterns:
            # NameError: m_acc is not defined
            m_acc = pypto.maximum(m_acc, x_tile)
        y[idx * 16:, :] = m_acc


def main() -> None:
    import torch_npu  # noqa: F401

    device_id = _get_device_id()
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    x = torch.randn(32, 32, dtype=torch.float32, device=device)
    y = torch.zeros(32, 32, dtype=torch.float32, device=device)
    state_scope_kernel(x, y)
    torch.npu.synchronize()
    print("state-scope repro kernel launch finished")


if __name__ == "__main__":
    main()

