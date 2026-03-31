#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Minimal repro: submit_before_loop on outer loop does not sync inner parallel tasks.

Algorithm: accumulate values across outer loop iterations using double-pointer trick.
  out[j] = sum over i of (input[i] * weight[j])  for all i,j

Outer loop over i (submit_before_loop=True), inner loop over j (parallel).
acc_in/acc_out point to same memory. Each iteration:
  acc_prev = view(acc_in, j_offset)
  acc_new = acc_prev + contribution
  assemble(acc_new, j_offset, acc_out)

Expected: all shapes PASS
Actual: PASS when inner loop has 1 iteration, FAIL when >1 iteration
"""

import os
import logging
import torch
import pypto

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(handler)

TILE = 64


@pypto.frontend.jit
def accumulate_kernel(
    src:     pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    acc_in:  pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    acc_out: pypto.Tensor([pypto.DYN, ...], pypto.DT_BF16),
    n_outer: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
    n_inner: pypto.Tensor([pypto.DYN], pypto.DT_INT32),
):
    """Accumulate src[i] into acc[j] for all (i,j) pairs."""
    n_i = n_outer.shape[0]
    n_j = n_inner.shape[0]

    # Outer loop: sequential (submit_before_loop=True)
    for i_idx in pypto.loop(n_i, name="LOOP_i", idx_name="i_idx",
                            submit_before_loop=True):
        pypto.set_vec_tile_shapes(TILE, TILE)
        src_tile = pypto.view(src, [TILE, TILE], [i_idx * TILE, 0])

        # Inner loop: parallel (no submit_before_loop)
        for j_idx in pypto.loop(n_j, name="LOOP_j", idx_name="j_idx"):
            j_off = j_idx * TILE
            pypto.set_vec_tile_shapes(TILE, TILE)

            # Read old accumulator value from acc_in
            prev = pypto.view(acc_in, [TILE, TILE], [j_off, 0])
            # Add contribution and write to acc_out
            pypto.assemble(pypto.add(prev, src_tile), [j_off, 0], acc_out)


def golden(src, n_i, n_j):
    """Pure torch reference: acc[j] = sum_i(src[i]) for all j."""
    total = src.reshape(n_i, TILE, TILE).sum(dim=0)  # [TILE, TILE]
    return total.unsqueeze(0).expand(n_j, -1, -1).reshape(n_j * TILE, TILE)


def run_test(n_i, n_j, device):
    """Run one test case."""
    torch.manual_seed(42)
    src = torch.randn(n_i * TILE, TILE, dtype=torch.bfloat16, device=device)
    acc = torch.zeros(n_j * TILE, TILE, dtype=torch.bfloat16, device=device)

    ni_t = torch.zeros(n_i, dtype=torch.int32, device=device)
    nj_t = torch.zeros(n_j, dtype=torch.int32, device=device)

    # Double pointer: acc_in and acc_out are the SAME tensor
    accumulate_kernel(src, acc, acc, ni_t, nj_t)

    ref = golden(src, n_i, n_j).to(device)
    diff = (acc.float() - ref.float()).abs().max().item()
    status = "PASS" if diff < 0.02 else "FAIL"
    logger.info(f"  n_i={n_i}, n_j={n_j}: max_diff={diff:.6f} -> {status}")
    return status == "PASS"


def main():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    import torch_npu
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    logger.info("=" * 60)
    logger.info("submit_before_loop sync issue - minimal repro")
    logger.info("=" * 60)

    all_pass = True
    # n_i=1: no cross-iteration dependency -> should PASS
    all_pass &= run_test(n_i=1, n_j=1, device=device)
    all_pass &= run_test(n_i=1, n_j=4, device=device)

    # n_i>1: cross-iteration dependency -> may FAIL
    all_pass &= run_test(n_i=2, n_j=2, device=device)
    all_pass &= run_test(n_i=4, n_j=4, device=device)
    all_pass &= run_test(n_i=8, n_j=2, device=device)

    logger.info("=" * 60)
    if all_pass:
        logger.info("All tests PASSED")
    else:
        logger.info("Some tests FAILED - submit_before_loop sync issue confirmed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
