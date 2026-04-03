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
Flash Attention Score with Online Softmax

This module implements Flash Attention using online softmax algorithm,
which avoids storing the full attention matrix and provides better
numerical stability through block-wise computation.
"""

import os
import sys
import math
import argparse
import logging
from typing import Optional
import torch
import numpy as np
from numpy.testing import assert_allclose
from flash_attention_score_impl import flash_attention_score_kernel_with_mask

logging.basicConfig(level=logging.INFO, format="%(message)s")


BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN_Q = 64
SEQ_LEN_KV = 128
HEAD_DIM = 64


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.info(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def flash_attention_score_golden_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flash Attention Score 参考实现 - Online Softmax 版本

    使用 online softmax 算法实现，避免存储完整的 attention matrix，
    通过分块计算和在线更新来提高数值稳定性。

    Args:
        query: Query tensor, shape [b, n, sq, d], dtype bfloat16
        key: Key tensor, shape [b, n, skv, d], dtype bfloat16
        value: Value tensor, shape [b, n, skv, d], dtype bfloat16
        atten_mask: Attention mask tensor, shape [sq, skv], dtype uint8
                   值为 1 表示不参与计算，值为 0 表示参与计算

    Returns:
        attention_out: Output tensor, shape [b, n, sq, d], dtype bfloat16
    """
    b, n, sq, d = query.shape
    _, _, skv, _ = key.shape

    scale = 1.0 / math.sqrt(d)

    query_fp32 = query.float()
    key_fp32 = key.float()
    value_fp32 = value.float()

    output = torch.zeros(b, n, sq, d, dtype=torch.float32, device=query.device)

    for b_idx in range(b):
        for n_idx in range(n):
            for q_idx in range(sq):
                q_vec = query_fp32[b_idx, n_idx, q_idx, :]

                max_score = float('-inf')
                sum_exp = 0.0
                output_vec = torch.zeros(d, dtype=torch.float32, device=query.device)

                for kv_idx in range(skv):
                    if atten_mask is not None and atten_mask[q_idx, kv_idx] == 1:
                        continue

                    k_vec = key_fp32[b_idx, n_idx, kv_idx, :]
                    score = torch.dot(q_vec, k_vec) * scale

                    new_max = max(max_score, score.item())

                    if new_max > max_score:
                        correction = math.exp(max_score - new_max)
                        sum_exp = sum_exp * correction
                        output_vec = output_vec * correction
                        max_score = new_max

                    exp_score = math.exp(score - max_score)
                    sum_exp += exp_score

                    v_vec = value_fp32[b_idx, n_idx, kv_idx, :]
                    output_vec += exp_score * v_vec

                if sum_exp > 0:
                    output[b_idx, n_idx, q_idx, :] = output_vec / sum_exp

    return output.to(torch.bfloat16)


def test_flash_attention_score(device_id=None, run_mode: str = "npu"):
    """Test Flash Attention Score"""
    logging.info("=" * 60)
    logging.info("Test: Flash Attention Score with Online Softmax")
    logging.info("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                      dtype=torch.bfloat16, device=device)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_KV, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)

    atten_mask = torch.zeros(SEQ_LEN_Q, SEQ_LEN_KV, dtype=torch.uint8, device=device)
    atten_mask[:, SEQ_LEN_KV // 2:] = 1

    output = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM,
                         dtype=torch.bfloat16, device=device)

    atten_mask_fp32 = atten_mask.float() if atten_mask is not None else None

    flash_attention_score_kernel_with_mask(query, key, value, atten_mask_fp32, output)

    golden = flash_attention_score_golden_online_softmax(query, key, value, atten_mask)

    logging.info(f"Input shape: query={query.shape}, key={key.shape}, value={value.shape}")
    logging.info(f"Output shape: {output.shape}")

    if run_mode == "npu":
        output_fp32 = output.float()
        golden_fp32 = golden.float()
        max_diff = (output_fp32 - golden_fp32).abs().max().item()
        mean_diff = (output_fp32 - golden_fp32).abs().mean().item()

        logging.info(f"Max difference: {max_diff:.6f}")
        logging.info(f"Mean difference: {mean_diff:.6f}")

        assert_allclose(
            output_fp32.cpu().numpy().flatten(),
            golden_fp32.cpu().numpy().flatten(),
            rtol=0.0078125,
            atol=0.0001
        )
        logging.info("✓ Flash Attention Score test passed!")


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Flash Attention Score Example",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        default='npu',
        choices=["npu", "sim"],
        help='Run mode: npu or sim (default: npu)'
    )
    args = parser.parse_args()

    logging.info("\n" + "=" * 60)
    logging.info("PyPTO Flash Attention Score Example")
    logging.info("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        logging.info(f"Running on NPU device {device_id}\n")

    try:
        test_flash_attention_score(device_id, args.run_mode)

        logging.info("=" * 60)
        logging.info("All tests passed!")
        logging.info("=" * 60)
    except Exception as e:
        logging.info(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
