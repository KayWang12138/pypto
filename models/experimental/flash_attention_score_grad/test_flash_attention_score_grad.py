#!/usr/bin/env python3
# coding: utf-8
"""
FlashAttentionScoreGrad PyPTO 算子测试

测试 PyPTO 实现与 PyTorch golden 参考实现的精度对比。
支持多个测试级别，验证不同规模下的正确性。
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from numpy.testing import assert_allclose

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(handler)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attention_score_grad_golden import (
    flash_attention_score_grad_golden,
    generate_forward_data,
)
from flash_attention_score_grad_impl import flash_attention_score_grad_wrapper


def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        logger.info("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        logger.info(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def run_test(name, B, N, S, D, device_id, run_mode="npu", rtol=1e-2, atol=2e-2):
    """运行单个测试用例。"""
    logger.info("=" * 60)
    logger.info(f"Test: {name} (B={B}, N={N}, S={S}, D={D})")
    logger.info("=" * 60)

    device = f"npu:{device_id}" if run_mode == "npu" and device_id is not None else "cpu"

    q, k, v, dy, sm, ss, ao, scale = generate_forward_data(B, N, S, D, device=device)

    logger.info(f"  Q shape:  {q.shape}")
    logger.info(f"  scale:    {scale:.6f}")

    # PyPTO 实现
    dq, dk, dv = flash_attention_score_grad_wrapper(
        q, k, v, dy, sm, ss, ao, scale, num_heads=N, head_dim=D
    )

    # Golden 参考
    dq_g, dk_g, dv_g = flash_attention_score_grad_golden(q, k, v, dy, sm, ss, ao, scale)

    logger.info(f"  dQ shape: {dq.shape}")
    logger.info(f"  dK shape: {dk.shape}")
    logger.info(f"  dV shape: {dv.shape}")

    # 精度对比
    results = []
    for grad_name, impl, golden in [("dQ", dq, dq_g), ("dK", dk, dk_g), ("dV", dv, dv_g)]:
        impl_np = impl.float().cpu().numpy()
        golden_np = golden.float().cpu().numpy()
        max_diff = np.abs(impl_np - golden_np).max()
        logger.info(f"  {grad_name} max diff: {max_diff:.6e}")
        results.append((grad_name, impl_np, golden_np, max_diff))

    # 精度判定
    try:
        for grad_name, impl_np, golden_np, _ in results:
            assert_allclose(impl_np, golden_np, rtol=rtol, atol=atol,
                            err_msg=f"{grad_name} precision check failed")
        logger.info(f"  [PRECISION_PASS]")
        logger.info(f"  ✓ {name} passed\n")
        return True
    except AssertionError as e:
        logger.error(f"  [PRECISION_FAIL] {e}")
        return False


def test_level0(device_id, run_mode="npu"):
    """Level 0: 最小功能验证"""
    return run_test("Level 0 (minimal)", B=1, N=8, S=128, D=64,
                    device_id=device_id, run_mode=run_mode)


def test_level1(device_id, run_mode="npu"):
    """Level 1: 典型小规模"""
    return run_test("Level 1 (typical)", B=2, N=8, S=128, D=64,
                    device_id=device_id, run_mode=run_mode)


def test_level2(device_id, run_mode="npu"):
    """Level 2: 中等规模"""
    return run_test("Level 2 (medium)", B=2, N=8, S=256, D=64,
                    device_id=device_id, run_mode=run_mode)


def main():
    parser = argparse.ArgumentParser(description="FlashAttentionScoreGrad PyPTO Test")
    parser.add_argument("level", type=int, nargs="?", default=None,
                        help="Test level (0/1/2). If not specified, run all.")
    parser.add_argument("--list", action="store_true", help="List test levels")
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu", "sim"],
                        help="Run mode")
    args = parser.parse_args()

    tests = {
        0: ("Level 0: 最小功能验证 (B=1,N=1,S=16,D=64)", test_level0),
        1: ("Level 1: 典型小规模 (B=2,N=8,S=64,D=64)", test_level1),
        2: ("Level 2: 中等规模 (B=2,N=8,S=128,D=128)", test_level2),
    }

    if args.list:
        logger.info("\nAvailable test levels:")
        for level, (desc, _) in tests.items():
            logger.info(f"  {level}: {desc}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("FlashAttentionScoreGrad PyPTO Test")
    logger.info("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            sys.exit(1)
        import torch_npu
        torch.npu.set_device(device_id)
        logger.info(f"Running on NPU:{device_id}\n")

    if args.level is not None:
        if args.level not in tests:
            logger.info(f"ERROR: Invalid level {args.level}. Use --list to see available levels.")
            sys.exit(1)
        desc, fn = tests[args.level]
        success = fn(device_id, args.run_mode)
    else:
        # 运行所有级别
        all_pass = True
        for level in sorted(tests.keys()):
            desc, fn = tests[level]
            if not fn(device_id, args.run_mode):
                all_pass = False
                break  # 低级别失败则不继续

        success = all_pass
        if all_pass:
            logger.info("=" * 60)
            logger.info("All tests passed!")
            logger.info("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
