#!/usr/bin/env python3
# coding: utf-8

"""PyPTO gated_delta_rule_backward operator test.

测试流程:
  1. 生成随机输入 (CPU)
  2. 运行 golden reference (CPU) 获取期望输出
  3. 移动输入到 NPU
  4. 调用 PyPTO kernel wrapper
  5. 移动输出回 CPU
  6. 逐张量对比精度 (atol=1e-3, rtol=1e-3)
  7. 输出 [PRECISION_PASS] 或 [PRECISION_FAIL]
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from gated_delta_rule_backward_golden import gated_delta_rule_backward_golden
from gated_delta_rule_backward_impl import gated_delta_rule_backward_wrapper


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID={device_id}")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0


def make_test_inputs(B, T, H, K, V, bt, seed=42):
    """构造测试输入。"""
    torch.manual_seed(seed)
    return {
        "q": torch.randn(B, T, H, K, dtype=torch.float32) * 0.5,
        "k": torch.randn(B, T, H, K, dtype=torch.float32) * 0.5,
        "v": torch.randn(B, T, H, V, dtype=torch.float32) * 0.5,
        "g_raw": torch.randn(B, T, H, dtype=torch.float32) * 0.1,
        "beta": torch.rand(B, T, H, dtype=torch.float32),
        "initial_state": torch.randn(B, H, K, V, dtype=torch.float32) * 0.1,
        "do": torch.randn(B, T, H, V, dtype=torch.float32) * 0.5,
        "dht": torch.randn(B, H, K, V, dtype=torch.float32) * 0.1,
    }


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────

def run_single_test(name, B, T, H, K, V, bt, device_id=None, run_mode="npu",
                    seed=42, rtol=1e-3, atol=1e-3):
    """运行单个测试用例。"""
    print("=" * 60)
    print(f"Test: {name}")
    print(f"  Config: B={B}, T={T}, H={H}, K={K}, V={V}, BT={bt}")
    print("=" * 60)

    inputs = make_test_inputs(B, T, H, K, V, bt, seed=seed)

    # ---- Golden reference (CPU) ----
    print("  Running golden reference...")
    golden_result = gated_delta_rule_backward_golden(
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        g_raw=inputs["g_raw"], beta=inputs["beta"],
        initial_state=inputs["initial_state"],
        do=inputs["do"], dht=inputs["dht"],
        bt=bt, use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
    )
    dq_golden, dk_golden, dv_golden, db_golden, dg_raw_golden, dh0_golden = golden_result

    # ---- Impl (NPU) ----
    if run_mode == "npu":
        import torch_npu
        torch.npu.set_device(device_id)

        print("  Running PyPTO impl on NPU...")
        impl_result = gated_delta_rule_backward_wrapper(
            q=inputs["q"], k=inputs["k"], v=inputs["v"],
            g_raw=inputs["g_raw"], beta=inputs["beta"],
            initial_state=inputs["initial_state"],
            do=inputs["do"], dht=inputs["dht"],
            bt=bt, use_qk_l2norm_in_kernel=True, l2_eps=1e-6,
        )
        dq_impl, dk_impl, dv_impl, db_impl, dg_raw_impl, dh0_impl = impl_result

        # ---- 精度对比 ----
        output_names = ["dq", "dk", "dv", "db", "dg_raw", "dh0"]
        golden_tensors = [dq_golden, dk_golden, dv_golden, db_golden, dg_raw_golden, dh0_golden]
        impl_tensors = [dq_impl, dk_impl, dv_impl, db_impl, dg_raw_impl, dh0_impl]

        all_pass = True
        for name_t, golden_t, impl_t in zip(output_names, golden_tensors, impl_tensors):
            g_np = golden_t.detach().cpu().numpy()
            i_np = impl_t.detach().cpu().numpy()

            max_diff = np.abs(g_np - i_np).max()
            mean_diff = np.abs(g_np - i_np).mean()
            has_nan = np.isnan(i_np).any()
            has_inf = np.isinf(i_np).any()
            has_nonzero = (np.abs(g_np) > 0).any()

            print(f"  {name_t}: shape={golden_t.shape}, max_diff={max_diff:.6e}, "
                  f"mean_diff={mean_diff:.6e}, nan={has_nan}, inf={has_inf}, "
                  f"golden_nonzero={has_nonzero}")

            if has_nan or has_inf:
                print(f"    ✗ FAIL: {name_t} contains NaN/Inf")
                all_pass = False
                continue

            try:
                assert_allclose(i_np, g_np, rtol=rtol, atol=atol)
                print(f"    ✓ PASS")
            except AssertionError as e:
                print(f"    ✗ FAIL: {e}")
                all_pass = False

        print()
        if all_pass:
            print("[PRECISION_PASS]")
            return True
        else:
            print("[PRECISION_FAIL]", file=sys.stderr)
            return False
    else:
        print("  Skipping NPU execution (run_mode != 'npu')")
        return None


# Level 0: 小规模基础功能验证
def test_gated_delta_rule_backward_level0(device_id=None, run_mode="npu"):
    """Level 0: B=1, T=128, H=4, K=128, V=128, BT=64"""
    return run_single_test(
        "Level0_Small", B=1, T=128, H=4, K=128, V=128, bt=64,
        device_id=device_id, run_mode=run_mode, seed=42,
    )


# Level 1: 典型场景验证
def test_gated_delta_rule_backward_level1(device_id=None, run_mode="npu"):
    """Level 1: B=1, T=512, H=4, K=128, V=128, BT=128"""
    return run_single_test(
        "Level1_Typical", B=1, T=512, H=4, K=128, V=128, bt=128,
        device_id=device_id, run_mode=run_mode, seed=777,
    )


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "gated_delta_rule_backward::test_gated_delta_rule_backward_level0": {
        "name": "Level 0 Small",
        "description": "B=1, T=128, H=4, K=128, V=128, BT=64",
        "function": test_gated_delta_rule_backward_level0,
    },
    "gated_delta_rule_backward::test_gated_delta_rule_backward_level1": {
        "name": "Level 1 Typical",
        "description": "B=1, T=512, H=4, K=128, V=128, BT=128",
        "function": test_gated_delta_rule_backward_level1,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO gated_delta_rule_backward operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("example_id", type=str, nargs="?",
                        help="Case ID to run")
    parser.add_argument("--list", action="store_true",
                        help="List available cases")
    parser.add_argument("--run_mode", "--run-mode",
                        type=str, default="npu", choices=["npu", "sim"],
                        help="Run mode (default: npu)")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return

    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        # Default: run level0 for quick verification
        to_run = [("gated_delta_rule_backward::test_gated_delta_rule_backward_level0",
                    EXAMPLES["gated_delta_rule_backward::test_gated_delta_rule_backward_level0"])]

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()

    try:
        results = []
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            result = info["function"](device_id, args.run_mode)
            results.append((key, result))

        print("\n" + "=" * 60)
        all_pass = all(r is True for _, r in results)
        if all_pass:
            print("All tests passed!")
        else:
            print("Some tests failed!")
            sys.exit(1)
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
