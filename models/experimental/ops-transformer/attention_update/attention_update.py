#!/usr/bin/env python3
# coding: utf-8

"""PyPTO attention_update 算子实现与测试。

将各 SP 域的局部 lse 和 localOut 合并为全局结果。

公式:
    lse_max = max(lse_i)
    lse_sum = sum_i(exp(lse_i - lse_max))
    lse_m = lse_max + log(lse_sum)
    O = sum_i(O_i * exp(lse_i - lse_m))
"""

import os
import sys
import argparse
from typing import List, Optional, Tuple

import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


# ============================================================================
# PyPTO Kernel 实现
# ============================================================================

def _attention_update_core_sp2(
    lse_0: pypto.Tensor(),
    lse_1: pypto.Tensor(),
    local_out_0: pypto.Tensor(),
    local_out_1: pypto.Tensor(),
    output: pypto.Tensor(),
    lse_out: pypto.Tensor(),
    d: int,
    update_type: int,
):
    """SP=2 的核心计算逻辑"""
    pypto.set_vec_tile_shapes(1, 8, 8, d)
    
    lse_max = pypto.maximum(lse_0, lse_1)
    
    lse_exp_0 = pypto.exp(lse_0 - lse_max)
    lse_exp_1 = pypto.exp(lse_1 - lse_max)
    
    lse_sum = lse_exp_0 + lse_exp_1
    
    lse_m = lse_max + pypto.log(lse_sum)
    
    final_exp_0 = pypto.exp(lse_0 - lse_m)
    final_exp_1 = pypto.exp(lse_1 - lse_m)
    
    final_exp_0_3d = pypto.unsqueeze(final_exp_0, 1)
    final_exp_1_3d = pypto.unsqueeze(final_exp_1, 1)
    
    output[:] = local_out_0 * final_exp_0_3d + local_out_1 * final_exp_1_3d
    
    if update_type == 1:
        lse_out[:] = lse_m


@pypto.frontend.jit
def _attention_update_kernel_sp2(
    lse_0: pypto.Tensor(),
    lse_1: pypto.Tensor(),
    local_out_0: pypto.Tensor(),
    local_out_1: pypto.Tensor(),
    output: pypto.Tensor(),
    lse_out: pypto.Tensor(),
    d: int,
    update_type: int,
):
    _attention_update_core_sp2(lse_0, lse_1, local_out_0, local_out_1, output, lse_out, d, update_type)


@pypto.frontend.jit
def _attention_update_kernel_sp4(
    lse_0: pypto.Tensor(),
    lse_1: pypto.Tensor(),
    lse_2: pypto.Tensor(),
    lse_3: pypto.Tensor(),
    local_out_0: pypto.Tensor(),
    local_out_1: pypto.Tensor(),
    local_out_2: pypto.Tensor(),
    local_out_3: pypto.Tensor(),
    output: pypto.Tensor(),
    lse_out: pypto.Tensor(),
    d: int,
    update_type: int,
):
    pypto.set_vec_tile_shapes(1, 8, 8, d)
    
    lse_max_01 = pypto.maximum(lse_0, lse_1)
    lse_max_23 = pypto.maximum(lse_2, lse_3)
    lse_max = pypto.maximum(lse_max_01, lse_max_23)
    
    lse_exp_0 = pypto.exp(lse_0 - lse_max)
    lse_exp_1 = pypto.exp(lse_1 - lse_max)
    lse_exp_2 = pypto.exp(lse_2 - lse_max)
    lse_exp_3 = pypto.exp(lse_3 - lse_max)
    
    lse_sum = lse_exp_0 + lse_exp_1 + lse_exp_2 + lse_exp_3
    
    lse_m = lse_max + pypto.log(lse_sum)
    
    final_exp_0 = pypto.unsqueeze(pypto.exp(lse_0 - lse_m), 1)
    final_exp_1 = pypto.unsqueeze(pypto.exp(lse_1 - lse_m), 1)
    final_exp_2 = pypto.unsqueeze(pypto.exp(lse_2 - lse_m), 1)
    final_exp_3 = pypto.unsqueeze(pypto.exp(lse_3 - lse_m), 1)
    
    output[:] = (local_out_0 * final_exp_0 + local_out_1 * final_exp_1 + 
                 local_out_2 * final_exp_2 + local_out_3 * final_exp_3)
    
    if update_type == 1:
        lse_out[:] = lse_m


@pypto.frontend.jit(debug_options={"runtime_debug_mode": 1})
def _attention_update_kernel_sp8(
    lse_list: pypto.Tensor(),
    local_out_list: pypto.Tensor(),
    output: pypto.Tensor(),
    lse_out: pypto.Tensor(),
    d: int,
    update_type: int,
):
    pypto.set_vec_tile_shapes(1, 8, 8, d)
    
    lse_max = pypto.amax(lse_list, dim=0, keepdim=False)
    
    lse_exp = pypto.exp(lse_list - pypto.unsqueeze(lse_max, 0))
    
    lse_sum = pypto.sum(lse_exp, dim=0, keepdim=False)
    
    lse_m = lse_max + pypto.log(lse_sum)
    
    final_exp = pypto.exp(lse_list - pypto.unsqueeze(lse_m, 0))
    final_exp_3d = pypto.unsqueeze(final_exp, 2)
    
    output[:] = pypto.sum(local_out_list * final_exp_3d, dim=0)
    
    if update_type == 1:
        lse_out[:] = lse_m


# ============================================================================
# Wrapper 函数
# ============================================================================

def attention_update_wrapper(
    lse_list: List[torch.Tensor],
    local_out_list: List[torch.Tensor],
    update_type: int,
    sp: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """attention_update wrapper 函数。
    
    Args:
        lse_list: 各 SP 域的局部 lse 列表，每个元素 shape=[BSN]
        local_out_list: 各 SP 域的局部输出列表，每个元素 shape=[BSN, D]
        update_type: 0=不输出 lseOut，1=输出 lseOut
        sp: 序列并行度（支持 2, 4, 8）
    
    Returns:
        output: 全局输出，shape=[BSN, D]
        lse_out: 全局 lse（可选），shape=[BSN]
    """
    bsn = lse_list[0].shape[0]
    d = local_out_list[0].shape[1]
    dtype = local_out_list[0].dtype
    device = lse_list[0].device
    
    output = torch.zeros(bsn, d, dtype=dtype, device=device)
    lse_out = torch.empty(bsn, dtype=torch.float32, device=device)
    
    if sp == 2:
        _attention_update_kernel_sp2(
            lse_list[0].float(),
            lse_list[1].float(),
            local_out_list[0].float(),
            local_out_list[1].float(),
            output.float(),
            lse_out,
            d,
            update_type,
        )
    elif sp == 4:
        _attention_update_kernel_sp4(
            lse_list[0].float(), lse_list[1].float(),
            lse_list[2].float(), lse_list[3].float(),
            local_out_list[0].float(), local_out_list[1].float(),
            local_out_list[2].float(), local_out_list[3].float(),
            output.float(),
            lse_out,
            d,
            update_type,
        )
    elif sp == 8:
        lse_stacked = torch.stack(lse_list, dim=0).float()
        local_out_stacked = torch.stack(local_out_list, dim=0).float()
        _attention_update_kernel_sp8(
            lse_stacked,
            local_out_stacked,
            output.float(),
            lse_out,
            d,
            update_type,
        )
    else:
        raise ValueError(f"Unsupported sp={sp}, only support 2, 4, 8")
    
    output = output.to(dtype)
    
    if update_type == 1:
        return output, lse_out
    else:
        return output, None


# ============================================================================
# Golden 参考实现
# ============================================================================

def attention_update_golden(
    lse_list: List[torch.Tensor],
    local_out_list: List[torch.Tensor],
    update_type: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch Golden 参考实现。
    
    Args:
        lse_list: 各 SP 域的局部 lse 列表，每个元素 shape=[BSN]
        local_out_list: 各 SP 域的局部输出列表，每个元素 shape=[BSN, D]
        update_type: 0=不输出 lseOut，1=输出 lseOut
    
    Returns:
        output: 全局输出，shape=[BSN, D]
        lse_out: 全局 lse（可选），shape=[BSN]
    """
    sp = len(lse_list)
    BSN = lse_list[0].shape[0]
    D = local_out_list[0].shape[1]
    
    lse_stacked = torch.stack(lse_list, dim=0)
    
    lse_max = torch.max(lse_stacked, dim=0)[0]
    
    lse_exp = torch.exp(lse_stacked - lse_max.unsqueeze(0))
    lse_sum = torch.sum(lse_exp, dim=0)
    
    lse_m = lse_max + torch.log(lse_sum + 1e-10)
    
    final_exp = torch.exp(lse_stacked - lse_m.unsqueeze(0))
    
    final_exp_broadcast = final_exp.unsqueeze(-1)
    
    out = torch.zeros(BSN, D, dtype=local_out_list[0].dtype, device=local_out_list[0].device)
    for i in range(sp):
        out = out + local_out_list[i] * final_exp_broadcast[i]
    
    if update_type == 1:
        return out, lse_m
    else:
        return out, None


# ============================================================================
# 测试函数
# ============================================================================

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def test_attention_update_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证（8-16 元素）。"""
    print("=" * 60)
    print("Test: attention_update Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(0)
    sp, B, S, N, D = 2, 1, 2, 2, 8
    BSN = B * S * N
    dtype = torch.float32
    
    lse_list = [torch.randn(BSN, dtype=torch.float32, device=device) for _ in range(sp)]
    local_out_list = [torch.randn(BSN, D, dtype=dtype, device=device) for _ in range(sp)]
    update_type = 0

    result_out, result_lse = attention_update_wrapper(lse_list, local_out_list, update_type, sp)
    golden_out, golden_lse = attention_update_golden(lse_list, local_out_list, update_type)

    print(f"  sp={sp}, BSN={BSN}, D={D}")
    print(f"  Output shape: {result_out.shape}")
    
    max_diff = np.abs(result_out.cpu().numpy() - golden_out.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_out.cpu().numpy(),
            golden_out.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ Passed\n")


def test_attention_update_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证（1K 元素）。"""
    print("=" * 60)
    print("Test: attention_update Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    sp, B, S, N, D = 4, 2, 16, 4, 64
    BSN = B * S * N
    dtype = torch.float32
    
    lse_list = [torch.randn(BSN, dtype=torch.float32, device=device) for _ in range(sp)]
    local_out_list = [torch.randn(BSN, D, dtype=dtype, device=device) for _ in range(sp)]
    update_type = 1

    result_out, result_lse = attention_update_wrapper(lse_list, local_out_list, update_type, sp)
    golden_out, golden_lse = attention_update_golden(lse_list, local_out_list, update_type)

    print(f"  sp={sp}, BSN={BSN}, D={D}")
    
    max_diff = np.abs(result_out.cpu().numpy() - golden_out.cpu().numpy()).max()
    print(f"  Out max diff: {max_diff:.6e}")
    
    if golden_lse is not None and result_lse is not None:
        lse_diff = np.abs(result_lse.cpu().numpy() - golden_lse.cpu().numpy()).max()
        print(f"  LSE max diff: {lse_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_out.cpu().numpy(),
            golden_out.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )
        if golden_lse is not None and result_lse is not None:
            assert_allclose(
                result_lse.cpu().numpy(),
                golden_lse.cpu().numpy(),
                rtol=1e-3, atol=1e-3,
            )

    print("  ✓ Passed\n")


def test_attention_update_level2(device_id=None, run_mode="npu"):
    """Level 2: 边界情况验证（大 sp）。"""
    print("=" * 60)
    print("Test: attention_update Level 2 (boundary)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(100)
    sp, B, S, N, D = 8, 1, 8, 2, 32
    BSN = B * S * N
    dtype = torch.float32
    
    lse_list = [torch.randn(BSN, dtype=torch.float32, device=device) for _ in range(sp)]
    local_out_list = [torch.randn(BSN, D, dtype=dtype, device=device) for _ in range(sp)]
    update_type = 0

    result_out, result_lse = attention_update_wrapper(lse_list, local_out_list, update_type, sp)
    golden_out, golden_lse = attention_update_golden(lse_list, local_out_list, update_type)

    print(f"  sp={sp}, BSN={BSN}, D={D}")
    
    max_diff = np.abs(result_out.cpu().numpy() - golden_out.cpu().numpy()).max()
    print(f"  Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result_out.cpu().numpy(),
            golden_out.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ Passed\n")


EXAMPLES = {
    "attention_update::test_attention_update_level0": {
        "name": "attention_update Level 0",
        "description": "小数据量基础功能验证",
        "function": test_attention_update_level0,
    },
    "attention_update::test_attention_update_level1": {
        "name": "attention_update Level 1",
        "description": "典型场景验证",
        "function": test_attention_update_level1,
    },
    "attention_update::test_attention_update_level2": {
        "name": "attention_update Level 2",
        "description": "边界情况验证",
        "function": test_attention_update_level2,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO attention_update operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s attention_update::test_attention_update_level0    Run Level 0
  %(prog)s --list                    List all cases
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
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
        to_run = list(sorted(EXAMPLES.items()))

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()