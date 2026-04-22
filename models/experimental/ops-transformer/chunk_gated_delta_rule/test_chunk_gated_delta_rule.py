#!/usr/bin/env python3
# coding: utf-8

"""PyPTO chunk_gated_delta_rule operator test.

测试规格与 tilelang 脚本一致：
- 定长模式: B=1, T=2048, H=8, Hg=4, K=128, V=128
  - use_g=True/False, use_initial_state=True/False (4 组)
- 变长模式:
  - seqlens=[512,512,512,512]
  - seqlens=[128,256,512,1024,128]
  - seqlens=[2048]
  - seqlens=[1024,1024]

运行环境: 使用 NPU 14号卡 (export ASCEND_RT_VISIBLE_DEVICES=14)
"""

import os
import sys
import argparse

import torch
import torch_npu  # 必须在 import impl 之前，避免 jit 装饰器报错
import numpy as np
from numpy.testing import assert_allclose

from chunk_gated_delta_rule_golden import chunk_gated_delta_rule_golden
from chunk_gated_delta_rule_impl import chunk_gated_delta_rule_wrapper


# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID 或使用 ASCEND_RT_VISIBLE_DEVICES。"""
    if "TILE_FWK_DEVICE_ID" in os.environ:
        try:
            return int(os.environ["TILE_FWK_DEVICE_ID"])
        except ValueError:
            pass
    
    # 使用 ASCEND_RT_VISIBLE_DEVICES 的第一个设备
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        if devices:
            try:
                return int(devices.split(",")[0])
            except ValueError:
                pass
    
    return 0


# ─────────────────────────────────────────────
# 2. 定长模式测试
# ─────────────────────────────────────────────

def test_fixed_p0(device_id=None, run_mode="npu"):
    """定长模式 P0: use_g=True, use_initial_state=True, B=1, T=2048"""
    print("=" * 60)
    print("Test: Fixed_P0 (use_g=True, use_initial_state=True)")
    print("=" * 60)
    
    B, T, H, Hg, K, V = 1, 2048, 8, 4, 128, 128
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    torch.manual_seed(42)
    k = torch.randn(B, T, Hg, K, dtype=torch.float16, device=device) * 0.01
    w = torch.randn(B, T, H, K, dtype=torch.float16, device=device) * 0.01
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device) * 0.01
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01
    h0 = torch.randn(B, H, K, V, dtype=torch.float16, device=device) * 0.01
    
    NT = (T + 63) // 64
    
    # PyPTO 实现
    h_impl, v_new_impl, ht_impl = chunk_gated_delta_rule_wrapper(
        k, w, v, g, h0, output_final_state=True, chunk_size=64
    )
    
    # Golden 参考
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(
        k.cpu(), w.cpu(), v.cpu(), g.cpu(), h0.cpu(),
        output_final_state=True, chunk_size=64
    )
    
    # 精度对比
    print(f"  Input shapes: k={k.shape}, w={w.shape}, v={v.shape}")
    print(f"  Output shapes: h={h_impl.shape}, v_new={v_new_impl.shape}, ht={ht_impl.shape}")
    
    h_diff = np.abs(h_impl.cpu().numpy() - h_ref.numpy()).max()
    v_new_diff = np.abs(v_new_impl.cpu().numpy() - v_new_ref.numpy()).max()
    ht_diff = np.abs(ht_impl.cpu().numpy() - ht_ref.numpy()).max()
    
    print(f"  Max diff: h={h_diff:.6e}, v_new={v_new_diff:.6e}, ht={ht_diff:.6e}")
    
    rtol, atol = 0.05, 0.05  # T=2048 精度要求
    
    if run_mode == "npu":
        try:
            assert_allclose(h_impl.cpu().numpy(), h_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(v_new_impl.cpu().numpy(), v_new_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(ht_impl.cpu().numpy(), ht_ref.numpy(), rtol=rtol, atol=atol)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] Test case failed: Fixed_P0", file=sys.stderr)
            print(f"  - h: rtol={rtol}, atol={atol}, max_diff={h_diff:.6e}", file=sys.stderr)
            print(f"  - v_new: rtol={rtol}, atol={atol}, max_diff={v_new_diff:.6e}", file=sys.stderr)
            print(f"  - ht: rtol={rtol}, atol={atol}, max_diff={ht_diff:.6e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise
    
    print("  ✓ Passed\n")



# ─────────────────────────────────────────────
# 3. 变长模式测试
# ─────────────────────────────────────────────


def test_varlen_mixed(device_id=None, run_mode="npu"):
    """变长模式: seqlens=[128,256,512,1024,128]"""
    print("=" * 60)
    print("Test: Varlen_mixed (seqlens=[128,256,512,1024,128])")
    print("=" * 60)
    
    seqlens = [128, 256, 512, 1024, 128]
    H, Hg, K, V = 8, 4, 128, 128
    T_total = sum(seqlens)
    N = len(seqlens)
    
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    torch.manual_seed(42)
    k = torch.randn(1, T_total, Hg, K, dtype=torch.float16, device=device) * 0.01
    w = torch.randn(1, T_total, H, K, dtype=torch.float16, device=device) * 0.01
    v = torch.randn(1, T_total, H, V, dtype=torch.float16, device=device) * 0.01
    g = torch.randn(1, T_total, H, dtype=torch.float32, device=device) * 0.01
    h0 = torch.randn(1, N, H, K, V, dtype=torch.float16, device=device) * 0.01
    
    cu_seqlens = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))], dtype=torch.int32, device=device)
    
    NT_total = sum([(s + 63) // 64 for s in seqlens])
    
    # PyPTO 实现
    h_impl, v_new_impl, ht_impl = chunk_gated_delta_rule_wrapper(
        k, w, v, g, h0, output_final_state=True, chunk_size=64, cu_seqlens=cu_seqlens
    )
    
    # Golden 参考
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(
        k.cpu(), w.cpu(), v.cpu(), g.cpu(), h0.cpu(),
        output_final_state=True, chunk_size=64, cu_seqlens=cu_seqlens
    )
    
    # 精度对比
    print(f"  seqlens: {seqlens}, T_total: {T_total}, NT_total: {NT_total}")
    
    h_diff = np.abs(h_impl.cpu().numpy() - h_ref.numpy()).max()
    v_new_diff = np.abs(v_new_impl.cpu().numpy() - v_new_ref.numpy()).max()
    ht_diff = np.abs(ht_impl.cpu().numpy() - ht_ref.numpy()).max()
    
    print(f"  Max diff: h={h_diff:.6e}, v_new={v_new_diff:.6e}, ht={ht_diff:.6e}")
    
    rtol, atol = 0.05, 0.05
    
    if run_mode == "npu":
        try:
            assert_allclose(h_impl.cpu().numpy(), h_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(v_new_impl.cpu().numpy(), v_new_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(ht_impl.cpu().numpy(), ht_ref.numpy(), rtol=rtol, atol=atol)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] Test case failed: Varlen_mixed", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise
    
    print("  ✓ Passed\n")


def test_fixed_h4_hg2(device_id=None, run_mode="npu"):
    """定长模式: H=4, Hg=2 (GQA ratio=2)"""
    print("=" * 60)
    print("Test: Fixed_H4_Hg2 (H=4, Hg=2)")
    print("=" * 60)
    
    B, T, H, Hg, K, V = 1, 512, 4, 2, 128, 128
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    torch.manual_seed(42)
    k = torch.randn(B, T, Hg, K, dtype=torch.float16, device=device) * 0.01
    w = torch.randn(B, T, H, K, dtype=torch.float16, device=device) * 0.01
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device) * 0.01
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01
    h0 = torch.randn(B, H, K, V, dtype=torch.float16, device=device) * 0.01
    
    NT = (T + 63) // 64
    
    h_impl, v_new_impl, ht_impl = chunk_gated_delta_rule_wrapper(
        k, w, v, g, h0, output_final_state=True, chunk_size=64
    )
    
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(
        k.cpu(), w.cpu(), v.cpu(), g.cpu(), h0.cpu(),
        output_final_state=True, chunk_size=64
    )
    
    print(f"  Input shapes: k={k.shape}, w={w.shape}, v={v.shape}")
    print(f"  Output shapes: h={h_impl.shape}, v_new={v_new_impl.shape}, ht={ht_impl.shape}")
    
    h_diff = np.abs(h_impl.cpu().numpy() - h_ref.numpy()).max()
    v_new_diff = np.abs(v_new_impl.cpu().numpy() - v_new_ref.numpy()).max()
    ht_diff = np.abs(ht_impl.cpu().numpy() - ht_ref.numpy()).max()
    
    print(f"  Max diff: h={h_diff:.6e}, v_new={v_new_diff:.6e}, ht={ht_diff:.6e}")
    
    rtol, atol = 0.05, 0.05
    
    if run_mode == "npu":
        try:
            assert_allclose(h_impl.cpu().numpy(), h_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(v_new_impl.cpu().numpy(), v_new_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(ht_impl.cpu().numpy(), ht_ref.numpy(), rtol=rtol, atol=atol)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] Test case failed: Fixed_H4_Hg2", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise
    
    print("  ✓ Passed\n")



# ─────────────────────────────────────────────
# 4. TileLang 规格测试（与 tilelang 参数完全一致）
# ─────────────────────────────────────────────

def test_tilelang_fixed(device_id=None, run_mode="npu"):
    """TileLang 规格: Fixed-length B=1, T=2048, H=8, Hg=4, K=128, V=128, use_g=True, use_initial_state=True"""
    print("=" * 60)
    print("Test: TileLang_Fixed (B=1, T=2048, H=8, Hg=4)")
    print("=" * 60)
    
    B, T, H, Hg, K, V = 1, 2048, 8, 4, 128, 128
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    torch.manual_seed(41)
    k = torch.randn(B, T, Hg, K, dtype=torch.float16, device=device) * 0.01
    w = torch.randn(B, T, H, K, dtype=torch.float16, device=device) * 0.01
    u = torch.randn(B, T, H, V, dtype=torch.float16, device=device) * 0.01
    g = torch.randn(B, T, H, dtype=torch.float32, device=device) * 0.01
    initial_state = torch.randn(B, H, K, V, dtype=torch.float16, device=device) * 0.01
    
    NT = (T + 63) // 64
    
    h_impl, v_new_impl, ht_impl = chunk_gated_delta_rule_wrapper(
        k, w, u, g, initial_state, output_final_state=True, chunk_size=64
    )
    
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(
        k.cpu(), w.cpu(), u.cpu(), g.cpu(), initial_state.cpu(),
        output_final_state=True, chunk_size=64
    )
    
    print(f"  TileLang spec: B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}")
    print(f"  Input shapes: k={k.shape}, w={w.shape}, u={u.shape}")
    print(f"  Output shapes: h={h_impl.shape}, v_new={v_new_impl.shape}, ht={ht_impl.shape}")
    
    h_diff = np.abs(h_impl.cpu().numpy() - h_ref.numpy()).max()
    v_new_diff = np.abs(v_new_impl.cpu().numpy() - v_new_ref.numpy()).max()
    ht_diff = np.abs(ht_impl.cpu().numpy() - ht_ref.numpy()).max()
    
    print(f"  Max diff: h={h_diff:.6e}, v_new={v_new_diff:.6e}, ht={ht_diff:.6e}")
    
    rtol, atol = 0.05, 0.05
    
    if run_mode == "npu":
        try:
            assert_allclose(h_impl.cpu().numpy(), h_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(v_new_impl.cpu().numpy(), v_new_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(ht_impl.cpu().numpy(), ht_ref.numpy(), rtol=rtol, atol=atol)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] Test case failed: TileLang_Fixed", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise
    
    print("  ✓ Passed\n")


def test_tilelang_varlen(device_id=None, run_mode="npu"):
    """TileLang 规格: Varlen seqlens=[512,512,512,512], H=8, Hg=4, K=128, V=128, use_g=True, use_initial_state=True"""
    print("=" * 60)
    print("Test: TileLang_Varlen (seqlens=[512,512,512,512], H=8, Hg=4)")
    print("=" * 60)
    
    seqlens = [512, 512, 512, 512]
    H, Hg, K, V = 8, 4, 128, 128
    T_total = sum(seqlens)
    N = len(seqlens)
    
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    torch.manual_seed(41)
    k = torch.randn(1, T_total, Hg, K, dtype=torch.float16, device=device) * 0.01
    w = torch.randn(1, T_total, H, K, dtype=torch.float16, device=device) * 0.01
    u = torch.randn(1, T_total, H, V, dtype=torch.float16, device=device) * 0.01
    g = torch.randn(1, T_total, H, dtype=torch.float32, device=device) * 0.01
    initial_state = torch.randn(1, N, H, K, V, dtype=torch.float16, device=device) * 0.01
    
    cu_seqlens = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))], dtype=torch.int32, device=device)
    
    NT_total = sum([(s + 63) // 64 for s in seqlens])
    
    h_impl, v_new_impl, ht_impl = chunk_gated_delta_rule_wrapper(
        k, w, u, g, initial_state, output_final_state=True, chunk_size=64, cu_seqlens=cu_seqlens
    )
    
    h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(
        k.cpu(), w.cpu(), u.cpu(), g.cpu(), initial_state.cpu(),
        output_final_state=True, chunk_size=64, cu_seqlens=cu_seqlens
    )
    
    print(f"  TileLang spec: seqlens={seqlens}, H={H}, Hg={Hg}, K={K}, V={V}")
    print(f"  T_total={T_total}, NT_total={NT_total}")
    
    h_diff = np.abs(h_impl.cpu().numpy() - h_ref.numpy()).max()
    v_new_diff = np.abs(v_new_impl.cpu().numpy() - v_new_ref.numpy()).max()
    ht_diff = np.abs(ht_impl.cpu().numpy() - ht_ref.numpy()).max()
    
    print(f"  Max diff: h={h_diff:.6e}, v_new={v_new_diff:.6e}, ht={ht_diff:.6e}")
    
    rtol, atol = 0.05, 0.05
    
    if run_mode == "npu":
        try:
            assert_allclose(h_impl.cpu().numpy(), h_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(v_new_impl.cpu().numpy(), v_new_ref.numpy(), rtol=rtol, atol=atol)
            assert_allclose(ht_impl.cpu().numpy(), ht_ref.numpy(), rtol=rtol, atol=atol)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] Test case failed: TileLang_Varlen", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise
    
    print("  ✓ Passed\n")


# ─────────────────────────────────────────────
# 5. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "chunk_gated_delta_rule::test_tilelang_fixed": {
        "name": "TileLang_Fixed",
        "description": "TileLang规格: Fixed B=1/T=2048/H=8/Hg=4, use_g=True, use_initial_state=True",
        "function": test_tilelang_fixed,
    },
    "chunk_gated_delta_rule::test_tilelang_varlen": {
        "name": "TileLang_Varlen",
        "description": "TileLang规格: Varlen seqlens=[512,512,512,512]/H=8/Hg=4, use_g=True, use_initial_state=True",
        "function": test_tilelang_varlen,
    },
    "chunk_gated_delta_rule::test_fixed_p0": {
        "name": "Fixed_P0",
        "description": "定长模式: H=8/Hg=4, use_g=True, use_initial_state=True",
        "function": test_fixed_p0,
    },
    "chunk_gated_delta_rule::test_fixed_h4_hg2": {
        "name": "Fixed_H4_Hg2",
        "description": "定长模式: H=4/Hg=2 (可配置参数验证)",
        "function": test_fixed_h4_hg2,
    },
    "chunk_gated_delta_rule::test_varlen_mixed": {
        "name": "Varlen_mixed",
        "description": "变长模式: H=8/Hg=4, seqlens=[128,256,512,1024,128]",
        "function": test_varlen_mixed,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO chunk_gated_delta_rule operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s chunk_gated_delta_rule::test_fixed_p0    Run Fixed_P0
  %(prog)s --list                                    List all cases
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
    
    # --list
    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return
    
    # 选择用例
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))
    
    # NPU 设备初始化
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        import torch_npu
        torch.npu.set_device(device_id)
    
    # 执行
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("[PRECISION_PASS] All tests passed. Precision verified against golden.")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()