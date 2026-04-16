#!/usr/bin/env python3
# coding: utf-8
"""
scaled_mm 测试样例

使用方法:
    python test_scaled_mm.py              # 运行所有测试（需要 NPU 环境）
    python test_scaled_mm.py --run_mode sim  # 使用模拟模式运行

scaled_mm 功能:
    执行 MXFP8 量化矩阵乘法: out = (mat_a * scale_a) @ (mat_b * scale_b)

数据类型约束:
    - mat_a, mat_b: torch.float8_e4m3fn 或 torch.float8_e5m2 (FP8 格式)
    - scale_a, scale_b: torch.float8_e8m0fnu (E8M0 缩放因子格式)
    - 输出: torch.float16 或 torch.float32 或 torch.bfloat16

Shape 约束:
    - mat_a: [M, K], K 必须是 64 的倍数
    - mat_b: [K, N]
    - scale_a: [M, K//64, 2]
    - scale_b: [K//64, N, 2]
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np


def get_run_mode():
    """从命令行参数获取运行模式"""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--run_mode="):
            return arg.split("=", 1)[1]
    return "npu"


def get_device_id():
    """获取设备 ID"""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("请设置环境变量 TILE_FWK_DEVICE_ID:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print(f"错误: TILE_FWK_DEVICE_ID 必须是整数")
        return None


def prepare_fp8_data(m, k, n, dtype="e4m3"):
    """
    准备 scaled_mm 的输入数据
    
    Args:
        m: 左矩阵的 M 维度
        k: 矩阵的 K 维度 (必须为 64 的倍数)
        n: 右矩阵的 N 维度
        dtype: FP8 类型, "e4m3" 或 "e5m2"
    
    Returns:
        mat_a, mat_b, scale_a, scale_b, golden (PyTorch tensors)
    """
    if k % 64 != 0:
        raise ValueError(f"K 维度 {k} 必须是 64 的倍数")
    
    torch_dtype = torch.float8_e4m3fn if dtype == "e4m3" else torch.float8_e5m2
    pypto_dtype = pypto.DT_FP8E4M3 if dtype == "e4m3" else pypto.DT_FP8E5M2
    
    mat_a = torch.randn(m, k, dtype=torch.float32).uniform_(-3, 3).to(torch_dtype)
    mat_b = torch.randn(k, n, dtype=torch.float32).uniform_(-3, 3).to(torch_dtype)
    
    scale_a_shape = [m, k // 64, 2]
    scale_b_shape = [k // 64, n, 2]
    
    scale_a = torch.randn(scale_a_shape, dtype=torch.float32).uniform_(0.5, 2.0).to(torch.float8_e8m0fnu)
    scale_b = torch.randn(scale_b_shape, dtype=torch.float32).uniform_(0.5, 2.0).to(torch.float8_e8m0fnu)
    
    scale_a_expanded = scale_a.view(torch.float32).unsqueeze(-1).expand(m, k // 64, 2, 32)
    scale_a_expanded = scale_a_expanded.reshape(m, k)
    scale_b_expanded = scale_b.view(torch.float32).unsqueeze(0).expand(32, k // 64, n, 2)
    scale_b_expanded = scale_b_expanded.reshape(k, n)
    
    mat_a_fp32 = mat_a.to(torch.float32)
    mat_b_fp32 = mat_b.to(torch.float32)
    
    scaled_a = mat_a_fp32 * scale_a_expanded
    scaled_b = scale_b_expanded * mat_b_fp32
    
    golden = torch.matmul(scaled_a, scaled_b)
    
    return mat_a, mat_b, scale_a, scale_b, golden


def test_scaled_mm_basic(device_id=None, run_mode="npu"):
    """
    测试基本的 scaled_mm 用法
    """
    print("\n" + "=" * 60)
    print("测试 1: 基本 scaled_mm 用法")
    print("=" * 60)
    
    m, k, n = 64, 128, 32
    print(f"Shape: mat_a[{m},{k}], mat_b[{k},{n}], scale_a[{m},{k//64},2], scale_b[{k//64},{n},2]")
    
    mat_a, mat_b, scale_a, scale_b, golden = prepare_fp8_data(m, k, n, dtype="e4m3")
    
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM if run_mode == "sim" else pypto.RunMode.NPU})
    def scaled_mm_kernel(
        a: pypto.Tensor([m, k], pypto.DT_FP8E4M3),
        b: pypto.Tensor([k, n], pypto.DT_FP8E4M3),
        sa: pypto.Tensor([m, k//64, 2], pypto.DT_FP8E8M0),
        sb: pypto.Tensor([k//64, n, 2], pypto.DT_FP8E8M0),
        out: pypto.Tensor([m, n], pypto.DT_FP16)
    ):
        pypto.set_cube_tile_shapes([64, 64], [128, 128], [32, 32])
        out[:] = pypto.scaled_mm(a, b, pypto.DT_FP16, sa, sb)
    
    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'
    
    out = torch.empty(m, n, dtype=torch.float16, device=device)
    
    if run_mode == "npu":
        mat_a_npu = mat_a.npu()
        mat_b_npu = mat_b.npu()
        scale_a_npu = scale_a.npu()
        scale_b_npu = scale_b.npu()
        out_npu = out.npu()
        scaled_mm_kernel(mat_a_npu, mat_b_npu, scale_a_npu, scale_b_npu, out_npu)
        out = out_npu.cpu()
    else:
        scaled_mm_kernel(mat_a, mat_b, scale_a, scale_b, out)
    
    print(f"输出形状: {out.shape}")
    print(f"输出 dtype: {out.dtype}")
    print(f"Golden 形状: {golden.shape}")
    print(f"Golden dtype: {golden.dtype}")
    
    out_fp32 = out.to(torch.float32)
    
    if run_mode == "npu":
        rtol, atol = 0.01, 0.01
        if torch.allclose(out_fp32, golden, rtol=rtol, atol=atol):
            print(f"✓ 精度验证通过 (rtol={rtol}, atol={atol})")
        else:
            diff = torch.abs(out_fp32 - golden)
            print(f"✗ 粁度验证失败")
            print(f"  最大差异: {diff.max().item():.6f}")
            print(f"  平均差异: {diff.mean().item():.6f}")
    else:
        print("✓ 模拟模式运行成功（未进行精度验证）")
    
    print("=" * 60)


def test_scaled_mm_with_bias(device_id=None, run_mode="npu"):
    """
    测试带 bias 的 scaled_mm
    """
    print("\n" + "=" * 60)
    print("测试 2: 带 bias 的 scaled_mm")
    print("=" * 60)
    
    m, k, n = 128, 256, 64
    print(f"Shape: mat_a[{m},{k}], mat_b[{k},{n}], scale_a[{m},{k//64},2], scale_b[{k//64},{n},2]")
    
    mat_a, mat_b, scale_a, scale_b, golden_no_bias = prepare_fp8_data(m, k, n, dtype="e5m2")
    
    bias = torch.randn(1, n, dtype=torch.float32).uniform_(-1, 1)
    golden = golden_no_bias + bias.expand(m, n)
    
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM if run_mode == "sim" else pypto.RunMode.NPU})
    def scaled_mm_bias_kernel(
        a: pypto.Tensor([m, k], pypto.DT_FP8E5M2),
        b: pypto.Tensor([k, n], pypto.DT_FP8E5M2),
        sa: pypto.Tensor([m, k//64, 2], pypto.DT_FP8E8M0),
        sb: pypto.Tensor([k//64, n, 2], pypto.DT_FP8E8M0),
        bias_tensor: pypto.Tensor([1, n], pypto.DT_FP32),
        out: pypto.Tensor([m, n], pypto.DT_FP32)
    ):
        pypto.set_cube_tile_shapes([128, 128], [256, 256], [64, 64])
        extend_params = {'bias_tensor': bias_tensor}
        out[:] = pypto.scaled_mm(a, b, pypto.DT_FP32, sa, sb, extend_params=extend_params)
    
    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'
    
    out = torch.empty(m, n, dtype=torch.float32, device=device)
    
    if run_mode == "npu":
        mat_a_npu = mat_a.npu()
        mat_b_npu = mat_b.npu()
        scale_a_npu = scale_a.npu()
        scale_b_npu = scale_b.npu()
        bias_npu = bias.npu()
        out_npu = out.npu()
        scaled_mm_bias_kernel(mat_a_npu, mat_b_npu, scale_a_npu, scale_b_npu, bias_npu, out_npu)
        out = out_npu.cpu()
    else:
        scaled_mm_bias_kernel(mat_a, mat_b, scale_a, scale_b, bias, out)
    
    print(f"输出形状: {out.shape}")
    print(f"输出 dtype: {out.dtype}")
    print(f"Golden 形状: {golden.shape}")
    
    if run_mode == "npu":
        rtol, atol = 0.01, 0.01
        if torch.allclose(out, golden, rtol=rtol, atol=atol):
            print(f"✓ 精度验证通过 (rtol={rtol}, atol={atol})")
        else:
            diff = torch.abs(out - golden)
            print(f"✗ 精度验证失败")
            print(f"  最大差异: {diff.max().item():.6f}")
            print(f"  平均差异: {diff.mean().item():.6f}")
    else:
        print("✓ 模拟模式运行成功（未进行精度验证）")
    
    print("=" * 60)


def test_scaled_mm_with_transpose(device_id=None, run_mode="npu"):
    """
    测试带转置的 scaled_mm
    """
    print("\n" + "=" * 60)
    print("测试 3: 带转置的 scaled_mm")
    print("=" * 60)
    
    m, k, n = 192, 128, 96
    print(f"Shape: mat_a[{m},{k}], mat_b[{k},{n}]")
    print(f"转置: a_trans=False, b_trans=True")
    
    mat_a, mat_b_no_trans, scale_a, scale_b_no_trans, golden_no_trans = prepare_fp8_data(m, k, n, dtype="e4m3")
    
    mat_b = mat_b_no_trans.T.contiguous()
    scale_b = scale_b_no_trans.permute(1, 0, 2).contiguous()
    
    golden = golden_no_trans
    
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM if run_mode == "sim" else pypto.RunMode.NPU})
    def scaled_mm_trans_kernel(
        a: pypto.Tensor([m, k], pypto.DT_FP8E4M3),
        b: pypto.Tensor([n, k], pypto.DT_FP8E4M3),
        sa: pypto.Tensor([m, k//64, 2], pypto.DT_FP8E8M0),
        sb: pypto.Tensor([n, k//64, 2], pypto.DT_FP8E8M0),
        out: pypto.Tensor([m, n], pypto.DT_BF16)
    ):
        pypto.set_cube_tile_shapes([192, 192], [128, 128], [96, 96])
        out[:] = pypto.scaled_mm(a, b, pypto.DT_BF16, sa, sb, b_trans=True, scale_b_trans=True)
    
    device = f'npu:{device_id}' if run_mode == "npu" and device_id is not None else 'cpu'
    
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    
    if run_mode == "npu":
        mat_a_npu = mat_a.npu()
        mat_b_npu = mat_b.npu()
        scale_a_npu = scale_a.npu()
        scale_b_npu = scale_b.npu()
        out_npu = out.npu()
        scaled_mm_trans_kernel(mat_a_npu, mat_b_npu, scale_a_npu, scale_b_npu, out_npu)
        out = out_npu.cpu()
    else:
        scaled_mm_trans_kernel(mat_a, mat_b, scale_a, scale_b, out)
    
    print(f"输出形状: {out.shape}")
    print(f"输出 dtype: {out.dtype}")
    
    out_fp32 = out.to(torch.float32)
    
    if run_mode == "npu":
        rtol, atol = 0.01, 0.01
        if torch.allclose(out_fp32, golden, rtol=rtol, atol=atol):
            print(f"✓ 精度验证通过 (rtol={rtol}, atol={atol})")
        else:
            diff = torch.abs(out_fp32 - golden)
            print(f"✗ 粁度验证失败")
            print(f"  最大差异: {diff.max().item():.6f}")
            print(f"  平均差异: {diff.mean().item():.6f}")
    else:
        print("✓ 模拟模式运行成功（未进行精度验证）")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PyPTO scaled_mm 测试样例")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="npu",
        choices=["npu", "sim"],
        help="运行模式: npu (需要 NPU) 或 sim (模拟)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["basic", "bias", "trans", "all"],
        help="要运行的测试: basic, bias, trans, all"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PyPTO scaled_mm 测试样例")
    print("=" * 60)
    
    run_mode = args.run_mode
    device_id = None
    
    if run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            print("\n无法获取设备 ID，请设置 TILE_FWK_DEVICE_ID 环境变量")
            print("或者使用 --run_mode sim 运行模拟模式")
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print(f"使用 NPU 设备: {device_id}")
    else:
        print("使用模拟模式运行")
    
    tests = {
        "basic": test_scaled_mm_basic,
        "bias": test_scaled_mm_with_bias,
        "trans": test_scaled_mm_with_transpose,
    }
    
    if args.test == "all":
        for test_name, test_func in tests.items():
            test_func(device_id, run_mode)
    else:
        tests[args.test](device_id, run_mode)
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()