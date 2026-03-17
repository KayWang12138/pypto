import os
import pypto
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

# ===== Golden Function (PyTorch Reference - BF16 version) =====
def embedding_head_quant_golden(weight, scale, eps=1e-4, min_v=-128.0, max_v=127.0):
    """PyTorch reference implementation for embedding head quantization (BF16 I/O, FP32 compute).

    Args:
        weight: Input weight tensor (N, M) in BF16
        scale: Quantization scale tensor (1, 1) scalar in BF16
        eps: Minimum scale threshold (default: 1e-4)
        min_v: Quantization lower bound (default: -128.0)
        max_v: Quantization upper bound (default: 127.0)

    Returns:
        Tuple of (quantized_weight, clamped, protected_scale) all in BF16
    """
    weight_fp32 = weight.float()
    scale_fp32 = scale.float()
    
    eps_tensor = torch.tensor(eps, device=scale_fp32.device, dtype=torch.float32)
    protected_scale = torch.where(scale_fp32 > eps_tensor, scale_fp32, eps_tensor)
    weight_normalized = weight_fp32 / protected_scale
    weight_rounded = (weight_normalized.round() - weight_normalized).detach() + weight_normalized
    clamped = torch.clamp(weight_rounded, min_v, max_v)
    output = clamped * protected_scale
    
    return output.to(torch.bfloat16), clamped.to(torch.bfloat16), weight_normalized.to(torch.bfloat16)

# ===== Backward JIT Kernel Implementation =====
def create_embedding_head_quant_backward_kernel(m, eps=1e-4, min_v=-128.0, max_v=127.0, run_mode="npu", unroll_list=None, enable_perf=False):
    """创建 embedding_head_quant 反向传播算子 kernel

    Args:
        m: 固定维度 M (weight 的第二维度)
        eps: 最小 scale 阈值
        min_v: 量化下界
        max_v: 量化上界
        run_mode: 运行模式
        unroll_list: unroll 参数列表
        enable_perf: 是否启用性能数据采集
    """
    N = pypto.frontend.dynamic("N")
    
    if unroll_list is None:
        unroll_list = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}")

    runtime_opts = {
        "run_mode": mode,
        "stitch_function_inner_memory": 512,
        "stitch_function_outcast_memory": 512,
        "stitch_function_num_initial": 128,
        "stitch_function_max_num": 128,
        "stitch_function_num_step": 20
    }

    debug_opts = {"runtime_debug_mode": 1} if enable_perf else {}

    @pypto.frontend.jit(runtime_options=runtime_opts, debug_options=debug_opts, )
    def embedding_head_quant_backward_kernel(
        grad_output: pypto.Tensor((N, m), pypto.DT_BF16),
        weight: pypto.Tensor((N, m), pypto.DT_BF16),
        scale: pypto.Tensor((1, 1), pypto.DT_BF16),
    ) -> (
        pypto.Tensor((N, m), pypto.DT_BF16),
        pypto.Tensor((1, 1), pypto.DT_BF16),
    ):
        # pypto.experimental.set_operation_options(combine_axis=True)
        grad_weight_out = pypto.Tensor((N, m), pypto.DT_BF16)
        grad_scale_out = pypto.Tensor((1, 1), pypto.DT_BF16)
        n = weight.shape[0]

        pypto.set_vec_tile_shapes(64, 256)
        scale_fp32 = pypto.cast(scale, pypto.DT_FP32)
        protected_scale = pypto.maximum(scale_fp32, eps)
        
        # 计算全局的 scale_mask
        scale_mask = pypto.ge(scale_fp32, eps)
        scale_mask_fp32 = pypto.where(scale_mask, 1.0, 0.0)
        
        # 在循环外初始化 FP32 的局部累加器
        grad_scale_acc = pypto.full([1, 1], 0.0, pypto.DT_FP32)
        
        for n_offset, unroll_length in pypto.loop_unroll(
            0, n, 1,
            name="BACKWARD_LOOP_N_UNROLL",
            idx_name="n_offset",
            unroll_list=unroll_list
        ):
            tile_n = unroll_length

            grad_out_tile = pypto.view(grad_output, [tile_n, m], [n_offset, 0])
            weight_tile = pypto.view(weight, [tile_n, m], [n_offset, 0])

            grad_out_fp32 = pypto.cast(grad_out_tile, pypto.DT_FP32)
            weight_fp32 = pypto.cast(weight_tile, pypto.DT_FP32)

            # 重算前向计算
            scale_n = pypto.expand_clone(protected_scale, [tile_n, 1])
            normalized = pypto.div(weight_fp32, scale_n)
            rounded = pypto.round(normalized, decimals=0)
            clamped = pypto.clip(rounded, min_v, max_v)

            # 计算mask: (rounded >= min_v) & (rounded <= max_v)
            # 等价与计算 equal(rounded, clamped) + where(相同, 1.0, 0.0)
            # 规避where, 求两者差，其都是整数型浮点数(xxx.0)
            diff = pypto.sub(rounded, clamped)
            abs_diff = pypto.abs(diff)
            out_of_bounds = pypto.clip(abs_diff, 0.0, 1.0)
            neg_out_of_bounds = pypto.mul(out_of_bounds, -1.0)
            mask_float = pypto.add(neg_out_of_bounds, 1.0)

            # ================= 计算 grad_weight =================
            grad_weight_fp32 = pypto.mul(grad_out_fp32, mask_float)
            grad_weight_tile = pypto.cast(grad_weight_fp32, pypto.DT_BF16)
            pypto.assemble(grad_weight_tile, [n_offset, 0], grad_weight_out)

            # ================= 计算 grad_scale =================
            # 乘法路径: grad_output * clamped 
            grad_scale_mul_tile = pypto.mul(grad_out_fp32, clamped) # [tile_n, m]
            grad_scale_mul_tile_m = pypto.sum(grad_scale_mul_tile, dim=1, keepdim=True)
            grad_scale_mul_tile_n = pypto.sum(grad_scale_mul_tile_m, dim=0, keepdim=True)

            # 除法路径: grad_output * mask * (-weight / protected_scale)
            neg_weight_fp32 = pypto.mul(weight_fp32, -1.0)
            weight_div_scale_tile = pypto.div(neg_weight_fp32, scale_n) 
            grad_scale_div_step1 = pypto.mul(grad_out_fp32, mask_float)
            grad_scale_div_tile = pypto.mul(grad_scale_div_step1, weight_div_scale_tile)
            grad_scale_div_tile_m = pypto.sum(grad_scale_div_tile, dim=1, keepdim=True)
            grad_scale_div_tile_n = pypto.sum(grad_scale_div_tile_m, dim=0, keepdim=True)

            # 合并两条路径
            grad_scale_tile = pypto.add(grad_scale_mul_tile_n, grad_scale_div_tile_n)
            # 将当前 Tile 的梯度累加到全局 FP32 寄存器中
            grad_scale_acc[:] = pypto.add(grad_scale_acc, grad_scale_tile)

        # 应用 scale_mask 掩码，并进行最终的类型转换和写回
        final_grad_scale_fp32 = pypto.mul(grad_scale_acc, scale_mask_fp32)
        final_grad_scale_bf16 = pypto.cast(final_grad_scale_fp32, pypto.DT_BF16)
        pypto.assemble(final_grad_scale_bf16, [0, 0], grad_scale_out)
        return grad_weight_out, grad_scale_out

    return embedding_head_quant_backward_kernel


# ===== Performance Test Cases =====
# Test configurations for different model sizes
PERFORMANCE_TEST_CONFIGS = {
    "3B": {
        "weight_shape": (153376, 2048),
        "description": "3B model embedding head"
    },
    "7B": {
        "weight_shape": (153376, 3072),
        "description": "7B model embedding head"
    },
    "30B": {
        "weight_shape": (75776, 2560),
        "description": "30B model embedding head"
    }
}

# ==========================================
# Backward Tests (Scaled up to large shapes)
# ==========================================
def test_backward_3b(device_id=None, run_mode="npu"):
    """Test 3B model embedding head backward: weight_shape=(153376, 2048)"""
    print("Test Backward: 3B Model - weight_shape=(153376, 2048)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    weight_shape = (153376, 2048)
    n, m = weight_shape

    torch.manual_seed(42)
    grad_output = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    weight = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16).requires_grad_(True)
    scale = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device).requires_grad_(True)

    # --- PyTorch AutoGrad 计算 ---
    output, _, _ = embedding_head_quant_golden(weight, scale)
    output.backward(grad_output)
    
    # 获取 PyTorch 算出的标准梯度
    expected_grad_weight = weight.grad.clone()
    expected_grad_scale = scale.grad.clone()
    
    # 清空 PyTorch 梯度
    weight.grad = None
    scale.grad = None

    kernel = create_embedding_head_quant_backward_kernel(m, run_mode=run_mode)
    grad_weight_out, grad_scale_out = kernel(grad_output, weight, scale)

    max_diff_weight = (grad_weight_out - expected_grad_weight).abs().max().item()
    print(f"  grad_weight max diff: {max_diff_weight:.6f}")
    assert_allclose(grad_weight_out.float().cpu().numpy(), expected_grad_weight.float().cpu().numpy(), rtol=3e-3, atol=3e-3)

    max_diff_scale = (grad_scale_out - expected_grad_scale).abs().max().item()
    print(f"  grad_scale max diff: {max_diff_scale:.6f}")
    assert_allclose(grad_scale_out.float().cpu().numpy(), expected_grad_scale.float().cpu().numpy(), rtol=3e-3, atol=3e-3)

    print("  ✓ Passed")


def test_backward_7b(device_id=None, run_mode="npu"):
    """Test 7B model embedding head backward: weight_shape=(153376, 3072)"""
    print("Test Backward: 7B Model - weight_shape=(153376, 3072)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    weight_shape = (153376, 3072)
    n, m = weight_shape

    torch.manual_seed(42)
    grad_output = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    weight = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16).requires_grad_(True)
    scale = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device).requires_grad_(True)

    # --- PyTorch AutoGrad 计算 ---
    output, _, _ = embedding_head_quant_golden(weight, scale)
    output.backward(grad_output)
    
    # 获取 PyTorch 算出的标准梯度
    expected_grad_weight = weight.grad.clone()
    expected_grad_scale = scale.grad.clone()
    
    # 清空 PyTorch 梯度
    weight.grad = None
    scale.grad = None

    kernel = create_embedding_head_quant_backward_kernel(m, run_mode=run_mode)
    grad_weight_out, grad_scale_out = kernel(grad_output, weight, scale)

    max_diff_weight = (grad_weight_out - expected_grad_weight).abs().max().item()
    print(f"  grad_weight max diff: {max_diff_weight:.6f}")
    assert_allclose(grad_weight_out.float().cpu().numpy(), expected_grad_weight.float().cpu().numpy(), rtol=3e-3, atol=3e-3)

    max_diff_scale = (grad_scale_out - expected_grad_scale).abs().max().item()
    print(f"  grad_scale max diff: {max_diff_scale:.6f}")
    assert_allclose(grad_scale_out.float().cpu().numpy(), expected_grad_scale.float().cpu().numpy(), rtol=3e-3, atol=3e-3)

    print("  ✓ Passed")


def test_backward_30b(device_id=None, run_mode="npu"):
    """Test 30B model embedding head backward: weight_shape=(75776, 2560)"""
    print("Test Backward: 30B Model - weight_shape=(75776, 2560)")

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    weight_shape = (75776, 2560)
    n, m = weight_shape

    torch.manual_seed(42)
    grad_output = (torch.randn(weight_shape, dtype=torch.float32, device=device)).to(torch.bfloat16)
    # 使用稍微大一点的值域触发不同的边界效果
    weight = (torch.randn(weight_shape, dtype=torch.float32, device=device) * 50).to(torch.bfloat16).requires_grad_(True)
    scale = torch.full((1, 1), 0.5, dtype=torch.bfloat16, device=device).requires_grad_(True)

    # --- PyTorch AutoGrad 计算 ---
    output, _, _ = embedding_head_quant_golden(weight, scale)
    output.backward(grad_output)
    
    # 获取 PyTorch 算出的标准梯度
    expected_grad_weight = weight.grad.clone()
    expected_grad_scale = scale.grad.clone()
    
    # 清空 PyTorch 梯度
    weight.grad = None
    scale.grad = None

    kernel = create_embedding_head_quant_backward_kernel(m, run_mode=run_mode)
    grad_weight_out, grad_scale_out = kernel(grad_output, weight, scale)

    max_diff_weight = (grad_weight_out - expected_grad_weight).abs().max().item()
    print(f"  grad_weight max diff: {max_diff_weight:.6f}")
    assert_allclose(grad_weight_out.float().cpu().numpy(), expected_grad_weight.float().cpu().numpy(), rtol=3e-3, atol=3e-3)

    max_diff_scale = (grad_scale_out - expected_grad_scale).abs().max().item()
    print(f"  grad_scale max diff: {max_diff_scale:.6f}")
    assert_allclose(grad_scale_out.float().cpu().numpy(), expected_grad_scale.float().cpu().numpy(), rtol=3e-3, atol=3e-3)

    print("  ✓ Passed")


# ===== Main Test Runner =====
def main():
    """Run all test cases for forward and backward operators."""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Head Quantization Operator Tests")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"],
                        help="Execution mode: npu (hardware) or sim (simulation)")
    parser.add_argument('--model', type=str, default="all", choices=["3b", "7b", "30b", "all"],
                        help="Model size to test: 3b, 7b, 30b, or all")
    parser.add_argument('--test', type=str, default="backward", choices=["forward", "backward", "all"],
                        help="Test type: forward, backward, or all")
    args = parser.parse_args()

    device_id = None
    if args.run_mode == "npu":
        if 'TILE_FWK_DEVICE_ID' not in os.environ:
            print("ERROR: TILE_FWK_DEVICE_ID not set. Please set environment variable:")
            print("  export TILE_FWK_DEVICE_ID=0")
            print("\nOr use simulation mode:")
            print("  python3 embedding_head_quant.py --run_mode sim")
            return
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        import torch_npu
        torch.npu.set_device(device_id)
        print(f"Using NPU device: {device_id}")
    else:
        print("Using simulation mode (CPU)")

    all_passed = True


    if args.test == "backward" or args.test == "all":
        print("=" * 70)
        print("Embedding Head Quantization Operator - Backward Propagation Tests")
        print("3B: (153376, 2048) | 7B: (153376, 3072) | 30B: (75776, 2560)")
        print("=" * 70)
        print()

        if args.model == "all" or args.model == "3b":
            try:
                test_backward_3b(device_id, args.run_mode)
                print()
            except Exception as e:
                print(f"  ❌ 3B Backward test failed: {e}")
                all_passed = False

        if args.model == "all" or args.model == "7b":
            try:
                test_backward_7b(device_id, args.run_mode)
                print()
            except Exception as e:
                print(f"  ❌ 7B Backward test failed: {e}")
                all_passed = False

        if args.model == "all" or args.model == "30b":
            try:
                test_backward_30b(device_id, args.run_mode)
                print()
            except Exception as e:
                print(f"  ❌ 30B Backward test failed: {e}")
                all_passed = False

    print("=" * 70)
    if all_passed:
        print("All tests passed successfully!")
    else:
        print("Some tests failed!")
    print("=" * 70)

if __name__ == "__main__":
    main()