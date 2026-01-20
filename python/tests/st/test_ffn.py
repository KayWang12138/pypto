# ffn_benchmark.py
import torch
import torch_npu
import pypto
import time
import os
import numpy as np


def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.nn.functional.silu(y_1) * y_2


def ffn_forward_torch(x, w1, w2, activation=torch.nn.functional.silu):
    """
    Simple FFN: x @ w1 -> activate -> @ w2
    Shapes:
      x: [M, H]
      w1: [H, F]
      w2: [F, H]
      out: [M, H]
    """
    # First linear: [M, H] @ [H, F] -> [M, F]
    hidden = torch.matmul(x, w1)
    # Activation
    hidden = swiglu(hidden)
    # Second linear: [M, F] @ [F, H] -> [M, H]
    out = torch.matmul(hidden, w2)
    return out


def ffn_forward_torch_wo_silu(x, w1):
    """
    Simple FFN: x @ w1 -> activate -> @ w2
    Shapes:
      x: [M, H]
      w1: [H, F]
      w2: [F, H]
      out: [M, H]
    """
    # First linear: [M, H] @ [H, F] -> [M, F]
    out = torch.matmul(x, w1)
    # Activation
    # hidden = swiglu(hidden)
    # Second linear: [M, F] @ [F, H] -> [M, H]
    # out = torch.matmul(hidden, w2)
    return out

@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_ffn_forward(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    w2: pypto.Tensor,
    out: pypto.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    
    pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_k, tile_k * 2], [tile_n * 2, tile_n * 2], True, False)
    pypto.set_matrix_size([tile_m, tile_k, tile_n * 2])

    hidden = pypto.matmul(x, w1, pypto.DT_FP32)
    hidden1 = hidden[:, :tile_n]
    hidden2 = hidden[:, tile_n:]
    pypto.set_vec_tile_shapes(tile_m, tile_n)
    activated = hidden1 * pypto.sigmoid(hidden1) * hidden2

    pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_n, tile_n], [tile_k, tile_k], True, False)

    pypto.set_matrix_size([tile_m, tile_n, tile_k])
    res = pypto.matmul(activated, w2, pypto.DT_FP32)
    pypto.assemble(res, [0, 0], out)


@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_ffn_forward_v1(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    w2: pypto.Tensor,
    out: pypto.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    hidden_size = x.shape[1]
    pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_k, tile_k * 2], [tile_n * 2, tile_n * 4], True, False)
    pypto.set_matrix_size([tile_m, tile_k, tile_n * 2])

    hidden = pypto.matmul(x, w1, pypto.DT_FP32)
    hidden1 = hidden[:, :hidden_size]
    hidden2 = hidden[:, hidden_size:]
    pypto.set_vec_tile_shapes(tile_m, tile_n)
    activated = hidden1 * pypto.sigmoid(hidden1) * hidden2

    pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_n, tile_n], [tile_k, tile_k], True, False)

    pypto.set_matrix_size([tile_m, tile_n, tile_k])
    res = pypto.matmul(activated, w2, pypto.DT_FP32)
    pypto.assemble(res, [0, 0], out)

@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_ffn_forward_v2(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    w2: pypto.Tensor,
    out: pypto.Tensor,
    tilling_token: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    task_id: int = 0,
) -> None:
    hidden_size = x.shape[1]
    ffn_hidden_size = w2.shape[0]
    total_tokens = x.shape[0]
    assert total_tokens % tilling_token == 0, "total_tokens must be divisible by tilling_token"
    num_tiles = total_tokens // tilling_token
    for idx in pypto.loop(0, num_tiles, name="token_loop_{}".format(task_id), unroll_list=[4]):
        start_token = idx * tilling_token

        x_tile = pypto.view(x, shape=[tilling_token, hidden_size], offsets=[start_token, 0])
        pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n * 2], True, False)
        pypto.set_matrix_size([tile_m, tile_k, tile_n * 2])


        hidden = pypto.matmul(x_tile, w1, pypto.DT_FP16)
        hidden1 = hidden[:, :ffn_hidden_size]
        hidden2 = hidden[:, ffn_hidden_size:]
        pypto.set_vec_tile_shapes(tile_m, tile_n)
        activated = hidden1 * pypto.sigmoid(hidden1) * hidden2

        pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_n, tile_n], [tile_k, tile_k], True, False)

        pypto.set_matrix_size([tile_m, tile_n, tile_k])
        res = pypto.matmul(activated, w2, pypto.DT_FP16)
        pypto.assemble(res, [start_token, 0], out)


@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_ffn_forward_v3(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    out: pypto.Tensor,
    tile_n: int,
    tile_k: int,
    task_id: int = 0,
    hidden_size: int = 6144,
    ffn_hidden_size: int = 6144,
    total_tokens: int = 6144,
    unroll_level: int = 32,
) -> None:

    for idx, loop_base in pypto.loop_unroll(total_tokens, name="token_loop_{}".format(task_id), unroll_list=[ unroll_level*4, unroll_level*2, unroll_level, 24,12, 6, 3, 1]):
        # start_token = idx * tilling_token
        pypto.set_vec_tile_shapes(loop_base, hidden_size)
        x_tile = pypto.view(x, shape=[loop_base, hidden_size], offsets=[idx, 0])
        w1_tile = pypto.view(w1, shape=[hidden_size, ffn_hidden_size], offsets=[0, 0])

        pypto.set_cube_tile_shapes([loop_base, loop_base], [tile_k, tile_k * 16], [tile_n, tile_n], True, False)
        pypto.set_matrix_size([loop_base, hidden_size, ffn_hidden_size])

        # res = pypto.matmul(x_tile_fp32, w1_tile_fp32, pypto.DT_FP32)
        res = pypto.matmul(x_tile, w1_tile, pypto.DT_FP16)
        # res_fp16 = pypto.cast(res, pypto.DT_FP16)
        pypto.assemble(res, [idx, 0], out)


@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
    # pass_options={"pg_parallel_lower_bound": 32}
)
def pypto_ffn_forward_v6(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    out: pypto.Tensor,
    tile_n: int,
    tile_k: int,
    task_id: int = 0,
    hidden_size: int = 4096,
    ffn_hidden_size: int = 4096,
    total_tokens: int = 8,
    unroll_level: int = 8,
) -> None:

    pypto.set_cube_tile_shapes([unroll_level, unroll_level], [tile_k, tile_k], [tile_n, tile_n], True, False)
    pypto.set_matrix_size([unroll_level, hidden_size, ffn_hidden_size])
    res = pypto.matmul(x, w1, pypto.DT_FP16)
    pypto.assemble(res, [0, 0], out)


# 优化版本 v4: 不使用循环，直接处理整个矩阵（减少循环和 view/assemble 开销）
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
)
def pypto_ffn_forward_v4(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    w2: pypto.Tensor,
    out: pypto.Tensor,
    tilling_token: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    """优化版本：不使用循环，直接处理整个矩阵"""
    # 第一个 matmul: [total_tokens, hidden_size] @ [hidden_size, ffn_hidden_size]
    pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_k, tile_k], [tile_n, tile_n], True, False)
    pypto.set_matrix_size([tile_m, tile_k, tile_n])
    hidden = pypto.matmul(x, w1, pypto.DT_FP16)
    
    # 第二个 matmul: [total_tokens, ffn_hidden_size] @ [ffn_hidden_size, hidden_size]
    pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_n, tile_n], [tile_k, tile_k], True, False)
    pypto.set_matrix_size([tile_m, tile_n, tile_k])
    res = pypto.matmul(hidden, w2, pypto.DT_FP16)
    pypto.assemble(res, [0, 0], out)


# 优化版本 v5: 改进的循环版本，优化 tile 配置和循环展开
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True},
)
def pypto_ffn_forward_v5(
    x: pypto.Tensor,
    w1: pypto.Tensor,
    w2: pypto.Tensor,
    out: pypto.Tensor,
    tilling_token: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    task_id: int = 0,
) -> None:
    """
    优化版本：改进的循环版本
    优化点：
    1. tile_m 应该匹配或接近 tilling_token，减少切分
    2. 增加循环展开因子（从4增加到16）
    3. 使用 FP16 而不是 FP32
    4. 优化 matrix_size 配置
    """
    hidden_size = x.shape[1]
    total_tokens = x.shape[0]
    assert total_tokens % tilling_token == 0, "total_tokens must be divisible by tilling_token"
    num_tiles = total_tokens // tilling_token
    
    # 优化：增加循环展开，减少循环开销
    unroll_factor = min(16, num_tiles)
    
    for idx in pypto.loop(0, num_tiles, name="token_loop_{}".format(task_id), unroll_list=[unroll_factor]):
        start_token = idx * tilling_token
        x_tile = pypto.view(x, shape=[tilling_token, hidden_size], offsets=[start_token, 0])
        
        # 优化：tile_m 应该匹配 tilling_token，减少切分
        actual_tile_m = min(tile_m, tilling_token)
        pypto.set_cube_tile_shapes([actual_tile_m, actual_tile_m], [tile_k, tile_k * 2], [tile_n, tile_n * 2], True, False)
        pypto.set_matrix_size([actual_tile_m, tile_k, tile_n])
        hidden = pypto.matmul(x_tile, w1, pypto.DT_FP16)
        
        pypto.set_cube_tile_shapes([actual_tile_m, actual_tile_m], [tile_n, tile_n * 2], [tile_k, tile_k * 2], True, False)
        pypto.set_matrix_size([actual_tile_m, tile_n, tile_k])
        res = pypto.matmul(hidden, w2, pypto.DT_FP16)
        pypto.assemble(res, [start_token, 0], out)


def test_ffn(device_id: int = 2):
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    print(f"FFN Test on {device}")
    total_tokens = 4096
    hidden_size = 2048
    ffn_hidden_size = 4096
    dtype = torch.float16
    print(f"Config: tokens={total_tokens}, H={hidden_size}, F={ffn_hidden_size}, dtype={dtype}")
    x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01
    w1 = torch.randn(hidden_size, ffn_hidden_size * 2, dtype=dtype, device=device) * 0.01 * 2 - 0.01
    w2 = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01
    out_torch = ffn_forward_torch(x, w1, w2)
    torch.npu.synchronize()
    print(f"Out torch shape: {out_torch.shape}")
    print(f"Out torch values: {out_torch.flatten().tolist()[:10]}")

        
    out_tensor = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)
    x_pypto = pypto.from_torch(x, name="x")
    w1_pypto = pypto.from_torch(w1, "w1")
    w2_pypto = pypto.from_torch(w2, "w2")
    out_pypto = pypto.from_torch(out_tensor, "out")
    # pypto_ffn_forward(x_pypto, w1_pypto, w2_pypto, out_pypto, tile_m=4, tile_n=32, tile_k=32)
    # pypto_ffn_forward_v1(x_pypto, w1_pypto, w2_pypto, out_pypto, tile_m=4, tile_n=32, tile_k=128)
    pypto_ffn_forward_v2(x_pypto, w1_pypto, w2_pypto, out_pypto, tilling_token=8, tile_m=4, tile_n=128, tile_k=128)

    torch.npu.synchronize()

    print(f"Out pypto shape: {out_tensor.shape}")
    print(f"Out pypto values: {out_tensor.flatten().tolist()[:10]}")

    # Convert bfloat16 to float32 for numpy comparison (numpy doesn't support bfloat16)
    out_tensor_np = out_tensor.detach().cpu().float().numpy()
    out_torch_np = out_torch.detach().cpu().float().numpy()
    print("out_tensor shape:", out_tensor.shape)
    print("out_torch shape:", out_torch.shape)
    abs_diff = torch.abs(out_tensor - out_torch)
    print(f"Max abs diff: {abs_diff.max().item():.6f}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6f}")
    np.testing.assert_allclose(
        out_tensor_np,
        out_torch_np,
        atol=1e-2,
        rtol=1e-2,
        err_msg="FFN mismatch!"
    )
    print(f"✅ test_ffn forward passed!")




def calculate_ffn_tflops(
    total_tokens: int,
    hidden_size: int,
    intermediate_size: int
) -> float:
    """
    计算标准 FFN 的 TFLOPs (Tera Floating Point Operations).
    
    FFN 包含两个主要的矩阵乘法和 SwiGLU 激活：
    1. x @ w1: [M, H] @ [H, 2F] → [M, 2F]
       FLOPs = 2 * M * H * (2F)
    2. swiglu_out @ w2: [M, F] @ [F, H] → [M, H]
       FLOPs = 2 * M * F * H
    3. SwiGLU: ~6 * M * F 元素操作（保守估计）
    
    Args:
        total_tokens (M): 输入 token 数量
        hidden_size (H): 隐藏维度
        intermediate_size (F): 中间层维度（注意 w1 输出是 2F）
        
    Returns:
        TFLOPs
    """
    M = total_tokens
    H = hidden_size
    F = intermediate_size

    mm1_flops = 2 * M * H * F      # x @ w1
    # mm2_flops = 2 * M * F * H            # activated @ w2
    # swiglu_flops = M * F * 6             # sigmoid + mul ≈ 6 ops per element

    # quant_dequant_flops = (mm1_flops + mm2_flops) * 0.02  # 估算量化开销

    total_flops = mm1_flops 
    # + mm2_flops 
    # + swiglu_flops 
    # + quant_dequant_flops
    tflops = total_flops / 1e12
    return tflops


def test_ffn_v2(device_id: int = 4, num_rounds: int = 5):
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    print(f"\n🚀 FFN Performance Test on {device} | Rounds: {num_rounds}")

    total_tokens = 6144
    hidden_size = 6144
    ffn_hidden_size = 6144
    # dtype = torch.float32
    dtype = torch.float16

    print(f"Config: tokens={total_tokens}, H={hidden_size}, F={ffn_hidden_size}, dtype={dtype}")

    # Warm-up
    print("\n🔥 Warm-up round...")
    x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01
    # w1 = torch.randn(hidden_size, ffn_hidden_size * 2, dtype=dtype, device=device) * 0.01 * 2 - 0.01
    w1 = torch.randn(hidden_size, ffn_hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01

    # w2 = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01

    # Warm-up PyTorch
    out_warm_torch = ffn_forward_torch_wo_silu(x, w1)
    torch.npu.synchronize()
    print(f"Out warm torch: {out_warm_torch.shape}")
    print(f"Out warm torch values: {out_warm_torch.flatten().tolist()[:10]}")

    # Warm-up PyPTO
    out_warm = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)
    x_p = pypto.from_torch(x, name="x_warm")
    w1_p = pypto.from_torch(w1, "w1_warm")
    # w2_p = pypto.from_torch(w2, "w2_warm")
    out_p = pypto.from_torch(out_warm, "out_warm")
    pypto_ffn_forward_v3(x_p, w1_p, out_p, unroll_level=48, tile_n=128, tile_k=48)

    # pypto_ffn_forward_v4(x_p, w1_p, w2_p, out_p, tilling_token=256, tile_m=32, tile_n=128, tile_k=128)
    torch.npu.synchronize()
    np.testing.assert_allclose(
        out_warm_torch.detach().cpu().float().numpy(),
        out_warm.detach().cpu().float().numpy(),
        atol=1e-2,
        rtol=1e-2,
        err_msg="FFN mismatch!"
    )
    print(f"✅ test_ffn forward passed!")

    # 清理缓存
    del x, w1, out_warm, x_p, w1_p, out_p
    torch.npu.empty_cache()

    # 存储结果
    torch_times = []
    torch_mem_peaks = []
    pypto_times = []
    pypto_mem_peaks = []

    tflops_per_run = calculate_ffn_tflops(total_tokens, hidden_size, ffn_hidden_size)
    print(f"\n📊 Estimated TFLOPs per run: {tflops_per_run:.4f} TFLOP")

    for r in range(num_rounds):
        print(f"\n🔁 Round {r+1}/{num_rounds}")

        # 重新初始化数据
        x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01
        # w1 = torch.randn(hidden_size, ffn_hidden_size * 2, dtype=dtype, device=device) * 0.01 * 2 - 0.01
        w1 = torch.randn(hidden_size, ffn_hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01

        # w2 = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype, device=device) * 0.01 * 2 - 0.01

         # --- PyTorch 测试 ---
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        start_time = time.perf_counter()
        torch.npu.synchronize()
        out_torch = ffn_forward_torch_wo_silu(x, w1)
        torch.npu.synchronize()
        end_time = time.perf_counter()
        print(f"Out torch: {out_torch.shape}")
        print(f"Out torch values: {out_torch.flatten().tolist()[:10]}")

        torch_time_s = end_time - start_time
        torch_mem_mb = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        torch_times.append(torch_time_s)
        torch_mem_peaks.append(torch_mem_mb)


        # --- PyPTO 测试 ---
        torch.npu.reset_peak_memory_stats(device)
        out_pypto_tensor = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)
        x_p = pypto.from_torch(x, name=f"x_{r}")
        w1_p = pypto.from_torch(w1, f"w1_{r}")
        # w2_p = pypto.from_torch(w2, f"w2_{r}")
        out_p = pypto.from_torch(out_pypto_tensor, f"out_{r}")
        torch.npu.synchronize()
        start_time2 = time.perf_counter()
        torch.npu.synchronize()
        pypto_ffn_forward_v3(x_p, w1_p, out_p, unroll_level=48, tile_n=128, tile_k=48)
        # pypto_ffn_forward_v4(x_p, w1_p, w2_p, out_p, tilling_token=256, tile_m=32, tile_n=128, tile_k=128)

        torch.npu.synchronize()
        end_time2 = time.perf_counter()
        
        pypto_time_s = end_time2 - start_time2
        pypto_mem_mb = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        pypto_times.append(pypto_time_s)
        pypto_mem_peaks.append(pypto_mem_mb)


        # 验证数值一致性（可选，可注释掉以加速）
        abs_diff = torch.abs(out_pypto_tensor - out_torch)
        max_diff = abs_diff.max().item()
        if max_diff > 1e-2:
            print(f"⚠️ Warning: Max diff = {max_diff:.6f} in round {r+1}")

        print(f"  PyTorch: {torch_time_s*1000:.2f} ms | Mem: {torch_mem_mb:.1f} MB")
        print(f"  PyPTO  : {pypto_time_s*1000:.2f} ms | Mem: {pypto_mem_mb:.1f} MB")

        # 清理
        del x, w1, out_torch, out_pypto_tensor, x_p, w1_p, out_p
        torch.npu.empty_cache()

    # 计算平均值
    avg_torch_time = sum(torch_times) / len(torch_times)
    avg_pypto_time = sum(pypto_times) / len(pypto_times)
    avg_torch_mem = sum(torch_mem_peaks) / len(torch_mem_peaks)
    avg_pypto_mem = sum(pypto_mem_peaks) / len(pypto_mem_peaks)

    torch_tflops = tflops_per_run / avg_torch_time
    pypto_tflops = tflops_per_run / avg_pypto_time

    print("\n" + "="*60)
    print("📈 Final Results (Average over {} rounds)".format(num_rounds))
    print("="*60)
    print(f"PyTorch: {avg_torch_time*1000:.2f} ms | {avg_torch_mem:.1f} MB | {torch_tflops:.2f} TFLOP/s")
    print(f"PyPTO  : {avg_pypto_time*1000:.2f} ms | {avg_pypto_mem:.1f} MB | {pypto_tflops:.2f} TFLOP/s")
    print(f"Speedup: {avg_torch_time / avg_pypto_time:.2f}x")
    print("="*60)



if __name__ == "__main__":
    # Typical MoE FFN config
    # benchmark_ffn(
    #     total_tokens=16384,      # 4 * 4096
    #     hidden_size=4096,
    #     ffn_hidden_size=16384,
    #     dtype=torch.float16,
    # )
    # test_ffn(device_id=2)
    test_ffn_v2(device_id=2, num_rounds=5)




# def create_ffn_forward_pypto_fp16(run_mode: str = "npu"):
#     if run_mode == "npu":
#         run_mode = pypto.RunMode.NPU
#     elif run_mode == "cpu":
#         run_mode = pypto.RunMode.SIM
#     else:
#         raise ValueError(f"Invalid run mode: {run_mode}")

#     total_tokens = pypto.frontend.dynamic("total_tokens")
#     hidden_size = pypto.frontend.dynamic("hidden_size")
#     ffn_hidden_size = pypto.frontend.dynamic("ffn_hidden_size")
    

#     @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
#     def ffn_forward_pypto_bf16_kernel(
#         x: pypto.tensor((total_tokens, hidden_size), dtype=pypto.DT_BF16), 
#         w1: pypto.tensor((hidden_size, ffn_hidden_size * 2), dtype=pypto.DT_BF16),
#         w2: pypto.tensor((ffn_hidden_size, hidden_size), dtype=pypto.DT_BF16)
#     ):
#         pypto.set_cube_tile_shapes([4, 4], [32, 32], [32, 32])
#         pypto.set_matrix_size([total_tokens, hidden_size, ffn_hidden_size * 2])
#         hidden = pypto.matmul(x, w1)
#         hidden1 = hidden[:, :ffn_hidden_size]
#         hidden2 = hidden[:, ffn_hidden_size:]
#         activated = hidden1 * pypto.sigmoid(hidden1) * hidden2
#         pypto.set_matrix_size([total_tokens, ffn_hidden_size, hidden_size])
#         out = pypto.matmul(activated, w2)
#         return out

#     return ffn_forward_pypto_bf16_kernel


# def benchmark_ffn(
#     total_tokens=16384,      # e.g., batch=4, seq=4096
#     hidden_size=4096,
#     ffn_hidden_size=14336,   # common in LLaMA/Mixtral
#     dtype=torch.float16,
#     num_warmup=5,
#     num_iter=20,
# ):
#     # 确保使用 NPU
#     device = "npu:0"
#     torch.npu.set_device(device)
#     print(f"FFN Benchmark on {device}")
#     print(f"Config: tokens={total_tokens}, H={hidden_size}, F={ffn_hidden_size}, dtype={dtype}")

#     # Create tensors on NPU
#     x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)
#     w1 = torch.randn(hidden_size, ffn_hidden_size, dtype=dtype, device=device)
#     w2 = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype, device=device)

#     # Warmup
#     print("Warming up...")
#     for _ in range(num_warmup):
#         out = ffn_forward(x, w1, w2)
#         torch.npu.synchronize()


#     total_time = 0.0
#     total_mem_gb = 0.0
#     for _ in range(num_iter):
#         # Reset memory stats
#         torch.npu.empty_cache()
#         torch.npu.reset_peak_memory_stats(device)
            
#         # Create tensors on NPU
#         x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)
#         w1 = torch.randn(hidden_size, ffn_hidden_size, dtype=dtype, device=device)
#         w2 = torch.randn(ffn_hidden_size, hidden_size, dtype=dtype, device=device)

#         # Benchmark
#         print("Running benchmark...")
#         torch.npu.synchronize()
#         start_time = time.perf_counter()
#         out = ffn_forward(x, w1, w2)
#         torch.npu.synchronize()

#         end_time = time.perf_counter()
#         total_time += end_time - start_time

#         peak_mem_gb = torch.npu.max_memory_allocated(device) / (1024**3)
#         total_mem_gb += peak_mem_gb

#         del x, w1, w2, out
    
#     avg_time = total_time / num_iter
#     tokens_per_sec = total_tokens / avg_time
#     tflops = (
#         2 * total_tokens * hidden_size * ffn_hidden_size * 2  # two GEMMs, each ~2*H*F*M flops
#         + 2 * total_tokens * ffn_hidden_size                  # silu is ~2 flops per element (approx)
#     ) / (avg_time * 1e12)

#     # Memory
#     avg_mem_gb = total_mem_gb / num_iter

#     print("\n=== FFN Benchmark Results ===")
#     print(f"Total tokens: {total_tokens}")
#     print(f"Avg forward time: {avg_time * 1000:.2f} ms")
#     print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
#     print(f"Approx TFLOPS: {tflops:.2f} TFLOPS")
#     print(f"Peak NPU Memory: {avg_mem_gb:.2f} GB")

#     return tokens_per_sec, tflops, avg_mem_gb



# @pypto.jit(
#     codegen_options={"support_dynamic_aligned": True},
#     # pass_options={"pg_parallel_lower_bound": 32}
# )
# def pypto_ffn_forward_native(
#     x: pypto.Tensor,
#     w1: pypto.Tensor,
#     w2: pypto.Tensor,
#     out: pypto.Tensor,
#     tile_m: int,
#     tile_n: int,
#     tile_k: int,
# ) -> None:
    
#     # pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_k, tile_k * 2], [tile_n * 2, tile_n * 2], True, False)
#     # pypto.set_matrix_size([tile_m, tile_k, tile_n * 2])

#     hidden = pypto.matmul(x, w1, pypto.DT_FP32)
#     hidden1 = hidden[:, :tile_n]
#     hidden2 = hidden[:, tile_n:]
#     # pypto.set_vec_tile_shapes(tile_m, tile_n)
#     activated = hidden1 * pypto.sigmoid(hidden1) * hidden2

#     # pypto.set_cube_tile_shapes([tile_m, tile_m], [tile_n, tile_n], [tile_k, tile_k], True, False)

#     # pypto.set_matrix_size([tile_m, tile_n, tile_k])
#     res = pypto.matmul(activated, w2, pypto.DT_FP32)
#     pypto.assemble(res, [0, 0], out)