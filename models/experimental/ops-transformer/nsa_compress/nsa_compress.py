#!/usr/bin/env python3
# coding: utf-8
"""
NSA Compress PyPTO Kernel Implementation

算子功能：训练场景下，使用NSA Compress算法减轻long-context的注意力计算，
         实现在KV序列维度进行压缩。

计算公式：
    output[token, n, d] = sum(input_window[:, n, d] * weight[:, n]) / compressBlockSize

其中：
    - input_window 为 [compressBlockSize, N, D] 的滑动窗口
    - 窗口间隔为 compressStride
"""

import os
import sys
import argparse
import pypto
import torch


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=14")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])


HEAD_DIM = 32
HEAD_NUM = 4
COMPRESS_BLOCK_SIZE = 32
COMPRESS_STRIDE = 16
TOTAL_TOKENS = 64
OUTPUT_TOKENS = 3
TOTAL_DIM = HEAD_NUM * HEAD_DIM


@pypto.frontend.jit(
    pass_options={"pg_upper_bound": 5000000},
    debug_options={"runtime_debug_mode": 1}
)
def nsa_compress_kernel(
    input: pypto.Tensor((TOTAL_TOKENS, TOTAL_DIM), pypto.DT_BF16),
    weight: pypto.Tensor((COMPRESS_BLOCK_SIZE, HEAD_NUM), pypto.DT_BF16),
    output: pypto.Tensor((OUTPUT_TOKENS, TOTAL_DIM), pypto.DT_BF16),
):
    """
    NSA Compress kernel
    
    计算流程：
        1. 遍历每个输出 token
        2. 获取对应的滑动窗口
        3. 对窗口内数据与权重相乘后求和
        4. 除以 compressBlockSize 得到平均值
    """
    pypto.set_vec_tile_shapes(32, 64)
    
    for token_idx in pypto.loop(0, OUTPUT_TOKENS, 1, name="LOOP_TOKEN", idx_name="token_idx"):
        result_fp32 = pypto.tensor([TOTAL_DIM], pypto.DT_FP32, "result_fp32")
        
        for row_idx in pypto.loop(0, COMPRESS_BLOCK_SIZE, 1, name="LOOP_ROW", idx_name="row_idx"):
            seq_idx = token_idx * COMPRESS_STRIDE + row_idx
            
            row_data = pypto.view(input, [1, TOTAL_DIM], 
                                 [seq_idx, 0],
                                 valid_shape=[1, TOTAL_DIM])
            row_data_1d = pypto.reshape(row_data, [TOTAL_DIM])
            row_fp32 = pypto.cast(row_data_1d, pypto.DT_FP32)
            
            weight_row = pypto.view(weight, [1, HEAD_NUM], 
                                   [row_idx, 0],
                                   valid_shape=[1, HEAD_NUM])
            weight_row_1d = pypto.reshape(weight_row, [HEAD_NUM])
            weight_t = pypto.reshape(weight_row_1d, [HEAD_NUM, 1])
            weight_exp = pypto.expand_clone(weight_t, [HEAD_NUM, HEAD_DIM])
            weight_flat = pypto.reshape(weight_exp, [TOTAL_DIM])
            weight_fp32 = pypto.cast(weight_flat, pypto.DT_FP32)
            
            weighted_row = pypto.mul(row_fp32, weight_fp32)
            
            if pypto.is_loop_begin(row_idx):
                result_fp32[:] = weighted_row
            else:
                result_fp32[:] = pypto.add(result_fp32, weighted_row)
        
        avg_result = pypto.div(result_fp32, COMPRESS_BLOCK_SIZE)
        avg_bf16 = pypto.cast(avg_result, pypto.DT_BF16)
        output[token_idx:token_idx+1, :] = pypto.reshape(avg_bf16, [1, TOTAL_DIM])


def nsa_compress_golden(
    input: torch.Tensor,
    weight: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
) -> torch.Tensor:
    """
    NSA Compress Golden 参考实现
    
    Args:
        input: [T, N, D] 输入张量 (TND layout)
        weight: [compressBlockSize, N] 压缩权重
        compress_block_size: 压缩滑窗大小
        compress_stride: 两次压缩滑窗间隔大小
    
    Returns:
        output: [T', N, D] 压缩后的结果
    """
    T, N, D = input.shape
    input_float = input.float()
    weight_float = weight.float()
    
    output_tokens = max(0, (T - compress_block_size) // compress_stride + 1)
    output = torch.zeros(output_tokens, N, D, dtype=input.dtype, device=input.device)
    output_float = torch.zeros(output_tokens, N, D, dtype=torch.float32, device=input.device)
    
    for i in range(output_tokens):
        start_idx = i * compress_stride
        window = input_float[start_idx:start_idx + compress_block_size, :, :]
        weight_expanded = weight_float.unsqueeze(-1).expand(-1, -1, D)
        compressed = (window * weight_expanded).sum(dim=0) / compress_block_size
        output_float[i] = compressed
    
    output = output_float.to(input.dtype)
    return output


def test_nsa_compress(device_id=None, run_mode: str = "npu"):
    print("=" * 60)
    print("Test: NSA Compress")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    device = f'npu:{device_id}' if device_id is not None else 'cpu'
    
    print(f"\n配置:")
    print(f"  TOTAL_TOKENS: {TOTAL_TOKENS}")
    print(f"  HEAD_NUM: {HEAD_NUM}")
    print(f"  HEAD_DIM: {HEAD_DIM}")
    print(f"  COMPRESS_BLOCK_SIZE: {COMPRESS_BLOCK_SIZE}")
    print(f"  COMPRESS_STRIDE: {COMPRESS_STRIDE}")
    print(f"  OUTPUT_TOKENS: {OUTPUT_TOKENS}")
    
    input_3d = torch.randn(TOTAL_TOKENS, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=device)
    input_tensor = input_3d.reshape(TOTAL_TOKENS, TOTAL_DIM)
    weight_tensor = torch.randn(COMPRESS_BLOCK_SIZE, HEAD_NUM, dtype=torch.bfloat16, device=device)
    output_tensor = torch.empty(OUTPUT_TOKENS, TOTAL_DIM, dtype=torch.bfloat16, device=device)
    
    print(f"\n输入 shape: {input_tensor.shape} (原始: {input_3d.shape})")
    print(f"权重 shape: {weight_tensor.shape}")
    print(f"输出 shape: {output_tensor.shape}")
    
    nsa_compress_kernel(input_tensor, weight_tensor, output_tensor)
    
    golden_output = nsa_compress_golden(
        input_3d, weight_tensor,
        COMPRESS_BLOCK_SIZE, COMPRESS_STRIDE
    )
    golden_output = golden_output.reshape(OUTPUT_TOKENS, TOTAL_DIM)
    
    if run_mode == "npu":
        diff = (output_tensor - golden_output).abs().max().item()
        print(f"\n精度对比 (与 Golden 最大差异): {diff:.6f}")
        
        atol = 0.001
        rtol = 0.01
        is_close = torch.allclose(output_tensor.float(), golden_output.float(), atol=atol, rtol=rtol)
        
        if is_close:
            print(f"\n✓ NSA Compress 测试通过! (atol={atol}, rtol={rtol})")
        else:
            print(f"\n✗ NSA Compress 精度测试未通过")


def main():
    parser = argparse.ArgumentParser(description="PyPTO NSA Compress Kernel")
    parser.add_argument('--run_mode', type=str, default='npu', choices=["npu"])
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PyPTO NSA Compress Kernel")
    print("=" * 60 + "\n")
    
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print(f"Running on NPU:{device_id}...")
    
    test_nsa_compress(device_id, args.run_mode)


if __name__ == "__main__":
    main()