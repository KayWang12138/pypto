# https://github.com/triton-lang/triton/blob/1c15d61c2f8c731356d06a53020ccc22f22ec8d9/python/tutorials/02-fused-softmax.py
import os
import torch
import triton
import triton.language as tl
import triton_pypto


@triton_pypto.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask)
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor, block_size: int):
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']), )
    softmax_kernel[grid](x, output, x.shape[1], x.shape[1], x.shape[0], x.shape[1], BLOCK_SIZE=block_size, num_stages=1)
    return output


def test_softmax():
    M = int(os.environ.get("SIZE_M", "13"))
    N = int(os.environ.get("SIZE_N", "1024"))
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32)
    output = softmax(x, N)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(output, ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    test_softmax()
