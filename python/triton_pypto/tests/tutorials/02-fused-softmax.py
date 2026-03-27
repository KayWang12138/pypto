# https://github.com/triton-lang/triton/blob/1c15d61c2f8c731356d06a53020ccc22f22ec8d9/python/tutorials/02-fused-softmax.py

import os

import torch
import triton
import triton.language as tl
import triton_pypto

SIZE_M = int(os.environ.get("SIZE_M", "13"))
SIZE_N = int(os.environ.get("SIZE_N", "1024"))
BLOCK_SIZE = SIZE_N


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
        # row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.load(input_ptrs, mask=mask)
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor):
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SIZE']), )
    softmax_kernel[grid](x, output, x.shape[1], x.shape[1], x.shape[0], x.shape[1], BLOCK_SIZE=BLOCK_SIZE, num_stages=1)
    return output


def test_softmax(dynamic: bool, unroll_factor: int):
    softmax_kernel.set_options(dynamic=dynamic, unroll_factor=unroll_factor)
    torch.manual_seed(0)
    x = torch.randn((SIZE_M, SIZE_N), dtype=torch.float32)

    ref = torch.softmax(x, dim=-1)
    out = softmax_triton(x)

    atol, rtol = 1e-5, 1e-4
    if not torch.allclose(ref, out, atol=atol, rtol=rtol):
        max_diff = (ref - out).abs().max().item()
        print("Input:")
        print("x =", x)
        print("\nReference output:", ref)
        print("\nTriton output:   ", out)
        print(f"\nMax absolute difference: {max_diff:.2e}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        assert False, "test_softmax: FAILED"
    print("test_softmax: PASSED")


if __name__ == "__main__":
    test_softmax(True, 1)
