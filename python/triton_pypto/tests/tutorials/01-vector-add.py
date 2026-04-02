# https://github.com/triton-lang/triton/blob/1c15d61c2f8c731356d06a53020ccc22f22ec8d9/python/tutorials/01-vector-add.py
import os
import torch
import triton
import triton.language as tl
import triton_pypto


@triton_pypto.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output


def test_add():
    SIZE = int(os.environ.get("SIZE", "1024"))
    BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "32"))
    torch.manual_seed(0)
    x = torch.randn(SIZE, dtype=torch.float32)
    y = torch.randn_like(x)
    output = add(x, y, BLOCK_SIZE)
    ref = x + y
    torch.testing.assert_close(output, ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_add()
