# https://github.com/triton-lang/triton/blob/1c15d61c2f8c731356d06a53020ccc22f22ec8d9/python/tutorials/01-vector-add.py

import os

import torch
import triton
import triton.language as tl
import triton_pypto

SIZE = int(os.environ.get("SIZE", "1024"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "32"))


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


def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def test_add(dynamic: bool, unroll_factor: int):
    add_kernel.set_options(dynamic=dynamic, unroll_factor=unroll_factor)
    torch.manual_seed(0)
    size = SIZE
    x = torch.randn(size, dtype=torch.float32)
    y = torch.randn_like(x)
    output = add(x, y)
    ref = x + y

    atol, rtol = 1e-6, 1e-6
    if not torch.allclose(ref, output, atol=atol, rtol=rtol):
        max_diff = (ref - output).abs().max().item()
        print("Inputs:")
        print("x =", x)
        print("y =", y)
        print("\nReference output:", ref)
        print("\nTriton output:   ", output)
        print(f"\nMax absolute difference: {max_diff:.2e}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        assert False, "test_add: FAILED"
    print("test_add: PASSED")


if __name__ == "__main__":
    test_add(True, 1)
