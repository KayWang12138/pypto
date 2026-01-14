# https://github.com/triton-lang/triton/blob/1c15d61c2f8c731356d06a53020ccc22f22ec8d9/python/tutorials/03-matrix-multiplication.py

import torch
import triton.language as tl
import triton_pypto


@triton_pypto.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    if matmul_kernel.options.dynamic:  # pid is symbolic scalar
        group_size_m = (num_pid_m - first_pid_m).min(GROUP_SIZE_M)
    else:
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    if not matmul_kernel.options.dynamic:  # __mod__ will not work properly if affine layout offset is symbolic scalar
        offs_am %= M
        offs_bn %= N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def test_matmul(dynamic: bool, unroll_factor: int):
    matmul_kernel.set_options(dynamic=dynamic, unroll_factor=unroll_factor)
    torch.manual_seed(0)
    M, N, K = 16, 16, 16
    BLOCK_SIZE = 8

    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    c = torch.zeros((M, N), dtype=torch.float32)
    ref_c = torch.matmul(a, b)

    grid = lambda META: (tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE),)
    matmul_kernel[grid](
        a, b, c,
        M=M, N=N, K=K,
        stride_am=K, stride_ak=1,
        stride_bk=N, stride_bn=1,
        stride_cm=N, stride_cn=1,
        BLOCK_SIZE_M=BLOCK_SIZE,
        BLOCK_SIZE_N=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE,
        GROUP_SIZE_M=1
    )

    atol, rtol = 1e-5, 1e-5
    if not torch.allclose(ref_c, c, atol=atol, rtol=rtol):
        max_diff = (ref_c - c).abs().max().item()
        print("Inputs:")
        print("a =", a)
        print("b =", b)
        print("\nReference output:", ref_c)
        print("\nTriton output:   ", c)
        print(f"\nMax absolute difference: {max_diff:.2e}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        assert False, "test_matmul: FAILED"
    print("test_matmul: PASSED")


if __name__ == "__main__":
    test_matmul(True, 1)
