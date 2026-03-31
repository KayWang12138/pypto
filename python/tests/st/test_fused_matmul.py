import torch
import torch_npu
import pypto
import time
import os
import numpy as np
from typing import List


def create_mm_kernel(M, K, N):
    @pypto.frontend.jit(
        # debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a: pypto.Tensor([M, K], pypto.DT_FP16),
        b: pypto.Tensor([K, N], pypto.DT_FP16),
        out: pypto.Tensor([M, N], pypto.DT_FP16),
    ):
        pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], 
            # enable_multi_data_load=True, 
            enable_split_k=False
            )
        tensorC = pypto.matmul(a, b, out_dtype=pypto.DT_FP16)
        out[:, :] = tensorC
    return matmul_pto

def create_mm_kernel_with_l2_split(M, K, N, m_view, n_view, dynamic=True):
    if dynamic:
        M = pypto.frontend.dynamic("M")
        N = pypto.frontend.dynamic("N")
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
    )
    def matmul_pto(
        a: pypto.Tensor([M, K], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        b: pypto.Tensor([K, N], pypto.DT_FP16, format=pypto.TileOpFormat.TILEOP_ND),
        out: pypto.Tensor([M, N], pypto.DT_FP16),
    ):
        # pypto.set_cube_tile_shapes([128, 128], [64, 512, 256], [256, 256], 
        pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], 

            # enable_multi_data_load=True, 
            enable_split_k=False
            )
        m_loop = (M + m_view - 1) // m_view
        n_loop = (N + n_view - 1) // n_view
        # outTensor = pypto.Tensor([M, N], pypto.DT_FP16)
        for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_LO_mIdx", idx_name="m_idx"):
            for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_LO_nIdx", idx_name="n_idx"):
                a_view = a[m_idx * m_view : m_idx * m_view + m_view, :]
                b_view = b[:, n_idx * n_view : n_idx * n_view + n_view]
                out_view = pypto.matmul(a_view, b_view, out_dtype=pypto.DT_FP16)
                out[m_idx * m_view : m_idx * m_view + m_view, n_idx * n_view : n_idx * n_view + n_view] = out_view
    return matmul_pto

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _run_case_multi_round(
    case_name: str,
    runner,
    M: int,
    K: int,
    N: int,
    warmup_rounds: int,
    num_rounds: int,
    device: str,
    atol: float,
    rtol: float,
):
    total_rounds = warmup_rounds + num_rounds
    pypto_times, pypto_mems = [], []
    torch_times, torch_mems = [], []

    print("\n" + "-" * 90)
    print(f"[CASE] {case_name} | warmup={warmup_rounds}, rounds={num_rounds}, device={device}")
    print(f"       M={M}, K={K}, N={N}")
    print("-" * 90)

    for r in range(total_rounds):
        torch.manual_seed(2026 + r)
        a = torch.randn([M, K], dtype=torch.float16, device=device)
        b = torch.randn([K, N], dtype=torch.float16, device=device)
        out = torch.empty([M, N], dtype=torch.float16, device=device)

        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        # torch.npu.synchronize()
        runner(a, b, out)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        pypto_t = t1 - t0
        pypto_m = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        # torch.npu.synchronize()
        golden = torch.matmul(a, b)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        torch_t = t1 - t0
        torch_m = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        assert torch.allclose(
            out.detach().cpu().to(torch.float32),
            golden.detach().cpu().to(torch.float32),
            atol=atol,
            rtol=rtol,
        ), f"[{case_name}] round={r + 1} output mismatch"

        if r >= warmup_rounds:
            pypto_times.append(pypto_t)
            pypto_mems.append(pypto_m)
            torch_times.append(torch_t)
            torch_mems.append(torch_m)
            print(
                f"Round {r + 1:02d} | "
                f"PyPTO: {pypto_t * 1000:.2f} ms / {pypto_m:.1f} MB | "
                f"Torch: {torch_t * 1000:.2f} ms / {torch_m:.1f} MB"
            )

        del a, b, out, golden
        torch.npu.empty_cache()

    avg_pypto_t = _mean(pypto_times)
    avg_pypto_m = _mean(pypto_mems)
    avg_torch_t = _mean(torch_times)
    avg_torch_m = _mean(torch_mems)

    print(f"[CASE-SUMMARY] {case_name}")
    print(f"  Torch: {avg_torch_t * 1000:.2f} ms | {avg_torch_m:.1f} MB")
    print(f"  PyPTO: {avg_pypto_t * 1000:.2f} ms | {avg_pypto_m:.1f} MB")
    if avg_pypto_t > 0:
        print(f"  Speedup(Torch/PyPTO): {avg_torch_t / avg_pypto_t:.2f}x")

    return {
        "case_name": case_name,
        "avg_pypto_t": avg_pypto_t,
        "avg_pypto_m": avg_pypto_m,
        "avg_torch_t": avg_torch_t,
        "avg_torch_m": avg_torch_m,
    }


def test_native_mm_multi_round():
    M = int(os.environ.get("MM_M", 6144))
    K = int(os.environ.get("MM_K", 6144))
    N = int(os.environ.get("MM_N", 6144))
    warmup_rounds = int(os.environ.get("MM_WARMUP_ROUNDS", 3))
    num_rounds = int(os.environ.get("MM_NUM_ROUNDS", 10))
    atol = float(os.environ.get("MM_ATOL", 1e-3))
    rtol = float(os.environ.get("MM_RTOL", 1e-3))
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)

    runner = create_mm_kernel(M, K, N)
    return _run_case_multi_round(
        case_name="native_mm",
        runner=runner,
        M=M,
        K=K,
        N=N,
        warmup_rounds=warmup_rounds,
        num_rounds=num_rounds,
        device=device,
        atol=atol,
        rtol=rtol,
    )


def test_mm_with_l2_split_multi_round(dynamic=True):
    M = int(os.environ.get("MM_M", 6144))
    K = int(os.environ.get("MM_K", 6144))
    N = int(os.environ.get("MM_N", 6144))
    m_view = int(os.environ.get("MM_VIEW_M", 128 * 6))
    n_view = int(os.environ.get("MM_VIEW_N", 256 * 4))
    warmup_rounds = int(os.environ.get("MM_WARMUP_ROUNDS", 3))
    num_rounds = int(os.environ.get("MM_NUM_ROUNDS", 10))
    atol = float(os.environ.get("MM_ATOL", 1e-3))
    rtol = float(os.environ.get("MM_RTOL", 1e-3))
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)

    runner = create_mm_kernel_with_l2_split(M, K, N, m_view, n_view, dynamic)
    return _run_case_multi_round(
        case_name="mm_with_l2_split",
        runner=runner,
        M=M,
        K=K,
        N=N,
        warmup_rounds=warmup_rounds,
        num_rounds=num_rounds,
        device=device,
        atol=atol,
        rtol=rtol,
    )

if __name__ == "__main__":
    print("\n🚀 FusedMatmul multi-round perf test")
    all_stats = []
    # all_stats.append(test_native_mm_multi_round())
    # all_stats.append(test_mm_with_l2_split_multi_round(dynamic=True))
    all_stats.append(test_mm_with_l2_split_multi_round(dynamic=False))


    avg_pypto_t = _mean([x["avg_pypto_t"] for x in all_stats])
    avg_torch_t = _mean([x["avg_torch_t"] for x in all_stats])
    avg_pypto_m = _mean([x["avg_pypto_m"] for x in all_stats])
    avg_torch_m = _mean([x["avg_torch_m"] for x in all_stats])

    print("\n" + "=" * 90)
    print("📈 [GLOBAL SUMMARY] FusedMatmul PyPTO vs Torch")
    print(f"Torch: {avg_torch_t * 1000:.2f} ms | {avg_torch_m:.1f} MB")
    print(f"PyPTO: {avg_pypto_t * 1000:.2f} ms | {avg_pypto_m:.1f} MB")
    if avg_pypto_t > 0:
        print(f"Speedup(Torch/PyPTO): {avg_torch_t / avg_pypto_t:.2f}x")
    print("=" * 90)