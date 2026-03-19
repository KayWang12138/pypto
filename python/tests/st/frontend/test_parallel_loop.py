import logging
import pypto
import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format='', force=True)
torch.manual_seed(0)


def gen_add_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a, dtype=torch.float32)
    
    # 三层循环
    for i in range(a.shape[0]):    # dim 0: 2
        for j in range(a.shape[1]):# dim 1: 4
            for k in range(a.shape[2]): # dim 2:4
                out[i, j, k] = a[i, j, k] + b[i, j, k]
    return out



def prep_env():

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    torch_npu.npu.config.allow_internal_format = True

def parallel_add_compute_single_parallel(left: pypto.Tensor, right: pypto.Tensor, res: pypto.Tensor):
    n0 = left.shape[0]
    n1 = left.shape[1]
    n2 = left.shape[2]
    for n0_idx in pypto.loop(0, 2, 1, name="N0_LOOP", idx_name="n0_loop"):
        for n1_idx in pypto.loop(0, 2, 1, name="N1_LOOP", idx_name="n1_loop", parallel_for=True):
            for n2_idx in pypto.loop(0, 2, 1, name="N2_LOOP", idx_name="n2_loop"):
                pypto.set_vec_tile_shapes(1, 1, 2)
                left_view = pypto.view(left, [1, 2, 2], [n0_idx, n1_idx * 2, n2_idx * 2])
                right_view = pypto.view(right, [1, 2, 2], [n0_idx, n1_idx * 2, n2_idx * 2])
                output = left_view + right_view
                pypto.assemble(output, [n0_idx, n1_idx * 2, n2_idx * 2], res)


def parallel_add_compute_double_parallel(left: pypto.Tensor, right: pypto.Tensor, res: pypto.Tensor):
    n0 = left.shape[0]
    n1 = left.shape[1]
    n2 = left.shape[2]
    for n0_idx in pypto.loop(0, 2, 1, name="N0_LOOP", idx_name="n0_loop", parallel_for=True):
        for n1_idx in pypto.loop(0, 2, 1, name="N1_LOOP", idx_name="n1_loop"):
            for n2_idx in pypto.loop(0, 2, 1, name="N2_LOOP", idx_name="n2_loop"):
                pypto.set_vec_tile_shapes(1, 1, 2)
                left_view = pypto.view(left, [1, 2, 2], [n0_idx, n1_idx * 2, n2_idx * 2])
                right_view = pypto.view(right, [1, 2, 2], [n0_idx, n1_idx * 2, n2_idx * 2])
                output = left_view + right_view
                pypto.assemble(output, [n0_idx, n1_idx * 2, n2_idx * 2], res)
    
    for second_n0_idx in pypto.loop(0, 2, 1, name="N0_LOOP_SECOND", idx_name="n0_loop_second", parallel_for=True):
        for second_n1_idx in pypto.loop(0, 2, 1, name="N1_LOOP_SECOND", idx_name="n1_loop_second"):
            for second_n2_idx in pypto.loop(0, 2, 1, name="N2_LOOP_SECOND", idx_name="n2_loop_second"):
                pypto.set_vec_tile_shapes(1, 1, 2)
                second_left_view = pypto.view(left, [1, 2, 2], [second_n0_idx, second_n1_idx * 2, second_n2_idx * 2])
                second_right_view = pypto.view(right, [1, 2, 2], [second_n0_idx, second_n1_idx * 2, second_n2_idx * 2])
                second_output = second_left_view + second_right_view
                pypto.assemble(second_output, [second_n0_idx, second_n1_idx * 2, second_n2_idx * 2], res)


def test_parallel_add_double_parallel():
    prep_env()
    a = torch.rand((2, 4, 4), dtype=torch.float32) * 2 - 1  # [-1, 1]
    b = torch.rand((2, 4, 4), dtype=torch.float32) * 2 - 1  # [-1, 1]

    res_golden = gen_add_golden(a, b)

    res = x = torch.empty((2, 4, 4), dtype=torch.float32).uniform_(-1, 1).npu()

    @pypto.frontend.jit
    def parallel_add(
        left: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
        right: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
        res: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ):
        """
        JIT-compiled MLA Prolog quantization for decode phase.

        Optimized version for decode phase with specific pass configurations.
        Processes single or few tokens at a time for low latency.

        Args:
            token_x: Input token tensor, shape (t, h), dtype BF16
            w_dq: Down-projection weight for query, NZ format
            w_uq_qr: Up-projection weight for query and RoPE, NZ format
            dequant_scale: Dequantization scale for w_uq_qr, FP32
            w_uk: Up-projection weight for key, BF16
            w_dkv_kr: Down-projection weight for key-value and RoPE, NZ format
            gamma_cq: RMSNorm scale for query, BF16
            gamma_ckv: RMSNorm scale for key-value, BF16
            cos: Cosine values for RoPE, BF16
            sin: Sine values for RoPE, BF16
            cache_index: Cache index for scatter update, INT64
            kv_cache: Key-value cache input/output, INT8
            kr_cache: Key RoPE cache input/output, BF16
            k_scale_cache: Key scale cache input/output, FP16
            q_norm_out: Output normalized query, INT8
            q_norm_scale_out: Output query normalization scale, FP32
            query_nope_out: Output query without RoPE, BF16
            query_rope_out: Output query with RoPE, BF16
            kv_cache_out: Output key-value cache
            kr_cache_out: Output key RoPE cache
            k_scale_cache_out: Output key scale cache
            epsilon_cq: RMSNorm epsilon for query
            epsilon_ckv: RMSNorm epsilon for key-value
            cache_mode: Cache mode ("PA_BSND" or "PA_NZ")
            tile_config: MlaTileConfig object
            rope_cfg: RopeTileShapeConfig object
        Note:
            Configured for decode phase with optimized memory and latency settings.
        """
        parallel_add_compute_double_parallel(left, right, res)
    a_npu = a.npu()
    b_npu = b.npu()
    parallel_add(a_npu, b_npu, res)
    assert torch.allclose(res.cpu(), res_golden, atol=0.000025, rtol=0.005)


def test_parallel_add_single_parallel():
    prep_env()
    a = torch.rand((2, 4, 4), dtype=torch.float32) * 2 - 1  # [-1, 1]
    b = torch.rand((2, 4, 4), dtype=torch.float32) * 2 - 1  # [-1, 1]

    res_golden = gen_add_golden(a, b)

    res = x = torch.empty((2, 4, 4), dtype=torch.float32).uniform_(-1, 1).npu()

    @pypto.frontend.jit
    def parallel_add(
        left: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
        right: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
        res: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    ):
        """
        JIT-compiled MLA Prolog quantization for decode phase.

        Optimized version for decode phase with specific pass configurations.
        Processes single or few tokens at a time for low latency.

        Args:
            token_x: Input token tensor, shape (t, h), dtype BF16
            w_dq: Down-projection weight for query, NZ format
            w_uq_qr: Up-projection weight for query and RoPE, NZ format
            dequant_scale: Dequantization scale for w_uq_qr, FP32
            w_uk: Up-projection weight for key, BF16
            w_dkv_kr: Down-projection weight for key-value and RoPE, NZ format
            gamma_cq: RMSNorm scale for query, BF16
            gamma_ckv: RMSNorm scale for key-value, BF16
            cos: Cosine values for RoPE, BF16
            sin: Sine values for RoPE, BF16
            cache_index: Cache index for scatter update, INT64
            kv_cache: Key-value cache input/output, INT8
            kr_cache: Key RoPE cache input/output, BF16
            k_scale_cache: Key scale cache input/output, FP16
            q_norm_out: Output normalized query, INT8
            q_norm_scale_out: Output query normalization scale, FP32
            query_nope_out: Output query without RoPE, BF16
            query_rope_out: Output query with RoPE, BF16
            kv_cache_out: Output key-value cache
            kr_cache_out: Output key RoPE cache
            k_scale_cache_out: Output key scale cache
            epsilon_cq: RMSNorm epsilon for query
            epsilon_ckv: RMSNorm epsilon for key-value
            cache_mode: Cache mode ("PA_BSND" or "PA_NZ")
            tile_config: MlaTileConfig object
            rope_cfg: RopeTileShapeConfig object
        Note:
            Configured for decode phase with optimized memory and latency settings.
        """
        parallel_add_compute_single_parallel(left, right, res)
    a_npu = a.npu()
    b_npu = b.npu()
    parallel_add(a_npu, b_npu, res)
    assert torch.allclose(res.cpu(), res_golden, atol=0.000025, rtol=0.005)


if __name__ == "__main__":
    test_parallel_add_single_parallel()
    test_parallel_add_double_parallel()