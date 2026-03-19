import pypto
import time
import torch
import torch_npu
import torch.nn.functional as F
import pytest
from numpy.testing import assert_allclose
from dataclasses import dataclass

def trans_nd_to_fractal_nz(data: torch.Tensor, keep_m_dim=False):
    def _gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def _ceil_div(a, b):
        return (a + b - 1) // b

    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    m0 = 16
    n0 = 32 // data.dtype.itemsize
    if data.dtype == torch.int32:
        n0 = 16
    m1, n1 = _ceil_div(m_ori, m0), _ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    if not keep_m_dim:
        pad_list = [0, padding_n, 0, padding_m] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
        data = data.reshape(batch_ori + (m1, m0, n1, n0)).permute(*array_trans).contiguous()
    else:
        pad_list = [0, padding_n, 0, 0] + [0, 0] * batch_num
        data = F.pad(data, pad_list, "constant")
        array_trans = _gen_axes_for_transpose(len(data.shape) - 2, [1, 0, 2])
        data = data.reshape(batch_ori + (m_ori, n1, n0)).permute(*array_trans).contiguous()
    return data

@dataclass
class QuantMatmulReduceSumConfig:
    ori: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    x2_format_nz: bool = True
    vec_tile_shapes: list = None
    description: str = ""
    
    def __post_init__(self):
        if self.vec_tile_shapes is None:
            self.vec_tile_shapes = [1, 256, 256]

def quant_matmul_reduce_sum_pypto(config: QuantMatmulReduceSumConfig):
    batch, m, k, n = config.ori
    x1_shape = [batch, m, k]
    x2_shape = [batch, k, n]
    x1_scale_shape = [batch, m]
    x2_scale_shape = [n]
    out_shape = [m, n]
    x2_format = pypto.TileOpFormat.TILEOP_NZ if config.x2_format_nz else pypto.TileOpFormat.TILEOP_ND
    
    @pypto.frontend.jit(
        debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1}
    )
    def quant_matmul_reduce_sum_impl(
        x1: pypto.Tensor(x1_shape, pypto.DT_INT8),
        x2: pypto.Tensor(x2_shape, pypto.DT_INT8, format=x2_format),
        x1_scale: pypto.Tensor(x1_scale_shape, pypto.DT_FP32),
        x2_scale: pypto.Tensor(x2_scale_shape, pypto.DT_BF16)
    ) -> pypto.Tensor(out_shape, pypto.DT_BF16):
        pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
        pypto.set_vec_tile_shapes(*config.vec_tile_shapes)

        if config.x2_format_nz:
            pypto.set_matrix_size([m, k, n])

        # matmul 计算并立即转换为 FP32
        matmul_result = pypto.matmul(x1, x2, pypto.DT_INT32)
        matmul_result_fp32 = pypto.cast(matmul_result, pypto.DT_FP32)

        # 将 x2_scale 转换为 FP32 并广播
        x2_scale_fp32 = pypto.cast(x2_scale, pypto.DT_FP32)
        x2_scale_2d = pypto.unsqueeze(x2_scale_fp32, 0)
        x2_scale_broadcast = pypto.expand_clone(x2_scale_2d, [m, n])

        # 广播 x1_scale
        x1_scale_2d = pypto.unsqueeze(x1_scale, 2)
        x1_scale_broadcast = pypto.expand_clone(x1_scale_2d, [batch, m, n])

        # 先计算 scale_mul
        scale_mul = pypto.mul(x1_scale_broadcast, x2_scale_broadcast)

        # 融合乘法和累加
        out = pypto.mul(matmul_result_fp32, scale_mul)
        out_fp32 = pypto.sum(out, 0)
        out_bf16 = pypto.cast(out_fp32, pypto.DT_BF16)

        return out_bf16
    return quant_matmul_reduce_sum_impl

def golden_quant_matmul_reduce_sum(x1, x2, x1_scale, x2_scale):
    batch, m, k = x1.shape
    _, _, n = x2.shape

    result = torch.zeros((m, n), dtype=torch.float32)

    for i in range(batch):
        matmul_result = torch.matmul(x1[i].float(), x2[i].float())
        scale_broadcast = x1_scale[i].unsqueeze(1).float() * x2_scale.unsqueeze(0).float()
        result += matmul_result * scale_broadcast

    return result.to(torch.bfloat16)

def run_quant_matmul_reduce_sum_case(config: QuantMatmulReduceSumConfig):
    b, m, k, n = config.ori

    torch.manual_seed(42)
    x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8).npu()
    x2_nd = torch.randint(-10, 10, (b, k, n), dtype=torch.int8).npu()
    x2_nz = trans_nd_to_fractal_nz(x2_nd).npu()
    x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0.5, 1.5).npu()
    x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0.5, 1.5).npu()

    # 根据x2_format_nz选择使用x2_nd还是x2
    if config.x2_format_nz:
        x2_input = x2_nz
    else:
        x2_input = x2_nd

    b, m, k, n = config.ori

    pypto_out = quant_matmul_reduce_sum_pypto(config)(x1, x2_input, x1_scale, x2_scale)
    golden_out = golden_quant_matmul_reduce_sum(x1.cpu(), x2_nd.cpu(), x1_scale.cpu(), x2_scale.cpu())

    pypto_out_cpu = pypto_out.cpu().float()
    golden_out_cpu = golden_out.cpu().float()

    assert_allclose(pypto_out_cpu, golden_out_cpu, rtol=0.001, atol=0.001)
    print(f"Test passed for {config.description}")

if __name__ == "__main__":
    # 15.14 23.58 0.642
    # run_quant_matmul_reduce_sum_case(QuantMatmulReduceSumConfig([5, 16, 32, 32], [128, 128], [256, 256], [256, 256], pypto.DT_INT8, pypto.DT_BF16, True, [256, 256, 256], "testcase1"))

    # 14.54 16.44 0.88
    # run_quant_matmul_reduce_sum_case(QuantMatmulReduceSumConfig([5, 16, 128, 128], [128, 128], [256, 256], [256, 256], pypto.DT_INT8, pypto.DT_BF16, True, [1024, 1024, 1024], "testcase2"))

    # 21.5 29.14 0.737
    # run_quant_matmul_reduce_sum_case(QuantMatmulReduceSumConfig([7, 32, 1024, 128], [128, 128], [128, 128], [128, 128], pypto.DT_INT8, pypto.DT_BF16, True, [16, 16, 16], "testcase3"))

    # 12.8 21.02 0.608
    # run_quant_matmul_reduce_sum_case(QuantMatmulReduceSumConfig([4, 64, 256, 64], [128, 128], [256, 256], [128, 128], pypto.DT_INT8, pypto.DT_BF16, True, [4096, 4096, 4096], "testcase4"))

    # 17.62 22 0.8
    run_quant_matmul_reduce_sum_case(QuantMatmulReduceSumConfig([2, 128, 128, 128], [128, 128], [128, 128], [128, 128], pypto.DT_INT8, pypto.DT_BF16, True, [64, 64, 64], "testcase5"))

    # 0.733
