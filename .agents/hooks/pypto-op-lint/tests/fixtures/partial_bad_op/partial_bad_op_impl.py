"""partial_bad_op_impl.py — 有 jit 装饰但内部违规"""
import pypto
import torch


@pypto.frontend.jit
def partial_bad_op_kernel(
    input_tensor: pypto.Tensor([], pypto.DT_FP32),
    output_tensor: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    # OL02 FAIL: 对参数名直接赋值（应使用 output_tensor[:] = ...）
    result = min(input_tensor, 1.0)  # OL06 FAIL: 原生 min()
    output_tensor = result  # OL02 FAIL
    return result  # OL03 FAIL


def partial_bad_op_wrapper(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    partial_bad_op_kernel(x, output)
    return output
