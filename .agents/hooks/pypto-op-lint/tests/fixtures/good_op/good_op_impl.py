"""good_op_impl.py — 全部通过 OL01-OL08 的 impl fixture"""
import pypto
import torch


@pypto.frontend.jit
def good_op_kernel(
    input_tensor: pypto.Tensor([], pypto.DT_FP32),
    output_tensor: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    output_tensor[:] = pypto.sin(input_tensor)


def good_op_wrapper(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    good_op_kernel(x, output)
    return output
