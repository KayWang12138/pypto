"""missing_files_op_impl.py — 只有 impl，缺少 golden/test/README"""
import pypto
import torch


@pypto.frontend.jit
def missing_files_op_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(8, 8)
    out[:] = pypto.sin(x)


def missing_files_op_wrapper(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    missing_files_op_kernel(x, output)
    return output
