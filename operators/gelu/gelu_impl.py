"""
GELU 算子 PyPTO 实现

直接使用 PyPTO 内置 gelu() API。
"""
import pypto
import torch


@pypto.frontend.jit
def gelu_kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
):
    """
    GELU JIT kernel - 使用内置 API
    """
    # 设置 tile shape
    pypto.set_vec_tile_shapes(1, 128, 1, 64)

    # 使用内置 gelu
    result = pypto.gelu(input_tensor)

    # 写回输出
    output_tensor[:] = result


def gelu_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    GELU wrapper

    Args:
        x: 输入 torch.Tensor

    Returns:
        GELU 激活后的 torch.Tensor
    """
    output = torch.empty_like(x)
    gelu_kernel(x, output)
    return output
