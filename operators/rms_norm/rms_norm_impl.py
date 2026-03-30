#!/usr/bin/env python3
# coding: utf-8

"""PyPTO rms_norm kernel implementation.

实现说明:
  - 使用 PyPTO 内置 pypto.rms_norm API（design.md 推荐的方案 A）。
  - 导出函数 rms_norm_wrapper() 供 test_rms_norm.py 调用。
  - kernel 使用 @pypto.frontend.jit 装饰。
"""

import pypto
import torch


# ----------------------------------------------------------------
# JIT Kernel
# ----------------------------------------------------------------

@pypto.frontend.jit
def rms_norm_kernel(
    x: pypto.Tensor(),
    gamma: pypto.Tensor(),
    output: pypto.Tensor(),
    eps: float,
):
    """PyPTO jit kernel.

    使用 PyPTO 内置 rms_norm API。
    - Tensor 描述符使用 pypto.Tensor()（shape 自动推断）。
    - 必须配置 tiling: pypto.set_vec_tile_shapes(...)。
    - 输出写回使用 pypto.assemble(result, offset, output)。
    """
    # 配置 tiling
    pypto.set_vec_tile_shapes(64, 128)

    # 使用 PyPTO 内置 rms_norm API
    result = pypto.rms_norm(x, gamma, eps)

    # 输出写回 - 对于任意维度，使用 [0, 0] offset
    pypto.assemble(result, [0, 0], output)


# ----------------------------------------------------------------
# Wrapper 函数（导出接口）
# ----------------------------------------------------------------

def rms_norm_wrapper(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """算子 wrapper，供 test_rms_norm.py 调用。

    负责:
    1. 构造输出 torch.Tensor
    2. 调用 JIT kernel
    3. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor, shape [b, s, d] 或 [b, d]。
        weight: 缩放参数 gamma, shape [d]。
        eps: 防止除零的小常数，默认 1e-6。

    Returns:
        输出 torch.Tensor, shape 与 x 相同。
    """
    # 构造输出 tensor
    output = torch.empty_like(x)

    # 调用 kernel
    rms_norm_kernel(x, weight, output, eps)

    return output
