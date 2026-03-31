#!/usr/bin/env python3
# coding: utf-8

"""PyPTO {op} kernel implementation.

模板说明：
  - 本文件是 {op}_impl.py 的固定模板，由 pypto-op-develop 在 Stage 3 生成。
  - 所有 {op} 占位符需替换为实际算子名称。
  - 导出函数 {op}_wrapper() 供 test_{op}.py 调用。
  - kernel 使用 @pypto.frontend.jit 装饰，内部使用 pypto API。
  - 参考 examples/ 中的 kernel 实现风格（activation、softmax、layer_norm 等）。
"""

import sys
import pypto
import torch


def _peek_run_mode_from_argv(default: str = "npu") -> str:
    """从命令行参数获取 run_mode，供装饰器使用。"""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value in ("npu", "sim"):
                return value
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value in ("npu", "sim"):
                return value
    return default


global_run_mode = pypto.RunMode.NPU
if _peek_run_mode_from_argv("npu") == "sim":
    global_run_mode = pypto.RunMode.SIM

# ─────────────────────────────────────────────
# 1. 核心计算函数（可选，复杂算子拆分用）
# ─────────────────────────────────────────────

def {op}_core(x: pypto.Tensor) -> pypto.Tensor:
    """核心计算逻辑，供 kernel 调用。

    根据 design.md 中的 API 映射实现。
    使用 pypto 基础 API（如 pypto.exp, pypto.sum, pypto.amax 等）。

    Args:
        x: pypto.Tensor 输入。
           根据实际算子需求调整参数列表。

    Returns:
        pypto.Tensor 计算结果。
    """
    # TODO: 替换为实际计算逻辑
    # 示例（SiLU）:  return x * pypto.sigmoid(x)
    # 示例（Softmax）:
    #   row_max = pypto.amax(x, dim=-1, keepdim=True)
    #   exp = pypto.exp(x - row_max)
    #   return exp / pypto.sum(exp, dim=-1, keepdim=True)
    return x

# ─────────────────────────────────────────────
# 2. JIT Kernel
# ─────────────────────────────────────────────

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def {op}_kernel(
    input_tensor: pypto.Tensor,
    output_tensor: pypto.Tensor,
):
    """PyPTO jit kernel。

    根据 design.md 实现。
    - Tensor 描述符使用 `pypto.Tensor`（让框架自动推断 shape 和 dtype）或
      `pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)`（显式指定）。
    - **注意**：`pypto.Tensor()` 空括号形式在类型注解中不支持。
    - 必须配置 tiling：pypto.set_vec_tile_shapes(...) 或 pypto.set_cube_tile_shapes(...)。
    - 输出写回使用 output_tensor[:] = result 或 pypto.assemble(result, offset, output_tensor)。
    """
    # TODO: 根据 design.md 配置 tiling
    # 示例: pypto.set_vec_tile_shapes(64, 128)

    # TODO: 替换为实际 kernel 逻辑
    result = {op}_core(input_tensor)
    output_tensor[:] = result

# ─────────────────────────────────────────────
# 3. Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def {op}_wrapper(x: torch.Tensor) -> torch.Tensor:
    """算子 wrapper，供 test_{op}.py 调用。

    负责：
    1. 构造输出 torch.Tensor
    2. 调用 JIT kernel
    3. 返回结果 torch.Tensor

    Args:
        x: 输入 torch.Tensor。
           根据实际算子需求调整参数列表（可多输入）。

    Returns:
        输出 torch.Tensor。
        根据实际算子需求调整返回值（可多输出）。
    """
    # TODO: 根据实际算子调整 output shape 和 dtype
    output = torch.empty_like(x)

    # 调用 kernel
    {op}_kernel(x, output)

    return output
