#!/usr/bin/env python3
# coding: utf-8

"""PyPTO {op} kernel implementation (complex-kernel template).

模板说明:
  - 本文件是复杂算子（attention / recurrent / fused 类）的 {op}_impl.py 起点模板。
  - 简单算子请使用 ../impl-template.py（92 行的精简版）。
  - 所有 {op} 占位符需替换为实际算子名称。
  - 配套 golden 文件: <op>_golden.py（由 pypto-golden-generate 生成，纯 torch 实现）。
  - 配套 test 文件: test_<op>.py（由 ../test-template.py 派生）。
  - 设计文档: ../../references/kernel-layer-format.md（layers G–K 详细规范）。

文件职责（layers G–K）:
  - Layer G: prepare_buffers_for_pypto — cache / layout 桥接
  - Layer H: pypto_* 命名的 PyPTO 子内核（小而命名清晰的区域）
  - Layer I: _{op}_kernel_impl — pypto.loop 嵌套
  - Layer J: {op}_kernel_npu — @pypto.frontend.jit 入口
  - Layer K: pypto_function — 主机包装器（**禁止 for ... in range** 驱动 batch/seq/tile 计算）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pypto


# ─────────────────────────────────────────────
# 全局占位符 — 替换为你的算子标识
# ─────────────────────────────────────────────

OP_NAME: str = "{op}"

# 实现完成、可端到端运行后置 True。
KERNEL_READY: bool = False


# ═══════════════════════════════════════════════════════════════════
# Layer G — Cache / layout 桥接（可选）
# ═══════════════════════════════════════════════════════════════════
#
# 角色:
#   将 reference 的 cache（嵌套 list、per-tile tensor、Python 标量）转换为
#   JIT 入口期望的 flat / multi-buffer layout。简单情况下可直接合并到 Layer K。
#
# 注意:
#   - flatten 顺序（batch / head / seq）必须与 Layer I 中 pypto.view 一致。
#   - 只在跨 staged file 的 cache 复杂场景下需要此层。
#
# ═══════════════════════════════════════════════════════════════════


def prepare_buffers_for_pypto(
    cache: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> List[Optional[torch.Tensor]]:
    """规范化 cache entries 用于设备路径。

    Agent: 返回与 kernel 输入顺序匹配的 list/tuple；或将逻辑内联到 pypto_function。
    """
    raise NotImplementedError(
        f"[{OP_NAME}] Agent: implement prepare_buffers_for_pypto (Layer G) or inline in pypto_function."
    )


# ═══════════════════════════════════════════════════════════════════
# Layer H — PyPTO 子内核（小命名区域）
# ═══════════════════════════════════════════════════════════════════
#
# 角色:
#   每个函数完成 **一个** 概念步骤:
#     - 用 pypto.view 切片
#     - matmul / elementwise 块
#     - 可选的 pypto.set_vec_tile_shapes / pypto.set_pass_options
#
# 简单算子可以只有一个 pypto_* 函数，或者直接内联到 Layer I。
#
# 规则:
#   - 返回下一步需要的所有 tensor（避免隐式全局）。
#   - 命名应当与 reference helper 对应（grep 友好）。
#
# 关键约束:
#   - set_vec_tile_shapes 传正整数 tile 维度（参考 docs/api/config/pypto-set_vec_tile_shapes.md）
#   - pypto.view / 内部 tensor: 必要时保持 ≤ 4D
#   - 在真实代码中给 tensor 行加 shape 注释（例: # [B, S, H]）
#   - 写 PyPTO 代码前先读 .agents/skills/pypto-general-debug/references/debug-playbook.md §9 的对应小节
#     （JIT §9.1, view §9.4, matmul §9.19 等）
#
# ═══════════════════════════════════════════════════════════════════


def pypto_stage_placeholder() -> None:
    """替换为真实的 pypto_* 函数，或合并到 Layer I。"""
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement pypto_* stages (Layer H).")


# ═══════════════════════════════════════════════════════════════════
# Layer I — Kernel 实现（pypto.loop 嵌套，无 @jit）
# ═══════════════════════════════════════════════════════════════════
#
# 角色:
#   设备侧配方: pypto.loop 跨 batch / tile / dim 迭代，
#   调用 Layer H 的函数，写入 Layer J 提供的输出 buffer。
#
# 为何与 Layer J 分离:
#   - 不重新进入 JIT shell 即可复用 / 测试。
#   - 保持 J 精简（仅签名 + options）。
#
# 关键约束:
#   - **算法迭代** 必须在 pypto.loop 中（这里 Layer I 或 Layer J 之后），
#     **绝不** 在 pypto_function（Layer K）的 Python for 中。
#   - 进展式集成: 一次实现 **一个** 语义模块；后续 stage 用 golden boundary
#     tensor stub，直到当前模块匹配 golden。
#   - **禁止**: 在任何中间 boundary 与 golden 匹配前，把所有 pypto_* helper
#     一次性串入 kernel_impl。
#   - 遇到不透明错误（Errcode、FFFFF…）: 跟随 .agents/skills/pypto-general-debug/references/debug-playbook.md。
#
# ═══════════════════════════════════════════════════════════════════


def _{op}_kernel_impl(
    x_in: pypto.Tensor,
    y_out: pypto.Tensor,
) -> None:
    """完整的 PyPTO 实现（此函数不加 @jit）。

    参数必须与 {op}_kernel_npu（Layer J）一一对应。

    使用 **pypto.loop** 处理 batch/sequence/tile 迭代 — 不要在 pypto_function（Layer K）
    用 Python for ... in range(...) 来做这件事。
    """
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement _{{op}}_kernel_impl (Layer I).")


# ═══════════════════════════════════════════════════════════════════
# Layer J — JIT 入口（@pypto.frontend.jit）
# ═══════════════════════════════════════════════════════════════════
#
# 角色:
#   - 声明 tensor 类型: dynamic vs static 维度、dtypes。
#   - 设置 runtime_options（memory、stitch、run_mode 等）。
#   - 设置 debug_options（开发期）。
#   - 函数体: 仅委托给 _{op}_kernel_impl。
#
# 规则:
#   - 优先把输出 tensor 作为参数预分配。
#   - 主机侧 reshape 留在 Layer K。
#   - 最终交付 kernel 应当只有 **一个** 生产用 @pypto.frontend.jit 入口。
#
# Staged 文件:
#   - 开发期 staged 文件（custom/<op>/staged/<op>_module<k>_impl.py）每个可以包含
#     一个 @jit 用于该累积 scope；不要在所有输出通过 detailed_tensor_compare 前推进。
#
# ═══════════════════════════════════════════════════════════════════


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_inner_memory": 128 * 16,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 128 * 16,
        "device_sched_mode": 1,
        "run_mode": pypto.RunMode.NPU,
    },
    debug_options={"runtime_debug_mode": 1},
)
def {op}_kernel_npu(
    x_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    y_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
) -> None:
    """JIT 编译入口（重命名以匹配你的算子）。

    用真实 buffer 替换参数，按需扩展列表。

    Agent: 给每个 I/O 显式 pypto.Tensor([...], pypto.DT_*); 不要用 *args。
    """
    _{op}_kernel_impl(x_in, y_out)


# ═══════════════════════════════════════════════════════════════════
# Layer K — 主机包装器（pypto_function）
# ═══════════════════════════════════════════════════════════════════
#
# 角色:
#   torch 侧 **仅做 pre-JIT 编排**:
#     1. device / dtype 放置
#     2. flatten / transpose / pad 到 Layer J 期望的 layout
#     3. 分配输出（torch.empty 等）
#     4. 调用 {op}_kernel_npu(...)（每次公共 API 调用通常 **一次**，无主机 tile 循环）
#     5. reshape 输出回到用户面 API
#
# 这是集成测试和产品代码的常用 **Python 入口**。
#
# **禁止规则（强制）**:
#   - **绝不** 在此函数中使用 for ... in range(...) 来驱动 kernel 工作（batch / sequence /
#     chunk / tile）。该逻辑必须在 _{op}_kernel_impl / JIT 图（{op}_kernel_npu）中
#     用 pypto.loop 完成。
#   - 允许的 Python 循环: 只有 **附带性** 用途（例如遍历一个固定的小输出名称
#     列表，zip 预构建的 tensor）— **不允许** dynamic range(B), range(S), range(NT)
#     等用于计算编排（用 Layer I 的 pypto.loop）。
#   - 把 I/O reshape 和分配留在这里; JIT 入口（Layer J）保持精简。
#
# ═══════════════════════════════════════════════════════════════════


def pypto_function(
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """打包输入，调用 {op}_kernel_npu，解包输出。

    **不要** 在这里用 for ... in range(...) 实现设备侧迭代 — 用 _{op}_kernel_impl
    中的 pypto.loop（见模块头部）。

    Agent: 让返回结构匹配 test_{op}.py 测试和公共 API 文档。
    """
    raise NotImplementedError(f"[{OP_NAME}] Agent: implement pypto_function (Layer K).")
