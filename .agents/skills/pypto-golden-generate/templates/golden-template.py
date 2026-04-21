#!/usr/bin/env python3
# coding: utf-8

"""PyPTO {op} golden reference implementation.

模板说明：
  - 所有 {op} 占位符需替换为实际算子名称。
  - golden 必须是纯 PyTorch 实现，禁止引入 pypto。
  - 导出函数 {op}_golden() 供 test_{op}.py 调用。
  - 参考 examples/ 中的 golden 函数风格（activation、layernorm 等）。
"""

import torch

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def {op}_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现。

    根据算子规格中的数学公式实现。
    仅使用 torch 标准操作，不依赖 pypto。

    Args:
        x: 输入 tensor。
           根据实际算子需求调整参数列表（可多输入、可带 gamma/beta/eps 等参数）。

    Returns:
        计算结果 tensor。
        根据实际算子需求调整返回值（可多输出、可返回 tuple）。

    注意（PyPTO 友好的 golden）：
      - 全量计算实现：直接对整个 tensor 计算（最常见）。
      - 或者：可选择用 for loop 将输入分块成小 tiles，对每个 tile 单独计算，
        最后用 torch.cat / concatenate 组合结果。这种分块实现更接近 PyPTO kernel
        的实际执行方式（PyPTO kernel 也是 tile-by-tile 处理），可以更好地
        验证 PyPTO 边界处理和累积逻辑。

      选择分块实现时的示例模式（如果算子支持分块累加）：
        # 示例：按 batch 维度分块处理（如 attention、matmul 等）
        output = []
        for b in range(x.shape[0]):
            tile = x[b:b+1, ...]  # 切出一个小 tile
            result_tile = _compute_single_tile(tile, **params)  # 计算
            output.append(result_tile)
        return torch.cat(output, dim=0)  # 组合结果

      或者按多维分块（如 seq_len 和 head 维度）：
        output = torch.zeros_like(x)
        for i in range(0, x.shape[1], tile_h):
            for j in range(0, x.shape[2], tile_w):
                h_end = min(i + tile_h, x.shape[1])
                w_end = min(j + tile_w, x.shape[2])
                tile = x[:, i:h_end, j:w_end, ...]
                output[:, i:h_end, j:w_end, ...] = _compute_single_tile(tile)
        return output
    """
    # TODO: 替换为实际 golden 逻辑
    # 示例（SiLU）:  return x * torch.sigmoid(x)
    # 示例（LayerNorm）:
    #   mean = x.mean(dim=-1, keepdim=True)
    #   var = x.var(dim=-1, keepdim=True, unbiased=False)
    #   normalized = (x - mean) / torch.sqrt(var + eps)
    #   return normalized * gamma + beta
    #
    # 示例（分块实现 - 如果算子支持）:
    #   output = []
    #   for b in range(x.shape[0]):
    #       tile = x[b:b+1, ...]
    #       result_tile = _compute_tile(tile, params)
    #       output.append(result_tile)
    #   return torch.cat(output, dim=0)
    return x


# ==========================================
# 验证
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    print("=" * 60)
    print("{op}_golden 验证报告")
    print("=" * 60)

    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    # TODO: 按算子规格中的典型配置生成验证

    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    # TODO: 按动态轴采样范围验证

    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    # TODO: 验证输出值域

    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    # TODO: 大值、小值、零值等极端输入

    # -- 5. API 对比（如适用）--
    print("\n[API 对比]")
    # TODO: 与 PyTorch 等价 API 对比

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
