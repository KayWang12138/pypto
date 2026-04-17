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
    """
    # TODO: 替换为实际 golden 逻辑
    # 示例（SiLU）:  return x * torch.sigmoid(x)
    # 示例（LayerNorm）:
    #   mean = x.mean(dim=-1, keepdim=True)
    #   var = x.var(dim=-1, keepdim=True, unbiased=False)
    #   normalized = (x - mean) / torch.sqrt(var + eps)
    #   return normalized * gamma + beta
    return x


# ==========================================
# 验证
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    import numpy as np
    from numpy.testing import assert_allclose

    print("=" * 60)
    print("{op}_golden 验证报告")
    print("=" * 60)

    passed = 0
    failed = 0

    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    try:
        x = torch.randn({typical_shape}, dtype=torch.{typical_dtype})
        result = {op}_golden(x)
        assert result.shape == x.shape, f"shape 不匹配: {result.shape} != {x.shape}"
        assert result.dtype == x.dtype, f"dtype 不匹配: {result.dtype} != {x.dtype}"
        assert torch.isfinite(result).all(), "输出包含 NaN 或 Inf"
        print(f"  ✅ 典型 case 通过: shape={x.shape}, dtype={x.dtype}")
        passed += 1
    except Exception as e:
        print(f"  ❌ 典型 case 失败: {e}")
        failed += 1

    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    for shape in {(2, 8), (4, 64), (1, 1024)}:
        try:
            x = torch.randn(shape, dtype=torch.{typical_dtype})
            result = {op}_golden(x)
            assert result.shape == x.shape, f"shape 不匹配: {result.shape} != {x.shape}"
            assert torch.isfinite(result).all(), "输出包含 NaN 或 Inf"
            print(f"  ✅ shape={shape} 通过")
            passed += 1
        except Exception as e:
            print(f"  ❌ shape={shape} 失败: {e}")
            failed += 1

    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    try:
        x = torch.randn(4, 16, dtype=torch.{typical_dtype})
        result = {op}_golden(x)
        assert torch.isfinite(result).all(), "输出包含 NaN 或 Inf"
        print(f"  ✅ 值域检查通过")
        passed += 1
    except Exception as e:
        print(f"  ❌ 值域检查失败: {e}")
        failed += 1

    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    for name, tensor_fn in [
        ("large_values", lambda: torch.randn(4, 16) * 1e6),
        ("small_values", lambda: torch.randn(4, 16) * 1e-6),
        ("zeros", lambda: torch.zeros(4, 16)),
    ]:
        try:
            x = tensor_fn().to(torch.{typical_dtype})
            result = {op}_golden(x)
            assert torch.isfinite(result).all(), f"{name}: 输出包含 NaN 或 Inf"
            print(f"  ✅ {name} 通过")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            failed += 1

    # -- 5. API 对比（如适用）--
    print("\n[API 对比]")
    try:
        x = torch.randn(4, 16, dtype=torch.{typical_dtype})
        result = {op}_golden(x)
        ref = torch.{torch_equiv_api}(x)
        assert_allclose(result.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)
        print(f"  ✅ 与 torch.{torch_equiv_api} 对比通过")
        passed += 1
    except AttributeError:
        print("  ⏭️ 无直接等价 torch API，跳过")
    except Exception as e:
        print(f"  ❌ API 对比失败: {e}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"验证完成: {passed} passed, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
