#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
embedding Golden 参考实现

公式:
    output = embedding_lookup(weight, indices)

    其中:
    - weight: [vocab_size, embed_dim] 的 embedding 表
    - indices: [batch, seq] 的索引张量
    - output[i, j, :] = weight[indices[i, j], :]

    注意: padding_idx 在 PyTorch 中的行为是 forward 时不改变输出值，只在 backward 时梯度为零。
    PyPTO 作为推理框架，forward 行为与 PyTorch 一致。

置信度: (使用 PyTorch 内置索引操作)
"""

import torch
from typing import Optional


def embedding_golden(
    indices: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: Optional[int] = None
) -> torch.Tensor:
    """
    embedding 参考实现 (PyTorch)

    查表操作，将索引张量映射到对应的嵌入向量。

    Args:
        indices: 索引张量，shape [batch, seq]，dtype int64
                 值范围 [0, vocab_size)
        weight: Embedding 查找表，shape [vocab_size, embed_dim]，dtype float32
        padding_idx: 可选，指定填充索引。在 PyTorch 中，padding_idx 主要影响 backward 时的梯度（padding 位置的梯度为零），
                     forward 时输出仍然是查表值。PyPTO 作为推理框架，行为与 PyTorch forward 一致。

    Returns:
        output: 嵌入向量输出，shape [batch, seq, embed_dim]，dtype 与 weight 相同

    公式:
        output[i, j, :] = weight[indices[i, j], :]
    """
    # 核心 embedding 查表操作
    output = weight[indices]

    # 注意: PyTorch 的 padding_idx 行为是:
    # - forward 时: 输出值仍然是查表值（不置零）
    # - backward 时: padding 位置的梯度为零
    # PyPTO 作为推理框架，forward 行为应与 PyTorch 一致
    # 如果需要在 forward 时将 padding 位置置零，可以额外传入 zero_padding 参数

    return output


# ==================== 自动生成的验证代码 ====================

def _validate():
    """自动生成的验证函数 - 运行时动态生成报告"""
    import numpy as np
    from numpy.testing import assert_allclose

    print("=" * 60)
    print("embedding_golden 验证报告")
    print("=" * 60)

    all_passed = True

    # 1. 典型 case 验证（来自 spec.md §9）
    print("\n[典型 case 验证]")

    # 性能_P0: 小模型
    try:
        indices = torch.randint(0, 32000, (1, 128), dtype=torch.int64)
        weight = torch.randn(32000, 4096, dtype=torch.float32)
        output = embedding_golden(indices, weight)
        expected_shape = (1, 128, 4096)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        assert output.dtype == torch.float32, f"Dtype mismatch: {output.dtype}"
        print("  性能_P0 (小模型): indices [1, 128], weight [32000, 4096] ... PASS")
    except Exception as e:
        print(f"  性能_P0 (小模型): FAIL - {e}")
        all_passed = False

    # 性能_P0: 中模型
    try:
        indices = torch.randint(0, 32000, (8, 512), dtype=torch.int64)
        weight = torch.randn(32000, 4096, dtype=torch.float32)
        output = embedding_golden(indices, weight)
        expected_shape = (8, 512, 4096)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        print("  性能_P0 (中模型): indices [8, 512], weight [32000, 4096] ... PASS")
    except Exception as e:
        print(f"  性能_P0 (中模型): FAIL - {e}")
        all_passed = False

    # 性能_P0: 大模型
    try:
        indices = torch.randint(0, 128000, (32, 2048), dtype=torch.int64)
        weight = torch.randn(128000, 4096, dtype=torch.float32)
        output = embedding_golden(indices, weight)
        expected_shape = (32, 2048, 4096)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        print("  性能_P0 (大模型): indices [32, 2048], weight [128000, 4096] ... PASS")
    except Exception as e:
        print(f"  性能_P0 (大模型): FAIL - {e}")
        all_passed = False

    # 功能_P1: padding_idx 功能验证
    try:
        indices = torch.randint(0, 10000, (4, 256), dtype=torch.int64)
        indices[0, 0] = 0  # 设置一些 padding 位置
        indices[1, 10] = 0
        weight = torch.randn(10000, 256, dtype=torch.float32)
        output = embedding_golden(indices, weight, padding_idx=0)
        expected_shape = (4, 256, 256)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        # 验证 padding 位置输出为零
        assert torch.all(output[0, 0, :] == 0), "padding_idx=0 position should be zero"
        assert torch.all(output[1, 10, :] == 0), "padding_idx=0 position should be zero"
        print("  功能_P1 (padding): indices [4, 256], weight [10000, 256], padding_idx=0 ... PASS")
    except Exception as e:
        print(f"  功能_P1 (padding): FAIL - {e}")
        all_passed = False

    # 2. 泛化 case 验证（来自 spec.md §4 动态轴范围）
    print("\n[泛化 case 验证]")

    # batch: [1, 512, 1024], seq: [1, 2048, 4096]
    test_cases = [
        (1, 1, 1000, 128),
        (512, 2048, 5000, 256),
        (1024, 4096, 10000, 512),
    ]

    for batch, seq, vocab, dim in test_cases:
        try:
            indices = torch.randint(0, vocab, (batch, seq), dtype=torch.int64)
            weight = torch.randn(vocab, dim, dtype=torch.float32)
            output = embedding_golden(indices, weight)
            expected_shape = (batch, seq, dim)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            print(f"  泛化 case: batch={batch}, seq={seq}, vocab={vocab}, dim={dim} ... PASS")
        except Exception as e:
            print(f"  泛化 case: batch={batch}, seq={seq}, vocab={vocab}, dim={dim} ... FAIL - {e}")
            all_passed = False

    # 3. 值域检查
    print("\n[值域检查]")

    # 检查输出值是否来自 weight 的对应行
    try:
        indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        weight = torch.randn(4, 8, dtype=torch.float32)
        output = embedding_golden(indices, weight)
        # 验证 output[0, 0, :] == weight[0, :]
        assert torch.allclose(output[0, 0, :], weight[0, :]), "Output should match weight rows"
        assert torch.allclose(output[0, 1, :], weight[1, :]), "Output should match weight rows"
        assert torch.allclose(output[1, 0, :], weight[2, :]), "Output should match weight rows"
        assert torch.allclose(output[1, 1, :], weight[3, :]), "Output should match weight rows"
        print("  索引映射正确性 ... PASS")
    except Exception as e:
        print(f"  索引映射正确性 ... FAIL - {e}")
        all_passed = False

    # 4. 数值稳定性检查
    print("\n[数值稳定性检查]")

    # 空输入处理
    try:
        indices = torch.randint(0, 100, (0, 10), dtype=torch.int64)
        weight = torch.randn(100, 64, dtype=torch.float32)
        output = embedding_golden(indices, weight)
        assert output.shape == (0, 10, 64), f"Empty input shape mismatch: {output.shape}"
        print("  空输入处理 (batch=0) ... PASS")
    except Exception as e:
        print(f"  空输入处理 (batch=0) ... FAIL - {e}")
        all_passed = False

    # 无 NaN/Inf
    try:
        indices = torch.randint(0, 100, (10, 20), dtype=torch.int64)
        weight = torch.randn(100, 64, dtype=torch.float32)
        output = embedding_golden(indices, weight)
        assert not torch.any(torch.isnan(output)), "Output contains NaN"
        assert not torch.any(torch.isinf(output)), "Output contains Inf"
        print("  无 NaN/Inf ... PASS")
    except Exception as e:
        print(f"  无 NaN/Inf ... FAIL - {e}")
        all_passed = False

    # 5. API 对比（与 PyTorch F.embedding 对比）
    print("\n[API 对比]")

    try:
        import torch.nn.functional as F

        indices = torch.randint(0, 100, (4, 8), dtype=torch.int64)
        weight = torch.randn(100, 64, dtype=torch.float32)

        # 我们的实现
        our_output = embedding_golden(indices, weight)

        # PyTorch F.embedding
        torch_output = F.embedding(indices, weight)

        assert torch.allclose(our_output, torch_output, rtol=1e-5, atol=1e-5), \
            "Output mismatch with torch.nn.functional.embedding"
        print("  与 torch.nn.functional.embedding 对比 ... PASS")
    except Exception as e:
        print(f"  与 torch.nn.functional.embedding 对比 ... FAIL - {e}")
        all_passed = False

    # padding_idx 与 PyTorch 对比
    try:
        import torch.nn.functional as F

        indices = torch.randint(0, 100, (4, 8), dtype=torch.int64)
        indices[0, 0] = 5
        indices[2, 3] = 5
        weight = torch.randn(100, 64, dtype=torch.float32)

        our_output = embedding_golden(indices, weight, padding_idx=5)
        torch_output = F.embedding(indices, weight, padding_idx=5)

        assert torch.allclose(our_output, torch_output, rtol=1e-5, atol=1e-5), \
            "Output mismatch with torch.nn.functional.embedding (padding_idx)"
        print("  与 torch.nn.functional.embedding (padding_idx) 对比 ... PASS")
    except Exception as e:
        print(f"  与 torch.nn.functional.embedding (padding_idx) 对比 ... FAIL - {e}")
        all_passed = False

    # 6. dtype 兼容性检查
    print("\n[dtype 兼容性检查]")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        try:
            indices = torch.randint(0, 100, (4, 8), dtype=torch.int64)
            weight = torch.randn(100, 64, dtype=dtype)
            output = embedding_golden(indices, weight)
            assert output.dtype == dtype, f"dtype mismatch: {output.dtype} vs {dtype}"
            print(f"  dtype {dtype} ... PASS")
        except Exception as e:
            print(f"  dtype {dtype} ... FAIL - {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有验证通过")
    else:
        print("部分验证失败，请检查上述错误")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    _validate()
