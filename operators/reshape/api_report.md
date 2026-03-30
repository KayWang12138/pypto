# API 探索报告: reshape

## 1. 概述

| 项目 | 内容 |
|------|------|
| 算子名称 | reshape |
| 算子分类 | tensor_manipulation / shape |
| 核心功能 | 改变 Tensor 形状，保持数据不变，仅改变维度视图 |
| API 可行性 | **可行** - PyPTO 提供原生 `pypto.reshape` API |
| 实现复杂度 | 低 - 纯 shape 变换，不涉及计算 |

## 2. 计算逻辑分解

### 2.1 原子操作序列

reshape 是原子操作，无需分解为子操作。

```
输入: x [d1, d2, ..., dn], shape [s1, s2, ..., sm]
输出: y [s1, s2, ..., sm]

约束: ∏(d_i) = ∏(s_i)
特殊: shape 中最多一个维度为 -1，表示自动推断
```

### 2.2 操作类型

| 类型 | 是否涉及 | 说明 |
|------|----------|------|
| elementwise | 否 | 不涉及逐元素计算 |
| reduction | 否 | 不涉及归约 |
| matmul | 否 | 不涉及矩阵乘法 |
| **shape** | **是** | 纯 shape 变换操作 |
| index | 否 | 不涉及索引操作 |

## 3. API 映射

### 3.1 主 API

| PyTorch API | PyPTO API | 映射状态 | 备注 |
|-------------|-----------|----------|------|
| `torch.reshape(x, shape)` | `pypto.reshape(input, shape, ...)` | **直接映射** | 完全对应 |
| `Tensor.view(shape)` | `pypto.reshape(input, shape, ...)` | **直接映射** | reshape 更通用 |
| `Tensor.flatten()` | `pypto.reshape(input, [-1])` | **组合实现** | flatten = reshape([-1]) |

### 3.2 API 签名

```python
pypto.reshape(
    input: Tensor,
    shape: List[int],
    *,
    valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None,
    inplace: bool = False
) -> Tensor
```

### 3.3 参数对照

| PyPTO 参数 | PyTorch 对应 | 说明 |
|------------|--------------|------|
| `input` | 第一个位置参数 | 输入 Tensor |
| `shape` | 第二个位置参数 | 目标形状，支持 -1 自动推断 |
| `valid_shape` | 无对应 | PyPTO 特有，用于动态 shape 场景 |
| `inplace` | 无直接对应 | 是否原地操作，有特殊约束 |

### 3.4 特性支持对照

| 特性 | PyTorch | PyPTO | 状态 |
|------|---------|-------|------|
| 负维度推断 (-1) | 支持 | 支持 | 兼容 |
| 动态 shape | 支持 | 支持 (通过 valid_shape) | 兼容 |
| 多 dtype | 支持 | 支持 | 兼容 |
| 非连续 Tensor | 支持但可能拷贝 | 需 contiguous | 需注意 |

## 4. 约束分析

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 来源 |
|--------|------|------|
| dtype | FP16/BF16/FP32/FP64/INT8-64/UINT8-64/BOOL | `docs/api/others/pypto-from_torch.md` |
| contiguous | 必须 `tensor.is_contiguous() == True` | `docs/api/others/pypto-from_torch.md` |
| 空 Tensor | 不支持 | API 文档 |
| Shape Size | 不大于 INT32_MAX | API 文档 |

### 4.2 API 约束 (pypto.reshape)

| 约束项 | 要求 | 来源 |
|--------|------|------|
| 输入 dtype | PyPTO 支持的所有数据类型 | `docs/api/operation/pypto-reshape.md` |
| 输入 shape | 非空，Size 不大于 INT32_MAX | API 文档 |
| 目标 shape | Size 不大于 INT32_MAX | API 文档 |
| -1 维度 | 支持自动推导 | API 文档 |
| inplace=True | 输入输出必须是当前 loop 的输入输出；输出不可作为 Function 输出 | API 文档 |

### 4.3 Tiling 约束

reshape 是纯 shape 操作，**不需要** tiling 配置（set_vec_tile_shapes / set_cube_tile_shapes）。

| Tiling 类型 | 是否需要 | 说明 |
|-------------|----------|------|
| Vector (set_vec_tile_shapes) | 否 | 不涉及向量计算 |
| Cube (set_cube_tile_shapes) | 否 | 不涉及矩阵乘法 |

### 4.4 约束检查清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 输入非空 | 必须检查 | 空 Tensor 不支持 |
| 元素总数匹配 | 必须检查 | ∏(input.shape) = ∏(target_shape) |
| -1 维度数量 | 必须检查 | 最多一个维度为 -1 |
| Shape Size 上限 | 必须检查 | 不超过 INT32_MAX |
| contiguous | 必须保证 | from_torch 要求 |

## 5. 动态 Shape 支持

### 5.1 动态轴处理

根据 spec.md，需要支持以下动态轴：
- **batch**: 批次大小，推理时可变
- **seq_len**: 序列长度，推理时可变

### 5.2 实现方式

```python
# 方式1: 使用 SymbolicScalar (推荐)
from pypto import SymbolicScalar
b = SymbolicScalar("batch")
s = SymbolicScalar("seq_len")
output = pypto.reshape(input, [b, s, -1])

# 方式2: 使用 valid_shape 参数
output = pypto.reshape(input, [max_b, max_s, d], valid_shape=[actual_b, actual_s, d])
```

### 5.3 动态 shape 约束

| 场景 | 处理方式 |
|------|----------|
| 输入动态 | 通过 `from_torch(..., dynamic_axis=[0, 1])` 标记 |
| 输出动态 | 自动从输入 shape 推断，或使用 valid_shape 显式指定 |

## 6. 参考实现

### 6.1 最佳匹配: glm_attention.py

| 项目 | 内容 |
|------|------|
| 路径 | `models/glm_v4_5/glm_attention.py` |
| 相似度 | 高 |
| 置信度 | 高 |
| 可复用点 | inplace reshape 用法、动态 shape 处理、loop 内 reshape |

**关键代码片段**:

```python
# 行 301-303: inplace reshape 用法
k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
q_2d_shape = (b_scalar * s1_scalar * nq, dn)

k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
q_2d = pypto.reshape(q, q_2d_shape, inplace=True)

# 行 402-404: 常规 reshape 用法
oi_final_3d = pypto.cast(
    pypto.reshape(oi_final, [1, g_tile, dn]),
    dtype)
```

### 6.2 次选: attention.py (examples)

| 项目 | 内容 |
|------|------|
| 路径 | `examples/03_advanced/advanced_nn/attention/attention.py` |
| 相似度 | 高 |
| 置信度 | 高 |
| 可复用点 | 多维 reshape、与 transpose 配合使用 |

**关键代码片段**:

```python
# 行 213-215: 多维 reshape
q = pypto.reshape(q_flat, [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM])
k = pypto.reshape(k_flat, [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM])
v = pypto.reshape(v_flat, [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM])

# 行 236: reshape 与其他操作组合
context_flat = pypto.reshape(context, [tile_b, SEQ_LEN, NUM_HEADS * HEAD_DIM])
```

### 6.3 参考实现总结

| 参考文件 | reshape 用法 | inplace | 动态 shape | 推荐场景 |
|----------|--------------|---------|------------|----------|
| glm_attention.py | 2D 变换 | 是 | 是 | 高性能场景 |
| attention.py | 多维变换 | 否 | 否 | 常规场景 |

## 7. 风险与限制

### 7.1 已知风险

| 风险 | 严重度 | 缓解措施 |
|------|--------|----------|
| 非连续 Tensor | 中 | 调用前确保 contiguous |
| inplace 约束 | 中 | 仅在 loop 内使用 inplace=True |
| 动态 shape 推断 | 低 | 使用 valid_shape 显式指定 |

### 7.2 限制条件

| 限制 | 说明 |
|------|------|
| 最大维度 | PyPTO Tensor 最多支持 8 维 |
| 元素总数 | 不超过 INT32_MAX |
| inplace 输出 | 不能作为 Function 的最终输出 |

### 7.3 无替代方案

reshape 有直接的 PyPTO API，无需 substitute 方案。

## 8. 证据索引

| 章节 | 证据来源 | 路径 |
|------|----------|------|
| API 存在性 | API 列表 | `docs/api/operation/index.md` (行 80) |
| API 签名 | reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| 入口约束 | from_torch 文档 | `docs/api/others/pypto-from_torch.md` |
| inplace 用法 | glm_attention.py | `models/glm_v4_5/glm_attention.py` (行 301-303) |
| 多维 reshape | attention.py | `examples/03_advanced/advanced_nn/attention/attention.py` (行 213-215) |

## 9. 结论

### 9.1 可行性判定

**结论: 完全可行**

| 判定项 | 状态 | 说明 |
|--------|------|------|
| API 存在 | 通过 | `pypto.reshape` 原生支持 |
| 功能覆盖 | 通过 | 支持负维度推断、动态 shape |
| 约束满足 | 通过 | 满足 spec.md 所有需求 |
| 参考实现 | 通过 | 多个生产级代码可参考 |

### 9.2 实现建议

1. **优先使用非 inplace 模式**，除非在 loop 内且有明确性能需求
2. **确保输入 contiguous**，避免运行时错误
3. **动态 shape 场景**使用 valid_shape 参数显式指定
4. **负维度推断**直接使用 -1，PyPTO 自动处理

### 9.3 下一步

- Stage 3: 生成 golden 参考实现
- Stage 4: 设计具体实现方案（包含 dtype 处理、边界检查）

---

*生成时间: 2026-03-28T15:30:00Z*
*API 探索工具: pypto-api-explorer*
