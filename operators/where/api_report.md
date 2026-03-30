# API 探索报告

> **生成时间**: 2026-03-28T08:15:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: where
- **算子分类**: comparison (element-wise)
- **数学公式**: $out_i = condition_i \ ? \ x_i \ : \ y_i$
- **功能描述**: 根据条件张量从 x 或 y 中选择元素。当 condition 对应位置为 True 时选择 x 的元素，为 False 时选择 y 的元素。
- **关键特性**: 广播支持、标量支持、动态轴支持
- **支持 dtype**: float32, float16, bfloat16, int32

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 逐元素条件选择操作，不涉及矩阵乘法，仅需要 Vector 类型的 tiling 配置

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | $result_i = condition_i \ ? \ input_i \ : \ other_i$ | 逐元素条件选择，基于布尔掩码选择 input 或 other |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | $result_i = condition_i \ ? \ input_i \ : \ other_i$ | `pypto.where(condition, input, other)` | direct | ✓ |

### 3.2 Substitute 配方

无需 Substitute，PyPTO 提供直接 API。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype (condition) | DT_BOOL | bool | ✓ |
| dtype (input/other) | DT_FP32/DT_FP16/DT_BF16 | float32/float16/bfloat16 | ✓ |
| dtype (from_torch) | FP16/BF16/FP32/INT8-64/BOOL | 符合 | ✓ |
| contiguous | 必须 | — | 需确保 |
| shape 维度 | 2-4 维 | [b, s, n, d] (4维) | ✓ |
| shape size | ≤ INT32_MAX | 符合 | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.where | condition dtype | DT_BOOL | ✓ |
| pypto.where | input/other dtype | DT_FP32, DT_FP16, DT_BF16 | ✓ |
| pypto.where | shape 维度 | 2-4 维 | ✓ |
| pypto.where | shape size | ≤ INT32_MAX | ✓ |
| pypto.where | 广播规则 | 只支持单轴广播 | ⚠ 需注意 |
| pypto.where | 标量 input/other | 支持 float/Element | ✓ |

### 4.3 重要约束说明

1. **广播限制**: PyPTO where 只支持单轴广播，需注意 spec 中的广播场景是否符合此限制
2. **标量处理**: 建议优先使用 Element 类型传入标量，对于 fp16 场景，直接传入 float 不保证正确性
3. **不支持空 Tensor**: 所有输入 Tensor 不能为空

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

### 5.1 TileShape 配置说明

- TileShape 维度应与输出一致
- 示例：输出为 [m, n]，TileShape 设置为 [m1, n1]，则 m1, n1 分别用于切分 m, n 轴
- 广播场景：输出为 [m, n]，TileShape 仍设置为 [m1, n1]

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/glm_v4_5/glm_select_experts.py` | models | 高 | 高 | where API 调用、广播场景、标量 other |
| `models/glm_v4_5/glm_moe_fusion.py` | models | 高 | 高 | where API 调用、动态 shape 处理 |

### 6.2 可复用模式

**从 glm_select_experts.py 提取 (L158)**:
```python
# where 使用模式 - 标量 other
pypto.set_vec_tile_shapes(view_first, ne)
twm_not = pypto.logical_not(twm_reshape)
topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)
```

- **API 调用模式**: `pypto.where(condition, scalar_value, tensor_value)` 或 `pypto.where(condition, tensor_a, tensor_b)`
- **Tiling 策略**: 在 where 调用前设置 `set_vec_tile_shapes`，维度与输出一致
- **Loop 结构**: 在 loop 内部使用，支持动态 batch 维度
- **边界处理**: 配合 `logical_not` 使用实现 masked_fill 语义

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输入 shape | 2D [bs, ne] | 4D [b, s, n, d] | TileShape 需设置为 4 维 |
| 动态轴 | 仅 batch | batch, seq_len | 使用 dynamic_axis 标记多个动态维度 |
| other 类型 | 标量 0.0 | Tensor 或标量 | 两种场景均需支持 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | API 直接支持 | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 单轴广播限制 | PyPTO where 只支持单轴广播，复杂广播场景可能需要预处理 |
| 标量 fp16 精度 | 建议使用 Element 类型传入标量，避免直接使用 float |
| 4D shape 支持 | 需验证 4D shape 的 tiling 配置是否正确 |
| 动态轴处理 | 需在 from_torch 时正确标记 dynamic_axis |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (L104) |
| where API 文档 | `docs/api/operation/pypto-where.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 1 | `models/glm_v4_5/glm_select_experts.py` (L158) |
| 参考实现 2 | `models/glm_v4_5/glm_moe_fusion.py` (L214) |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题，API 直接支持
- **实现建议**:
  1. 使用 `pypto.where(condition, input, other)` 直接实现
  2. 对于标量场景，优先使用 Element 类型
  3. TileShape 设置为 4 维，与输出 shape 一致
  4. 在 from_torch 时标记 batch 和 seq_len 为动态轴
