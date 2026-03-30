# API 探索报告

> **生成时间**: 2026-03-28T22:06:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: mean_reduction
- **数学公式**: y = mean(x, dim) = sum(x_i) / N
- **功能描述**: 沿指定轴计算张量均值的归约操作
- **关键特性**: 动态轴支持 (b, s)、keepdim 参数

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 仅涉及 reduction (sum) 和 element-wise (div) 操作，无 matmul

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | reduction | sum_result = sum(x, dim, keepdim) | 沿 dim 轴求和 |
| 2 | element-wise | y = sum_result / N | 除以归约轴元素数量得到均值 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | sum(x, dim, keepdim) | `pypto.sum(input, dim, keepdim)` | direct | ✓ |
| 2 | sum_result / N | `pypto.div(input, other)` | direct | ✓ |

### 3.2 Substitute 配方

```
mean: sum(x, dim, keepdim) / x.shape[dim]
```

**实现说明**:
- PyPTO 无直接的 `mean` API
- 使用 `pypto.sum` 求和，再使用 `pypto.div` 除以元素数量
- 元素数量 N 通过 `x.shape[dim]` 获取（支持动态轴）

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32 | ✓ |
| contiguous | 必须 | — | 需确保 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.sum | dtype | DT_FP32, DT_INT32, DT_INT16 | ✓ |
| pypto.sum | shape | 2-4维, Size ≤ INT32_MAX | ✓ |
| pypto.sum | dim | 支持任意单轴 | ✓ |
| pypto.sum | TileShape | ≤ 64KB | ✓ |
| pypto.sum | 尾轴对齐 | 32 bytes 对齐 | 需确保 |
| pypto.sum | 次尾轴 | ≤ 255 | 需确保 |
| pypto.div | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| pypto.div | shape | 2-4维, Size ≤ INT32_MAX | ✓ |
| pypto.div | other | 不支持 nan, inf | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 策略建议**:
- TileShape 维度应与输入 input 一致
- 典型配置: 对于 [b, s, n, d] 形状，使用 `pypto.set_vec_tile_shapes(tile_b, tile_s, tile_n, tile_d)`
- 尾轴需 32 bytes 对齐 (float32 时为 8 元素)
- 次尾轴 ≤ 255

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/compute/reduce_ops.py` | examples | 高 | 高 | sum API 调用模式、keepdim 处理、Tiling 配置 |
| `models/experimental/vector/BNTrainingReduce/bn_training_reduce.py` | models | 中 | 高 | sum 归约 + 动态 shape 处理、Tiling 策略选择 |

### 6.2 可复用模式

- **API 调用模式**：
  ```python
  # sum 归约
  sum_result = pypto.sum(x, dim=dim, keepdim=keepdim)
  # 除法得到均值
  mean_result = pypto.div(sum_result, N)
  ```

- **Tiling 策略**：
  ```python
  # 根据输入维度设置 TileShape
  tile_shapes = [8 for _ in range(len(x.shape))]
  pypto.set_vec_tile_shapes(*tile_shapes)
  ```

- **动态轴处理**：
  ```python
  # 通过 from_torch 标记动态轴
  x_pto = pypto.from_torch(x, dynamic_axis=[0, 1])  # b, s 为动态
  # N 通过 shape 获取（支持动态）
  N = x.shape[dim]
  ```

- **边界处理**：尾轴 32 bytes 对齐，次尾轴 ≤ 255

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 归约结果 | 仅 sum | sum / N | 增加 div 操作 |
| 动态轴 | 部分示例无 | 需支持 b, s | 使用 from_torch 的 dynamic_axis 参数 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | - | - |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 尾轴对齐 | sum API 要求尾轴 32 bytes 对齐，需在 Tiling 时确保 |
| 次尾轴限制 | sum API 要求次尾轴 ≤ 255，大 shape 需注意 |
| 动态轴 shape 获取 | N = x.shape[dim] 在动态轴场景下返回 SymbolicScalar，可用于 div |
| keepdim 后续操作 | keepdim=False 后需重设 TileShape 再调用其他 operation |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 列表 | `docs/api/operation/index.md` |
| pypto.sum 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.div 文档 | `docs/api/operation/pypto-div.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 (sum) | `examples/01_beginner/compute/reduce_ops.py` |
| 参考实现 (BN reduce) | `models/experimental/vector/BNTrainingReduce/bn_training_reduce.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题
- **实现策略**: 使用 `pypto.sum` + `pypto.div` 的 substitute 方案实现 mean
- **关键约束**: 尾轴 32 bytes 对齐、次尾轴 ≤ 255、TileShape ≤ 64KB
