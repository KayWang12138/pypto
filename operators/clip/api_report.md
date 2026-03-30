# API 探索报告

> **生成时间**: 2026-03-29T00:00:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: clip
- **数学公式**: y = min(max(x, min_val), max_val)
- **功能描述**: 将输入张量的每个元素限制在 [min_val, max_val] 范围内
- **输入规格**:
  - x: 任意shape张量, float32, 所有维度动态
  - min_val: 标量, float32
  - max_val: 标量, float32
- **输出规格**: y: 与x相同shape, float32

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 逐元素操作，不涉及矩阵乘法(matmul)，仅需 set_vec_tile_shapes()

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | y = min(max(x, min_val), max_val) | 单步逐元素裁剪操作 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | y = clip(x, min, max) | `pypto.clip(input, min, max)` | direct | ✓ |

### 3.2 替代方案

PyPTO 也支持使用 `pypto.maximum` 和 `pypto.minimum` 组合实现:

```python
# 方案1: 直接使用 pypto.clip (推荐)
y = pypto.clip(x, min_val, max_val)

# 方案2: 组合 maximum + minimum
y = pypto.minimum(pypto.maximum(x, min_val), max_val)
```

**推荐使用方案1**: `pypto.clip` 是 PyPTO 原生支持的一体化 API，性能更优。

---

## 4. 约束检查

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | float32 | ✓ |
| contiguous | 必须 | — | 需确保 |
| 非空Tensor | 必须 | — | 需确保 |

### 4.2 API 约束 (pypto.clip)

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| input dtype | DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16 | float32 (DT_FP32) | ✓ |
| input 维度 | 2-4维 | 任意维度 | ⚠ 需注意 |
| input 元素个数 | ≤ UINT32_MAX | — | 需确保 |
| min/max 类型 | int/float/Element/Tensor | 标量 float | ✓ |
| min/max 类型一致性 | 同时为 Element 或同时为 Tensor | 标量 | ✓ |

### 4.3 关键约束说明

1. **维度限制**: `pypto.clip` 要求 input 为 2-4 维。若需支持 1 维输入，需要先 reshape 到 2 维再操作。
2. **min/max 缺省**: min 和 max 可同时缺省，返回原值。
3. **NaN/INF 处理**: NaN/INF 仅在浮点数运算时有定义。

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

### 5.1 TileShape 设置建议

- TileShape 维度应和输出一致
- 非广播场景: 输入 shape 为 [m, n]，TileShape 设置为 [m1, n1]
- 广播场景: 输入 shape 为 [m, n]，min/max 为 [m, 1]，TileShape 设置为 [m1, n1]

```python
# 示例: 2D 输入
pypto.set_vec_tile_shapes(4, 16)

# 示例: 3D 输入
pypto.set_vec_tile_shapes(2, 8, 16)

# 示例: 4D 输入
pypto.set_vec_tile_shapes(1, 1, 8, 8)
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/compute/elementwise_ops.py` | examples | 高 | 高 | clip_kernel 函数签名、Tiling 配置、标量支持 |
| `models/glm_v4_5/glm_moe_distributed_dispatch_combine.py` | models | 中 | 高 | 生产环境使用案例 |

### 6.2 可复用模式

- **API 调用模式**:
  ```python
  @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
  def clip_kernel(
      x: pypto.Tensor([], pypto.DT_FP32),
      min_: pypto.Tensor([], pypto.DT_FP32),
      max_: pypto.Tensor([], pypto.DT_FP32),
      out: pypto.Tensor([], pypto.DT_FP32)):
      pypto.set_vec_tile_shapes(2, 8)
      out[:] = pypto.clip(x, min_, max_)
  ```

- **Tiling 策略**: 使用 `pypto.set_vec_tile_shapes(2, 8)` 作为基础配置
- **Loop 结构**: 无显式循环，由框架自动处理
- **边界处理**: 自动处理 min > max 情况（输出为 max）

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| min/max 类型 | Tensor 形式 | 标量形式 | 直接传入 float/int 标量即可 |
| 动态轴 | 未明确标记 | 所有维度动态 | 使用 `dynamic_axis` 参数标记 |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题。PyPTO 直接支持 `pypto.clip` API。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 维度限制 | input 仅支持 2-4 维，1 维输入需 reshape |
| 标量支持 | min/max 支持标量形式，无需创建 Tensor |
| 精度要求 | atol=0.001, rtol=0.001，满足 spec 要求 |
| 动态轴 | 需通过 `pypto.from_torch(..., dynamic_axis=[...])` 标记 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (行23) |
| pypto.clip 文档 | `docs/api/operation/pypto-clip.md` |
| pypto.maximum 文档 | `docs/api/operation/pypto-maximum.md` |
| pypto.minimum 文档 | `docs/api/operation/pypto-minimum.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 (examples) | `examples/01_beginner/compute/elementwise_ops.py` (行241-271) |
| 参考实现 (models) | `models/glm_v4_5/glm_moe_distributed_dispatch_combine.py` (行394) |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无
- **推荐方案**: 直接使用 `pypto.clip(x, min_val, max_val)` 实现
- **预期复杂度**: 低（单 API 调用，无特殊处理需求）
