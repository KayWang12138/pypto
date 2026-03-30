# API 探索报告

> **生成时间**: 2026-03-30T04:15:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: abs
- **数学公式**: y = |x|（逐元素取绝对值）
- **分类**: element-wise
- **输入**: x [*, D] float32
- **输出**: y [*, D] float32（与输入 shape 相同）
- **动态轴**: 所有维度支持 pypto.DYNAMIC
- **实现约束**: @pypto.frontend.jit, pypto.set_vec_tile_shapes(), output[:] = result

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: abs 为逐元素运算，不涉及矩阵乘法，仅需调用 `pypto.set_vec_tile_shapes()`

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | y = |x| | 逐元素计算输入张量的绝对值 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | y = \|x\| | `pypto.abs(input)` | direct | ✓ |

### 3.2 Substitute 配方

无需 substitute，API 直接支持。

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
| `pypto.abs` | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| `pypto.abs` | shape | 2-4维，非空Tensor | ✓ |
| `pypto.abs` | shape size | ≤ INT32_MAX | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**TileShape 设置要求**:
- TileShape 维度应和输出一致
- 每个维度必须大于 0
- 最多不超过 4 个 inputs

**示例配置**:
- 2D 输入 [m, n]: `pypto.set_vec_tile_shapes(m1, n1)`
- 4D 输入 [b, s, h, d]: `pypto.set_vec_tile_shapes(b1, s1, h1, d1)`

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/compute/elementwise_ops.py` | examples | 高 | 高 | 完整的 abs kernel 实现模式 |
| `models/glm_v4_5/glm_ffn_common_interface.py` | models | 中 | 高 | abs 在量化场景中的使用 |

### 6.2 可复用模式

**首选参考实现**（来自 `examples/01_beginner/compute/elementwise_ops.py`）:

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def abs_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.abs(x)
```

- **API 调用模式**: `out[:] = pypto.abs(x)` — 输出写回模式
- **Tiling 策略**: `pypto.set_vec_tile_shapes(2, 8)` — 根据实际 shape 调整
- **装饰器模式**: `@pypto.frontend.jit` — JIT 编译装饰器
- **隐式 shape 推断**: `pypto.Tensor([], pypto.DT_FP32)` — 空列表表示隐式推断

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 动态轴 | 示例未显式使用动态轴 | 支持所有维度动态 | 在 from_torch 时使用 `dynamic_axis` 参数 |
| Tiling 配置 | 固定值 (2, 8) | 需根据实际 shape 配置 | 根据输入 shape 选择合适的 tile 大小 |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题。API 直接支持，参考实现完整。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 动态轴声明 | 使用 `pypto.from_torch(x, dynamic_axis=[0, 1, ...])` 标记动态维度 |
| Shape 维度限制 | API 仅支持 2-4 维，需确保输入 shape 符合要求 |
| 输出写回模式 | 必须使用 `out[:] = result` 形式，而非直接返回 |
| contiguous 要求 | 输入 tensor 必须连续，否则需先调用 `.contiguous()` |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| abs 文档 | `docs/api/operation/pypto-abs.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 1 | `examples/01_beginner/compute/elementwise_ops.py` |
| 参考实现 2 | `models/glm_v4_5/glm_ffn_common_interface.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无
- **实现建议**:
  1. 使用 `pypto.abs()` 直接映射
  2. 配置 `pypto.set_vec_tile_shapes()` 根据输入 shape 设置
  3. 使用 `out[:] = pypto.abs(x)` 输出写回模式
  4. 支持动态轴需要在 `from_torch` 时声明
