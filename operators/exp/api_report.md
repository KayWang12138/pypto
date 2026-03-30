# API 探索报告

> **生成时间**: 2026-03-30T05:08:00Z

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

- **算子名称**: exp
- **数学公式**: y = exp(x) = e^x
- **算子分类**: element-wise
- **输入规格**: x [b, s, n, d] 或 [m, n], dtype=FP32/FP16
- **输出规格**: y [与输入相同shape], dtype=与输入相同
- **特殊要求**: 支持动态轴 (pypto.DYNAMIC), 4D 输入使用 reshape 策略

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 纯逐元素计算，不涉及矩阵乘法，使用 `pypto.set_vec_tile_shapes()` 配置 tiling

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | y = exp(x) | 逐元素计算 e 的指数 |

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | y = exp(x) | `pypto.exp(x)` | direct | ✓ |
| 2 | reshape 4D→2D | `pypto.reshape(x, [-1, d])` | direct | ✓ |
| 3 | reshape 2D→4D | `pypto.reshape(y, [b, s, n, d])` | direct | ✓ |

### 3.2 Substitute 配方

无需 substitute，所有操作均有直接 API 支持。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32 | FP32/FP16 | ✓ |
| contiguous | 必须 | 需确保 | ✓ 需确保 |
| 空 Tensor | 不支持 | 非空 | ✓ |
| Shape 维度 | 2-4 维 | 2D/4D | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.exp | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| pypto.exp | Shape 维度 | 2-4 维 | ✓ |
| pypto.exp | Shape Size | ≤ INT32_MAX | ✓ |
| pypto.reshape | Shape Size | ≤ INT32_MAX | ✓ |
| pypto.reshape | -1 维度 | 支持自动推导 | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes(64, 128)` |

**Tiling 策略说明**:
- 2D 输入: 使用 `pypto.set_vec_tile_shapes(64, 128)`
- 4D 输入: 先 reshape 为 2D，再使用 2D tiling
- 最后一维 (d/n) 使用 128，其他维度根据实际 shape 调整

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/operators/softmax/softmax.py` | examples | 高 | 高 | pypto.exp 调用方式、tiling 配置、动态轴声明 |
| `operators/sigmoid/sigmoid_impl.py` | operators | 高 | 高 | element-wise 实现模式、wrapper 函数结构、tiling 配置函数 |
| `examples/01_beginner/compute/elementwise_ops.py` | examples | 中 | 高 | 基础 element-wise 操作示例 |

### 6.2 可复用模式

- **API 调用模式**:
  ```python
  exp = pypto.exp(x)  # 直接调用
  ```

- **Tiling 策略**:
  ```python
  # 2D tiling
  pypto.set_vec_tile_shapes(64, 128)
  ```

- **动态轴声明**:
  ```python
  @pypto.frontend.jit
  def kernel(x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, ...], pypto.DT_FP32), ...):
  ```

- **Wrapper 函数结构**:
  ```python
  def op_wrapper(x: torch.Tensor) -> torch.Tensor:
      if not x.is_contiguous():
          x = x.contiguous()
      output = torch.empty_like(x)
      kernel(x, output)
      return output
  ```

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 4D 输入 | softmax 直接处理 4D | 需要 reshape 到 2D | 在 kernel 内部添加 reshape 逻辑 |
| dtype | softmax 使用 FP32 | 支持 FP32/FP16 | 在 wrapper 中处理 dtype |

---

<!-- REQUIRED -->
## 7. 风险评估

### 7.1 阻断问题

无阻断问题。所有 API 均直接支持。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 4D reshape 策略 | spec 要求 4D 输入 reshape 到 2D 后调用 2D kernel，避免 4D tiling 编译问题 |
| 动态轴限制 | pypto.exp 仅支持 2-4 维，动态轴需在 from_torch 时声明 |
| 数值范围 | exp 可能产生极大值，建议输入已做数值稳定化处理 |
| Tiling 配置 | 使用统一的 2D tiling: `pypto.set_vec_tile_shapes(64, 128)` |

---

<!-- REQUIRED -->
## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (line 31: pypto-exp) |
| pypto.exp 文档 | `docs/api/operation/pypto-exp.md` |
| pypto.reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 | `examples/02_intermediate/operators/softmax/softmax.py` |
| 参考实现 | `operators/sigmoid/sigmoid_impl.py` |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: ✓ 可行
- **主要问题**: 无
- **实现建议**:
  1. 使用 `pypto.exp` 直接 API
  2. 4D 输入使用 reshape(-1, last_dim) 策略
  3. 使用 `pypto.set_vec_tile_shapes(64, 128)` 配置 tiling
  4. 动态轴在 from_torch 时声明
