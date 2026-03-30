# API 探索报告

> **生成时间**: 2026-03-28

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: relu
- **算子分类**: element-wise
- **数学公式**: res_i = max(0, input_i)
- **功能描述**: 对 input 的每个元素进行整流线性单元（Rectified Linear Unit）运算，只保留正数部分，负数变为0
- **输入 Shape**: [m, n] 或 [b, m, n] 或 [b, s, m, n]（2-4维）
- **数据类型**: DT_FP16/DT_FP32/DT_BF16
- **精度要求**: atol=3e-3, rtol=3e-3

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: ReLU 是纯逐元素操作（element-wise），不涉及矩阵乘法，仅需要对每个输入元素执行 max(0, x) 计算，因此使用 Vector 类型的 tiling 配置。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | res_i = max(0, input_i) | 逐元素取最大值，负数置零 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | max(0, x) | `pypto.relu(input)` | direct | ✓ |

### 3.2 Substitute 配置

无需 substitute，PyPTO 提供直接的 `pypto.relu()` API。

---

## 4. 约束检查

### 4.1 入口约束（from_torch）

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP16/FP32/BF16 | ✓ |
| contiguous | 必须 | — | 需确保 |
| 非空 Tensor | 必须 | spec 约束不支持空 Tensor | ✓ |

### 4.2 API 约束（pypto.relu）

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.relu | dtype | DT_FP16, DT_FP32, DT_BF16 | ✓ |
| pypto.relu | shape | 仅支持 2-4 维 | ✓ (spec: 2-4维) |
| pypto.relu | shape size | ≤ INT32_MAX (2147483647) | ✓ (spec: ≤ INT32_MAX) |
| pypto.relu | 空Tensor | 不支持 | ✓ (spec: 不支持空Tensor) |
| pypto.relu | 特殊值 | 不支持 nan/inf | ✓ (spec: 不支持nan/inf) |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**TileShape 配置说明**（来自 pypto.relu 文档）:
- TileShape 维度应和输出一致
- 示例：输入 input shape 为 [m, n]，输出为 [m, n]，TileShape 设置为 [m1, n1]，则 m1, n1 分别用于切分 m, n 轴
- 每个维度必须 > 0
- 最多 4 个 inputs

**推荐 TileShape**:
- 2D: `pypto.set_vec_tile_shapes(32, 128)`
- 3D: `pypto.set_vec_tile_shapes(1, 32, 128)`
- 4D: `pypto.set_vec_tile_shapes(1, 1, 32, 128)`

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/compute/elementwise_ops.py` | examples | 高 | 高 | element-wise kernel 模式、tiling 配置、jit 装饰器用法、测试框架结构 |
| `examples/02_intermediate/operators/activation/activation.py` | examples | 高 | 高 | 激活函数 kernel 模式、configure_tiling 函数设计、golden 参考实现模式 |
| `examples/02_intermediate/operators/softmax/softmax.py` | examples | 中 | 高 | 完整算子开发模板、动态 shape 处理、loop 结构 |

### 6.2 可复用模式

- **API 调用模式**：
  ```python
  @pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
  def kernel(x: pypto.Tensor(), out: pypto.Tensor()):
      pypto.set_vec_tile_shapes(...)
      out[:] = pypto.relu(x)
  ```

- **Tiling 策略**：
  ```python
  def configure_tiling(x):
      if len(x.shape) >= 2:
          tile_list = [32 for _ in range(len(x.shape))]
          pypto.set_vec_tile_shapes(*tile_list)
  ```

- **Loop 结构**：ReLU 为简单 element-wise 操作，无需显式 loop，框架自动处理分块

- **边界处理**：无需特殊边界处理，框架自动处理

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| API 复杂度 | 示例使用组合 API（如 sigmoid + mul） | 单一 API（relu） | 更简单，直接调用 pypto.relu |
| 动态 shape | softmax 示例使用 pypto.DYNAMIC | spec 要求所有维度可动态 | 参考 softmax 示例的动态 shape 处理方式 |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题。`pypto.relu` API 完整支持需求规格。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 输入必须 contiguous | 调用 from_torch 前需确保 tensor.is_contiguous() == True |
| 不支持 nan/inf | 输入不应包含 nan/inf 特殊值，需在 golden 测试中注意 |
| TileShape 维度匹配 | TileShape 维度数应与输入 shape 维度数一致 |
| dtype 一致性 | 输出 dtype 与输入 dtype 相同，无需显式 cast |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (line 78: pypto-relu) |
| relu API 文档 | `docs/api/operation/pypto-relu.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| elementwise 示例 | `examples/01_beginner/compute/elementwise_ops.py` |
| activation 示例 | `examples/02_intermediate/operators/activation/activation.py` |
| softmax 示例 | `examples/02_intermediate/operators/softmax/softmax.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无

PyPTO 提供直接的 `pypto.relu()` API，完全满足 spec 中的所有约束条件（dtype、shape、特殊值处理）。实现难度低，可直接参考 elementwise_ops.py 和 activation.py 中的 kernel 模式进行开发。
