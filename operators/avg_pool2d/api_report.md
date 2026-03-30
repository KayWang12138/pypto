# API 探索报告

> **生成时间**: 2026-03-29

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: avg_pool2d
- **数学公式**: `output[n, c, oh, ow] = (1 / (k_h * k_w)) * Σ_{i=0}^{k_h-1} Σ_{j=0}^{k_w-1} input[n, c, oh*s_h+i, ow*s_w+j]`
- **输入**: x [batch_size, channels, in_h, in_w], float32
- **输出**: y [batch_size, channels, out_h, out_w], float32
- **动态轴**: batch_size, channels
- **参数**: kernel_size, stride, padding_mode (SAME/VALID)
- **精度要求**: rtol=1e-3, atol=1e-3

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 公式仅含索引切片、sum 归约和 div 逐元素操作，无矩阵乘法，属于纯 Vector 类型

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | index/slice | `window = input[n, c, oh*s_h:oh*s_h+k_h, ow*s_w:ow*s_w+k_w]` | 提取滑动窗口 |
| 2 | reduction | `sum_h = sum(window, axis=h)` | 对高度维度求和 |
| 3 | reduction | `sum_w = sum(sum_h, axis=w)` | 对宽度维度求和 |
| 4 | elementwise | `avg = sum_w / (k_h * k_w)` | 除以窗口面积得到平均值 |
| 5 | assemble | `output[n, c, oh, ow] = avg` | 将结果写入输出位置 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | 窗口切片 | `Tensor.__getitem__` / `pypto.view` | direct | ✓ |
| 2 | 高度方向求和 | `pypto.sum(input, dim=1, keepdim=True)` | direct | ✓ |
| 3 | 宽度方向求和 | `pypto.sum(input, dim=2, keepdim=True)` | direct | ✓ |
| 4 | 除法运算 | `pypto.div(sum_val, k_h * k_w)` | direct | ✓ |
| 5 | 结果写入 | `pypto.assemble(val, offsets, output)` | direct | ✓ |
| - | 动态轴定义 | `pypto.frontend.dynamic("name")` | direct | ✓ |
| - | 循环展开 | `pypto.loop_unroll(start, stop, step, unroll_list=[...])` | direct | ✓ |
| - | reshape | `pypto.reshape(input, shape, inplace=True)` | direct | ✓ |
| - | 零填充 | `pypto.zeros(shape, dtype)` | direct | ✓ |

### 3.2 Substitute 配方

无需 substitute，所有操作均有直接 API 支持。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | float32 | ✓ |
| contiguous | 必须 | — | 需确保 |
| shape size | ≤ INT32_MAX | [b, c, h, w] | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.sum` | dtype | DT_FP32, DT_INT32, DT_INT16 | ✓ |
| `pypto.sum` | shape | 2-4维, Size ≤ INT32_MAX | ✓ |
| `pypto.sum` | TileShape | ≤ 64KB, 尾轴32B对齐, 次尾轴≤255 | 需配置 |
| `pypto.div` | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| `pypto.div` | other | 不支持 nan/inf | ✓ |
| `pypto.assemble` | offsets | 需小于 out shape | 需确保 |
| `pypto.zeros` | dtype | DT_FP32, DT_INT32, DT_INT16, DT_FP16, DT_BF16 | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

### 5.1 Tiling 配置建议

基于参考实现分析，推荐以下 tiling 策略：

| 场景 | TileShape | 说明 |
|------|-----------|------|
| batch*channel 循环 | `(unroll_length, 4, 128)` | 根据 unroll_factor 动态调整 |
| 窗口内高度求和 | `(16, 16, 128)` | 处理 h 维度 |
| 结果组装 | `(unroll_length, 4, 128)` | 写入输出位置 |

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/experimental/vector/AvgPool2d/avg_pool2d.py` | models/experimental | 高 | 中 | 完整实现模式 |
| `examples/02_intermediate/controlflow/others/dynamic.py` | examples | 中 | 高 | 动态轴处理模式 |
| `examples/01_beginner/transform/add_scalar_loop_view_assemble.py` | examples | 中 | 高 | loop + view + assemble 模式 |

**注意**: `models/experimental/` 为实验性实现，未充分验证，仅供参考。优先参考 `examples/` 中的模式。

### 6.2 可复用模式

从 `models/experimental/vector/AvgPool2d/avg_pool2d.py` 提取：

- **API 调用模式**:
  - 动态轴定义: `batch_size = pypto.frontend.dynamic("batch_size")`
  - reshape: `input_reshaped = pypto.reshape(input_tensor, [batch_size * channels, in_h, in_w], inplace=True)`
  - 循环展开: `for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, unroll_list=[8, 4, 2, 1])`
  - 分步求和: `pypto.sum(input_single_row, 1, keepdim=True)` + `pypto.sum(window, dim=2, keepdim=True)`
  - 除法: `avg_val = sum_val / (k_h * k_w)`
  - 组装: `pypto.assemble(avg_val, [bc_idx, oh, ow], output_tmp)`

- **Tiling 策略**:
  - 主循环前: `pypto.set_vec_tile_shapes(16, 16, 4, 128)`
  - 窗口处理: `pypto.set_vec_tile_shapes(16, 16, 128)`
  - 结果组装: `pypto.set_vec_tile_shapes(unroll_length, 4, 128)`

- **Loop 结构**:
  - 外层: `loop_unroll` 遍历 batch*channel 维度
  - 中层: `for oh in range(out_h)` 遍历输出高度
  - 内层: `for ow in range(out_w)` 遍历输出宽度

- **边界处理**:
  - SAME padding: 计算上下左右 padding，使用 `max()` 和 `min()` clamp 边界
  - 空窗口: 当 `cur_k_h <= 0` 或 `cur_k_w <= 0` 时，使用 `pypto.zeros()` 填充

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 无 | 完整匹配 | 完整匹配 | 可直接复用示例代码结构 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | 所有 API 均可映射 | - |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| TileShape 配置 | 需根据实际 shape 动态调整，确保尾轴 32B 对齐 |
| sum 后 TileShape 重置 | `keepdim=False` 时维度减少，需重置 TileShape |
| padding 边界 | SAME padding 需正确计算边界，避免越界访问 |
| 动态轴使用 | 使用 `pypto.frontend.dynamic()` 或 `pypto.DYNAMIC` 定义动态维度 |
| loop_unroll 展开 | 选择合适的 unroll_list 以平衡编译时间和性能 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| sum 文档 | `docs/api/operation/pypto-sum.md` |
| div 文档 | `docs/api/operation/pypto-div.md` |
| reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| assemble 文档 | `docs/api/operation/pypto-assemble.md` |
| zeros 文档 | `docs/api/operation/pypto-zeros.md` |
| loop_unroll 文档 | `docs/api/controlflow/pypto-loop_unroll.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Tensor 构造 | `docs/api/tensor/pypto-Tensor_constructor.md` |
| SymbolicScalar | `docs/api/symbolic/pypto-SymbolicScalar_constructor.md` |
| 参考实现 | `models/experimental/vector/AvgPool2d/avg_pool2d.py`（实验性，仅供参考） |
| 动态轴示例 | `examples/02_intermediate/controlflow/others/dynamic.py` |
| assemble 示例 | `examples/01_beginner/transform/add_scalar_loop_view_assemble.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题，所有必需 API 均有直接支持
- **实现建议**:
  1. 参考 `models/experimental/vector/AvgPool2d/avg_pool2d.py` 的整体结构
  2. 使用 `pypto.frontend.dynamic()` 定义 batch_size 和 channels 为动态轴
  3. 使用 `loop_unroll` 优化 batch*channel 维度的并行度
  4. 分步使用 `pypto.sum` 完成窗口内求和
  5. 注意边界处理和 TileShape 配置
