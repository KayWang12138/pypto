# API 探索报告

> **生成时间**: 2026-03-29
> **算子名称**: adaptive_avg_pool2d

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

- **算子名称**: adaptive_avg_pool2d
- **数学公式**: 自适应平均池化，将任意尺寸输入池化到指定输出尺寸
- **输入**: input [N, C, H, W] float32, output_size (oH, oW) 或 int
- **输出**: output [N, C, oH, oW] float32
- **关键特性**: 动态轴支持 (N, H, W)、非对齐池化窗口、输出尺寸可配置

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 核心计算为逐元素的 sum 和 div 操作，不涉及 matmul，因此使用 Vector 类型的 tiling

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | shape | 解析 output_size | 若为 int 则 oH=oW，若为 tuple 则 (oH, oW) |
| 2 | compute | h_start = floor(oh * H / oH) | 计算窗口起始行 |
| 3 | compute | h_end = ceil((oh+1) * H / oH) | 计算窗口结束行 |
| 4 | compute | w_start = floor(ow * W / oW) | 计算窗口起始列 |
| 5 | compute | w_end = ceil((ow+1) * W / oW) | 计算窗口结束列 |
| 6 | reduction | sum(input[h_start:h_end, w_start:w_end]) | 窗口内求和 |
| 7 | elementwise | sum / ((h_end-h_start) * (w_end-w_start)) | 计算平均值 |
| 8 | shape | assemble 到输出位置 | 将结果写入输出 tensor |

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | 解析 output_size | Python 原生 | - | ✓ |
| 2-5 | floor/ceil 计算 | `pypto.floor` / `pypto.ceil` | direct | ✓ |
| 6 | 窗口求和 | `pypto.sum` | direct | ✓ |
| 7 | 除法 | `pypto.div` | direct | ✓ |
| 8 | 结果组装 | `pypto.assemble` | direct | ✓ |
| - | reshape | `pypto.reshape` | direct | ✓ |
| - | 创建零张量 | `pypto.zeros` | direct | ✓ |

### 3.2 Substitute 配方

**adaptive_avg_pool2d 无直接 API**，需要通过以下组合实现：

```
adaptive_avg_pool2d(input, output_size):
  1. 使用 pypto.floor/ceil 计算动态窗口边界
  2. 使用 pypto.sum 对窗口内元素求和
  3. 使用 pypto.div 计算平均值
  4. 使用 pypto.assemble 组装输出
  5. 使用 pypto.reshape 处理维度变换
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32 | ✓ |
| contiguous | 必须 | 需确保 | ⚠ 需确保输入连续 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.sum` | dtype | DT_FP32, DT_INT32, DT_INT16 | ✓ |
| `pypto.sum` | shape | 2-4维, Shape Size <= INT32_MAX | ✓ |
| `pypto.sum` | TileShape | 不超过 64KB, 尾轴 32bytes 对齐, 次尾轴 <= 255 | ⚠ 需合理设置 |
| `pypto.div` | dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| `pypto.div` | other | 不支持 nan/inf | ✓ |
| `pypto.floor` | dtype | DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16 | ✓ |
| `pypto.floor` | shape | 2-4维 | ✓ |
| `pypto.ceil` | dtype | DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16 | ✓ |
| `pypto.ceil` | shape | 2-4维 | ✓ |
| `pypto.assemble` | Shape Size | <= INT32_MAX | ✓ |
| `pypto.reshape` | Shape Size | <= INT32_MAX | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 策略建议**：
- 外层循环遍历 batch 和 channel（可用 `pypto.loop_unroll` 优化）
- 中层循环遍历输出位置 (oh, ow)
- 内层使用 sum 进行窗口内归约
- TileShape 需根据实际输入尺寸动态调整，建议：`pypto.set_vec_tile_shapes(16, 16, 128)`

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/experimental/vector/AvgPool2d/avg_pool2d.py` | models/experimental | 高 | 中 | sum/div/assemble 用法、loop 结构、tiling 策略 |
| `examples/01_beginner/compute/reduce_ops.py` | examples | 中 | 高 | sum API 调用模式、tiling 设置 |

**注意**: `models/experimental/` 为实验性实现，置信度标记为中，但提供了最接近的参考。

### 6.2 可复用模式

从 `avg_pool2d.py` 提取的可复用模式：

- **API 调用模式**：
  ```python
  # 窗口求和
  input_single_row_1 = pypto.sum(input_single_row, 1, keepdim=True)
  sum_val = pypto.sum(window, dim=2, keepdim=True)
  # 计算平均值
  avg_val = sum_val / (k_h * k_w)
  ```

- **Tiling 策略**：
  ```python
  pypto.set_vec_tile_shapes(16, 16, 4, 128)  # 外层循环
  pypto.set_vec_tile_shapes(16, 16, 128)     # 内层计算
  pypto.set_vec_tile_shapes(unroll_length, 4, 128)  # 组装输出
  ```

- **Loop 结构**：
  ```python
  for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1, name="LOOP_BC",
                                                   idx_name="bc_idx", unroll_list=[8, 4, 2, 1]):
      for oh in range(out_h):
          for ow in range(out_w):
              # 计算窗口并求平均
  ```

- **边界处理**：
  ```python
  h_start_clamped = max(h_start, 0)
  h_end_clamped = min(h_end, in_h)
  cur_k_h = h_end_clamped - h_start_clamped
  if cur_k_h > 0 and cur_k_w > 0:
      # 正常计算
  else:
      # 填充零值
      zero_val = pypto.zeros([unroll_length, 1, 1], dtype=pypto.DT_FP32)
  ```

### 6.3 差异分析

| 差异点 | avg_pool2d 做法示例 | adaptive_avg_pool2d 需求 | 调整建议 |
|--------|---------------------|-------------------------|----------|
| 窗口计算 | 固定 kernel_size 和 stride | 动态计算窗口边界 | 使用 floor/ceil 公式计算 h_start, h_end 等 |
| 输出尺寸 | 由 kernel/stride/padding 计算 | 由 output_size 参数直接指定 | 无需 padding，直接指定 oH, oW |
| 循环结构 | 嵌套循环 + loop_unroll | 相同模式 | 复用 loop_unroll 优化 |

---

<!-- REQUIRED -->
## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无直接 API | PyPTO 无 `adaptive_avg_pool2d` 直接 API | 通过 sum/div/assemble 组合实现 |
| 动态窗口计算 | 窗口边界需要运行时计算 | 使用 Python 表达式 + SymbolicScalar |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 实验性参考 | 参考实现位于 `models/experimental/`，需额外验证稳定性 |
| Tiling 约束 | sum API 要求尾轴 32bytes 对齐，次尾轴 <= 255 |
| 动态轴处理 | 使用 `pypto.frontend.dynamic()` 声明动态轴 |
| 非对齐窗口 | 每个输出位置的窗口大小可能不同，需动态计算除数 |

---

<!-- REQUIRED -->
## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 列表 | `docs/api/operation/index.md` |
| pypto.sum 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.div 文档 | `docs/api/operation/pypto-div.md` |
| pypto.floor 文档 | `docs/api/operation/pypto-floor.md` |
| pypto.ceil 文档 | `docs/api/operation/pypto-ceil.md` |
| pypto.assemble 文档 | `docs/api/operation/pypto-assemble.md` |
| pypto.reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| pypto.zeros 文档 | `docs/api/operation/pypto-zeros.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 (experimental) | `models/experimental/vector/AvgPool2d/avg_pool2d.py` |
| 参考实现 (examples) | `examples/01_beginner/compute/reduce_ops.py` |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: 可行
- **主要问题**: 无直接 API，需通过 sum/div/assemble 组合实现
- **实现策略**:
  1. 使用 loop_unroll 优化 batch/channel 维度遍历
  2. 使用 floor/ceil 公式动态计算窗口边界
  3. 使用 sum 进行窗口内归约
  4. 使用 div 计算平均值
  5. 使用 assemble 组装输出
- **风险等级**: 中（参考实现来自 experimental 目录，需额外验证）
