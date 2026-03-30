# API 探索报告

> **生成时间**: 2026-03-28

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: max_pool2d
- **功能**: 2D 最大池化，对输入 tensor 在空间维度（H 和 W）上进行滑动窗口最大值计算
- **数学公式**: output[n, c, h, w] = max_{i,j}(input[n, c, h*s_h + i*d_h - p_h, w*s_w + j*d_w - p_w])
- **支持特性**: kernel_size, stride, padding, dilation, ceil_mode
- **动态轴**: N (batch), H_in, W_in

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 核心计算为滑动窗口内的 max 归约操作，不涉及 matmul，属于 Vector 类型算子

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | shape | reshape | 将 [N, C, H, W] reshape 为 [N*C, H, W] |
| 2 | shape | padding | 对 H/W 维度进行填充（可选） |
| 3 | index | sliding window | 提取每个输出位置对应的输入窗口 |
| 4 | reduction | amax | 对窗口内数据求最大值 |
| 5 | shape | assemble | 将结果写入输出位置 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | reshape | `pypto.reshape` | direct | ✓ |
| 2 | padding | `pypto.pad` | substitute | ⚠ 仅支持右侧/底部填充 |
| 3 | sliding window | `pypto.tensor` 切片 | direct | ✓ |
| 4 | max reduce | `pypto.amax` | direct | ✓ |
| 5 | write output | `pypto.assemble` | direct | ✓ |
| - | loop control | `pypto.loop_unroll` | direct | ✓ |
| - | create zero tensor | `pypto.zeros` | direct | ✓ |
| - | elementwise max | `pypto.maximum` | direct | ✓ |

### 3.2 Substitute 配方

**padding 处理**：
```
max_pool2d padding: 由于 pypto.pad 仅支持右侧/底部填充，对于 max_pool2d 需要特殊处理：
- 方案 A: 使用 pypto.pad 时，仅支持 padding=(0, right, 0, bottom) 格式
- 方案 B: 在 kernel 中通过边界检查和条件判断模拟左/上 padding
- 推荐方案: 方案 B，在滑动窗口时通过 clamping 处理边界，避免显式 padding
```

**max 窗口计算**：
```
由于 pypto.amax 仅支持单轴归约，需要对窗口的两个维度分别归约：
- 步骤 1: 对窗口 H 维度做 amax(dim=1, keepdim=True)
- 步骤 2: 对结果 W 维度做 amax(dim=2, keepdim=True)
- 或者: 使用循环逐元素比较，pypto.maximum 累积最大值
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | DT_FP16/DT_FP32/DT_BF16/DT_INT32/... | float16/float32 | ✓ |
| contiguous | 必须 | — | 需确保 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.amax` | dtype | DT_FP16/DT_BF16/DT_FP32/DT_INT32/DT_INT16 | ✓ |
| `pypto.amax` | shape | 2-4维，非空，≤INT32_MAX | ✓ |
| `pypto.amax` | TileShape | ≤64KB，尾轴32B对齐，次尾轴≤255 | 需注意 |
| `pypto.pad` | dtype | DT_FP32/DT_FP16/DT_BF16 | ✓ |
| `pypto.pad` | shape | 1-4维，非空，≤INT32_MAX | ✓ |
| `pypto.pad` | mode | 仅 'constant' | ⚠ 限制 |
| `pypto.pad` | value | 仅 -inf/inf/0.0 | ✓ (max用-inf) |
| `pypto.maximum` | dtype | DT_FP16/DT_BF16/DT_INT16/DT_INT32/DT_FP32 | ✓ |
| `pypto.maximum` | shape | 2-4维，非空，≤INT32_MAX | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**推荐 TileShape 配置**：
- 输入处理: `pypto.set_vec_tile_shapes(16, 16, 128)` 用于 [bc, h, w] 形状
- 窗口计算: 根据窗口大小动态调整
- 输出组装: `pypto.set_vec_tile_shapes(unroll_length, 4, 128)`

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/experimental/vector/AvgPool2d/avg_pool2d.py` | models | 高 | 高 | loop_unroll模式、边界处理、assemble用法、动态shape支持 |

### 6.2 可复用模式

- **API 调用模式**：
  - 使用 `pypto.loop_unroll` 对 batch*channel 维度进行分块处理
  - 使用 `pypto.reshape` 将 4D tensor reshape 为 3D 进行处理
  - 使用 `pypto.assemble` 将结果写入输出位置

- **Tiling 策略**：
  - 主循环前设置: `pypto.set_vec_tile_shapes(16, 16, 4, 128)`
  - 单行处理时: `pypto.set_vec_tile_shapes(16, 16, 128)`
  - 输出组装时: `pypto.set_vec_tile_shapes(unroll_length, 4, 128)`

- **Loop 结构**：
  ```python
  for bc_idx, unroll_length in pypto.loop_unroll(0, bc_total, 1,
                                                   name="LOOP_BC",
                                                   idx_name="bc_idx",
                                                   unroll_list=[8, 4, 2, 1]):
      # 处理 bc_idx 到 bc_idx + unroll_length 的数据
  ```

- **边界处理**：
  ```python
  h_start = oh * s_h - t_pad
  h_end = h_start + k_h
  h_start_clamped = max(h_start, 0)
  h_end_clamped = min(h_end, in_h)
  cur_k_h = h_end_clamped - h_start_clamped
  ```

### 6.3 差异分析

| 差异点 | 示例做法 (avg_pool2d) | 本算子需求 (max_pool2d) | 调整建议 |
|--------|----------------------|------------------------|----------|
| 归约操作 | `pypto.sum` 求和后除以窗口大小 | `pypto.amax` 求最大值 | 替换为 amax |
| padding | SAME/VALID 模式 | 支持 int/tuple 形式的显式 padding | 适配 padding 参数解析 |
| dilation | 不支持 | 支持空洞池化 | 在窗口计算中加入 dilation 因子 |
| ceil_mode | 不支持 | 支持 ceil 模式计算输出大小 | 添加 ceil_mode 逻辑 |
| 初始值 | 窗口无效时填充 0 | 窗口无效时应填充 -inf | 使用极小值初始化 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无直接 max_pool2d API | PyPTO 未提供原生池化 API | 使用 amax + loop 组合实现 |
| pypto.pad 不支持左/上填充 | API 仅支持右侧/底部填充 | 在 kernel 内通过边界检查处理 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 动态轴支持 | 使用 `pypto.frontend.dynamic()` 标记动态维度 |
| TileShape 调整 | amax 后维度变化，需重设 TileShape |
| 精度问题 | float16 需注意数值范围，最大值计算相对稳定 |
| dilation 支持 | 窗口索引需乘以 dilation 因子 |
| ceil_mode | 需在参数计算阶段确定输出 shape |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| pypto.amax 文档 | `docs/api/operation/pypto-amax.md` |
| pypto.pad 文档 | `docs/api/operation/pypto-pad.md` |
| pypto.maximum 文档 | `docs/api/operation/pypto-maximum.md` |
| pypto.assemble 文档 | `docs/api/operation/pypto-assemble.md` |
| pypto.sum 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| pypto.zeros 文档 | `docs/api/operation/pypto-zeros.md` |
| pypto.loop_unroll 文档 | `docs/api/controlflow/pypto-loop_unroll.md` |
| pypto.set_vec_tile_shapes 文档 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| 参考实现 | `models/experimental/vector/AvgPool2d/avg_pool2d.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无原生 max_pool2d API，需基于 amax/maximum 组合实现；pypto.pad 不支持左/上填充，需在 kernel 内处理边界
- **实现建议**: 参考 AvgPool2d 实现模式，将 sum 替换为 amax，添加 dilation 和 ceil_mode 支持，通过边界检查处理 padding
