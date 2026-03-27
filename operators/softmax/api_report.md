# API 探索报告

> **生成时间**: 2026-03-27

---

## 1. 概述

### 1.1 输入摘要

算子名称：softmax  
数学公式：$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$  
计算逻辑：数值稳定的 softmax 归一化，沿指定轴计算  
输入规格：x [batch, seq, ...dims] float32，dim int  
输出规格：y [batch, seq, ...dims] float32

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 仅包含逐元素操作（exp, sub, div）和归约操作（amax, sum），不涉及矩阵乘法

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | reduction | max_val = max(x, axis=dim) | 沿指定轴求最大值 |
| 2 | elementwise | x_shifted = x - max_val | 数值稳定：减去最大值 |
| 3 | elementwise | exp_x = exp(x_shifted) | 计算指数 |
| 4 | reduction | sum_exp = sum(exp_x, axis=dim) | 沿指定轴求和 |
| 5 | elementwise | y = exp_x / sum_exp | 归一化 |

---

## 3. API 映射

### 3.1 方案 A：直接使用 softmax API（推荐）

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1-5 | softmax(x, dim) | `pypto.softmax(x, dim)` | direct | ✓ |

**优势**：
- 代码简洁，一行完成
- 官方优化实现
- 内置数值稳定性

**劣势**：
- 灵活性较低
- 仅支持 FP32

### 3.2 方案 B：手动实现（参考官方示例）

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | max(x, dim, keepdim) | `pypto.amax(x, dim, keepdim=True)` | direct | ✓ |
| 2 | x - max_val | `pypto.sub(x, max_val)` | direct | ✓ |
| 3 | exp(x_shifted) | `pypto.exp(x_shifted)` | direct | ✓ |
| 4 | sum(exp_x, dim, keepdim) | `pypto.sum(exp_x, dim, keepdim=True)` | direct | ✓ |
| 5 | exp_x / sum_exp | `pypto.div(exp_x, sum_exp)` | direct | ✓ |

**优势**：
- 更灵活，可自定义实现细节
- 可支持更多 dtype（FP16/BF16）
- 更好的学习价值

**劣势**：
- 代码较长
- 需要手动处理 Tiling 和 Loop

### 3.3 推荐方案

**推荐使用方案 A（直接 API）**，原因：
1. PyPTO 已提供优化的 softmax API
2. 代码更简洁，维护成本低
3. 官方 API 更稳定可靠
4. 如有特殊需求，可随时切换到方案 B

---

## 4. 约束检查

### 4.1 入口约束（from_torch）

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/FP64/INT8-64/BOOL | FP32 | ✓ |
| contiguous | 必须 | 需确保 | ⚠ 需调用前检查 |
| tensor type | torch.Tensor 或子类 | torch.Tensor | ✓ |
| 空Tensor | 不支持 | 非空 | ✓ |
| shape size | ≤ INT32_MAX | 待确认 | ⚠ 需根据实际输入验证 |

### 4.2 API 约束

#### 方案 A：pypto.softmax

| 约束项 | 要求 | 结果 |
|--------|------|------|
| dtype | DT_FP32 | ✓ |
| dim 范围 | [-input.dim, input.dim-1] | ✓ |
| shape size | ≤ INT32_MAX | ⚠ 需验证 |

#### 方案 B：pypto.amax

| 约束项 | 要求 | 结果 |
|--------|------|------|
| dtype | DT_FP16, DT_BF16, DT_FP32 | ✓ |
| shape 维度 | 2-4 维 | ⚠ 需验证输入维度 |
| shape size | ≤ INT32_MAX | ⚠ 需验证 |
| TileSize | ≤ 64KB | ⚠ 需配置 |
| 尾轴对齐 | 32 bytes | ⚠ 需验证 |
| 次尾轴 | ≤ 255 | ⚠ 需验证 |

---

## 5. Tiling 需求

### 方案 A：pypto.softmax

**需要调用**：`pypto.set_vec_tile_shapes()`

**Tiling 策略**：由框架自动处理，用户需根据输入 shape 设置合适的 tile shapes

**参考配置**（来自官方示例）：
```python
pypto.set_vec_tile_shapes(1, 4, 1, 64)  # [tile_b, tile_s, tile_h, tile_d]
```

### 方案 B：手动实现

**需要调用**：`pypto.set_vec_tile_shapes()`

**Tiling 策略**：
- 需要为每个操作单独考虑 Tiling
- amax/sum 操作后需要重设 TileShape（keepdim=False 时）
- 建议使用 Loop 处理大批次数据

**参考配置**（来自官方示例）：
```python
pypto.set_vec_tile_shapes(1, 4, 1, 64)  # [tile_b, tile_s, tile_h, tile_d]
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/operators/softmax/softmax.py` | examples | 100% | 高 | 完整实现、Tiling配置、Loop结构、测试方法 |

### 6.2 可复用模式

**API 调用模式**：
```python
# 方案 B：手动实现
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    sub = x - row_max
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return exp / esum
```

**Tiling 策略**：
```python
# 设置 Vector 类型的 TileShape
pypto.set_vec_tile_shapes(1, 4, 1, 64)  # 根据实际输入调整
```

**Loop 结构**：
```python
# 使用 pypto.loop 处理大批次
bs, seqlen, head, dim = input_tensor.shape
tile_b = 1
b_loop = bs // tile_b

for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
    b_offset = idx * tile_b
    input_view = input_tensor[b_offset:b_offset+tile_b, ...]
    # ... 处理逻辑
```

**动态轴标记**：
```python
# 在 kernel 函数签名中标记动态轴
def softmax_kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)):
    # ...
```

**边界处理**：
- 使用 keepdim=True 保持维度，避免维度不匹配
- 使用切片操作进行数据分块

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| API 选择 | 手动实现 | 可使用直接 API | 推荐使用 `pypto.softmax`，如需灵活性则手动实现 |
| dim 参数 | 固定为 -1 | 支持任意 dim | 传递 dim 参数，支持负索引 |
| dtype | FP32 | FP32/FP16 | 如使用直接 API，仅支持 FP32；手动实现可支持 FP16 |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题。所有必需 API 均可用。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| dtype 限制 | 方案 A（直接 API）仅支持 FP32，方案 B 可支持 FP16/BF16 |
| contiguous | 输入 tensor 必须连续，需在 from_torch 前检查 |
| TileShape 配置 | 需根据实际输入 shape 设置合适的 tile shapes |
| 动态轴标记 | 需在 kernel 函数签名中正确标记动态轴 |
| 维度限制 | 方案 B 的 amax/sum 操作仅支持 2-4 维输入 |
| 内存对齐 | 方案 B 需确保尾轴 32 bytes 对齐，次尾轴 ≤ 255 |
| dim 范围 | 需验证 dim 在 [-input.dim, input.dim-1] 范围内 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| softmax API 文档 | `docs/api/operation/pypto-softmax.md` |
| amax API 文档 | `docs/api/operation/pypto-amax.md` |
| exp API 文档 | `docs/api/operation/pypto-exp.md` |
| sum API 文档 | `docs/api/operation/pypto-sum.md` |
| sub API 文档 | `docs/api/operation/pypto-sub.md` |
| div API 文档 | `docs/api/operation/pypto-div.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| 参考实现 | `examples/02_intermediate/operators/softmax/softmax.py` |

---

## 9. 结论

- **可行性**: ✓ 可行
- **推荐方案**: 方案 A - 直接使用 `pypto.softmax` API
- **主要优势**: 
  - PyPTO 已提供优化的 softmax API
  - 实现简单，代码量少
  - 官方保证稳定性和性能
- **备选方案**: 方案 B - 手动实现（如需支持 FP16/BF16 或更多自定义）
- **下一步**: 进入 Stage 3 - Golden 参考实现生成
