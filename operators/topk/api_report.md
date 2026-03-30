# API 探索报告

> **生成时间**: 2026-03-30T01:07:00Z

---

## 1. 概述

### 1.1 输入摘要

- **算子名称**: topk
- **功能**: 返回输入张量在指定维度上最大（或最小）的 k 个元素及其索引
- **输入**: input [b, s, n, d] float32, k (int), dim (int), largest (bool), sorted (bool)
- **输出**: values [b, s, n, k] float32, indices [b, s, n, k] int64
- **动态轴**: b, s, n

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: topk 是选择/排序类操作，不涉及矩阵乘法，属于 Vector 计算类型

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | selection | `values, indices = topk(input, k, dim, largest)` | 在 dim 维度上选取最大/最小的 k 个元素 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | `topk(input, k, dim, largest)` | `pypto.topk` | direct | ⚠ 部分约束 |

### 3.2 API 详细对比

| 特性 | PyTorch torch.topk | PyPTO pypto.topk | 兼容性 |
|------|-------------------|------------------|--------|
| k 参数 | int | int | ✓ |
| dim 参数 | 任意维度 | **仅支持最后一维** | ⚠ 受限 |
| largest 参数 | bool | bool | ✓ |
| sorted 参数 | bool | **不支持** | ⚠ 缺失 |
| 返回值 | (values, indices) | (values, indices) | ✓ |
| indices dtype | int64 | int32 | ⚠ 不同 |

### 3.3 缺失功能处理

| 缺失功能 | 优先级 | 处理方案 |
|----------|--------|----------|
| sorted 参数 | P1 | 若 sorted=True，需对 topk 输出结果额外调用 sort |
| dim 任意维度 | P0 | 若 dim != -1，需先 transpose 将目标维度移到最后，计算后再 transpose 回去 |
| indices int64 | P1 | PyPTO 返回 int32，可使用 pypto.cast 转换 |

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32 | ✓ |
| contiguous | 必须 | — | 需确保 |
| 空Tensor | 不支持 | 非空 | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.topk | dtype | DT_FP32 | ✓ |
| pypto.topk | Shape 维度 | 2-4 维 | ✓ |
| pypto.topk | Shape Size | ≤ INT32_MAX | ✓ |
| pypto.topk | dim | **仅支持 -1 或最后一维** | ⚠ 受限 |
| pypto.topk | k | 1 ≤ k ≤ input.shape[dim] | ✓ |
| pypto.topk | sorted | **不支持** | ⚠ 需额外处理 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

### 5.1 Tiling 策略建议

```python
# 对于 [b, s, n, d] 输入，在 dim=-1 上操作
# 建议的 tiling 配置
pypto.set_vec_tile_shapes(b_tile, s_tile, n_tile, d)
# 其中 d 是完整的最后一维大小，因为 topk 需要完整遍历该维度
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/glm_v4_5/glm_select_experts.py` | models | 高 | 高 | topk API 调用、Tiling 配置、循环结构 |
| `models/deepseek_v32_exp/deepseekv32_lightning_indexer_quant.py` | models | 中 | 高 | torch.topk golden 参考、索引比较方法 |

### 6.2 可复用模式

**来自 glm_select_experts.py**:

```python
# API 调用模式
pypto.set_vec_tile_shapes(view_first, num_expert_group)
_, topk_group_indices = pypto.topk(group_weight, topk_group, -1, True)  # (2, topk_group) int32

# Tiling 策略
# - 在调用 topk 前设置 set_vec_tile_shapes
# - tile shapes 需覆盖完整的操作维度

# Loop 结构
# - 使用 pypto.loop 进行批处理
for bs_index in pypto.loop(bs_loop, name="LOOP_NAME", idx_name="bs_idx"):
    # 分块处理逻辑
```

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| dim 维度 | 固定使用 -1 | 需支持任意 dim | 若 dim != -1，先 transpose 再 topk |
| sorted 参数 | 未使用 | 需支持 | 若 sorted=True，对结果额外排序 |
| 动态轴 | 使用 DYNAMIC | 需支持 b,s,n | 使用 from_torch 的 dynamic_axis 参数 |
| indices dtype | int32 | int64 | 使用 pypto.cast 转换 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无阻断问题 | — | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| dim 限制 | PyPTO topk 仅支持 dim=-1，需通过 transpose 间接支持其他维度 |
| sorted 缺失 | PyPTO topk 不保证排序输出，若需排序需额外调用 sort |
| indices 类型 | PyPTO 返回 int32 索引，PyTorch 返回 int64，需类型转换 |
| Tiling 配置 | topk 需要完整遍历操作维度，tile shape 应覆盖该维度完整大小 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| topk 文档 | `docs/api/operation/pypto-topk.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling API | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 1 | `models/glm_v4_5/glm_select_experts.py` |
| 参考实现 2 | `models/deepseek_v32_exp/deepseekv32_lightning_indexer_quant.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**:
  1. PyPTO topk 仅支持 dim=-1，需通过 transpose 间接支持任意维度
  2. PyPTO topk 不支持 sorted 参数，需额外实现排序逻辑
  3. indices 返回 int32 而非 int64，需类型转换

- **实现策略**:
  1. 若 dim != -1，先 transpose 将目标维度移到最后
  2. 调用 pypto.topk 获取 values 和 indices
  3. 若 sorted=True，对结果进行排序
  4. 若 dim != -1，transpose 回原维度顺序
  5. 若需要 int64 索引，使用 pypto.cast 转换

---

## 10. 实现路线图

### 10.1 核心实现步骤

```
Step 1: 输入处理
  - 获取 input tensor 和参数 (k, dim, largest, sorted)
  - 计算 actual_dim = dim if dim >= 0 else rank + dim

Step 2: 维度转换（若 dim != -1）
  - 若 dim != -1: transpose(input, dim, -1) -> input_transposed

Step 3: TopK 计算
  - pypto.set_vec_tile_shapes(...)
  - values, indices = pypto.topk(input_transposed, k, -1, largest)

Step 4: 排序处理（若 sorted=True）
  - 若 sorted=True: sort(values, indices) by values

Step 5: 维度还原（若 dim != -1）
  - 若 dim != -1: transpose back

Step 6: 类型转换
  - 若需要 int64: cast(indices, DT_INT64)
```

### 10.2 动态轴支持

```python
# 使用 from_torch 的 dynamic_axis 参数
input_pto = pypto.from_torch(input, dynamic_axis=[0, 1, 2])  # b, s, n 动态
```
