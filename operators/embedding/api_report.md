# PyPTO API 探索报告：embedding

## 1. 概述

| 项目 | 内容 |
|------|------|
| 算子名称 | embedding |
| 类别 | embedding (查表类) |
| 复杂度 | medium |
| 可行性 | 可行 |
| 探索时间 | 2026-03-30 |

### 核心结论

embedding 算子可通过 PyPTO 的 `pypto.gather` API 实现。该 API 支持动态轴和必要的索引操作，满足 embedding 查表的核心需求。padding_idx 功能可通过 `pypto.eq` + `pypto.unsqueeze` + `pypto.where` 组合实现。

---

## 2. 计算逻辑分解

### 2.1 公式分解

```
embedding 查表操作:
  output[i, j, :] = weight[indices[i, j], :]

可选 padding_idx 处理:
  if padding_idx is not None:
    mask = (indices == padding_idx).unsqueeze(-1)  # [batch, seq] -> [batch, seq, 1]
    output = output.masked_fill(mask, 0.0)
```

### 2.2 原子操作序列

| 步骤 | 操作类型 | PyPTO API | 输入 | 输出 | 说明 |
|------|----------|-----------|------|------|------|
| 1 | index | `pypto.gather` | weight, indices | output | 核心查表操作 |
| 2 | compare | `pypto.eq` | indices, padding_idx | mask | padding 检测 (P1) |
| 3 | shape | `pypto.unsqueeze` | mask | mask_3d | 维度扩展 (P1) |
| 4 | select | `pypto.where` | mask_3d, 0.0, output | output_final | 条件填充 (P1) |

---

## 3. API 映射

### 3.1 核心 API：pypto.gather

| 项目 | 内容 |
|------|------|
| API | `pypto.gather(input, dim, index)` |
| 映射关系 | input=weight, dim=0, index=indices |
| 支持状态 | 支持 |
| 置信度 | 高 |

**API 约束检查**：

| 约束项 | 要求 | embedding 需求 | 状态 |
|--------|------|----------------|------|
| input dtype | FP32/FP16/BF16/INT16/INT32 | FP32 | OK |
| index dtype | INT32/INT64 | INT64 | OK |
| input shape | 2-4维 | 2维 [vocab_size, embed_dim] | OK |
| index shape | 2-4维，与 input 维度相同 | 2维 [batch, seq] | 需适配 |
| shape size | <= INT32_MAX | 满足 | OK |
| dim 轴不可切 | viewshape[dim] >= max(input.shape[dim], index.shape[dim]) | 需在 tiling 中处理 | 注意 |

**关键约束说明**：
1. `index.dim = input.dim`：indices 需要与 weight 维度一致，embedding 需要扩展 indices 到 2D
2. `index.shape[i] <= input.shape[i] (i != dim)`：batch 和 seq 维度需满足约束
3. dim=0 轴不可切：vocab_size 轴需要全载

### 3.2 padding_idx 支持 API

#### pypto.eq

| 项目 | 内容 |
|------|------|
| API | `pypto.eq(input, other)` |
| 用途 | 检测 indices == padding_idx |
| 支持状态 | 支持 |
| 约束 | input/other 类型一致，支持 1D 广播 |

**注意**：`pypto.eq` 要求 input 为浮点类型 (FP16/BF16/FP32)，indices 为 INT64。需要先 cast indices 为 FP32 再比较。

#### pypto.unsqueeze

| 项目 | 内容 |
|------|------|
| API | `pypto.unsqueeze(input, dim)` |
| 用途 | 扩展 mask 维度 [batch, seq] -> [batch, seq, 1] |
| 支持状态 | 支持 |

#### pypto.where

| 项目 | 内容 |
|------|------|
| API | `pypto.where(condition, input, other)` |
| 用途 | 条件填充 padding 位置为 0 |
| 支持状态 | 支持 |
| 约束 | condition 为 BOOL 类型 |

### 3.3 辅助 API

#### pypto.cast

| 项目 | 内容 |
|------|------|
| API | `pypto.cast(input, dtype)` |
| 用途 | INT64 -> FP32 转换用于 eq 比较 |
| 支持状态 | 支持 |

---

## 4. 约束分析

### 4.1 from_torch 入口约束

| 约束项 | 要求 | embedding 需求 | 状态 |
|--------|------|----------------|------|
| tensor 类型 | torch.Tensor | 满足 | OK |
| contiguous | 必须连续 | 需保证输入连续 | OK |
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32/INT64 | OK |

### 4.2 gather API 约束

| 约束项 | 说明 | 影响 |
|--------|------|------|
| dim 轴不可切 | vocab_size 轴需全载 | Tiling 需保证 viewshape[0] >= vocab_size |
| index 维度匹配 | index.dim = input.dim | indices 需保持 2D |
| 索引合法性 | index 值 < input.shape[dim] | 用户需保证 indices < vocab_size |

### 4.3 Tiling 约束

**Vector 类型**（非 matmul 操作）：
- 使用 `pypto.set_vec_tile_shapes()`
- TileShape 维度与输出一致
- 每维 > 0，最多 4 维

**embedding Tiling 策略**：
```
输入 weight: [vocab_size, embed_dim]  (静态)
输入 indices: [batch, seq]  (动态)
输出 output: [batch, seq, embed_dim]  (动态)

Tiling 切分:
- vocab_size 轴 (dim=0) 不可切，需全载
- batch, seq, embed_dim 轴可切
- TileShape: [tile_batch, tile_seq, tile_embed_dim]
```

---

## 5. 风险与替代方案

### 5.1 已识别风险

| 风险 | 级别 | 说明 | 缓解措施 |
|------|------|------|----------|
| gather dim=0 不可切 | 中 | vocab_size 轴需全载，可能影响大词表性能 | 分块处理 batch/seq 维度 |
| eq 不支持 INT64 输入 | 低 | 需要 cast indices 为 FP32 | 使用 pypto.cast 转换 |
| gather 要求 2-4 维 | 低 | indices 和 weight 需为 2D | 确保输入 shape 正确 |

### 5.2 替代方案

| 方案 | API | 说明 |
|------|-----|------|
| 主方案 | pypto.gather | 直接索引查表，推荐使用 |
| 备选方案 | pypto.index_select | 支持 1-2 维 index，但语义略有不同 |

**方案对比**：
- `pypto.gather`：index shape 与 output shape 一致，更符合 embedding 语义
- `pypto.index_select`：index 只能是 1D 或 2D，输出 shape 在 dim 维度与 index 长度一致

---

## 6. 参考实现

### 6.1 官方参考

| 路径 | 相似度 | 置信度 | 可复用点 |
|------|--------|--------|----------|
| models/glm_v4_5/glm_select_experts.py | 高 | 高 | gather 用法、动态轴、tiling 配置、循环结构 |
| examples/02_intermediate/operators/softmax/softmax.py | 中 | 高 | 动态轴标记、jit 装饰器、loop 结构 |

### 6.2 关键代码模式

**gather 用法参考** (来自 glm_select_experts.py):
```python
# tw_gather = pypto.gather(topk_weights, 1, topk_ids)  # (bs, 8)
tw_gather = pypto.gather(topk_weights, 1, topk_ids)
```

**动态轴标记参考**:
```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)):
    # kernel implementation
```

**Tiling 配置参考**:
```python
pypto.set_vec_tile_shapes(1, 4, 1, 64)  # 根据输出 shape 配置
```

**Loop 结构参考**:
```python
for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
    b_offset = idx * tile_b
    # process tile
```

---

## 7. 实现建议

### 7.1 实现策略

1. **核心查表**：使用 `pypto.gather(weight, 0, indices)` 实现 embedding 查表
2. **动态轴**：batch 和 seq 维度标记为 DYNAMIC
3. **Tiling**：vocab_size 轴不可切，按 batch/seq 分块处理
4. **padding_idx**：通过 eq + unsqueeze + where 组合实现

### 7.2 实现步骤

```
1. 输入处理:
   - indices: [batch, seq] int64, 动态轴 [0, 1]
   - weight: [vocab_size, embed_dim] float32, 静态

2. 核心计算:
   - output = pypto.gather(weight, dim=0, index=indices)

3. padding_idx 处理 (可选):
   - indices_fp32 = pypto.cast(indices, pypto.DT_FP32)
   - mask = pypto.eq(indices_fp32, float(padding_idx))  # [batch, seq]
   - mask_3d = pypto.unsqueeze(mask, dim=-1)  # [batch, seq, 1]
   - output = pypto.where(mask_3d, 0.0, output)

4. 输出:
   - output: [batch, seq, embed_dim] float32, 动态轴 [0, 1]
```

### 7.3 Tiling 配置建议

```python
# 输出 shape: [batch, seq, embed_dim]
# vocab_size 轴 (dim=0 of weight) 不可切
pypto.set_vec_tile_shapes(tile_batch, tile_seq, tile_embed_dim)
# 例如：
pypto.set_vec_tile_shapes(1, 128, 256)
```

---

## 8. 证据索引

| 信息 | 来源 | 路径/引用 |
|------|------|-----------|
| gather API 定义 | 官方文档 | docs/api/operation/pypto-gather.md |
| gather 约束 | 官方文档 | docs/api/operation/pypto-gather.md L43-50 |
| from_torch 约束 | 官方文档 | docs/api/others/pypto-from_torch.md |
| set_vec_tile_shapes | 官方文档 | docs/api/config/pypto-set_vec_tile_shapes.md |
| gather 使用示例 | 官方示例 | models/glm_v4_5/glm_select_experts.py L164 |
| 动态轴示例 | 官方示例 | examples/02_intermediate/operators/softmax/softmax.py L92-94 |
| where API | 官方文档 | docs/api/operation/pypto-where.md |
| eq API | 官方文档 | docs/api/operation/pypto-eq.md |
| unsqueeze API | 官方文档 | docs/api/operation/pypto-unsqueeze.md |
| cast API | 官方文档 | docs/api/operation/pypto-cast.md |

---

## 9. 结论

### 9.1 可行性评估

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 核心功能 | 可行 | pypto.gather 可实现 embedding 查表 |
| 动态轴 | 可行 | batch 和 seq 可标记为 DYNAMIC |
| padding_idx | 可行 | 通过 eq + where 组合实现 |
| Tiling | 可行 | 需注意 vocab_size 轴不可切 |

### 9.2 实现优先级

| 功能 | 优先级 | API | 状态 |
|------|--------|-----|------|
| 核心 embedding 查表 | P0 | pypto.gather | 可行 |
| padding_idx 支持 | P1 | eq + unsqueeze + where | 可行 |
| 动态轴支持 | P0 | from_torch(dynamic_axis=...) | 可行 |

### 9.3 风险等级

**总体风险：低**

主要风险点：
1. vocab_size 轴不可切，对大词表场景可能有性能影响
2. eq 不直接支持 INT64，需要 cast 转换

### 9.4 建议

1. **推荐实现**：使用 pypto.gather 作为核心 API
2. **性能优化**：优先按 batch/seq 维度分块，保证 vocab_size 全载
3. **测试覆盖**：重点测试动态轴、padding_idx、大词表场景
