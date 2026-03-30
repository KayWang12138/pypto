# PyPTO API Exploration Report: scaled_dot_product_attention

**生成时间**: 2026-03-28T00:00:00Z
**算子名称**: scaled_dot_product_attention
**算子分类**: attention
**复杂度**: hard

---

## 1. 概述

本报告分析 `scaled_dot_product_attention` 算子在 PyPTO 框架中的 API 映射、约束条件和实现策略。

**核心计算**:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

**关键特性**:
- scale: 缩放因子 (默认 1/sqrt(d))
- attn_mask: 支持 bool 和 float 类型掩码
- is_causal: 因果注意力掩码
- dropout_p: Dropout 概率
- numerical_stable_softmax: 数值稳定的 softmax 实现
- 动态轴支持: batch, num_heads, seq_len

---

## 2. 公式分解

### 原子操作序列

| 步骤 | 操作 | 描述 | Shape 变化 |
|------|------|------|-----------|
| 1 | transpose(K) | K 转置 | [N, H, S, E] -> [N, H, E, S] |
| 2 | matmul(Q, K^T) | 计算注意力分数 | [N, H, L, E] @ [N, H, E, S] -> [N, H, L, S] |
| 3 | mul(scores, scale) | 缩放 | [N, H, L, S] * scalar -> [N, H, L, S] |
| 4 | add(scores, attn_bias) | 应用掩码 | [N, H, L, S] + [N, H, L, S] -> [N, H, L, S] |
| 5 | softmax(scores, dim=-1) | 归一化 | [N, H, L, S] -> [N, H, L, S] |
| 6 | dropout(attn_weights, p) (可选) | Dropout | [N, H, L, S] -> [N, H, L, S] |
| 7 | matmul(attn_weights, V) | 计算输出 | [N, H, L, S] @ [N, H, S, Ev] -> [N, H, L, Ev] |

---

## 3. API 映射

### 核心操作映射表

| 操作 | PyPTO API | 参数 | 约束 | 可行性 | 置信度 |
|------|-----------|------|------|--------|--------|
| transpose | `pypto.transpose(input, dim0, dim1)` | dim0=2, dim1=3 | 4D: 支持(2,3)交换 | ✓ 支持 | ✓ 高 |
| matmul (QK^T) | `pypto.matmul(input, mat2, out_dtype, b_trans=True)` | b_trans=True | 需要 cube tiling | ✓ 支持 | ✓ 高 |
| mul (scale) | `pypto.mul(input, other)` | other=scale_factor | - | ✓ 支持 | ✓ 高 |
| add (mask) | `pypto.add(input, other)` | other=attn_bias | - | ✓ 支持 | ✓ 高 |
| softmax | `pypto.softmax(input, dim)` | dim=-1 | **仅支持 DT_FP32** | ⚠ 需 cast | ✓ 高 |
| matmul (Attn@V) | `pypto.matmul(input, mat2, out_dtype)` | - | 需要 cube tiling | ✓ 支持 | ✓ 高 |

### 特殊操作映射表

| 操作 | PyPTO API | 参数 | 约束 | 可行性 | 置信度 |
|------|-----------|------|------|--------|--------|
| causal mask | `pypto.tril` 或组合实现 | - | PyPTO 无直接 tril API | ⚠ 需组合 | ⚠ 中 |
| bool mask | `pypto.where` 或 `pypto.add` | - | bool->float 转换 | ⚠ 需转换 | ⚠ 中 |
| dropout | **不支持** | - | PyPTO 无 dropout API | ✗ 不支持 | ✓ 高 |
| dtype cast | `pypto.cast(input, dtype)` | dtype=DT_FP32/DT_FP16/DT_BF16 | - | ✓ 支持 | ✓ 高 |

---

## 4. 约束分析

### 4.1 入口约束 (from_torch)

| 约束项 | 要求 | 来源 |
|--------|------|------|
| dtype | DT_FP16 / DT_BF16 / DT_FP32 / DT_INT8-64 / BOOL | `docs/api/others/pypto-from_torch.md` |
| shape | 非空 Tensor | `docs/api/others/pypto-from_torch.md` |
| contiguous | 必须连续 | `docs/api/others/pypto-from_torch.md` |

### 4.2 API 约束

#### pypto.matmul 约束

| 约束项 | 要求 | 影响 |
|--------|------|------|
| dtype | DT_FP16 / DT_BF16 / DT_FP32 / DT_INT8 | - |
| 维度 | 2D/3D/4D | ✓ 支持 4D |
| TileShape | **必须**调用 `set_cube_tile_shapes` | **阻塞** |
| format | ND/NZ | - |

**关键约束**:
- 调用 matmul 前必须设置 `pypto.set_cube_tile_shapes(m, k, n)`
- 3D/4D matmul 需要同时设置 `pypto.set_vec_tile_shapes`

#### pypto.softmax 约束

| 约束项 | 要求 | 影响 |
|--------|------|------|
| **dtype** | **仅支持 DT_FP32** | **阻塞** |

**关键问题**:
- PyPTO softmax **仅支持 FP32**，需要：
  - 输入 FP16/BF16 -> cast to FP32
  - softmax 计算
  - cast back to FP16/BF16

#### pypto.transpose 约束

| 约束项 | 要求 | 影响 |
|--------|------|------|
| 维度 | 2D-5D | ✓ 支持 4D |
| 4D 交换 | 支持: (0,2), (1,3), (2,3), (1,2) | ✓ 支持 (2,3) |
| **不支持** | (0,3), (0,1) | 需规避 |

**K^T 转置分析**:
- K shape: [N, H, S, E]
- 需要转置为: [N, H, E, S]
- 交换维度 2 和 3: **支持** ✓

### 4.3 Tiling 约束

#### Cube Tiling (matmul)

**必须调用**: `pypto.set_cube_tile_shapes(m, k, n)`

| 参数 | 说明 | 约束 |
|------|------|------|
| m | L0/L1 上的 M 维 TileShape | 列表 [m_l0, m_l1] |
| k | L0/L1 上的 K 维 TileShape | 列表 [k_l0, k_l1] |
| n | L0/L1 上的 N 维 TileShape | 列表 [n_l0, n_l1] |

**参考值** (from examples):
```python
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
```

#### Vector Tiling (softmax, add, mul, transpose)

**必须调用**: `pypto.set_vec_tile_shapes(*args)`

| 参数 | 说明 | 约束 |
|------|------|------|
| *args | 每个维度的 TileShape | 最多 4 个，每个 > 0 |

**参考值** (from examples):
```python
# 4D tensor [batch, heads, seq, dim]
pypto.set_vec_tile_shapes(1, 8, 16, 64)
```

---

## 5. 实现策略

### 5.1 数据类型处理

**Softmax 精度问题**:

由于 PyPTO softmax **仅支持 FP32**，需要：

```python
# 方案 1: 完整精度保护
scores_fp32 = pypto.cast(scores, pypto.DT_FP32)
attn_weights_fp32 = pypto.softmax(scores_fp32, dim=-1)
attn_weights = pypto.cast(attn_weights_fp32, pypto.DT_BF16)

# 方案 2: 混合精度 (参考 GLM-4.5 实现)
# matmul 输出 FP32，直接 softmax
scores_fp32 = pypto.matmul(q, k_t, pypto.DT_FP32, b_trans=True)  # out_dtype=FP32
scores_scaled = pypto.mul(scores_fp32, scale_factor)
attn_weights_fp32 = pypto.softmax(scores_scaled, dim=-1)
attn_weights = pypto.cast(attn_weights_fp32, pypto.DT_BF16)
output = pypto.matmul(attn_weights, v, pypto.DT_BF16)
```

**推荐**: 方案 2 (参考 GLM-4.5 实现)

### 5.2 Causal Mask 实现

**问题**: PyPTO 无直接 `tril` API

**解决方案**:

```python
# 方案 1: 预计算 causal mask (推荐)
# 在 Python 端预计算下三角掩码，传入 kernel
causal_mask = torch.tril(torch.ones(L, S))

# 方案 2: 组合实现 (复杂)
# 使用 arange + broadcast + compare
# 但效率较低，不推荐
```

**推荐**: 方案 1 (预计算)

### 5.3 Attention Mask 处理

```python
# bool mask: True -> 0.0, False -> -inf
if attn_mask.dtype == bool:
    attn_bias = pypto.where(attn_mask, 0.0, float('-inf'))
else:
    attn_bias = attn_mask

scores = pypto.add(scores, attn_bias)
```

### 5.4 Tiling 策略

**基于参考实现** (GLM-4.5 和 examples):

```python
# Cube tiling for matmul
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

# Vector tiling for elementwise ops
pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)
```

**动态轴处理**:
- batch 维度: 使用 `pypto.loop` 循环
- heads 维度: 可并行或循环
- seq_len 维度: 使用 tiling 切分

### 5.5 Dropout 处理

**问题**: PyPTO **不支持** dropout API

**解决方案**:
1. **P2 优先级**: 在第一个版本中**不实现** dropout
2. 如果必须实现，需要自定义随机数生成和 mask (复杂，不推荐)

---

## 6. 参考实现

### 参考实现 1: GLM-4.5 Attention (Paged Attention)

**文件**: `models/glm_v4_5/glm_attention.py`

**相似度**: ⭐⭐⭐⭐ (4/5)
**置信度**: ✓ 高
**可复用点**:
1. **Flash Attention 实现**: Online softmax + 分块计算
2. **数值稳定性**: FP32 累加器 + max 减法
3. **动态轴处理**: `pypto.loop` + `pypto.view`
4. **Tiling 配置**: Cube + Vector tiling 示例
5. **matmul + softmax + matmul 结构**: 完整的 attention 计算流

**代码片段**:
```python
# Flash Attention core logic (from glm_attention.py)
sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
sij_scale = pypto.mul(sij, softmax_scale)
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
tsub = pypto.sub(sij_scale, tilda_mij)
tilda_pij = pypto.exp(tsub)
tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
# ... online update logic ...
oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)
```

**差异点**:
- GLM-4.5 是 **Paged Attention** (需要 block_table)
- 我们的实现是 **标准 Attention** (连续 K/V cache)
- 需要简化 block_table 逻辑

### 参考实现 2: Scaled Dot-Product Attention Example

**文件**: `examples/03_advanced/advanced_nn/attention/attention.py`

**相似度**: ⭐⭐⭐⭐⭐ (5/5)
**置信度**: ✓ 高
**可复用点**:
1. **标准 attention 实现**: Q @ K^T -> softmax -> @ V
2. **API 调用模式**: transpose + matmul + mul + softmax + matmul
3. **Tiling 配置**: 完整的 cube + vector tiling 示例
4. **完整测试**: golden reference + 精度验证

**代码片段**:
```python
# Standard attention (from attention.py)
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)

k_t = pypto.transpose(k, 2, 3)
scores = pypto.matmul(q, k_t, out_dtype=pypto.DT_BF16)
scores_scaled = pypto.mul(scores, scale)
attn_weights = pypto.softmax(scores_scaled, dim=-1)
output = pypto.matmul(attn_weights, v, out_dtype=pypto.DT_BF16)
```

**注意**: 示例中 softmax 直接使用 BF16 输入，**违反了 softmax 仅支持 FP32 的约束**，需要修正。

---

## 7. 风险与限制

### 高风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **Softmax 仅支持 FP32** | 阻塞 | matmul 输出 FP32，或 cast BF16->FP32 |
| **Dropout 不支持** | 功能缺失 | P2 优先级，暂不实现 |
| **Causal mask 无直接 API** | 实现复杂 | 预计算 mask tensor |

### 中风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Transpose 4D 约束 | 限制某些转置 | 使用支持的 (2,3) 交换 |
| 动态轴循环性能 | 性能影响 | 优化 tiling 配置 |

### 低风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Bool mask 转换 | 需额外处理 | 使用 where 或 add |

---

## 8. 证据索引

### API 文档

| API | 文件路径 | 关键信息 |
|-----|----------|----------|
| matmul | `docs/api/operation/pypto-matmul.md` | 需要 cube tiling, 支持 4D |
| softmax | `docs/api/operation/pypto-softmax.md` | **仅支持 DT_FP32** |
| transpose | `docs/api/operation/pypto-transpose.md` | 4D: 支持 (2,3) 交换 |
| mul | `docs/api/operation/pypto-mul.md` | 支持广播 |
| add | `docs/api/operation/pypto-add.md` | 支持广播 |
| set_cube_tile_shapes | `docs/api/config/pypto-set_cube_tile_shapes.md` | matmul 前必须调用 |
| set_vec_tile_shapes | `docs/api/config/pypto-set_vec_tile_shapes.md` | elementwise 前调用 |

### 参考实现

| 实现 | 文件路径 | 相似度 | 可复用点 |
|------|----------|--------|----------|
| GLM-4.5 Attention | `models/glm_v4_5/glm_attention.py` | 4/5 | Flash Attention, Online softmax, Tiling |
| Standard Attention | `examples/03_advanced/advanced_nn/attention/attention.py` | 5/5 | API 调用模式, Tiling 配置 |

---

## 9. 结论

### 可行性: ✓ 可行

**核心功能实现**:
- ✓ 基础 attention (Q, K, V): 完全支持
- ✓ Scale: 完全支持
- ⚠ Softmax: 需要精度处理 (FP32)
- ⚠ Attention mask: 需要类型转换
- ⚠ Causal mask: 需要预计算或组合实现
- ✗ Dropout: **不支持** (P2 优先级，可暂缓)

### 实现建议

1. **精度处理**:
   - matmul 输出 FP32 (推荐)
   - softmax 使用 FP32
   - 输出 cast 回 BF16/FP16

2. **Tiling 策略**:
   - 参考 `examples/03_advanced/advanced_nn/attention/attention.py`
   - Cube: `[64, 64] x [64, 64] x [64, 64]`
   - Vector: `[1, 8, 16, HEAD_DIM]`

3. **Causal Mask**:
   - Python 端预计算 `torch.tril`
   - 作为参数传入 kernel

4. **Dropout**:
   - **不实现** (标记为 P2/P3)
   - 文档中说明限制

### 下一步行动

1. ✅ Stage 2 完成: api_report.md 已生成
2. → Stage 3: 生成 `scaled_dot_product_attention_golden.py`
3. → Stage 4: 生成 `design.md` (tiling 策略, loop 结构)
4. → Stage 5: 生成 `scaled_dot_product_attention_impl.py`

---

**生成时间**: 2026-03-28T00:00:00Z
**状态**: Stage 2 完成
