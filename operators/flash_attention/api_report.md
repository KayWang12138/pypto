# PyPTO API Exploration Report: flash_attention

**生成时间**: 2026-03-29T00:00:00Z
**算子名称**: flash_attention
**算子分类**: attention
**复杂度**: hard

---

## 1. 概述

### 1.1 输入摘要

Flash Attention 是一种内存高效的注意力计算方法，核心思想是通过分块计算和在线 Softmax 算法减少 HBM 访问次数。

**数学公式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**关键特性**:
1. **tiling_strategy**: 将 Q/K/V 分块，每次只加载一个 tile 到 SRAM
2. **online_softmax**: 使用在线 Softmax 算法，增量更新 max 值和累加值
3. **memory_efficient**: 避免存储完整 N^2 的注意力矩阵
4. **causal_mask**: 因果注意力掩码
5. **attn_mask**: 自定义注意力掩码

### 1.2 算子分类

- **类型**: 混合 (Cube + Vector)
- **判断依据**:
  - 包含 matmul 操作 (Q @ K^T, P @ V) → 需要 Cube Tiling
  - 包含逐元素操作 (scale, exp, sub, add) 和归约操作 (amax, sum) → 需要 Vector Tiling
  - Flash Attention 的核心是分块计算 + 在线 Softmax，两者都需要

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | shape | K^T | K 转置: [N, H, S, d] -> [N, H, d, S] |
| 2 | matmul | Q @ K^T | 计算注意力分数: [N, H, L, d] @ [N, H, d, S] -> [N, H, L, S] |
| 3 | elementwise | scores * scale | 缩放: scores / sqrt(d) |
| 4 | elementwise | scores + mask | 应用掩码 (可选) |
| 5 | reduction | rowmax(scores) | 在线 Softmax: 计算 max 值 |
| 6 | elementwise | scores - max | 数值稳定减法 |
| 7 | elementwise | exp(scores - max) | 计算指数 |
| 8 | reduction | rowsum(exp_scores) | 在线 Softmax: 计算累加值 |
| 9 | elementwise | exp_scores / sum | 归一化 |
| 10 | matmul | attn_weights @ V | 计算输出: [N, H, L, S] @ [N, H, S, d] -> [N, H, L, d] |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | K^T | `pypto.transpose(input, 2, 3)` | direct | ✓ 4D 支持 (2,3) 交换 |
| 2 | Q @ K^T | `pypto.matmul(q, k_t, out_dtype, b_trans=True)` | direct | ✓ 支持 4D |
| 3 | scores * scale | `pypto.mul(scores, scale)` | direct | ✓ 支持广播 |
| 4 | scores + mask | `pypto.add(scores, mask)` | direct | ✓ 支持广播 |
| 5 | rowmax(scores) | `pypto.amax(scores, dim=-1, keepdim=True)` | direct | ✓ 支持归约 |
| 6 | scores - max | `pypto.sub(scores, max_val)` | direct | ✓ 支持广播 |
| 7 | exp(scores) | `pypto.exp(scores)` | direct | ✓ 支持 2-4D |
| 8 | rowsum(exp) | `pypto.sum(exp_scores, dim=-1, keepdim=True)` | direct | ✓ 支持归约 |
| 9 | exp / sum | `pypto.div(exp_scores, sum_val)` | direct | ✓ 支持广播 |
| 10 | attn @ V | `pypto.matmul(attn, v, out_dtype)` | direct | ✓ 支持 4D |
| 11 | dtype cast | `pypto.cast(input, dtype)` | direct | ✓ 支持 FP32/BF16/FP16 |
| 12 | max(old, new) | `pypto.maximum(old_max, new_max)` | direct | ✓ 支持逐元素 |

### 3.2 在线 Softmax 实现

Flash Attention 的核心是在线 Softmax 算法，PyPTO 需要以下 API 组合实现:

```python
# 在线 Softmax 更新公式
# m_new = max(m_old, m_ij)
# l_new = exp(m_old - m_new) * l_old + exp(m_ij - m_new) * l_ij
# O_new = (exp(m_old - m_new) * l_old / l_new) * O_old + (exp(m_ij - m_new) / l_new) * P_ij @ V_j

# PyPTO 实现
m_new = pypto.maximum(m_old, m_ij)                    # max 更新
exp_old = pypto.exp(pypto.sub(m_old, m_new))          # exp(m_old - m_new)
exp_new = pypto.exp(pypto.sub(m_ij, m_new))           # exp(m_ij - m_new)
l_new = pypto.add(pypto.mul(exp_old, l_old),          # 累加值更新
                  pypto.mul(exp_new, l_ij))
o_scale = pypto.div(pypto.mul(exp_old, l_old), l_new) # 输出缩放因子
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32 | FP32/BF16/FP16 | ✓ |
| contiguous | 必须 | — | ✓ 需确保 |
| shape | 非空 Tensor | [N, H, L, d] | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| matmul | TileShape | **必须**调用 `set_cube_tile_shapes` | ⚠ 必须设置 |
| matmul | 4D 支持 | 支持 2D/3D/4D | ✓ |
| softmax | dtype | **仅支持 DT_FP32** | ⚠ 需要 cast |
| transpose | 4D 交换 | 支持 (2,3)，不支持 (0,1) 和 (0,3) | ✓ K^T 使用 (2,3) |
| amax/sum | TileShape | 尾轴 32 字节对齐 | ⚠ 需注意 |
| exp | dtype | FP16/BF16/FP32 | ✓ |

### 4.3 关键约束详情

**Softmax 约束 (关键)**:
- PyPTO softmax **仅支持 DT_FP32**
- BF16/FP16 输入必须先 cast 到 FP32，计算后再 cast 回原类型

**Transpose 4D 约束**:
- 只支持: (0,2), (1,3), (2,3), (1,2)
- 不支持: (0,3), (0,1)
- K^T 转置: [N, H, S, d] -> [N, H, d, S] 使用 (2,3) ✓

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Cube (matmul) | `pypto.set_cube_tile_shapes([m, m], [k, k], [n, n])` |
| Vector (其他) | `pypto.set_vec_tile_shapes(*args)` |

### 5.1 Cube Tiling 配置

```python
# 参考 GLM-4.5 Flash Attention 实现
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```

### 5.2 Vector Tiling 配置

```python
# 4D tensor [batch, heads, seq, dim]
pypto.set_vec_tile_shapes(1, 8, 16, 128)  # 根据 head_dim 调整
```

### 5.3 Flash Attention 分块策略

参考 GLM-4.5 实现:
- **Br (Q 分块)**: 128 (mL0/mL1)
- **Bc (K/V 分块)**: 512 (s2_tile)
- **循环结构**: 外层遍历 K/V 块，内层遍历 Q 块

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/glm_v4_5/glm_attention.py` | models | 5/5 | 高 | 在线 Softmax, 分块计算, Tiling 配置, Loop 结构 |
| `examples/03_advanced/advanced_nn/attention/attention.py` | examples | 4/5 | 高 | 标准 Attention API 调用, Tiling 配置 |
| `operators/scaled_dot_product_attention/` | operators | 4/5 | 高 | 项目内参考实现, 完整测试流程 |

### 6.2 可复用模式

- **API 调用模式**:
  - `pypto.matmul(q, k_t, pypto.DT_FP32, b_trans=True)` - 使用 b_trans 避免显式转置
  - `pypto.amax(scores, dim=-1, keepdim=True)` - 在线 Softmax max 计算
  - `pypto.sum(exp_scores, dim=-1, keepdim=True)` - 在线 Softmax sum 计算
  - `pypto.cast(x, dtype)` - 精度转换

- **Tiling 策略**:
  ```python
  # Cube tiling for matmul
  pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

  # Vector tiling for elementwise/reduction
  pypto.set_vec_tile_shapes(1, 8, 16, HEAD_DIM)
  ```

- **Loop 结构** (参考 GLM-4.5):
  ```python
  for b_idx in pypto.loop(batch_size, name="LOOP_b"):
      for s1_idx in pypto.loop(seq_len_q, name="LOOP_s1"):
          for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", unroll_list=[8, 4, 2, 1]):
              # 分块计算 + 在线 Softmax 更新
              if pypto.is_loop_begin(s2_idx):
                  # 首次迭代: 初始化
              else:
                  # 后续迭代: 在线更新
              if pypto.is_loop_end(s2_idx):
                  # 最后迭代: 归一化输出
  ```

- **边界处理**:
  - 使用 `valid_shape` 参数处理尾块
  - 使用 `pypto.min()` 计算实际块大小

### 6.3 差异分析

| 差异点 | GLM-4.5 实现 | Flash Attention 需求 | 调整建议 |
|--------|--------------|----------------------|----------|
| Paged KV Cache | 使用 block_table 管理分页 KV | 标准 Flash Attention (连续 KV) | 简化 block_table 逻辑，直接使用连续 K/V |
| GQA 支持 | 支持 Grouped Query Attention | 基础版本暂不支持 | 移除 GQA 相关逻辑 |
| 动态轴 | 使用 pypto.DYNAMIC | 需要支持动态 batch/seq | 保留动态轴支持 |
| 在线 Softmax | 完整实现在线更新算法 | 需要完整复用 | 直接复用在线更新逻辑 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | - | - |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| Softmax 仅支持 FP32 | 必须在 softmax 前将 BF16/FP16 cast 到 FP32，计算后再 cast 回原类型 |
| matmul 需要 Tiling | 调用 matmul 前必须调用 `set_cube_tile_shapes` |
| 在线 Softmax 复杂度 | 需要正确处理循环边界条件和状态更新 |
| 动态轴处理 | 使用 `pypto.loop` + `pypto.view` 处理动态 batch/seq_len |
| 尾块对齐 | Vector Tiling 尾轴需要 32 字节对齐 |
| Causal Mask | 需要预计算或在循环内动态生成 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 列表 | `docs/api/operation/index.md` |
| matmul 文档 | `docs/api/operation/pypto-matmul.md` |
| softmax 文档 | `docs/api/operation/pypto-softmax.md` |
| transpose 文档 | `docs/api/operation/pypto-transpose.md` |
| mul 文档 | `docs/api/operation/pypto-mul.md` |
| add 文档 | `docs/api/operation/pypto-add.md` |
| sub 文档 | `docs/api/operation/pypto-sub.md` |
| exp 文档 | `docs/api/operation/pypto-exp.md` |
| amax 文档 | `docs/api/operation/pypto-amax.md` |
| sum 文档 | `docs/api/operation/pypto-sum.md` |
| div 文档 | `docs/api/operation/pypto-div.md` |
| maximum 文档 | `docs/api/operation/pypto-maximum.md` |
| cast 文档 | `docs/api/operation/pypto-cast.md` |
| triu 文档 | `docs/api/operation/pypto-triu.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Cube Tiling | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| GLM-4.5 Flash Attention | `models/glm_v4_5/glm_attention.py` |
| Standard Attention | `examples/03_advanced/advanced_nn/attention/attention.py` |
| 项目参考实现 | `operators/scaled_dot_product_attention/` |

---

## 9. 结论

- **可行性**: ✓ 可行
- **主要问题**: 无阻断问题，需要注意 softmax 的 FP32 约束和在线 Softmax 的实现复杂度

### 实现建议

1. **精度处理**:
   - matmul 输出 FP32 (推荐)
   - softmax 使用 FP32
   - 输出 cast 回 BF16/FP16

2. **Tiling 策略**:
   - 参考 GLM-4.5 的分块配置
   - Cube: `[128, 128] x [128, 128] x [128, 128]`
   - Vector: `[1, 8, 16, HEAD_DIM]`

3. **在线 Softmax**:
   - 参考 GLM-4.5 的在线更新算法
   - 使用 `pypto.is_loop_begin()` 和 `pypto.is_loop_end()` 处理边界

4. **Causal Mask**:
   - 使用 `pypto.triu()` 生成上三角掩码
   - 或在 Python 端预计算后传入

5. **分阶段实现**:
   - P0: 基础 Flash Attention (无 mask)
   - P1: 添加 causal mask 支持
   - P1: 添加自定义 attn_mask 支持

### 下一步行动

1. ✅ Stage 2 完成: api_report.md 已生成
2. → Stage 3: 生成 `flash_attention_golden.py`
3. → Stage 4: 生成 `design.md`
4. → Stage 5: 生成 `flash_attention_impl.py`

---

**生成时间**: 2026-03-29T00:00:00Z
**状态**: Stage 2 完成
