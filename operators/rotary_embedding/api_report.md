# API 探索报告

> **生成时间**: 2026-03-28T00:00:00Z

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

- **算子名称**: rotary_embedding
- **计算逻辑**: Rotary Position Embedding (RoPE) - 通过旋转矩阵编码位置信息
- **输入**: x [batch, seq_len, num_heads, head_dim], cos [seq_len, head_dim], sin [seq_len, head_dim]
- **输出**: y [batch, seq_len, num_heads, head_dim]

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 核心计算为逐元素乘法和加法运算，不涉及矩阵乘法，公式: y = x * cos + rotate_half(x) * sin

  - 无 matmul 操作 → Vector 类型
  - 仅需设置 `set_vec_tile_shapes()`

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | slice | x1 = x[..., :head_dim//2] | 分割输入张量为前后两半 |
| 2 | slice | x2 = x[..., head_dim//2:] | 分割输入张量为前后两半 |
| 3 | neg | -x2 | 对后半部分取负 |
| 4 | concat | [-x2, x1] | 将负的后半部分与前半部分拼接 |
| 5 | mul | x * cos | 输入与 cos 相乘 |
| 6 | mul | rotated_x * sin | 旋转后的x与 sin 相乘 |
| 7 | add | x * cos + rotated_x * sin | 两个乘积结果相加 |

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | x[..., :d//2] | `pypto.view` | direct | ✓ |
| 2 | x[..., d//2:] | `pypto.view` | direct | ✓ |
| 3 | -x2 | `pypto.neg` | direct | ✓ |
| 4 | [-x2, x1] | `pypto.concat` | direct | ✓ |
| 5 | x * cos | `pypto.mul` | direct | ✓ |
| 6 | rotated_x * sin | `pypto.mul` | direct | ✓ |
| 7 | x * cos + rotated_x * sin | `pypto.add` | direct | ✓ |

| 8 | cos/sin 生成 (可选) | cos(freq), sin(freq) | `pypto.cos`, `pypto.sin` | substitute | ✓ |

| 9 | freq 生成 | freq = 1/(base^(2i/d)) | `pypto.pow` + `pypto.div` | substitute | ⚠ |

**说明**:
- 步骤 1-7 为核心计算路径，所有 API 直接支持，dtype 为 FP16/BF16/FP32
- 步骤 8-9 为可选路径，若 cos/sin 已预计算，则跳过
- PyPTO 无内置 `outer` API，但 PyTorch 使用 `torch.outer` 实现

- 步骤 9 鶉freq` 生成涉及 `pypto.pow` 和 `pypto.div`，但需注意:
  - `pypto.pow` 只支持 `base` 为标量或 `input` 为 Tensor
  - `pypto.div` 只支持标量除以 Tensor
  - 需要在运行时预计算频率，或者传入 cos/sin

- **Substitute 配方**:

```
freq_gen: base^(2i/d) → pow(base, 2i/d) -> div(1, pow_result)
arange: arange(0, d//2) -> freqs = outer(arange, freqs) -> cos/sin(freqs)
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT8-64/BOOL | FP32 (默认), BF16 (推荐) | ✓ |
| contiguous | 必须 | — | 需确保 |

| format | ND (推荐) | ND | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.mul | dtype | FP16/BF16/FP32/INT16/INT32 | ✓ |
| pypto.add | dtype | FP16/BF16/FP32/INT16/INT32 | ✓ |
| pypto.neg | dtype | FP16/BF16/FP32/INT16/INT32 | ✓ |
| pypto.concat | dtype | FP16/BF16/FP32/INT8/INT16/INT32 | ✓ |
| pypto.view | 维度 | 2-4维 | ✓ |
| pypto.cos | dtype | FP32 only | ⚠ 仅FP32 |
| pypto.sin | dtype | FP32 only | ⚠ 仅FP32 |

**说明**:
- `pypto.cos` 和 `pypto.sin` 仅支持 FP32，若输入为 BF16/FP16，需要先 cast 到 FP32 再计算
- 对于性能关键路径，建议预计算 cos/sin 并传入 kernel

- `pypto.pow` 仅支持标量 base，若需要动态生成频率，需要使用 PyTorch 预计算

- `pypto.div` 仅支持标量除以 Tensor，频率生成需外部处理

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**Tiling 建议**:
- 输入 shape: [b, s, n, d] (4维)
- 推荐设置: `pypto.set_vec_tile_shapes(b_tile, s_tile, n_tile, d_tile)`
- 建议配置:
  - b_tile: 1-8 (batch 维度通常较小或建议每个batch独立处理)
  - s_tile: 64-512 (根据序列长度调整)
  - n_tile: 1-32 (head数量)
  - d_tile: 64-128 (head_dim，通常是64或128)

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/deepseek_v32_exp/deepseekv32_lightning_indexer_prolog_quant.py` | models | 高 | 高 | rotate_half函数实现、mul/add/concat/neg API组合模式 |
| `models/glm_v4_5/utils/golden/attn_golden.py` | models | 高 | 高 | RoPE golden实现、attention中的RoPE应用 |
| `operators/scaled_dot_product_attention/` | operators | 中 | 高 | 多head attention实现模式、动态轴处理 |

### 6.2 可复用模式

- **API 调用模式**: 使用 `pypto.view` 分割张量，使用 `pypto.neg` 取负, 使用 `pypto.concat` 拼接
- **Tiling 策略**: 设置 `set_vec_tile_shapes` 范盖所有计算维度
- **Loop 结构**: 外层循环 batch/seq， 内层循环 heads
- **边界处理**: 使用 `pypto.view` 的 valid_shape 参数处理动态长度

- **动态轴处理**: 使用 `dynamic_axis` 参数标记 batch 和 seq_len 维度为动态

### 6.3 巷异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输入shape | [b, t, n, d] | [b, s, n, d] | 调整维度顺序 |
| cos/sin shape | [b, s, 1, d] | [s, d] 或 [1, s, 1, d] | 需要广播或unsqueeze处理 |
| rotate_half | 使用 chunk | 使用 view + concat | 使用 view 分割, concat 拼接 |
| 数据类型 | BF16 | FP32/BF16/FP16 | 支持 BF16/FP16，需精度验证 |

---

## 7. 騂评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无直接 RoPE API | PyPTO 暂无内置 rotary_embedding API | 使用组合 API 实现 |
| cos/sin 仅支持 FP32 | cos/sin API 仅支持 FP32 dtype | 黺议预计算 cos/sin 或传入 kernel，或在 kernel 中 cast 到 FP32 计算 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| 广播处理 | cos/sin 需要广播到 x 的 shape，使用 unsqueeze/expand 夑理 |
| 动态轴 | batch 和 seq_len 必须标记为动态轴，使用 SymbolicScalar |
| 精度问题 | BF16/FP16 计算时注意精度损失，建议使用 FP32 中间计算 |
| 性能优化 | 考虑融合 cos/sin 生成或使用预计算表量 |

---

<!-- REQUIRED -->
## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| mul 文档 | `docs/api/operation/pypto-mul.md` |
| add 文档 | `docs/api/operation/pypto-add.md` |
| neg 文档 | `docs/api/operation/pypto-neg.md` |
| concat 文档 | `docs/api/operation/pypto-concat.md` |
| view 文档 | `docs/api/operation/pypto-view.md` |
| sin 文档 | `docs/api/operation/pypto-sin.md` |
| cos 文档 | `docs/api/operation/pypto-cos.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 参考实现 | `models/deepseek_v32_exp/deepseekv32_lightning_indexer_prolog_quant.py` |
| 参考实现 | `models/glm_v4_5/utils/golden/attn_golden.py` |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断问题，所有核心 API 均支持
- **实现建议**:
  1. 使用 `pypto.view` + `pypto.neg` + `pypto.concat` 实现 rotate_half
  2. 使用 `pypto.mul` + `pypto.add` 实现旋转计算
  3. 建议预计算 cos/sin 并传入 kernel 以提升性能
  4. cos/sin 计算建议在 FP32 进行，计算完成后 cast 回目标 dtype
  5. 支持动态轴: batch 和 seq_len 使用 `dynamic_axis` 参数
