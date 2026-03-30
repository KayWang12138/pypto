## 算子需求规范

### 1. 基础信息
- **算子名称**: scaled_dot_product_attention
- **算子分类**: attention
- **数学公式**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
- **功能描述**: 计算缩放点积注意力，是 Transformer 架构的核心算子。支持可选的注意力掩码、dropout、因果掩码等特性。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| scale | 需要 | 高 | 缩放因子，默认 1/sqrt(d) | P0 |
| attn_mask | 需要 | 高 | 支持 bool 和 float 类型掩码 | P1 |
| is_causal | 需要 | 高 | 因果注意力掩码，与 attn_mask 互斥 | P1 |
| dropout_p | 需要 | 中 | Dropout 概率，默认 0.0 | P2 |
| enable_gqa | 不需要 | 低 | Grouped Query Attention，实验性功能 | P3 |
| numerical_stable_softmax | 需要 | 高 | 数值稳定的 softmax 实现 | P0 |
| 动态轴支持 | 需要 | 高 | batch, num_heads, seq_len 动态 | P0 |

### 3. 算法描述

```
Algorithm: Scaled Dot Product Attention
────────────────────────────────────
输入: Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
输出: output

1. 计算缩放因子: scale_factor = 1/sqrt(d) if scale is None else scale
2. 初始化注意力偏置: attn_bias = zeros(L, S)
3. if is_causal:
     3.1 生成下三角掩码: temp_mask = tril(ones(L, S))
     3.2 应用因果掩码: attn_bias[~temp_mask] = -inf
4. if attn_mask is not None:
     4.1 if attn_mask.dtype == bool:
           attn_bias[~attn_mask] = -inf
         else:
           attn_bias = attn_mask + attn_bias
5. 计算注意力分数: attn_weight = Q @ K^T * scale_factor
6. 应用偏置: attn_weight = attn_weight + attn_bias
7. Softmax: attn_weight = softmax(attn_weight, dim=-1)
8. Dropout: if dropout_p > 0: attn_weight = dropout(attn_weight, dropout_p)
9. 计算输出: output = attn_weight @ V
10. return output
```

### 4. 数据流图

```
        Q                    K                    V
   [N, H, L, E]        [N, H, S, E]        [N, H, S, Ev]
        │                    │                    │
        │                    │                    │
        │              ┌─────┴─────┐              │
        │              │ transpose │              │
        │              │  [N,H,E,S]│              │
        │              └─────┬─────┘              │
        │                    │                    │
        └────────┬───────────┘                    │
                 │                                │
                 ▼                                │
          ┌────────────┐                          │
          │  matmul    │                          │
          │ [N,H,L,S]  │                          │
          └─────┬──────┘                          │
                │                                 │
                ▼                                 │
          ┌────────────┐                          │
          │  * scale   │                          │
          └─────┬──────┘                          │
                │                                 │
                │      attn_mask                  │
                │    [N,H,L,S] or [L,S]           │
                │         │                       │
                ▼         ▼                       │
          ┌───────────────────┐                   │
          │   apply_mask      │                   │
          │   (add bias)      │                   │
          └─────────┬─────────┘                   │
                    │                             │
                    ▼                             │
          ┌───────────────────┐                   │
          │     softmax       │                   │
          │   (dim=-1)        │                   │
          └─────────┬─────────┘                   │
                    │                             │
                    ▼                             │
          ┌───────────────────┐                   │
          │  dropout (opt)    │                   │
          └─────────┬─────────┘                   │
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                            ┌────────────┐
                            │  matmul    │
                            │ [N,H,L,Ev] │
                            └─────┬──────┘
                                  │
                                  ▼
                             输出 output
                            [N, H, L, Ev]
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| query | [N, H, L, E] | float32 / float16 / bfloat16 | N, H, L | Query 张量 |
| key | [N, H, S, E] | float32 / float16 / bfloat16 | N, H, S | Key 张量 |
| value | [N, H, S, Ev] | float32 / float16 / bfloat16 | N, H, S | Value 张量 |
| attn_mask | [N, H, L, S] or [L, S] | bool / float | N, H, L, S (可选) | 注意力掩码 (可选) |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| output | [N, H, L, Ev] | float32 / float16 / bfloat16 | N, H, L | 注意力输出 |

**Shape 说明**:
- N: Batch size
- H: Number of attention heads
- L: Target sequence length (query length)
- S: Source sequence length (key/value length)
- E: Embedding dimension of query and key (head_dim)
- Ev: Embedding dimension of value (通常 Ev = E)

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | 是 | 0.001 | 0.001 | 默认，高精度 |
| float16 | 是 | 0.01 | 0.01 | 需要 numerical_stable_softmax |
| bfloat16 | 是 | 0.01 | 0.01 | 需要 numerical_stable_softmax |

### 7. 精度要求
- **atol**: 0.001 (float32) / 0.01 (float16/bfloat16)
- **rtol**: 0.001 (float32) / 0.01 (float16/bfloat16)

### 8. 动态轴说明
- **动态轴**: N (batch), H (num_heads), L (seq_len_q), S (seq_len_kv)
- **轴含义**:
  - N: Batch size，批处理大小
  - H: Number of attention heads，注意力头数
  - L: Query sequence length，查询序列长度
  - S: Key/Value sequence length，键值序列长度
- **取值范围**:
  - N: [1, 1024]
  - H: [1, 128]
  - L: [1, 4096]
  - S: [1, 4096]
  - E: [64, 256] (通常为 64, 128)

### 9. 边界条件处理
- **零值**: 正常计算
- **极值 (Inf)**: softmax 前的 -inf 用于 mask，softmax 后处理
- **NaN/Inf**:
  - 输入检测: 若输入含 NaN/Inf，行为未定义
  - 输出保证: softmax 使用数值稳定实现，避免溢出

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍
- **内存优化**: 支持分块计算以减少内存占用
- **并行度**: 充分利用 NPU 多核并行

### 11. 参考信息
- **参考实现**: PyTorch `torch.nn.functional.scaled_dot_product_attention`
- **论文**: "Attention Is All You Need" (Vaswani et al., 2017)
- **类似算子**:
  - Flash Attention (优化实现)
  - Memory-Efficient Attention
  - Multi-Head Attention

### 12. 应用场景
- **目标模型**: Transformer, BERT, GPT, LLaMA, etc.
- **使用位置**: Multi-Head Attention 模块的核心计算

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | scale=None, dropout_p=0.0 | Q:[1,8,1024,128], K:[1,8,1024,128], V:[1,8,1024,128] | [1,8,1024,128] | 性能核心场景 |
| 功能_P0 | 功能 | P0 | scale=None, dropout_p=0.0 | Q:[2,4,512,64], K:[2,4,512,64], V:[2,4,512,64] | [2,4,512,64] | 基础功能验证 |
| 因果注意力_P1 | 功能 | P1 | is_causal=True, scale=None | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] | 因果掩码验证 |
| 掩码注意力_P1 | 功能 | P1 | attn_mask=[1,1,512,512] | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] | 自定义掩码验证 |
| 长序列_P2 | 功能 | P2 | scale=None | Q:[1,4,4096,64], K:[1,4,4096,64], V:[1,4,4096,64] | [1,4,4096,64] | 长序列测试 |
| Dropout_P2 | 功能 | P2 | dropout_p=0.1 | Q:[1,8,256,128], K:[1,8,256,128], V:[1,8,256,128] | [1,8,256,128] | Dropout 功能验证 |

---
*生成时间: 2026-03-28T00:00:00Z*
*确认状态: 已确认（非交互模式，使用默认值）*
