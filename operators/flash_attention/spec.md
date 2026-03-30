## 算子需求规范

### 1. 基础信息
- **算子名称**: flash_attention
- **算子分类**: attention
- **数学公式**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
- **功能描述**: Flash Attention 是一种内存高效的注意力计算方法，通过分块计算和在线 Softmax 算法，将 HBM 访问复杂度从 O(N^2) 降低到 O(N)，显著减少内存读写次数，提升大序列长度场景下的性能。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| tiling_strategy | 需要 | 高 | 将 Q/K/V 分块，每次只加载一个 tile 到 SRAM 进行计算 | P0 |
| online_softmax | 需要 | 高 | 使用在线 Softmax 算法，增量更新 max 值和累加值 | P0 |
| memory_efficient | 需要 | 高 | 避免存储完整 N^2 的注意力矩阵，仅需 O(N) 中间存储 | P0 |
| scale | 需要 | 高 | 缩放因子，默认 1/sqrt(d) | P0 |
| causal_mask | 需要 | 中 | 因果注意力掩码，用于自回归生成任务 | P1 |
| attn_mask | 需要 | 中 | 支持自定义注意力掩码 | P1 |
| dropout_p | 不需要 | 低 | Dropout 概率，PyPTO 不支持，标记为 P3 | P3 |
| 动态轴支持 | 需要 | 高 | batch, num_heads, seq_len 动态 | P0 |

### 3. 算法描述

```
Algorithm: Flash Attention (Forward)
────────────────────────────────────
输入: Q, K, V in R^{N x H x L x d}, 分块大小 Br, Bc, scale
输出: O in R^{N x H x L x d}

1. 初始化:
   - 将 Q 分为 Tr = ceil(L / Br) 块 (Q_1, ..., Q_Tr)
   - 将 K, V 分为 Tc = ceil(S / Bc) 块 (K_1, ..., K_Tc), (V_1, ..., V_Tc)
   - 初始化 O = zeros(N, H, L, d), l = zeros(N, H, L, 1), m = -inf(N, H, L, 1)

2. 外层循环: 遍历 K/V 块
   for j = 1 to Tc:
     2.1 加载 K_j, V_j 到 SRAM (shape: [N, H, Bc, d])

3. 内层循环: 遍历 Q 块
   for i = 1 to Tr:
     3.1 加载 Q_i, O_i, l_i, m_i 到 SRAM

     3.2 计算当前块的注意力分数:
         S_ij = Q_i @ K_j^T * scale  (shape: [N, H, Br, Bc])

     3.3 应用掩码 (如果需要):
         if causal_mask:
             S_ij = apply_causal_mask(S_ij, i, j)
         if attn_mask:
             S_ij = S_ij + attn_mask_ij

     3.4 在线 Softmax 更新:
         - m_ij_new = max(m_i, rowmax(S_ij))
         - P_ij = exp(S_ij - m_ij_new)
         - l_ij_new = exp(m_i - m_ij_new) * l_i + rowsum(P_ij)

     3.5 更新输出:
         O_i = (exp(m_i - m_ij_new) * l_i / l_ij_new) * O_i
             + (P_ij / l_ij_new) @ V_j

     3.6 更新状态:
         m_i = m_ij_new
         l_i = l_ij_new

     3.7 写回 O_i, l_i, m_i 到 HBM

4. 返回 O
```

### 4. 数据流图

```
        Q                    K                    V
   [N, H, L, d]        [N, H, S, d]        [N, H, S, d]
        │                    │                    │
        │              ┌─────┴─────┐              │
        │              │ 分块加载   │              │
        │              │ K_j, V_j  │              │
        │              └─────┬─────┘              │
        │                    │                    │
        │    ┌───────────────┼────────────────────┤
        │    │               │                    │
        ▼    ▼               ▼                    │
   ┌─────────────┐     ┌─────────────┐           │
   │ 分块 Q_i    │     │ K_j^T      │           │
   └──────┬──────┘     └──────┬──────┘           │
          │                   │                  │
          └────────┬──────────┘                  │
                   │                             │
                   ▼                             │
            ┌────────────┐                       │
            │  Q_i @ K_j │                       │
            │  * scale   │                       │
            │ [Br, Bc]   │                       │
            └─────┬──────┘                       │
                  │                              │
                  │    causal_mask / attn_mask   │
                  │         │                    │
                  ▼         ▼                    │
            ┌───────────────────┐                │
            │   apply_mask      │                │
            └─────────┬─────────┘                │
                      │                          │
                      ▼                          │
            ┌───────────────────┐                │
            │  Online Softmax   │                │
            │  m_new, l_new     │                │
            └─────────┬─────────┘                │
                      │                          │
                      └──────────────┬───────────┘
                                     │
                                     ▼
                              ┌────────────┐
                              │  P_ij @ V_j│
                              │  更新 O_i  │
                              │ [Br, d]    │
                              └─────┬──────┘
                                    │
                                    ▼
                             输出 O_i
                            [Br, d]
                                    │
                      ┌─────────────┴─────────────┐
                      │  循环直到所有块处理完毕    │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                             最终输出 O
                            [N, H, L, d]
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| query | [N, H, L, d] | float32 / float16 / bfloat16 | N, H, L | Query 张量 |
| key | [N, H, S, d] | float32 / float16 / bfloat16 | N, H, S | Key 张量 |
| value | [N, H, S, d] | float32 / float16 / bfloat16 | N, H, S | Value 张量 |
| attn_mask | [N, H, L, S] or [L, S] | float / bool | N, H, L, S (可选) | 注意力掩码 (可选) |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| output | [N, H, L, d] | float32 / float16 / bfloat16 | N, H, L | 注意力输出 |

**Shape 说明**:
- N: Batch size (批大小)
- H: Number of attention heads (注意力头数)
- L: Target sequence length (Query 序列长度)
- S: Source sequence length (Key/Value 序列长度，通常 S = L)
- d: Head dimension (每个头的维度，通常为 64, 128)

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | 是 | 0.001 | 0.001 | 默认，高精度 |
| float16 | 是 | 0.01 | 0.01 | 需要在 softmax 时转 FP32 |
| bfloat16 | 是 | 0.01 | 0.01 | 需要在 softmax 时转 FP32 |

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
  - L: [1, 8192] (Flash Attention 支持更长序列)
  - S: [1, 8192]
  - d: [64, 256] (通常为 64, 128)

### 9. 边界条件处理
- **零值**: 正常计算
- **极值 (Inf)**: softmax 前的 -inf 用于 mask，softmax 后处理
- **NaN/Inf**:
  - 输入检测: 若输入含 NaN/Inf，行为未定义
  - 输出保证: 使用数值稳定的在线 Softmax 实现，避免溢出

### 10. 性能要求
- **性能目标**: 相比标准 attention 实现，内存占用降低 O(N) -> O(1) (相对于序列长度)
- **HBM 访问优化**: 从 O(N^2) 降低到 O(N)
- **并行度**: 充分利用 NPU 多核并行

### 11. 参考信息
- **参考实现**: Flash Attention 官方实现 (https://github.com/Dao-AILab/flash-attention)
- **论文**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- **类似算子**:
  - scaled_dot_product_attention (标准实现)
  - Memory-Efficient Attention (PyTorch)
  - xFormers Attention

### 12. 应用场景
- **目标模型**: LLaMA, GPT, BERT, Transformer-XL 等大语言模型
- **使用位置**: Multi-Head Attention 模块的核心计算

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | scale=None, causal=False | Q:[1,8,1024,128], K:[1,8,1024,128], V:[1,8,1024,128] | [1,8,1024,128] | 性能核心场景 |
| 功能_P0 | 功能 | P0 | scale=None, causal=False | Q:[2,4,512,64], K:[2,4,512,64], V:[2,4,512,64] | [2,4,512,64] | 基础功能验证 |
| 因果注意力_P1 | 功能 | P1 | causal=True, scale=None | Q:[1,8,512,128], K:[1,8,512,128], V:[1,8,512,128] | [1,8,512,128] | 因果掩码验证 |
| 长序列_P1 | 功能 | P1 | scale=None | Q:[1,4,4096,64], K:[1,4,4096,64], V:[1,4,4096,64] | [1,4,4096,64] | 长序列测试 (Flash Attention 优势场景) |
| 超长序列_P2 | 功能 | P2 | scale=None | Q:[1,2,8192,64], K:[1,2,8192,64], V:[1,2,8192,64] | [1,2,8192,64] | 超长序列测试 |

---
*生成时间: 2026-03-29T00:00:00Z*
*确认状态: 已确认（非交互模式，使用默认值）*
