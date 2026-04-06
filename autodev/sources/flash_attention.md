# flash_attention

## 算法描述

FlashAttention 通过分块（tiling）方式计算标准 Scaled Dot-Product Attention，避免显式构造 N×N 注意力矩阵，将显存复杂度从 O(N²) 降至 O(N)。

数学公式与计算流程：

1. **标准 Attention 定义**:
   - Attn(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

2. **分块在线 softmax 计算**:
   - 将 Q 按行分块（block_size_q），K/V 按列分块（block_size_kv）
   - 对每个 Q 块，遍历所有 KV 块：
     - S_ij = Q_i @ K_j^T / sqrt(d_k)，计算局部注意力分数
     - 在线更新行最大值 m_i = max(m_i_prev, rowmax(S_ij))
     - 使用修正因子更新分母 l_i = exp(m_i_prev - m_i) * l_i_prev + rowsum(exp(S_ij - m_i))
     - 更新输出 O_i = diag(exp(m_i_prev - m_i)) * O_i_prev + exp(S_ij - m_i) @ V_j
   - 最终 O_i = diag(1/l_i) @ O_i

3. **基础版约定**: 无 causal mask、无 dropout，纯分块 attention 计算。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Self-Attention |
| LLaMA 3 / LLaMA 4 | Decoder-only Transformer | Self-Attention |
| Gemma 3 | Decoder-only Transformer | Self-Attention |
| DeepSeek-V3 | Decoder-only MoE | Self-Attention |
| GLM-4 | Decoder-only Transformer | Self-Attention |
| Mistral / Mixtral | Decoder-only Transformer / MoE | Self-Attention |
| SmolLM3 | Decoder-only Transformer | Self-Attention |

## 参考实现

- `torch.nn.functional.scaled_dot_product_attention`
  - 当输入满足条件时自动选择 FlashAttention 后端
- FlashAttention 论文 Algorithm 1（Dao et al., 2022）
- FlashAttention-2 论文 Algorithm 1（Dao, 2023）

## 输入输出规格

- 输入:
  - Q: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Query 张量
  - K: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Key 张量
  - V: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Value 张量
- 输出:
  - O: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: 注意力输出
- 典型 shape:
  - batch=1, num_heads=32, seq_len=2048/4096, head_dim=128
  - batch=1, num_heads=40, seq_len=8192, head_dim=128

## 输入约束

- `head_dim` 必须为 2 的幂次，常见值：64、128
- `seq_len` 须 ≥ 1；实际上限受设备显存限制，典型最大值 8192~131072
- Q / K / V 的 `num_heads` 必须一致（本算子不做 GQA 分组，GQA 由 grouped_query_attention 处理）
- 输入 dtype 必须为 float16 或 bfloat16（float32 因显存和计算量过大不适用于 FlashAttention）

## 精度要求

与 PyTorch scaled_dot_product_attention 的相对误差 ≤ 1e-3（float16/bfloat16）
