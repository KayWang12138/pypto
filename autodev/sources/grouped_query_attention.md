# grouped_query_attention

## 算法描述

Grouped-Query Attention (GQA) 是 Multi-Head Attention 的变体，将 query head 分组，每组共享同一对 KV head，在保持模型质量的同时减少 KV cache 的显存占用。

数学公式与计算流程：

1. **分组映射**:
   - 设 num_query_heads = H_q, num_kv_heads = H_kv
   - 每 H_q / H_kv 个 query head 共享同一个 KV head
   - group_size = H_q / H_kv

2. **计算过程**:
   - Q = x @ W_q, Q: [batch, seq_len, H_q, head_dim]
   - K = x @ W_k, K: [batch, seq_len, H_kv, head_dim]
   - V = x @ W_v, V: [batch, seq_len, H_kv, head_dim]
   - 对每个 query head i, 使用 KV head index = i // group_size
   - 等效于将 K, V 沿 head 维度 repeat group_size 次后执行标准 MHA:
     - K_expanded = K.repeat_interleave(group_size, dim=head_dim_axis)
     - V_expanded = V.repeat_interleave(group_size, dim=head_dim_axis)
     - O = softmax(Q @ K_expanded^T / sqrt(d_k)) @ V_expanded

3. **特殊情况**:
   - H_kv = H_q: 退化为标准 Multi-Head Attention
   - H_kv = 1: 退化为 Multi-Query Attention (MQA)

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | GQA |
| LLaMA 3 | Decoder-only Transformer | GQA |
| Gemma 3 | Decoder-only Transformer | GQA |
| GLM-4 | Decoder-only Transformer | GQA |
| Mistral / Mixtral | Decoder-only Transformer / MoE | GQA |
| SmolLM3 | Decoder-only Transformer | GQA |
| Cohere2 | Decoder-only Transformer | GQA |

## 参考实现

- `transformers/models/llama/modeling_llama.py::LlamaAttention`
  - `repeat_kv()` 函数用于将 KV head 扩展到与 Q head 相同数量
- `transformers/models/qwen2/modeling_qwen2.py::Qwen2Attention`
- `transformers/models/gemma2/modeling_gemma2.py::Gemma2Attention`
- `torch.nn.functional.scaled_dot_product_attention`（通过 enable_gqa=True 参数支持）

## 输入输出规格

- 输入:
  - Q: [batch_size, num_query_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Query 张量
  - K: [batch_size, num_kv_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Key 张量（head 数少于 Q）
  - V: [batch_size, num_kv_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Value 张量（head 数少于 Q）
- 输出:
  - O: [batch_size, num_query_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: 注意力输出
- 典型 shape:
  - LLaMA 3 8B: batch=1, H_q=32, H_kv=8, seq_len=4096, head_dim=128
  - Qwen2.5 7B: batch=1, H_q=28, H_kv=4, seq_len=4096, head_dim=128

## 输入约束

- `num_kv_heads` 必须能整除 `num_heads`，即 `num_heads % num_kv_heads == 0`
- `head_dim` 必须为 2 的幂次，常见值：64、128
- `seq_len` ≥ 1

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
