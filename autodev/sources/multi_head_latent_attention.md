# multi_head_latent_attention

## 算法描述

Multi-head Latent Attention (MLA) 是 DeepSeek 提出的注意力变体，将 KV 投影到低维潜在空间以压缩 KV cache，推理时通过上投影矩阵解压恢复。

数学公式与计算流程：

1. **KV 压缩（训练与 prefill）**:
   - c_kv = x @ W_dkv,  c_kv: [batch, seq_len, d_c], d_c << num_heads * head_dim
   - K = c_kv @ W_uk,  K: [batch, seq_len, num_heads, head_dim]
   - V = c_kv @ W_uv,  V: [batch, seq_len, num_heads, head_dim]

2. **Q 压缩（可选）**:
   - c_q = x @ W_dq,  c_q: [batch, seq_len, d_c_q]
   - Q = c_q @ W_uq,  Q: [batch, seq_len, num_heads, head_dim]

3. **RoPE 解耦**:
   - 将 head_dim 拆分为 rope 部分和 nope 部分
   - Q = [Q_nope, RoPE(Q_rope)]
   - K = [K_nope, RoPE(K_rope)]
   - K_rope 通过独立投影 W_kr 从 c_kv 获得，维度较小

4. **Attention 计算**:
   - O = softmax(Q @ K^T / sqrt(d_k)) @ V
   - 标准 scaled dot-product attention

5. **推理优化**:
   - KV cache 只需存储 c_kv（低维潜在向量），而非完整的 K/V
   - 推理时将 W_uk/W_uv 吸收进 W_q 和输出投影，避免显式解压

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| DeepSeek-V2 | Decoder-only MoE | MLA 首创 |
| DeepSeek-V3 | Decoder-only MoE | MLA |

## 参考实现

- `transformers/models/deepseek_v2/modeling_deepseek_v2.py::DeepseekV2Attention`
  - `kv_a_proj_with_mqa`: 下投影 W_dkv
  - `kv_b_proj`: 上投影 W_uk, W_uv
  - `q_a_proj` / `q_b_proj`: Q 的下/上投影
- `transformers/models/deepseek_v3/modeling_deepseek_v3.py::DeepseekV3Attention`

## 输入输出规格

- 输入:
  - x: [batch_size, seq_len, hidden_size], dtype: float16/bfloat16, 含义: 输入隐状态
  - W_dkv: [hidden_size, d_c + qk_rope_head_dim], dtype: float16/bfloat16, 含义: KV 下投影权重
  - W_uk: [d_c, num_heads * qk_nope_head_dim], dtype: float16/bfloat16, 含义: K 上投影权重
  - W_uv: [d_c, num_heads * v_head_dim], dtype: float16/bfloat16, 含义: V 上投影权重
- 输出:
  - O: [batch_size, seq_len, hidden_size], dtype: float16/bfloat16, 含义: 注意力输出
- 典型 shape:
  - DeepSeek-V2: batch=1, num_heads=128, seq_len=4096, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, d_c=512
  - KV cache 每 token: d_c + qk_rope_head_dim = 576 维（相比标准 MHA 的 128*128*2 = 32768 维大幅压缩）

## 输入约束

- `kv_lora_rank` 须远小于 `num_heads * head_dim`，常见值：512
- `q_lora_rank` 须远小于 `hidden_size`，常见值：1536
- `head_dim` 常见值：128
- `rope_head_dim` 须 ≤ `head_dim`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
