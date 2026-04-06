# cross_attention

## 算法描述

Cross Attention 是编码器-解码器架构中的注意力机制，Query 来自 decoder 隐状态，Key 和 Value 来自 encoder 输出，实现跨模态或跨序列的信息融合。

数学公式与计算流程：

1. **投影**:
   - Q = x_decoder @ W_q,  Q: [batch, tgt_len, num_heads, head_dim]
   - K = x_encoder @ W_k,  K: [batch, src_len, num_kv_heads, head_dim]
   - V = x_encoder @ W_v,  V: [batch, src_len, num_kv_heads, head_dim]

2. **Attention 计算**:
   - S = Q @ K^T / sqrt(d_k),  S: [batch, num_heads, tgt_len, src_len]
   - O = softmax(S) @ V,  O: [batch, num_heads, tgt_len, head_dim]

3. **特点**:
   - Q 和 KV 序列长度可不同（tgt_len != src_len）
   - 通常不使用 causal mask（decoder 需要完整访问 encoder 输出）
   - encoder 输出在推理时可缓存，每个 decode step 只需计算新的 Q

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Whisper | Encoder-Decoder Transformer | Decoder 对 Encoder 的交叉注意力 |
| Qwen3-VL | 多模态 VLM | 视觉-语言交叉注意力 |
| Gemma 3 | Decoder-only Transformer | 视觉 token 的交叉注意力 |

## 参考实现

- `transformers/models/whisper/modeling_whisper.py::WhisperDecoderLayer`
  - `encoder_attn`: Cross Attention 模块
  - `encoder_attn_layer_norm`: 对应 LayerNorm
- `transformers/models/t5/modeling_t5.py::T5Attention`
  - `is_cross_attention` 标志区分 self-attention 和 cross-attention
- `torch.nn.MultiheadAttention`（通过传入不同的 query 和 key_value 实现）

## 输入输出规格

- 输入:
  - x_decoder: [batch_size, tgt_len, hidden_size], dtype: float16/bfloat16, 含义: Decoder 隐状态（用于生成 Q）
  - x_encoder: [batch_size, src_len, hidden_size], dtype: float16/bfloat16, 含义: Encoder 输出（用于生成 K, V）
- 输出:
  - O: [batch_size, tgt_len, hidden_size], dtype: float16/bfloat16, 含义: Cross attention 输出
- 典型 shape:
  - Whisper Large-v3: batch=1, num_heads=20, tgt_len=448, src_len=1500, head_dim=64
  - T5-Large: batch=1, num_heads=16, tgt_len=512, src_len=512, head_dim=64

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
