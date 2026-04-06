# softmax

## 算法描述

沿最后一维计算 Softmax 归一化：

$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$$

计算步骤：
1. 求最后一维的最大值 $m = \max(x)$（用于数值稳定）
2. 计算指数 $e_i = \exp(x_i - m)$
3. 求和 $s = \sum_j e_j$
4. 归一化 $y_i = e_i / s$

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Attention score 归一化 |
| LLaMA 3 | Decoder-only Transformer | Attention score 归一化 |
| Gemma 3 | Decoder-only Transformer | Attention score 归一化 |
| GLM-4 | Decoder-only Transformer | Attention score 归一化 |
| DeepSeek-V3 | Decoder-only MoE | Attention score 归一化 |
| ModernBERT | Encoder-only Transformer | Attention score 归一化 |
| Whisper | Encoder-Decoder Transformer | Attention score 归一化 |

## 参考实现

- PyTorch：`torch.nn.functional.softmax(input, dim=-1)`
- 模块接口：`torch.nn.Softmax(dim=-1)`
- Transformers 中使用：
  ```python
  attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
  ```

## 输入输出规格

- 输入：
  - `input`: shape `[batch, num_heads, seq_len, seq_len]`（attention 场景）或任意 shape，dtype float16 / bfloat16 / float32
  - `dim`: int，归一化维度，默认 -1
- 输出：
  - `output`: shape 与 input 相同，dtype 同 input（通常强制 float32 计算）
- 典型 shape：
  - Attention: `[1, 32, 2048, 2048]`、`[1, 8, 512, 512]`
  - Decode: `[1, 32, 1, 2048]`

## 精度要求

与 PyTorch fp32 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
