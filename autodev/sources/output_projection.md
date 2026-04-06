# output_projection

## 算法描述

注意力输出投影（Output Projection）是 Multi-Head Attention 的最后一步，将多头注意力的拼接输出通过线性变换投影回原始隐藏维度。

数学公式：
$$
\text{output} = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_H) \cdot W_O + b_O
$$

等价于：
$$
\text{output} = \text{attn\_output} \cdot W_O + b_O
$$

其中：
- $\text{attn\_output} \in \mathbb{R}^{B \times S \times (H \times d_k)}$ 为多头注意力的拼接输出
- $W_O \in \mathbb{R}^{(H \times d_k) \times D}$ 为输出投影权重矩阵
- $b_O \in \mathbb{R}^{D}$ 为偏置项（部分模型无偏置）
- $D$ 为模型隐藏维度

计算流程：
1. 将多头注意力输出从 `[B, H, S, d_k]` transpose 并 reshape 为 `[B, S, H*d_k]`
2. 执行线性变换 `output = attn_output @ W_O + b_O`，输出 shape 为 `[B, S, D]`

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Attention 输出投影 |
| LLaMA 3 | Decoder-only Transformer | Attention 输出投影 |
| DeepSeek-V3 | Decoder-only MoE | Attention 输出投影 |

## 参考实现

- `transformers.models.llama.modeling_llama.LlamaAttention`：`self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)`
- `transformers.models.gpt2.modeling_gpt2.GPT2Attention`：`self.c_proj = Conv1D(n_embd, n_embd)`
- `transformers.models.bert.modeling_bert.BertSelfOutput`：`self.dense = nn.Linear(hidden_size, hidden_size)`
- `torch.nn.Linear`
- `torch.nn.functional.linear(input, weight, bias)`

## 输入输出规格

- 输入:
  - `attn_output`: shape `[batch, seq_len, num_heads * head_dim]`, dtype `float16/bfloat16/float32`，多头注意力拼接输出
  - `weight_o`: shape `[num_heads * head_dim, hidden_dim]`, dtype 同上，输出投影权重
  - `bias_o`（可选）: shape `[hidden_dim]`, dtype 同上，偏置项
- 输出:
  - `output`: shape `[batch, seq_len, hidden_dim]`, dtype 同输入，投影后的输出
- 典型 shape:
  - `attn_output: [1, 2048, 4096]` -> `output: [1, 2048, 4096]`（LLaMA-7B，hidden_dim=4096）
  - `attn_output: [1, 4096, 4096]` -> `output: [1, 4096, 4096]`（seq_len=4096）
  - `attn_output: [4, 512, 768]` -> `output: [4, 512, 768]`（BERT-base，hidden_dim=768）
  - `attn_output: [1, 1, 4096]` -> `output: [1, 1, 4096]`（decode 阶段）
  - `attn_output: [1, 2048, 5120]` -> `output: [1, 2048, 5120]`（LLaMA-13B，hidden_dim=5120）

## 精度要求

与 PyTorch Linear 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
