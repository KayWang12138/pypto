# lm_head

## 算法描述

语言模型输出头，将 Transformer 最后一层的隐藏状态映射到词表空间，生成 logits：

$$\text{logits} = \text{hidden\_states} \cdot W^T$$

其中 $W \in \mathbb{R}^{V \times D}$ 为投影权重，$V$ 为词表大小，$D$ 为隐藏维度。大部分现代模型不使用偏置。本质是一个大型矩阵乘法，将 hidden_size 维度投影到 vocab_size 维度。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | 语言模型输出头 |
| LLaMA 3 | Decoder-only Transformer | 语言模型输出头 |
| Gemma 3 | Decoder-only Transformer | 语言模型输出头 |
| GLM-4 | Decoder-only Transformer | 语言模型输出头 |
| DeepSeek-V3 | Decoder-only MoE | 语言模型输出头 |

## 参考实现

- PyTorch：`nn.Linear(hidden_size, vocab_size, bias=False)`
- Transformers 中使用：
  ```python
  self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
  logits = self.lm_head(hidden_states)
  ```

## 输入输出规格

- 输入：
  - `hidden_states`: shape `[batch, seq_len, hidden_size]`（prefill）或 `[batch, 1, hidden_size]`（decode），dtype float16 / bfloat16 / float32
  - `weight`: shape `[vocab_size, hidden_size]`
- 输出：
  - `logits`: shape `[batch, seq_len, vocab_size]` 或 `[batch, 1, vocab_size]`，dtype 同输入（通常强制 float32）
- 典型 shape：
  - Prefill: `[1, 2048, 4096]` → `[1, 2048, 128256]`
  - Decode: `[1, 1, 4096]` → `[1, 1, 128256]`
  - Weight: `[128256, 4096]`

## 精度要求

与 PyTorch Linear 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
