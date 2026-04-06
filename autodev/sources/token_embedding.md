# token_embedding

## 算法描述

词表嵌入查表操作，将离散的 token ID 映射为连续的稠密向量：

$$y = \text{EmbeddingTable}[\text{token\_id}]$$

从形状为 $[V, D]$ 的嵌入矩阵中，按 token_id 索引取出对应行向量。其中 $V$ 为词表大小，$D$ 为嵌入维度。这是一个纯查表操作（gather），无浮点运算。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Token Embedding 层 |
| LLaMA 3 | Decoder-only Transformer | Token Embedding 层 |
| GLM-4 | Decoder-only Transformer | Token Embedding 层 |
| ModernBERT | Encoder-only Transformer | Token Embedding 层 |
| Whisper | Encoder-Decoder Transformer | Token Embedding 层 |

## 参考实现

- PyTorch：`torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)`
- 函数式接口：`torch.nn.functional.embedding(input, weight, padding_idx=None)`
- Transformers 中使用：
  ```python
  self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
  inputs_embeds = self.embed_tokens(input_ids)
  ```

## 输入输出规格

- 输入：
  - `input_ids`: shape `[batch, seq_len]`，dtype int32 / int64，值域 $[0, V)$
  - `weight` (EmbeddingTable): shape `[vocab_size, hidden_size]`，dtype float16 / bfloat16 / float32
- 输出：
  - `embeddings`: shape `[batch, seq_len, hidden_size]`，dtype 同 weight
- 典型 shape：
  - `input_ids`: `[1, 2048]`、`[8, 512]`
  - `weight`: `[151936, 4096]`（Qwen2.5-7B）、`[128256, 4096]`（LLaMA3-8B）
  - `output`: `[1, 2048, 4096]`、`[8, 512, 4096]`

## 精度要求

查表操作，无精度损失
