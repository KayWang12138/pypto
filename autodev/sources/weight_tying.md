# weight_tying

## 算法描述

权重绑定（Weight Tying / Weight Sharing）是指输入嵌入层（embed_tokens）和输出投影层（lm_head）共享同一组权重参数：

$$W_{\text{lm\_head}} = W_{\text{embed\_tokens}}$$

即：
- 输入嵌入：$e = W[\text{token\_id}]$（查表，gather 操作）
- 输出投影：$\text{logits} = h \cdot W^T$（矩阵乘法）

两者使用完全相同的权重矩阵 $W \in \mathbb{R}^{V \times D}$，但执行不同的操作。

核心优势：
1. 减少参数量（节省一个 $V \times D$ 的矩阵，可达数百 MB）
2. 输入输出语义空间一致性，通常能提升生成质量
3. 正则化效果，有助于防止过拟合

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Embedding 与 LM Head 权重共享 |
| Gemma 3 | Decoder-only Transformer | Embedding 与 LM Head 权重共享 |

## 参考实现

- Transformers 中的实现：
  ```python
  # 模型初始化时
  self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
  self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  # 权重绑定
  self.lm_head.weight = self.embed_tokens.weight  # 共享同一 Parameter 对象

  # 在 PreTrainedModel 中
  def tie_weights(self):
      self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())
  ```
- 源码路径：`transformers/modeling_utils.py` → `PreTrainedModel.tie_weights()`
- 配置项：`config.tie_word_embeddings = True`

## 输入输出规格

- 共享权重：
  - `weight`: shape `[vocab_size, hidden_size]`，dtype float16 / bfloat16 / float32
- 作为 Embedding 使用时：
  - 输入：`input_ids` shape `[batch, seq_len]`，dtype int32/int64
  - 输出：`embeddings` shape `[batch, seq_len, hidden_size]`
- 作为 lm_head 使用时：
  - 输入：`hidden_states` shape `[batch, seq_len, hidden_size]`
  - 输出：`logits` shape `[batch, seq_len, vocab_size]`
- 典型 shape：
  - `weight`: `[151936, 3584]`（Qwen2.5-7B）、`[262144, 2304]`（Gemma3）

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
