# embedding

## 算法描述

通用 Embedding 查表操作，将离散整数索引映射为连续稠密向量。这是 token_embedding 的泛化版本，适用于各种类型的嵌入查表场景：

$$y = \text{EmbeddingTable}[\text{index}]$$

从形状为 $[N, D]$ 的嵌入矩阵中，按索引取出对应行向量。其中：
- $N$：嵌入表条目数（num_embeddings）
- $D$：嵌入维度（embedding_dim）

支持的功能：
- **padding_idx**：指定索引位置的输出恒为零向量，梯度恒为零
- **max_norm**：对取出的向量进行范数裁剪
- **scale_grad_by_freq**：按词频缩放梯度（训练场景）
- **sparse**：使用稀疏梯度更新（训练场景）

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Token Embedding |
| LLaMA 3 | Decoder-only Transformer | Token Embedding |
| Gemma 3 | Decoder-only Transformer | Token Embedding |
| ModernBERT | Encoder-only Transformer | Token + Position Embedding |

## 参考实现

- PyTorch：`torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)`
- 函数式接口：`torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)`
- 源码路径：`torch/nn/modules/sparse.py` → `Embedding`

## 输入输出规格

- 输入：
  - `indices`: 任意形状 `[*]` 的整数 Tensor，dtype int32 / int64，值域 $[0, N)$
  - `weight`: shape `[num_embeddings, embedding_dim]`，dtype float16 / bfloat16 / float32
  - `padding_idx`（可选）: int，指定 padding 位置
- 输出：
  - `output`: shape `[*, embedding_dim]`，即 indices 的原始 shape 追加一个 embedding_dim 维度
- 典型 shape：
  - Token embedding: indices `[1, 2048]` → output `[1, 2048, 4096]`
  - Position embedding: indices `[1, 512]` → output `[1, 512, 768]`
  - Token type embedding: indices `[1, 512]` → output `[1, 512, 768]`

## 精度要求

查表操作，无精度损失
