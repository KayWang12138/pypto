# moe_topk_router

## 算法描述

Top-K 路由器将每个 token 分配到 K 个最优专家。计算过程：

1. 计算门控分数：$S = \text{softmax}(X \cdot W_{\text{gate}})$，其中 $X \in \mathbb{R}^{T \times D}$ 为输入 token，$W_{\text{gate}} \in \mathbb{R}^{D \times E}$ 为门控权重，$E$ 为专家数量
2. 对每个 token 选择分数最高的 K 个专家：$\text{indices}_i, \text{weights}_i = \text{TopK}(S_i, K)$
3. 对选出的权重重新归一化：$\hat{w}_i = \frac{w_i}{\sum_{j \in \text{TopK}} w_j}$

最终每个 token 获得 K 个专家索引和对应的归一化路由权重。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| DeepSeek-V3 | Decoder-only MoE | Top-K 专家路由 |
| Qwen3-MoE | Decoder-only MoE | Top-K 专家路由 + 共享专家 |
| Mixtral | Decoder-only MoE | Top-K 专家路由 |
| Llama 4 | Decoder-only MoE | Top-K 专家路由 |
| Mistral Small 4 | Decoder-only MoE | 128 专家 Top-4 路由 |

## 参考实现

- `transformers/models/mixtral/modeling_mixtral.py::MixtralSparseMoeBlock`
- `transformers/models/deepseek_v3/modeling_deepseek_v3.py` 路由逻辑
- `torch.nn.functional.softmax` + `torch.topk`

## 输入输出规格

- 输入:
  - `hidden_states`: shape `[batch_size * seq_len, hidden_dim]`, dtype `float32/bfloat16`, 输入 token 的隐藏状态
  - `gate_weight`: shape `[hidden_dim, num_experts]`, dtype `float32/bfloat16`, 门控投影权重
- 输出:
  - `topk_weights`: shape `[batch_size * seq_len, top_k]`, dtype `float32`, 归一化的路由权重
  - `topk_indices`: shape `[batch_size * seq_len, top_k]`, dtype `int64`, 选中的专家索引
- 典型 shape:
  - `[1 * 2048, 4096]` 输入，`[4096, 8]` 门控（Mixtral-8x7B）
  - `[1 * 2048, 5120]` 输入，`[5120, 256]` 门控（DeepSeek-V3）
  - `[1 * 2048, 2560]` 输入，`[2560, 128]` 门控（Qwen3-MoE）

## 输入约束

- `num_experts` ≥ 2，常见值：8、16、64、128
- `top_k` ≥ 1 且 ≤ `num_experts`，常见值：1、2、4
- 输入 logits shape 须为 `[num_tokens, num_experts]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
