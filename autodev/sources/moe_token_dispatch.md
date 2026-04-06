# moe_token_dispatch

## 算法描述

Token 分发将 token 按路由结果分配到对应的专家。计算过程：

1. 根据路由器输出的 `topk_indices`，为每个专家收集分配给它的 token
2. 构建分发映射：$\text{dispatch}[e] = \{i \mid e \in \text{topk\_indices}[i]\}$，即专家 $e$ 需要处理的 token 集合
3. 将 token 按专家分组重排：对每个专家 $e$，提取其对应的 token 子集 $X_e = X[\text{dispatch}[e]]$

分发后每个专家独立处理各自的 token 子集，支持并行计算。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| DeepSeek-V3 | Decoder-only MoE | Token 到专家的分发 |
| Qwen3-MoE | Decoder-only MoE | Token 到专家的分发 |
| Mixtral | Decoder-only MoE | Token 到专家的分发 |

## 参考实现

- `transformers/models/mixtral/modeling_mixtral.py::MixtralSparseMoeBlock.forward` 中的 token 分发逻辑
- `torch.index_select` / `torch.gather` 用于按索引提取 token

## 输入输出规格

- 输入:
  - `hidden_states`: shape `[num_tokens, hidden_dim]`, dtype `float32/bfloat16`, 所有 token 的隐藏状态
  - `topk_indices`: shape `[num_tokens, top_k]`, dtype `int64`, 每个 token 选中的专家索引
  - `topk_weights`: shape `[num_tokens, top_k]`, dtype `float32`, 每个 token 对应的路由权重
- 输出:
  - `dispatched_inputs`: 每个专家对应的 token 子集，shape `[num_tokens_for_expert_e, hidden_dim]`, dtype 同输入
  - `dispatch_mask`: shape `[num_experts, num_tokens]`, dtype `bool`, 分发掩码
- 典型 shape:
  - `[2048, 4096]` 输入，Top-2 分发到 8 个专家（Mixtral-8x7B）
  - `[2048, 5120]` 输入，Top-6 分发到 256 个专家（DeepSeek-V3）
  - `[2048, 2560]` 输入，Top-4 分发到 128 个专家（Qwen3-MoE）

## 输入约束

- `expert_indices` 中的值须在 `[0, num_experts)` 范围内
- `top_k` 须与 `expert_indices` 的最后一维大小一致

## 精度要求

与 PyTorch 参考实现的最大绝对误差 ≤ 1e-6（索引精确匹配）
