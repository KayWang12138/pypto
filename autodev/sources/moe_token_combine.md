# moe_token_combine

## 算法描述

Token 合并将各专家的输出按路由权重加权求和，得到最终的 MoE 层输出。计算公式：

$$y_i = \sum_{k=1}^{K} w_{i,k} \cdot \text{Expert}_{e_{i,k}}(x_i)$$

其中 $w_{i,k}$ 为 token $i$ 分配给第 $k$ 个选中专家的归一化路由权重，$e_{i,k}$ 为对应的专家索引，$\text{Expert}_{e}(x)$ 为专家 $e$ 的输出。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| DeepSeek-V3 | Decoder-only MoE | 专家输出的加权合并 |
| Qwen3-MoE | Decoder-only MoE | 专家输出的加权合并 |
| Mixtral | Decoder-only MoE | 专家输出的加权合并 |

## 参考实现

- `transformers/models/mixtral/modeling_mixtral.py::MixtralSparseMoeBlock.forward` 中的加权合并逻辑
- 核心操作：`torch.einsum` 或逐专家 `weight * expert_output` 后 `torch.sum`

## 输入输出规格

- 输入:
  - `expert_outputs`: 每个专家对 token 的输出，shape `[num_tokens, top_k, hidden_dim]`, dtype `float32/bfloat16`, 各专家的计算结果
  - `topk_weights`: shape `[num_tokens, top_k]`, dtype `float32`, 归一化路由权重
- 输出:
  - `combined_output`: shape `[num_tokens, hidden_dim]`, dtype `float32/bfloat16`, 加权合并后的最终输出
- 典型 shape:
  - `[2048, 2, 4096]` 专家输出 + `[2048, 2]` 权重（Mixtral-8x7B，Top-2）
  - `[2048, 6, 5120]` 专家输出 + `[2048, 6]` 权重（DeepSeek-V3，Top-6）
  - `[2048, 4, 2560]` 专家输出 + `[2048, 4]` 权重（Qwen3-MoE，Top-4）

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
