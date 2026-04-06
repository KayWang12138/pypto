# per_layer_experts

## 算法描述

每层独立专家（Per-Layer Experts）是指 MoE 架构中每个 Transformer 层拥有独立的专家参数集合。计算过程：

对第 $l$ 层，给定路由结果 $(e_{i,1}, \dots, e_{i,K})$ 和权重 $(w_{i,1}, \dots, w_{i,K})$：

$$y_i^{(l)} = \sum_{k=1}^{K} w_{i,k} \cdot \text{FFN}^{(l)}_{e_{i,k}}(x_i)$$

其中每个 $\text{FFN}^{(l)}_e$ 为第 $l$ 层第 $e$ 个专家的前馈网络：

$$\text{FFN}^{(l)}_e(x) = W_2^{(l,e)} \cdot \sigma(W_1^{(l,e)} \cdot x)$$

每层的专家数量、路由策略可以不同（例如部分层使用 dense FFN，部分层使用 MoE）。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| DeepSeek-V3 | Decoder-only MoE | 各层独立专家配置 |
| Llama 4 | Decoder-only MoE | 各层独立专家配置 |

## 参考实现

- `transformers/models/mixtral/modeling_mixtral.py::MixtralSparseMoeBlock`（`self.experts = nn.ModuleList`）
- `transformers/models/llama4/modeling_llama4.py` 中 MoE 层的专家列表
- 每个专家为独立的 `torch.nn.Linear` 层组合

## 输入输出规格

- 输入:
  - `hidden_states`: shape `[num_tokens, hidden_dim]`, dtype `float32/bfloat16`, 输入 token
  - `expert_weights`: 每个专家的参数集合，`W_1`: shape `[num_experts, hidden_dim, ffn_dim]`, `W_2`: shape `[num_experts, ffn_dim, hidden_dim]`, dtype `float32/bfloat16`
  - `topk_indices`: shape `[num_tokens, top_k]`, dtype `int64`, 路由选中的专家索引
  - `topk_weights`: shape `[num_tokens, top_k]`, dtype `float32`, 路由权重
- 输出:
  - `output`: shape `[num_tokens, hidden_dim]`, dtype `float32/bfloat16`, MoE 层输出
- 典型 shape:
  - `[2048, 4096]` 输入，8 专家 Top-2（Mixtral-8x7B，ffn_dim=14336）
  - `[2048, 6144]` 输入，128 专家 Top-4（Mistral4）
  - `[2048, 5120]` 输入，256 专家 Top-6（DeepSeek-V3，ffn_dim=12288）

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
