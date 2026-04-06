# shared_expert

## 算法描述

共享专家（Shared Expert）是 MoE 架构中所有 token 都必须经过的固定专家，其输出与路由专家的输出相加。计算过程：

$$y_i = \text{SharedExpert}(x_i) + \sum_{k=1}^{K} w_{i,k} \cdot \text{Expert}_{e_{i,k}}(x_i)$$

其中 SharedExpert 通常为标准 FFN 结构：

$$\text{SharedExpert}(x) = W_2 \cdot \sigma(W_1 \cdot x) $$

$\sigma$ 为激活函数（如 SiLU），$W_1 \in \mathbb{R}^{D \times D_{\text{ff}}}$，$W_2 \in \mathbb{R}^{D_{\text{ff}} \times D}$。

共享专家保证所有 token 都有基线表达能力，避免路由不均导致的信息丢失。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3-MoE | Decoder-only MoE | 共享专家（所有 token 都经过） |
| DeepSeek-V3 | Decoder-only MoE | 共享专家 |

## 参考实现

- `transformers/models/deepseek_v2/modeling_deepseek_v2.py::DeepseekV2MoE` 中的 `shared_expert`
- `transformers/models/qwen2_moe/modeling_qwen2_moe.py::Qwen2MoeSparseMoeBlock` 中的 `shared_expert`
- 底层为标准 `torch.nn.Linear` + 激活函数组合

## 输入输出规格

- 输入:
  - `hidden_states`: shape `[batch_size * seq_len, hidden_dim]`, dtype `float32/bfloat16`, 输入 token 的隐藏状态
- 输出:
  - `shared_output`: shape `[batch_size * seq_len, hidden_dim]`, dtype `float32/bfloat16`, 共享专家的输出，与路由专家输出相加
- 典型 shape:
  - `[2048, 5120]` 输入/输出（DeepSeek-V3，hidden_dim=5120）
  - `[2048, 2560]` 输入/输出（Qwen3-MoE，hidden_dim=2560）
  - `[1, 5120]` 输入/输出（DeepSeek-V3 decode 阶段）

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
