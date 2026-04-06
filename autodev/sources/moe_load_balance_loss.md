# moe_load_balance_loss

## 算法描述

负载均衡损失用于训练时惩罚 token 在专家间分布不均的情况，鼓励均匀利用所有专家。计算公式（Switch Transformer 风格）：

$$\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot p_e$$

其中：
- $E$ 为专家数量
- $f_e = \frac{1}{T} \sum_{i=1}^{T} \mathbb{1}[e \in \text{TopK}_i]$ 为专家 $e$ 被选中的 token 比例
- $p_e = \frac{1}{T} \sum_{i=1}^{T} S_{i,e}$ 为所有 token 对专家 $e$ 的平均路由概率
- $\alpha$ 为负载均衡损失系数（超参数）
- $T$ 为 token 总数

当 token 完全均匀分布时 $\mathcal{L}_{\text{balance}}$ 取最小值。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| DeepSeek-V3 | Decoder-only MoE | 负载均衡辅助损失 |
| Qwen3-MoE | Decoder-only MoE | 负载均衡辅助损失 |
| Mixtral | Decoder-only MoE | 负载均衡辅助损失 |
| OLMoE | Decoder-only MoE | 负载均衡辅助损失 |

## 参考实现

- `transformers/models/mixtral/modeling_mixtral.py::load_balancing_loss_func`
- `transformers/models/switch_transformers/modeling_switch_transformers.py::load_balancing_loss_func`
- 核心操作：`torch.mean` + `torch.sum` + 逐元素乘法

## 输入输出规格

- 输入:
  - `router_probs`: shape `[num_tokens, num_experts]`, dtype `float32`, softmax 后的路由概率矩阵
  - `expert_mask`: shape `[num_tokens, num_experts]`, dtype `bool/float32`, 每个 token 是否被分配到各专家的掩码
- 输出:
  - `loss`: 标量, dtype `float32`, 负载均衡损失值
- 典型 shape:
  - `[2048, 8]` 路由概率（Mixtral-8x7B）
  - `[2048, 256]` 路由概率（DeepSeek-V3）
  - `[2048, 128]` 路由概率（Qwen3-MoE）

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-5（float32）
