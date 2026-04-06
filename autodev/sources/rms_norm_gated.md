# rms_norm_gated

## 算法描述

带门控机制的 RMSNorm，将 RMSNorm 的输出与门控信号（sigmoid 激活）逐元素相乘：

$$y = \text{RMSNorm}(x) \cdot \sigma(\text{gate})$$

展开为：

$$y = \frac{x}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}} \cdot \gamma \cdot \sigma(\text{gate})$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 为 Sigmoid 函数。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3.5 | SSM-Attention Hybrid | GatedDeltaNet 层后的门控归一化 |
| Qwen3-Next | SSM-Attention Hybrid | GatedDeltaNet 层后的门控归一化 |
| Falcon-H1 | SSM-Attention Hybrid | Mamba2 层后的门控归一化 |

## 参考实现

- Transformers 路径：`transformers/models/qwen3_next/modeling_qwen3_next.py` → `RMSNormGated`
- 等效 PyTorch 实现：
  ```python
  class RMSNormGated(nn.Module):
      def __init__(self, hidden_size, eps=1e-6):
          super().__init__()
          self.weight = nn.Parameter(torch.ones(hidden_size))
          self.eps = eps

      def forward(self, x, gate):
          variance = x.pow(2).mean(-1, keepdim=True)
          x_norm = x * torch.rsqrt(variance + self.eps)
          return self.weight * x_norm * torch.sigmoid(gate)
  ```

## 输入输出规格

- 输入：
  - `input` ($x$): shape `[batch, seq_len, hidden_size]`，dtype float16 / bfloat16 / float32
  - `gate`: shape `[batch, seq_len, hidden_size]`，与 input 同 shape 和 dtype
  - `weight` ($\gamma$): shape `[hidden_size]`
  - `eps`: float，默认 1e-6
- 输出：
  - `output`: shape `[batch, seq_len, hidden_size]`，与 input 同 dtype
- 典型 shape：
  - `input/gate`: `[1, 2048, 4096]`、`[1, 4096, 8192]`
  - `weight`: `[4096]`、`[8192]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
