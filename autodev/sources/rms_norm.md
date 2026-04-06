# rms_norm

## 算法描述

Root Mean Square Layer Normalization，仅使用均方根进行归一化，无均值中心化和偏置：

$$y = \frac{x}{\sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}} \cdot \gamma$$

其中：
- $\text{RMS}(x) = \sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \epsilon}$
- $\gamma$：可学习的缩放参数，shape `[hidden_size]`
- $\epsilon$：防止除零的小常数，默认 1e-6

相比 LayerNorm，RMSNorm 省去均值计算和偏置参数，计算量更小，效果相当。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | Pre-RMSNorm |
| LLaMA 3 | Decoder-only Transformer | Pre-RMSNorm |
| Gemma 3 | Decoder-only Transformer | Pre-RMSNorm |
| GLM-4 | Decoder-only Transformer | Pre-RMSNorm |
| DeepSeek-V3 | Decoder-only MoE | Pre-RMSNorm |
| Mistral | Decoder-only Transformer | Pre-RMSNorm |
| SmolLM3 | Decoder-only Transformer | Pre-RMSNorm |

## 参考实现

- 各模型的 RMSNorm 类实现基本一致：
  ```python
  class RMSNorm(nn.Module):
      def __init__(self, hidden_size, eps=1e-6):
          super().__init__()
          self.weight = nn.Parameter(torch.ones(hidden_size))
          self.eps = eps

      def forward(self, x):
          variance = x.pow(2).mean(-1, keepdim=True)
          x = x * torch.rsqrt(variance + self.eps)
          return self.weight * x
  ```
- Transformers 路径：`transformers/models/llama/modeling_llama.py` → `LlamaRMSNorm`

## 输入输出规格

- 输入：
  - `input`: shape `[batch, seq_len, hidden_size]`，dtype float16 / bfloat16 / float32
  - `weight` ($\gamma$): shape `[hidden_size]`
  - `eps`: float，默认 1e-6
- 输出：
  - `output`: shape `[batch, seq_len, hidden_size]`，与 input 同 dtype
- 典型 shape：
  - `input`: `[1, 2048, 4096]`（LLaMA3-8B）、`[1, 2048, 8192]`（LLaMA3-70B）
  - `weight`: `[4096]`、`[8192]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
