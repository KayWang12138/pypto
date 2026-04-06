# gelu

## 算法描述

GELU（Gaussian Error Linear Unit）激活函数。本算子实现 **quick-gelu（sigmoid 近似）** 变体：

$$\text{GELU}_{\text{quick}}(x) = x \cdot \sigma(1.702 \cdot x)$$

其中 $\sigma$ 为 Sigmoid 函数。该近似来源于 OpenAI GPT 系列，系数 1.702 使得近似曲线与精确版高度吻合，且计算效率更高。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| ModernBERT | Encoder-only Transformer | FFN 激活 |
| Whisper | Encoder-Decoder Transformer | FFN 激活 |
| SigLIP 2 | Vision Encoder | MLP 激活 |
| Phi-3 / Phi-4 | Decoder-only Transformer | quick-gelu |

## 参考实现

- **Quick-GELU（transformers）**：`transformers.activations.QuickGELUActivation`
  ```python
  class QuickGELUActivation(nn.Module):
      def forward(self, input):
          return input * torch.sigmoid(1.702 * input)
  ```
- **PyTorch 标准 GELU**（用于精度对比）：`torch.nn.functional.gelu(input, approximate='none')`

## 输入输出规格

- **输入**:
  - `x`: shape `[*]`（任意 shape），dtype float16 / bfloat16 / float32
- **输出**:
  - `y`: shape 同输入，dtype 同输入
- **典型 shape**:
  - `[1, 2048, 4096]`（LLM 隐藏层）
  - `[32, 512, 3072]`（Batch 推理）
  - `[1, 197, 3072]`（ViT）

## 精度要求

与 PyTorch quick-gelu 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
