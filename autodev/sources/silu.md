# silu

## 算法描述

SiLU（Sigmoid Linear Unit），又称 Swish 激活函数，由 Google Brain 在 2017 年提出。其数学定义为：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 为 Sigmoid 函数。

SiLU 是 Swish-1 的特例（Swish-$\beta$ 定义为 $x \cdot \sigma(\beta x)$，SiLU 即 $\beta=1$）。该函数具有以下数学性质：
- **非单调**：在 $x \approx -1.278$ 处有全局最小值约 $-0.278$
- **处处光滑**：无限阶可导
- **自门控**：输入自身作为门控信号，无需额外参数
- **导数**：$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x)) = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))$

SiLU 广泛用于 SwiGLU（Swish-Gated Linear Unit）结构中，该结构将 SiLU 与门控线性单元结合：$\text{SwiGLU}(x, W_1, W_2) = \text{SiLU}(xW_1) \otimes xW_2$，是当前主流 LLM 的标准 FFN 激活。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | SwiGLU MLP 中的激活 |
| LLaMA 3 | Decoder-only Transformer | SwiGLU MLP 中的激活 |
| Gemma 3 | Decoder-only Transformer | SwiGLU MLP 中的激活 |
| GLM-4 | Decoder-only Transformer | SwiGLU MLP 中的激活 |
| Mistral / Mixtral | Decoder-only Transformer / MoE | SwiGLU MLP 中的激活 |
| DeepSeek-V3 | Decoder-only MoE | SwiGLU MLP 中的激活 |

## 参考实现

- **PyTorch 标准 API**：`torch.nn.functional.silu(input, inplace=False)`
- **PyTorch 模块**：`torch.nn.SiLU(inplace=False)`
- **transformers 中使用**：`transformers.activations.silu`（直接调用 `F.silu`）
- **LLaMA MLP 实现**：`transformers.models.llama.modeling_llama.LlamaMLP`，在 `forward()` 中调用 `self.act_fn(self.gate_proj(x)) * self.up_proj(x)`，其中 `act_fn` 即 SiLU

## 输入输出规格

- **输入**:
  - `x`: `torch.Tensor`，shape 为任意维度（1D~4D），dtype 为 `float32` / `float16` / `bfloat16`
  - 含义：待激活的特征张量，通常为 FFN gate projection 的输出

- **输出**:
  - `y`: `torch.Tensor`，shape 与输入相同，dtype 与输入相同
  - 含义：激活后的特征张量，将与 up projection 的输出做逐元素乘法

- **典型 shape**:
  - LLaMA3-8B 推理：`[1, seq_len, intermediate_size]`，如 `[1, 2048, 14336]`
  - Qwen2.5-7B 推理：`[1, seq_len, intermediate_size]`，如 `[1, 4096, 11008]`
  - Batch 推理：`[batch, seq_len, intermediate_size]`，如 `[32, 512, 14336]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
