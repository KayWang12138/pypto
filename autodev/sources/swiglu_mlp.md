# swiglu_mlp

## 算法描述

SwiGLU（Swish-Gated Linear Unit）MLP，是现代大语言模型中最主流的前馈网络结构。由 Noam Shazeer 在 2020 年提出，结合了 Swish（SiLU）激活函数和门控线性单元（GLU）：

$$\text{SwiGLU}(x) = \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})$$
$$\text{output} = \text{SwiGLU}(x) \cdot W_{\text{down}}$$

其中：
- $\text{SiLU}(z) = z \cdot \sigma(z) = z \cdot \frac{1}{1 + e^{-z}}$（Swish 激活）
- $W_{\text{gate}} \in \mathbb{R}^{D \times D_{\text{ff}}}$：门控投影
- $W_{\text{up}} \in \mathbb{R}^{D \times D_{\text{ff}}}$：上投影
- $W_{\text{down}} \in \mathbb{R}^{D_{\text{ff}} \times D}$：下投影
- $\odot$：逐元素乘法
- $D$：hidden_size，$D_{\text{ff}}$：intermediate_size（通常 $D_{\text{ff}} = \frac{8}{3} D$ 取整到某个倍数）

计算流程：
1. 并行计算 gate = $x W_{\text{gate}}$ 和 up = $x W_{\text{up}}$（两个线性投影）
2. gate 经过 SiLU 激活
3. 逐元素相乘：hidden = SiLU(gate) $\odot$ up
4. 下投影：output = hidden $\cdot W_{\text{down}}$

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | FFN |
| LLaMA 3 | Decoder-only Transformer | FFN |
| Gemma 3 | Decoder-only Transformer | FFN |
| GLM-4 | Decoder-only Transformer | FFN |
| DeepSeek-V3 | Decoder-only MoE | 专家 FFN |
| Mistral / Mixtral | Decoder-only Transformer / MoE | FFN |
| SmolLM3 | Decoder-only Transformer | FFN |
| Phi-3 / Phi-4 | Decoder-only Transformer | FFN |

## 参考实现

- LLaMA MLP：
  ```python
  class LlamaMLP(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
          self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
          self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
          self.act_fn = nn.SiLU()

      def forward(self, x):
          return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  ```
- 源码路径：`transformers/models/llama/modeling_llama.py` → `LlamaMLP`

## 输入输出规格

- 输入：
  - `input`: shape `[batch, seq_len, hidden_size]`，dtype float16 / bfloat16
  - `gate_proj.weight`: shape `[intermediate_size, hidden_size]`
  - `up_proj.weight`: shape `[intermediate_size, hidden_size]`
  - `down_proj.weight`: shape `[hidden_size, intermediate_size]`
- 输出：
  - `output`: shape `[batch, seq_len, hidden_size]`，dtype 同输入
- 典型 shape：
  - `input`: `[1, 2048, 4096]`
  - `gate/up weight`: `[14336, 4096]`
  - `down weight`: `[4096, 14336]`
  - `intermediate`: `[1, 2048, 14336]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
