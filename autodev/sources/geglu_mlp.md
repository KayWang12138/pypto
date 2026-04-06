# geglu_mlp

## 算法描述

GeGLU（GELU-Gated Linear Unit）MLP，使用 GELU 激活函数替代 SwiGLU 中的 SiLU，结构与 SwiGLU 完全一致：

$$\text{GeGLU}(x) = \text{GELU}(x W_{\text{gate}}) \odot (x W_{\text{up}})$$
$$\text{output} = \text{GeGLU}(x) \cdot W_{\text{down}}$$

其中：
- $\text{GELU}(z) = 0.5 \cdot z \cdot (1 + \text{erf}(z / \sqrt{2}))$
- $W_{\text{gate}} \in \mathbb{R}^{D \times D_{\text{ff}}}$：门控投影
- $W_{\text{up}} \in \mathbb{R}^{D \times D_{\text{ff}}}$：上投影
- $W_{\text{down}} \in \mathbb{R}^{D_{\text{ff}} \times D}$：下投影
- $\odot$：逐元素乘法

与 SwiGLU 的唯一区别在于激活函数从 SiLU 换为 GELU。GeGLU 在部分模型和任务上表现与 SwiGLU 相当或略优。

计算流程与 SwiGLU 完全一致：
1. 并行计算 gate 和 up 两个线性投影
2. gate 经过 GELU 激活
3. 逐元素相乘
4. 下投影得到输出

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Gemma 3 | Decoder-only Transformer | FFN（GEGLU 变体） |
| ModernBERT | Encoder-only Transformer | FFN |

## 参考实现

- 等效 PyTorch 实现（与 SwiGLU 结构相同，仅换激活函数）：
  ```python
  class GeGLU_MLP(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
          self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
          self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
          self.act_fn = nn.GELU()

      def forward(self, x):
          return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  ```
- 源码路径：类似 `transformers/models/falcon_h1/modeling_falcon_h1.py` 中的 MLP 类

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
  - `gate/up weight`: `[16384, 4096]`
  - `down weight`: `[4096, 16384]`
  - `intermediate`: `[1, 2048, 16384]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
