# sigmoid_act

## 算法描述

Sigmoid 激活函数，也称为 Logistic 函数，将任意实数映射到 $(0, 1)$ 区间。数学定义为：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

等价形式：
$$\sigma(x) = \frac{e^x}{e^x + 1}$$

核心数学性质：
- **值域**：$(0, 1)$，输出可解释为概率
- **对称性**：$\sigma(-x) = 1 - \sigma(x)$
- **导数**：$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$，导数可用输出自身表示
- **单调递增**：全定义域上严格单调
- **饱和性**：$|x|$ 较大时梯度趋近于 0

Sigmoid 在现代 LLM 中不常单独作为激活函数使用（因梯度消失问题），但作为 SiLU、Gating 机制、注意力权重归一化等的核心组件被广泛调用。在 SwiGLU 结构中，Sigmoid 是 SiLU 的内部计算步骤。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Qwen3 | Decoder-only Transformer | SiLU 的子运算 |
| LLaMA 3 | Decoder-only Transformer | SiLU 的子运算 |
| SigLIP 2 | Vision Encoder | Sigmoid 对比学习 |
| Falcon-H1 | SSM-Attention Hybrid | 门控机制 |

## 参考实现

- **PyTorch 标准 API**：`torch.sigmoid(input)` 或 `torch.Tensor.sigmoid()`
- **PyTorch 模块**：`torch.nn.Sigmoid()`
- **PyTorch 函数式**：`torch.nn.functional.sigmoid(input)`（已标记为 deprecated，推荐用 `torch.sigmoid`）
- **SiLU 内部调用**：`torch.nn.functional.silu` 内部等价于 `x * torch.sigmoid(x)`

## 输入输出规格

- **输入**:
  - `x`: `torch.Tensor`，shape 为任意维度（1D~4D），dtype 为 `float32` / `float16` / `bfloat16`
  - 含义：待激活的特征张量，可为门控信号、激活中间值等

- **输出**:
  - `y`: `torch.Tensor`，shape 与输入相同，dtype 与输入相同
  - 含义：映射到 $(0, 1)$ 的激活值

- **典型 shape**:
  - 作为 SiLU 组件：`[1, seq_len, intermediate_size]`，如 `[1, 2048, 14336]`
  - 作为门控信号：`[1, seq_len, hidden_size]`，如 `[1, 2048, 4096]`
  - SSM 选择性门控：`[batch, seq_len, d_inner]`，如 `[1, 2048, 4096]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
