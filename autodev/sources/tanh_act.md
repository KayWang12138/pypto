# tanh_act

## 算法描述

Tanh（双曲正切）激活函数，将输入映射到 $(-1, 1)$ 区间。数学定义为：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1} = 2\sigma(2x) - 1$$

其中最后一个等式揭示了 tanh 与 Sigmoid 的关系。

核心数学性质：
- **值域**：$(-1, 1)$，零中心化输出
- **奇函数**：$\tanh(-x) = -\tanh(x)$
- **导数**：$\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)$
- **在零点处**：$\tanh(0) = 0$，$\tanh'(0) = 1$，局部近似恒等映射
- **饱和性**：$|x|$ 较大时输出趋近 $\pm 1$，梯度趋近 0

Tanh 在经典 RNN/LSTM 中作为标准激活函数使用。在现代架构中，tanh 主要出现在以下场景：
- LSTM/xLSTM 的细胞状态激活和输出门激活
- GELU 的 tanh 近似版中作为子运算
- SSM（如 Mamba2）的某些门控路径
- 注意力分数的范围约束

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| xLSTM | SSM Hybrid | sLSTM/mLSTM 门控 |
| Recurrent Gemma | 循环架构 | RG-LRU 门控 |
| Whisper | Encoder-Decoder Transformer | 位置编码/激活 |

## 参考实现

- **PyTorch 标准 API**：`torch.tanh(input)` 或 `torch.Tensor.tanh()`
- **PyTorch 模块**：`torch.nn.Tanh()`
- **PyTorch 函数式**：`torch.nn.functional.tanh(input)`
- **xLSTM sLSTM 中的使用**：在 sLSTM cell 的 `forward()` 中，`c_t = f_t * c_{t-1} + i_t * tanh(z_t)` 以及 `h_t = o_t * tanh(c_t)`

## 输入输出规格

- **输入**:
  - `x`: `torch.Tensor`，shape 为任意维度（1D~4D），dtype 为 `float32` / `float16` / `bfloat16`
  - 含义：待激活的特征张量，通常为 LSTM 候选状态或门控中间值

- **输出**:
  - `y`: `torch.Tensor`，shape 与输入相同，dtype 与输入相同
  - 含义：映射到 $(-1, 1)$ 的激活值

- **典型 shape**:
  - sLSTM 细胞状态：`[batch, seq_len, hidden_size]`，如 `[1, 2048, 4096]`
  - GELU tanh 近似中间值：`[1, seq_len, intermediate_size]`，如 `[1, 2048, 11008]`
  - RNN 隐藏状态：`[batch, hidden_size]`，如 `[32, 4096]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
