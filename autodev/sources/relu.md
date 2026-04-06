# relu

## 算法描述

ReLU（Rectified Linear Unit）是最基础的激活函数之一，对输入逐元素取与零的最大值。

数学公式：
$$
\text{ReLU}(x) = \max(0, x)
$$

等价的分段函数表示：
$$
\text{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

ReLU 的导数：
$$
\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

ReLU 在 $x=0$ 处不可导，实践中通常定义 $\text{ReLU}'(0) = 0$。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| ModernBERT | Encoder-only Transformer | FFN 激活 |
| Whisper | Encoder-Decoder Transformer | FFN 激活 |

## 参考实现

- `torch.nn.functional.relu(input, inplace=False)`
- `torch.nn.ReLU(inplace=False)`
- `torch.clamp(input, min=0)`
- `torch.maximum(input, torch.zeros_like(input))`
- `transformers.activations.ACT2FN["relu"]`

## 输入输出规格

- 输入:
  - `input`: shape `[*]`（任意 shape）, dtype `float16/bfloat16/float32/int32`，输入张量
- 输出:
  - `output`: shape 同 `input`, dtype 同 `input`，ReLU 激活后的张量
- 典型 shape:
  - `[1, 2048, 11008]`（LLaMA-7B FFN 中间维度，若使用 ReLU）
  - `[4, 512, 3072]`（BERT-base FFN 中间维度）
  - `[1, 2048, 4096]`（通用隐藏层维度）
  - `[4, 512, 768]`（BERT-base 隐藏层维度）
  - `[1, 1, 4096]`（decode 阶段）
  - `[32, 10, 512]`（分类任务）

## 精度要求

与 PyTorch fp32 参考实现的最大绝对误差 ≤ 1e-6
