# mish

## 算法描述

Mish 是一种平滑、非单调激活函数，定义为：

`mish(x) = x * tanh(softplus(x))`

其中：
- `softplus(x) = log(1 + exp(x))`
- `tanh(z)` 为双曲正切函数

Mish 在负半轴保留小幅负响应，在正半轴近似线性，常被视为兼顾平滑性与表达能力的激活函数。它不依赖可训练参数，适合作为通用逐元素激活。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| SAM 3 | 图像分割 | 检测分支中的卷积激活 |

## 参考语义

- **PyTorch 标准 API**：`torch.nn.functional.mish(input, inplace=False)`
- **等价数学表达**：`x * torch.tanh(torch.nn.functional.softplus(x))`

## 输入输出规格

- **输入**:
  - `x`: 任意形状浮点 tensor
  - 推荐 dtype：`float32` / `float16` / `bfloat16`

- **输出**:
  - `y`: 与输入 shape、dtype 相同的 tensor

- **典型 shape**:
  - `[batch, hidden]`
  - `[batch, seq, hidden]`
  - `[batch, channel, height, width]`

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
