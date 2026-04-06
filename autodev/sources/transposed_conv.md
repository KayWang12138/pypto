# transposed_conv

## 算法描述

转置卷积（Transposed Convolution），也称反卷积（Deconvolution）或分数步长卷积（Fractional-strided Convolution），用于上采样特征图，将低分辨率特征映射到高分辨率空间。

数学本质：转置卷积是标准卷积的梯度操作。若标准卷积可表示为 $Y = C \cdot X$（其中 $C$ 为卷积矩阵），则转置卷积计算 $X' = C^T \cdot Y$。

计算过程（以 stride=2 为例）：
1. 在输入特征图的元素之间插入 (stride-1) 个零值：如 stride=2 时，$[a, b, c]$ → $[a, 0, b, 0, c]$
2. 在扩展后的特征图上执行标准卷积

输出尺寸：
$$H_{out} = (H_{in} - 1) \times s - 2p + K + \text{output\_padding}$$

其中 $s$ 为步长，$p$ 为填充，$K$ 为卷积核大小，$\text{output\_padding}$ 用于解决输出尺寸歧义。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| SAM 3 | 图像分割 | Mask Decoder 上采样 |

## 参考实现

- **PyTorch 函数**：`torch.nn.functional.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)`
- **PyTorch 模块**：`torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)`
- **SAM3**：`transformers/models/sam/modeling_sam.py` 中 MaskDecoder 的上采样层

## 输入输出规格

- **输入**:
  - `input`: `torch.Tensor`，shape `[N, C_in, H_in, W_in]`，dtype `float32/float16/bfloat16`，含义：低分辨率输入特征图
  - `weight`: `torch.Tensor`，shape `[C_in, C_out/groups, K_H, K_W]`，dtype 同输入，含义：转置卷积核权重（注意维度顺序与 Conv2D 相反）
  - `bias`: `torch.Tensor`（可选），shape `[C_out]`，dtype 同输入，含义：偏置
  - `stride`: `int` 或 `(int, int)`，含义：上采样步长
  - `padding`: `int` 或 `(int, int)`，含义：输入填充
  - `output_padding`: `int` 或 `(int, int)`，含义：输出填充（解决尺寸歧义）

- **输出**:
  - `output`: `torch.Tensor`，shape `[N, C_out, H_out, W_out]`，dtype 同输入，含义：上采样后的特征图

- **典型 shape**:
  - SAM3 Mask Decoder：`input=[1, 256, 16, 16], stride=2, kernel=2` → `output=[1, 256, 32, 32]`
  - UNet 上采样：`input=[1, 512, 32, 32], stride=2, kernel=4, padding=1` → `output=[1, 256, 64, 64]`
  - 最终 mask 输出：`input=[1, 32, 64, 64], stride=4, kernel=4` → `output=[1, 1, 256, 256]`

## 输入约束

- `kernel_size` ≥ 1
- `stride` ≥ 1
- `padding` ≥ 0
- `output_padding` < `stride`

## 精度要求

与 PyTorch ConvTranspose2d 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
