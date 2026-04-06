# conv2d

## 算法描述

二维卷积（2D Convolution），对输入特征图使用一组卷积核在空间维度上滑动，计算局部区域的加权求和。

数学公式：
$$Y(n, c_{out}, h, w) = \sum_{c_{in}=0}^{C_{in}-1} \sum_{kh=0}^{K_H-1} \sum_{kw=0}^{K_W-1} X(n, c_{in}, h \cdot s + kh - p, w \cdot s + kw - p) \cdot W(c_{out}, c_{in}, kh, kw) + b(c_{out})$$

其中：
- $X$ 为输入特征图，shape `[N, C_in, H_in, W_in]`
- $W$ 为卷积核权重，shape `[C_out, C_in, K_H, K_W]`
- $s$ 为步长（stride），$p$ 为填充（padding）
- $b$ 为偏置（可选）
- 输出尺寸：$H_{out} = \lfloor(H_{in} + 2p - K_H) / s\rfloor + 1$

本算子实现标准卷积（groups=1, dilation=1）。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| SAM 3 | 图像分割 | Neck、Mask Decoder 特征变换 |
| SigLIP 2 | Vision Encoder | Patch Embedding（stride=patch_size） |
| Qwen3-VL | 多模态 VLM | 视觉编码器 Patch Embedding |

## 参考实现

- **PyTorch 函数**：`torch.nn.functional.conv2d(input, weight, bias, stride, padding)`
- **PyTorch 模块**：`torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)`

## 输入输出规格

- **输入**:
  - `input`: shape `[N, C_in, H_in, W_in]`，dtype float16 / bfloat16 / float32
  - `weight`: shape `[C_out, C_in, K_H, K_W]`，dtype 同输入
  - `bias`（可选）: shape `[C_out]`，dtype 同输入
  - `stride`: int 或 (int, int)
  - `padding`: int 或 (int, int)
- **输出**:
  - `output`: shape `[N, C_out, H_out, W_out]`，dtype 同输入
- **典型 shape**:
  - Patch Embedding：`input=[1, 3, 224, 224], weight=[1024, 3, 14, 14], stride=14` → `output=[1, 1024, 16, 16]`
  - SAM3 Neck：`input=[1, 256, 64, 64], weight=[256, 256, 3, 3], stride=1, padding=1` → `output=[1, 256, 64, 64]`

## 输入约束

- `kernel_size` ≥ 1，常见值：1×1、3×3、5×5、7×7、14×14
- `stride` ≥ 1
- `padding` ≥ 0
- 输出空间维度须 ≥ 1：`(H_in + 2*padding - kernel_size) / stride + 1 ≥ 1`

## 精度要求

与 PyTorch Conv2d 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
