# bilinear_interpolation

## 算法描述

双线性插值（Bilinear Interpolation）是一种二维空间插值方法，用于将特征图从一个空间分辨率变换到另一个分辨率。通过在两个方向上分别进行线性插值来计算目标位置的值。

对于目标位置 $(x, y)$ 映射到源特征图的连续坐标 $(x_s, y_s)$：

1. 找到周围 4 个最近整数坐标点：$(x_0, y_0), (x_0, y_1), (x_1, y_0), (x_1, y_1)$
2. 计算插值权重：
   $$w_{00} = (x_1 - x_s)(y_1 - y_s)$$
   $$w_{01} = (x_1 - x_s)(y_s - y_0)$$
   $$w_{10} = (x_s - x_0)(y_1 - y_s)$$
   $$w_{11} = (x_s - x_0)(y_s - y_0)$$
3. 加权求和：
   $$f(x, y) = w_{00} \cdot f(x_0, y_0) + w_{01} \cdot f(x_0, y_1) + w_{10} \cdot f(x_1, y_0) + w_{11} \cdot f(x_1, y_1)$$

坐标映射（align_corners=False）：
$$x_s = \frac{(x + 0.5) \cdot W_{in}}{W_{out}} - 0.5$$

坐标映射（align_corners=True）：
$$x_s = \frac{x \cdot (W_{in} - 1)}{W_{out} - 1}$$

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| SAM 3 | 图像分割 | Mask 上采样至原始分辨率 |
| Qwen3-VL | 多模态 VLM | 视觉预处理图像 resize |

## 参考实现

- **PyTorch 函数**：`torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear', align_corners=False)`
- **PyTorch 模块**：`torch.nn.Upsample(size, scale_factor, mode='bilinear', align_corners)`
- **SAM3**：推理 pipeline 中 `F.interpolate(masks, original_size, mode='bilinear', align_corners=False)` 用于 mask 上采样

## 输入输出规格

- **输入**:
  - `input`: `torch.Tensor`，shape `[N, C, H_in, W_in]`，dtype `float32/float16/bfloat16`，含义：输入特征图
  - `size`: `(int, int)`（与 scale_factor 二选一），含义：目标输出尺寸 `(H_out, W_out)`
  - `scale_factor`: `float` 或 `(float, float)`（与 size 二选一），含义：缩放因子
  - `align_corners`: `bool`，含义：坐标对齐模式

- **输出**:
  - `output`: `torch.Tensor`，shape `[N, C, H_out, W_out]`，dtype 同输入，含义：插值后的特征图

- **典型 shape**:
  - SAM3 mask 上采样：`input=[1, 1, 64, 64]` → `output=[1, 1, 1024, 1024]`（16× 上采样）
  - VLM 图像 resize：`input=[1, 3, 480, 640]` → `output=[1, 3, 448, 448]`
  - FPN 特征对齐：`input=[1, 256, 16, 16]` → `output=[1, 256, 32, 32]`（2× 上采样）

## 输入约束

- 输入必须为 4D 张量 `[N, C, H_in, W_in]`
- `size` 和 `scale_factor` 二选一，不可同时指定
- 输出尺寸 `H_out` 和 `W_out` 须 ≥ 1
- `align_corners` 为 True 时，`H_out` 和 `W_out` 须 ≥ 2

## 精度要求

与 PyTorch F.interpolate 参考实现的相对误差 ≤ 1e-3（float16）/ 1e-5（float32）
