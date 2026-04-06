# gated_short_conv

## 算法描述

门控短卷积（Gated Short Convolution），LFM2（Liquid Foundation Model 2）的序列混合组件。通过短卷积配合门控机制实现高效的局部序列特征提取，作为 LFM2 架构中注意力/SSM 层的替代方案之一。

计算流程：
```
# 输入分支
x_conv = CausalConv1d(x, kernel_size=K)    # 因果短卷积

# 门控分支
g = sigmoid(Linear(x))                      # 门控信号

# 门控输出
y = x_conv ⊙ g                             # 逐元素门控

# 或者更完整的形式（含值投影）：
x_conv = CausalConv1d(x, kernel_size=K)    # 因果短卷积
x_conv = activation(x_conv)                 # 激活（SiLU/GELU）
g = sigmoid(Linear_g(x))                    # 门控
v = Linear_v(x)                             # 值投影
y = (x_conv ⊙ g) * v                       # 门控 + 值混合
```

其中：
- CausalConv1d 为因果一维卷积（同 causal_conv1d）
- K 为短卷积核大小，通常较小（K=3~7）
- 门控机制使得模型可以选择性地传递卷积特征
- 整体结构类似 GLU（Gated Linear Unit）的卷积变体

LFM2 中门控短卷积的设计意图：
- 提供高效的局部特征提取，补充全局注意力/SSM
- 短卷积核使得计算和内存开销极小
- 门控提供非线性选择能力

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| LFM2-VL | 混合架构 VLM | 门控短卷积 |

## 参考实现

- `transformers/models/lfm2_vl/modeling_lfm2_vl.py` — 门控短卷积实现
- Liquid Foundation Model 2 技术报告 — 架构描述

## 输入输出规格

- 输入:
  - `x`: (batch, seq_len, d_model) float16/bfloat16 — 输入序列
  - `conv_weight`: (d_conv, 1, kernel_size) float16/bfloat16 — 因果卷积权重（depthwise）
  - `conv_bias`: (d_conv,) float16/bfloat16 — 卷积偏置（可选）
  - `gate_weight`: (d_model, d_conv) float16/bfloat16 — 门控投影权重
  - `gate_bias`: (d_conv,) float16/bfloat16 — 门控偏置（可选）
  - `value_weight`: (d_model, d_out) float16/bfloat16 — 值投影权重（可选）
- 输出:
  - `y`: (batch, seq_len, d_out) float16/bfloat16 — 输出序列
- 典型 shape:
  - batch=1~16, seq_len=1024~4096, d_model=2048~4096, d_conv=d_model, kernel_size=4~7

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
