# causal_conv1d

## 算法描述

因果一维卷积（Causal 1D Convolution），仅使用当前时间步及之前的元素进行卷积运算，确保不会泄露未来信息。在所有 SSM/Mamba 模型中作为输入预处理层使用。

数学定义：
```
# 因果卷积（depthwise）
y[t, d] = Σ_{k=0}^{K-1} w[d, k] * x_pad[t + k, d]

# 其中 x_pad 为左 padding 后的输入：
x_pad[t, d] = x[t - (K-1) + k, d]   当 t - (K-1) + k >= 0
            = 0                       否则（左 padding）

# 等价表达：
y[t, d] = Σ_{k=0}^{K-1} w[d, k] * x[t - (K-1) + k, d]   (越界部分补零)
```

其中：
- K 为卷积核大小（通常 K=4，即仅看当前和前 3 个时间步）
- 采用 depthwise 方式：每个通道有独立的卷积核，通道间不交互
- 因果性保证：输出 y[t] 仅依赖 x[t], x[t-1], ..., x[t-K+1]
- 激活函数（SiLU/Swish）通常紧接卷积之后应用

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Mamba2 | SSM | 输入序列的因果卷积预处理 |
| Falcon-H1 | SSM-Attention Hybrid | Mamba2 层中的因果卷积 |
| Nemotron-H | SSM-Attention Hybrid | Mamba2 层中的因果卷积 |
| Bamba | SSM-Attention Hybrid | Mamba 层中的因果卷积 |
| Jamba | SSM-Attention-MoE Hybrid | Mamba 层中的因果卷积 |

## 参考实现

- `transformers/models/mamba2/modeling_mamba2.py` — `Mamba2Mixer` 中 `self.conv1d` 层
- `transformers/models/mamba/modeling_mamba.py` — `MambaMixer` 中 `self.conv1d` 层
- `transformers/models/jamba/modeling_jamba.py` — `JambaMambaMixer` 中的 conv1d
- `causal-conv1d` 库（Dao, 2024）— 高效 CUDA 实现

## 输入输出规格

- 输入:
  - `x`: (batch, d_inner, seq_len) float16/bfloat16 — 输入序列（注意 channel-first 布局）
  - `weight`: (d_inner, 1, kernel_size) float16/bfloat16 — depthwise 卷积权重
  - `bias`: (d_inner,) float16/bfloat16 — 卷积偏置（可选）
- 输出:
  - `y`: (batch, d_inner, seq_len) float16/bfloat16 — 卷积输出
- 典型 shape:
  - Mamba-2.8B: batch=1~16, d_inner=5120, seq_len=1024~8192, kernel_size=4
  - Mamba2: d_inner=2560~5120, kernel_size=4
  - Falcon-H1: d_inner=4096~8192, kernel_size=4

## 精度要求

与 PyTorch Conv1d 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
