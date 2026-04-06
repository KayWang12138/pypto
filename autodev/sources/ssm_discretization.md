# ssm_discretization

## 算法描述

状态空间模型的零阶保持（Zero-Order Hold, ZOH）离散化，将连续时间 SSM 参数转化为离散时间参数。这是 Mamba/SSM 模型中从连续参数空间到离散递推的关键步骤。

数学公式：
```
# 连续时间 SSM
dx/dt = A * x + B * u
y = C * x

# ZOH 离散化（精确解）
A_bar = exp(dt * A)
B_bar = (exp(dt * A) - I) * A^{-1} * B

# 当 A 为对角矩阵时，A^{-1} 简化为逐元素取倒数：
A_bar[i,i] = exp(dt * A[i,i])
B_bar[i,:] = (exp(dt * A[i,i]) - 1) / A[i,i] * B[i,:]

# 简化近似（Mamba 实际使用）：
A_bar = exp(dt * A)
B_bar = dt * B                     # 一阶近似，当 dt 较小时足够精确

# 离散递推
h_t = A_bar * h_{t-1} + B_bar * u_t
y_t = C * h_t
```

其中：
- dt ∈ R^{d_inner} 为输入依赖的时间步长（由 softplus 激活保证非负）
- A ∈ R^{d_inner×d_state} 为连续时间的状态转移矩阵（通常初始化为负值以保证稳定性）
- exp(dt * A) 对角结构下简化为逐元素指数运算
- 精确 ZOH 的 A^{-1} 项在 A 对角时退化为标量除法

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Mamba2 | SSM | 连续→离散状态空间参数转换 |
| Falcon-H1 | SSM-Attention Hybrid | Mamba2 层的离散化 |
| Nemotron-H | SSM-Attention Hybrid | Mamba2 层的离散化 |

## 参考实现

- `transformers/models/mamba2/modeling_mamba2.py` — `Mamba2Mixer` 中离散化部分
- `transformers/models/mamba/modeling_mamba.py` — `MambaMixer` 中 `dt_proj` + 离散化
- Mamba 论文（Gu & Dao, 2023）Section 3.2 — Discretization 部分
- S4 论文（Gu et al., 2022）— ZOH 离散化的理论推导

## 输入输出规格

- 输入:
  - `A`: (d_inner, d_state) float32 — 连续时间状态转移矩阵（通常为负值对角）
  - `B`: (batch, seq_len, d_state) float16/bfloat16 — 输入投影矩阵
  - `dt`: (batch, seq_len, d_inner) float16/bfloat16 — 时间步长（softplus 激活后，非负）
- 输出:
  - `A_bar`: (batch, seq_len, d_inner, d_state) float32 — 离散化后的状态转移矩阵
  - `B_bar`: (batch, seq_len, d_inner, d_state) float32 — 离散化后的输入矩阵
- 典型 shape:
  - Mamba-2.8B: batch=1~16, seq_len=1024~8192, d_inner=5120, d_state=16
  - Mamba2: d_inner=2560, d_state=64~128

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
