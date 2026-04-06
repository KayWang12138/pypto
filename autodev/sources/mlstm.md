# mlstm

## 算法描述

xLSTM 的 mLSTM（matrix LSTM）变体，将经典 LSTM 的标量记忆单元扩展为矩阵记忆，使用指数门控（exponential gating）提升记忆容量和检索能力。

核心递推公式：
```
# 指数门控
f_t = exp(w_f^T * x_t + b_f)       # 遗忘门（指数激活）
i_t = exp(w_i^T * x_t + b_i)       # 输入门（指数激活）

# 数值稳定化（log-space）
m_t = max(log(f_t) + m_{t-1}, log(i_t))
i'_t = exp(log(i_t) - m_t)         # 稳定化后的输入门
f'_t = exp(log(f_t) + m_{t-1} - m_t)  # 稳定化后的遗忘门

# 键值投影
k_t = W_k * x_t                     # 键向量
v_t = W_v * x_t                     # 值向量
q_t = W_q * x_t                     # 查询向量

# 矩阵记忆更新
C_t = f'_t * C_{t-1} + i'_t * (v_t @ k_t^T)   # C_t ∈ R^{d_v × d_k}

# 归一化状态
n_t = f'_t * n_{t-1} + i'_t * k_t              # n_t ∈ R^{d_k}

# 输出（covariance-based retrieval）
h_t = C_t @ q_t                     # 检索
h_t = h_t / max(|n_t^T @ q_t|, 1)  # 归一化
o_t = sigmoid(w_o^T * x_t + b_o)   # 输出门
y_t = o_t ⊙ h_t                    # 门控输出
```

其中：
- C_t ∈ R^{d_v × d_k} 为矩阵记忆，存储键值对的协方差信息
- 指数门控使得遗忘和写入的动态范围远大于 sigmoid 门控
- m_t 为 log-space 稳定化变量，防止 exp 溢出
- 矩阵记忆的检索方式类似线性注意力：C_t @ q_t

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| xLSTM | SSM Hybrid | mLSTM 矩阵记忆 LSTM |

## 参考实现

- `transformers/models/xlstm/modeling_xlstm.py` — `mLSTMCell` / `mLSTMBlock` 类
- xLSTM 官方仓库（NX-AI/xlstm）— 包含 CUDA 优化实现
- xLSTM 论文（Beck et al., 2024）Section 3 — mLSTM 定义

## 输入输出规格

- 输入:
  - `x`: (batch, seq_len, d_model) float16/bfloat16 — 输入序列
  - `W_q`: (d_model, d_k) float16/bfloat16 — 查询投影权重
  - `W_k`: (d_model, d_k) float16/bfloat16 — 键投影权重
  - `W_v`: (d_model, d_v) float16/bfloat16 — 值投影权重
  - `w_i`: (d_model,) float32 — 输入门投影权重
  - `w_f`: (d_model,) float32 — 遗忘门投影权重
  - `w_o`: (d_model,) float32 — 输出门投影权重
  - `b_i, b_f, b_o`: float32 — 各门偏置
- 输出:
  - `y`: (batch, seq_len, d_model) float16/bfloat16 — 输出序列
- 典型 shape:
  - xLSTM-7B: batch=1~16, seq_len=1024~8192, d_model=4096, d_k=d_v=128 (per head), num_heads=32

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
