# slstm

## 算法描述

xLSTM 的 sLSTM（scalar LSTM）变体，在经典 LSTM 基础上引入指数门控（exponential gating）和多个记忆单元（多头），保持标量记忆结构以实现细粒度序列处理。

核心递推公式：
```
# 门控计算（指数门控 + peephole）
i_t = exp(w_i^T * x_t + r_i * c_{t-1} + b_i)   # 输入门（指数激活）
f_t = exp(w_f^T * x_t + r_f * c_{t-1} + b_f)   # 遗忘门（指数激活）
o_t = sigmoid(w_o^T * x_t + r_o * c_t + b_o)    # 输出门（sigmoid）
z_t = tanh(w_z^T * x_t + b_z)                    # 候选记忆

# 数值稳定化（log-space）
m_t = max(log(f_t) + m_{t-1}, log(i_t))
i'_t = exp(log(i_t) - m_t)
f'_t = exp(log(f_t) + m_{t-1} - m_t)

# 标量记忆更新
c_t = f'_t * c_{t-1} + i'_t * z_t               # c_t ∈ R（标量）

# 归一化状态
n_t = f'_t * n_{t-1} + i'_t                      # 归一化因子

# 输出
h_t = o_t * (c_t / max(|n_t|, 1))               # 归一化后门控输出
```

其中：
- c_t ∈ R 为标量记忆单元（区别于 mLSTM 的矩阵记忆 C_t ∈ R^{d×d}）
- 支持多组独立的 (i, f) 门头（multi-head），增加混合能力
- r_i, r_f, r_o 为 peephole 权重，使门控可窥探记忆单元状态
- 指数门控的数值范围远大于 sigmoid，必须使用 log-space 稳定化
- sLSTM 与 mLSTM 在 xLSTM 架构中交替堆叠使用

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| xLSTM | SSM Hybrid | sLSTM 标量记忆 LSTM |

## 参考实现

- `transformers/models/xlstm/modeling_xlstm.py` — `sLSTMCell` / `sLSTMBlock` 类
- xLSTM 官方仓库（NX-AI/xlstm）— 包含 CUDA 优化实现
- xLSTM 论文（Beck et al., 2024）Section 3 — sLSTM 定义

## 输入输出规格

- 输入:
  - `x`: (batch, seq_len, d_model) float16/bfloat16 — 输入序列
  - `W_i, W_f, W_o, W_z`: (d_model, d_hidden) float16/bfloat16 — 各门投影权重
  - `R_i, R_f, R_o`: (d_hidden, d_hidden) float16/bfloat16 — 循环连接权重
  - `r_i, r_f, r_o`: (d_hidden,) float32 — peephole 权重
  - `b_i, b_f, b_o, b_z`: (d_hidden,) float32 — 偏置
  - `n_heads`: int — 多输入/遗忘门头数
- 输出:
  - `y`: (batch, seq_len, d_model) float16/bfloat16 — 输出序列
- 典型 shape:
  - batch=1~16, seq_len=1024~4096, d_model=2048~4096, d_hidden=128~512
  - 每个 sLSTM 块有 n_heads 组独立的 (i, f) 门

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
