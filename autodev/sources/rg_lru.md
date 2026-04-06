# rg_lru

## 算法描述

Recurrent Gemma 的线性循环单元（Real-Gated Linear Recurrent Unit, RG-LRU），使用实数对角循环矩阵和输入依赖的门控机制。

核心递推公式：
```
# 输入门控
r_t = sigmoid(W_r * x_t + b_r)     # 循环门（recurrence gate）
i_t = sigmoid(W_i * x_t + b_i)     # 输入门

# 衰减因子计算
a_t = sigmoid(a_log)               # 基础衰减参数，范围 (0, 1)
c = -8 * softplus(β)               # 缩放常数
log_a_t = c * r_t * log(a_t)       # log 空间计算
a_t = exp(log_a_t)                 # 最终衰减因子

# 状态递推
h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t ⊙ x_t)

# 输出
y_t = h_t
```

其中：
- h_t ∈ R^{d_lru} 为隐状态
- a_t ∈ (0, 1) 为输入依赖的衰减因子
- sqrt(1 - a_t^2) 为归一化因子，保证状态能量守恒
- r_t 控制衰减程度：r_t → 1 时保留更多历史，r_t → 0 时接收更多新输入
- 对角循环矩阵使得各 d_lru 维度独立，天然可并行

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Recurrent Gemma | 循环架构 | RG-LRU 线性循环单元 |

## 参考实现

- `transformers/models/recurrent_gemma/modeling_recurrent_gemma.py` — `RecurrentGemmaRglru` 类
  - `forward()` 方法中包含完整的 RG-LRU 递推实现

## 输入输出规格

- 输入:
  - `x`: (batch, seq_len, d_lru) float16/bfloat16 — 输入序列
  - `a_param`: (d_lru,) float32 — 可学习的衰减参数（log-sigmoid 形式）
  - `W_r`: (d_model, d_lru) float16/bfloat16 — 循环门权重
  - `b_r`: (d_lru,) float32 — 循环门偏置
  - `W_i`: (d_model, d_lru) float16/bfloat16 — 输入门权重
  - `b_i`: (d_lru,) float32 — 输入门偏置
- 输出:
  - `y`: (batch, seq_len, d_lru) float16/bfloat16 — 输出序列
- 典型 shape:
  - RecurrentGemma-2B: batch=1~16, seq_len=1024~8192, d_model=2560, d_lru=2560
  - RecurrentGemma-9B: d_model=4096, d_lru=4096

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
