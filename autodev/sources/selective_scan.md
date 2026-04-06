# selective_scan

## 算法描述

选择性扫描（Selective Scan），Mamba/Mamba2 的核心计算原语。与传统 SSM 使用固定参数不同，选择性扫描的 SSM 参数（B, C, dt）均依赖输入，使模型具备内容感知的选择能力。

通过并行前缀和（parallel prefix sum / parallel scan）实现高效计算：
```
# 离散化
A_bar_t = exp(dt_t * A)           # 离散化衰减矩阵
B_bar_t = dt_t * B_t              # 离散化输入矩阵（简化 ZOH）

# 状态递推（逻辑上逐步）
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
y_t = C_t * h_t

# 并行前缀和实现
# 将递推转化为二元结合运算 (a, b):
#   元素: (A_bar_t, B_bar_t * x_t)
#   结合律: (a1, b1) ⊕ (a2, b2) = (a2 * a1, a2 * b1 + b2)
# 使用 parallel prefix sum 在 O(log T) 深度内完成整个序列的扫描
```

其中：
- A ∈ R^{d_inner×d_state} 为可学习参数（通常取负值以保证稳定性）
- B_t, C_t 为输入依赖的投影，由输入 x 经线性变换得到
- dt_t 为输入依赖的时间步长，控制离散化精度
- 对角结构使得各 state 维度独立，可高效并行

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Mamba2 | SSM | 选择性状态空间扫描 |
| Jamba | SSM-Attention-MoE Hybrid | Mamba 层 |
| Zamba2 | SSM-Attention Hybrid | Mamba 层 |

## 参考实现

- `transformers/models/mamba2/modeling_mamba2.py` — `Mamba2Mixer` 类中的 selective scan 逻辑
- `transformers/models/jamba/modeling_jamba.py` — `JambaMambaMixer` 类
- Mamba 论文（Gu & Dao, 2023）Algorithm 2 — Selective Scan

## 输入输出规格

- 输入:
  - `x`: (batch, seq_len, d_inner) float16/bfloat16 — 输入序列
  - `A`: (d_inner, d_state) float32 — 状态转移矩阵（可学习，通常为负值）
  - `B`: (batch, seq_len, d_state) float16/bfloat16 — 输入依赖的投影矩阵
  - `C`: (batch, seq_len, d_state) float16/bfloat16 — 输出依赖的投影矩阵
  - `dt`: (batch, seq_len, d_inner) float16/bfloat16 — 输入依赖的时间步长
  - `D`: (d_inner,) float32 — 跳跃连接参数（可选）
- 输出:
  - `y`: (batch, seq_len, d_inner) float16/bfloat16 — 输出序列
- 典型 shape:
  - Mamba-2.8B: batch=1~16, seq_len=1024~8192, d_inner=5120, d_state=16
  - Jamba: d_inner=4096~8192, d_state=16
  - Zamba2: d_inner=4096, d_state=16~64

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
