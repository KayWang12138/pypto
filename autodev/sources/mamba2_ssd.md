# mamba2_ssd

## 算法描述

Mamba2 的状态空间对偶（State Space Duality, SSD）核心递推，采用对角结构的线性循环。

状态递推公式：
```
h_t = A_t * h_{t-1} + B_t * x_t
y_t = C_t * h_t + D * x_t
```

其中：
- A_t ∈ R^{n×n} 为对角矩阵（每个时间步输入依赖），控制状态衰减
- B_t ∈ R^{n×d_in} 为输入投影矩阵（输入依赖）
- C_t ∈ R^{d_out×n} 为输出投影矩阵（输入依赖）
- D ∈ R^{d_out×d_in} 为跳跃连接矩阵（可选，通常为标量或对角）
- h_t ∈ R^n 为隐状态
- x_t ∈ R^{d_in} 为输入，y_t ∈ R^{d_out} 为输出

SSD 的核心观察：当 A 为对角时，SSM 递推等价于一个半可分矩阵乘法，可在分块（chunk）内使用矩阵乘加速，chunk 间使用递推传递状态，实现线性复杂度与硬件友好的混合计算模式。

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| Mamba2 | SSM | SSD 核心递推 |
| Falcon-H1 | SSM-Attention Hybrid | Mamba2 SSM 层 |
| Nemotron-H | SSM-Attention Hybrid | Mamba2 SSM 层 |
| Bamba | SSM-Attention Hybrid | Mamba SSM 层 |

## 参考实现

- `transformers/models/mamba2/modeling_mamba2.py` — `Mamba2Mixer` 类
  - `slow_forward()` / `torch_forward()` 方法中包含 SSD 递推实现
- Mamba2 论文（Dao & Gu, 2024）Section 7 — SSD 算法描述

## 输入输出规格

- 输入:
  - `x`: (batch, seq_len, d_model) float16/bfloat16 — 输入序列
  - `A`: (num_heads,) 或 (num_heads, d_state) float32 — 对角衰减参数
  - `B`: (batch, seq_len, n_groups, d_state) float16/bfloat16 — 输入投影
  - `C`: (batch, seq_len, n_groups, d_state) float16/bfloat16 — 输出投影
  - `D`: (num_heads,) float32 — 跳跃连接参数（可选）
  - `dt`: (batch, seq_len, num_heads) float16/bfloat16 — 时间步长（离散化参数）
- 输出:
  - `y`: (batch, seq_len, d_model) float16/bfloat16 — 输出序列
- 典型 shape:
  - Mamba2-2.7B: batch=1~16, seq_len=1024~8192, d_model=2560, d_state=128, num_heads=80
  - Falcon-H1: d_model=4096, d_state=64~256, num_heads=64~128

## 输入约束

- `d_state` 须为正整数，常见值：64、128、256
- `num_heads` 须能整除 `d_model`
- `n_groups` 须能整除 `num_heads`
- chunk_size（分块大小）须为正整数，常见值：256

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
