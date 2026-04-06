# lightning_attention

## 算法描述

Lightning Attention 是 MiniMax 提出的线性注意力变体，通过累积 KV 状态矩阵实现线性时间复杂度的注意力计算，避免构造 N×N 的注意力矩阵。

数学公式与计算流程：

1. **线性注意力基础**:
   - 标准 attention: Attn = softmax(Q @ K^T) @ V
   - 线性 attention 移除 softmax，使用核函数 phi: Attn = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
   - 通过改变计算顺序，先算 phi(K)^T @ V（d_k × d_v），再与 phi(Q) 相乘，复杂度从 O(N^2) 降至 O(N)

2. **Lightning Attention 具体实现**:
   - 引入衰减因子 lambda (decay)，控制历史信息的遗忘速率
   - 状态更新（自回归形式）：
     - S_t = lambda * S_{t-1} + K_t^T @ V_t,  S 为累积 KV 状态矩阵
     - O_t = Q_t @ S_t
   - 其中 S: [d_k, d_v] 为固定大小的状态矩阵，与序列长度无关

3. **分块计算（Intra-chunk 和 Inter-chunk）**:
   - 将序列分为多个 chunk，chunk 内使用 intra-chunk attention（可用标准 attention）
   - chunk 间使用 inter-chunk 的累积状态 S 传递信息
   - 混合策略在保持线性复杂度的同时提升表达能力

4. **门控机制**: 部分实现中使用 gate 对输出进行调制：O = gate * O_attn

## 来源模型

| 模型 | 架构类型 | 使用位置 |
|------|----------|----------|
| MiniMax | SSM-Attention Hybrid + MoE | Lightning Attention 线性注意力 |

## 参考实现

- `transformers/models/minimax/modeling_minimax.py`
  - `MiniMaxLightningAttention` 类
  - 核心方法: `forward()` 中包含 intra-chunk 和 inter-chunk 计算
- MiniMax 官方实现:
  - `lightning_attn` 库（包含 CUDA kernel 和 Triton 实现）
- 关键函数:
  - `lightning_attention_decode()`: 推理阶段的线性 attention 计算
  - `lightning_attention_prefill()`: 预填充阶段的分块计算

## 输入输出规格

- 输入:
  - Q: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Query 张量
  - K: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Key 张量
  - V: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: Value 张量
  - slope / decay: [num_heads] 或 [num_heads, 1, 1], dtype: float32, 含义: 每个 head 的衰减率
  - state: (可选) [batch_size, num_heads, head_dim, head_dim], dtype: float32, 含义: 累积 KV 状态矩阵（推理时传入）
  - gate: (可选) [batch_size, seq_len, hidden_size], dtype: float16/bfloat16, 含义: 门控值
- 输出:
  - O: [batch_size, num_heads, seq_len, head_dim], dtype: float16/bfloat16, 含义: 注意力输出
  - updated_state: [batch_size, num_heads, head_dim, head_dim], dtype: float32, 含义: 更新后的累积状态
- 典型 shape:
  - MiniMax-01: batch=1, num_heads=64, seq_len=4096, head_dim=128
  - 状态矩阵: [1, 64, 128, 128]（固定大小，与序列长度无关）
  - Prefill chunk_size: 256 或 512

## 输入约束

- `head_dim` 须为正整数，常见值：128、256
- `num_heads` 须能整除 `hidden_size`
- 输入 dtype 须为 float16 或 bfloat16

## 精度要求

与 PyTorch 参考实现的相对误差 ≤ 1e-3（float16/bfloat16）
