## 算子需求规范

### 1. 基础信息
- **算子名称**: rotary_embedding
- **算子分类**: embedding
- **数学公式**:
  - 旋转角度: $\theta_i = \text{position} \times \text{freq}_i$, where $\text{freq}_i = 1 / (10000^{2i/d})$
  - 旋转变换: $\text{RoPE}(x, \text{position}) = x \cdot \cos(\theta) + \text{rotate\_half}(x) \cdot \sin(\theta)$
  - rotate_half: $[x_1, x_2, ..., x_d] \to [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]$
- **功能描述**: Rotary Position Embedding (RoPE) 是一种位置编码方法，通过旋转矩阵将位置信息编码到查询和键张量中。它基于位置对查询和键张量应用旋转变换，常用于 LLaMA、GPT-NeoX 等变压器架构中。旋转使用带频率带的 sin/cos 函数逐元素应用。支持交错格式和非交错（rotary halves）格式。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| dynamic_position | ✓ 需要 | ✓ 高 | 支持动态位置序列 | P0 |
| rotary_half_format | ✓ 需要 | ✓ 高 | 非交错格式（将head_dim分成前后两半） | P0 |
| interleaved_format | ✗ 不需要 | ⚠ 中 | 交错格式（相邻元素配对），首版暂不支持 | P2 |
| freq_base | ✓ 需要 | ✓ 高 | 频率基数，默认10000 | P1 |
| freq_scale | ✗ 不需要 | ⚠ 中 | 频率缩放因子，首版暂不支持 | P2 |
| partial_rotary | ✗ 不需要 | ⚠ 中 | 部分旋转维度，首版暂不支持 | P2 |
| multi_query_support | ✓ 需要 | ✓ 高 | 支持GQA/MQA（K/V的head数可以少于Q） | P1 |

### 3. 算法描述

```
Algorithm: Rotary Position Embedding (Forward)
────────────────────────────────────────────────
输入: x [batch, seq_len, num_heads, head_dim]
      cos [seq_len, head_dim]
      sin [seq_len, head_dim]
      position_ids [batch, seq_len] (可选)
输出: y [batch, seq_len, num_heads, head_dim]

1. 获取位置索引:
   1.1 若 position_ids 为空，则 position = [0, 1, 2, ..., seq_len-1]
   1.2 否则 position = position_ids  # [batch, seq_len]

2. 生成频率张量 (若 cos/sin 未预计算):
   2.1 inv_freq = 1.0 / (freq_base ** (torch.arange(0, head_dim, 2) / head_dim))
   2.2 freqs = torch.outer(position, inv_freq)  # [seq_len, head_dim/2] 或 [batch, seq_len, head_dim/2]
   2.3 cos = cos(freqs), sin = sin(freqs)

3. 应用旋转 (rotary half format):
   3.1 x1 = x[..., :head_dim//2]  # 前半部分
   3.2 x2 = x[..., head_dim//2:]  # 后半部分
   3.3 rotated_x = concat([-x2, x1], dim=-1)  # 旋转后的x
   3.4 y = x * cos + rotated_x * sin

4. return y
```

### 4. 数据流图

```
输入 x [b, s, n, d]          cos [s, d/2] 或 [s, d]        sin [s, d/2] 或 [s, d]
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ [batch, seq,    │          │ [seq, head_dim] │          │ [seq, head_dim] │
│  heads, dim]    │          │                 │          │                 │
│    float32      │          │    float32      │          │    float32      │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         │                            │                            │
         ▼                            ▼                            ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                        Rotary Embedding                            │
    │  1. 分割 x 为 x1 (前半) 和 x2 (后半)                               │
    │  2. rotated_x = concat([-x2, x1])                                 │
    │  3. y = x * cos + rotated_x * sin                                 │
    └────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ 输出 y           │
                              │ [b, s, n, d]    │
                              │    float32      │
                              └─────────────────┘

动态轴: batch, seq_len
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [batch, seq_len, num_heads, head_dim] | float32 | batch, seq_len | 输入张量（query或key） |
| cos | [seq_len, head_dim] 或 [1, seq_len, 1, head_dim] | float32 | seq_len | 余弦值（可预计算或运行时计算） |
| sin | [seq_len, head_dim] 或 [1, seq_len, 1, head_dim] | float32 | seq_len | 正弦值（可预计算或运行时计算） |
| position_ids | [batch, seq_len] | int64 | batch, seq_len | 位置索引（可选，默认为连续位置） |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [batch, seq_len, num_heads, head_dim] | float32 | batch, seq_len | 旋转后的输出张量 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认 |
| float16 | ✓ | 0.005 | 0.005 | 混合精度训练 |
| bfloat16 | ✓ | 0.01 | 0.01 | 混合精度训练 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: batch, seq_len, num_heads
- **轴含义**:
  - batch: 批次大小，表示并行处理的样本数
  - seq_len: 序列长度，表示输入序列的token数量
  - num_heads: 注意力头数，表示多头注意力中的头数量
- **取值范围**:
  - batch: [1, 65536]
  - seq_len: [1, 32768]
  - num_heads: [1, 128]
  - head_dim: [64, 256] (通常为64的倍数)

### 9. 边界条件处理
- **零值**: 正常计算，cos(0)=1, sin(0)=0
- **极值**: 正常计算，sin/cos函数值域为[-1, 1]
- **NaN/Inf**: 输入中的NaN/Inf会传播到输出

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的2倍

### 11. 参考信息
- **参考实现**:
  - PyTorch: 无内置实现，但 HuggingFace transformers 库有实现
  - LLaMA: https://github.com/facebookresearch/llama
  - HuggingFace: transformers/models/llama/modeling_llama.py
- **论文**: RoFormer: Enhanced Transformer with Rotary Position Embedding (https://arxiv.org/abs/2104.09864)
- **类似算子**: position_embedding, sinusoidal_position_embedding

### 12. 应用场景
- **目标模型**: LLaMA, GPT-NeoX, PaLM, Mistral, Qwen
- **使用位置**: Transformer 的 Self-Attention 层，应用于 Query 和 Key 投影后

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| LLaMA-7B | 性能 | P0 | head_dim=128, num_heads=32 | x:[1,4096,32,128], cos:[4096,128], sin:[4096,128] | y:[1,4096,32,128] | LLaMA-7B 典型配置 |
| LLaMA-70B | 性能 | P0 | head_dim=128, num_heads=64 | x:[1,4096,64,128], cos:[4096,128], sin:[4096,128] | y:[1,4096,64,128] | LLaMA-70B 典型配置 |
| 小规模验证 | 功能 | P0 | head_dim=64, num_heads=8 | x:[2,128,8,64], cos:[128,64], sin:[128,64] | y:[2,128,8,64] | 功能验证小规模 |
| 动态shape | 功能 | P1 | head_dim=64, num_heads=12 | x:[b,s,12,64], cos:[s,64], sin:[s,64] | y:[b,s,12,64] | 动态batch和seq |
| MQA支持 | 功能 | P1 | head_dim=64, q_heads=32, kv_heads=8 | x(Q):[1,1024,32,64], x(K):[1,1024,8,64] | y(Q):[1,1024,32,64], y(K):[1,1024,8,64] | 多查询注意力 |

---
*生成时间: 2026-03-28T00:00:00Z*
*确认状态: 已确认（非交互模式自动生成）*
