## 算子需求规范

### 1. 基础信息
- **算子名称**: layer_norm
- **算子分类**: normalization
- **数学公式**: $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$
- **功能描述**: 对输入张量的最后一个维度进行归一化处理，通过计算均值和方差进行标准化，然后应用可学习的缩放参数 gamma 和偏移参数 beta 进行仿射变换。广泛应用于 Transformer 等模型中，用于稳定训练过程。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| eps 参数 | ✓ 需要 | ✓ 高 | 数值稳定性常数，防止除零 | P0 |
| dynamic_axis | ✓ 需要 | ✓ 高 | 支持 batch, seq_len 动态轴 | P0 |
| gamma (weight) | ✓ 需要 | ✓ 高 | 可学习缩放参数 | P0 |
| beta (bias) | ✓ 需要 | ✓ 高 | 可学习偏移参数 | P0 |
| elementwise_affine | ✓ 需要 | ✓ 高 | 应用 gamma/beta 仿射变换 | P0 |

### 3. 算法描述

```
Algorithm: Layer Normalization
────────────────────────────────────
输入: x ∈ R^{batch×seq_len×normalized_shape}
      gamma, beta ∈ R^{normalized_shape}
      eps (scalar)
输出: y ∈ R^{batch×seq_len×normalized_shape}

1. 在最后一个维度上计算均值:
   mean = reduce_mean(x, axis=-1, keepdims=True)
   // shape: [batch, seq_len, 1]

2. 在最后一个维度上计算方差:
   var = reduce_mean((x - mean)^2, axis=-1, keepdims=True)
   // shape: [batch, seq_len, 1]

3. 归一化:
   x_norm = (x - mean) / sqrt(var + eps)
   // shape: [batch, seq_len, normalized_shape]

4. 仿射变换:
   y = x_norm * gamma + beta
   // gamma, beta 广播到 [batch, seq_len, normalized_shape]

5. return y
```

### 4. 数据流图

```
         输入 x                  gamma                beta
    ┌──────────────┐      ┌────────────┐      ┌────────────┐
    │  [b, s, n]   │      │    [n]     │      │    [n]     │
    │   float32    │      │  float32   │      │  float32   │
    └──────┬───────┘      └─────┬──────┘      └─────┬──────┘
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  mean(x,-1)  │            │                   │
    │    [b,s,1]   │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ x - mean(x)  │            │                   │
    │   [b,s,n]    │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ var(x,-1)    │            │                   │
    │   [b,s,1]    │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │sqrt(var+eps) │            │                   │
    │   [b,s,1]    │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ (x-mean)/σ   │            │                   │
    │   [b,s,n]    │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ├────────────────────┘                   │
           ▼                                        │
    ┌──────────────┐                                │
    │  * gamma     │                                │
    │   [b,s,n]    │                                │
    └──────┬───────┘                                │
           │                                        │
           ├────────────────────────────────────────┘
           ▼
    ┌──────────────┐
    │  + beta      │
    │   [b,s,n]    │
    └──────┬───────┘
           │
           ▼
       输出 y
    ┌──────────────┐
    │  [b, s, n]   │
    │   float32    │
    └──────────────┘

动态轴: b (batch), s (seq_len)
归一化轴: n (最后一个维度)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [batch, seq_len, normalized_shape] | float32 | batch, seq_len | 输入张量 |
| gamma (weight) | [normalized_shape] | float32 | 无 | 可学习缩放参数 |
| beta (bias) | [normalized_shape] | float32 | 无 | 可学习偏移参数 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [batch, seq_len, normalized_shape] | float32 | batch, seq_len | 归一化输出 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认 |
| float16 | ✓ | 0.001 | 0.001 | 混合精度场景 |
| bfloat16 | ✓ | 0.01 | 0.01 | 混合精度场景 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: batch, seq_len
- **轴含义**:
  - batch: 批次大小，表示一次处理的样本数量
  - seq_len: 序列长度，表示输入序列的长度
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算 (normal)
- **极值**: 正常计算 (normal)
- **NaN/Inf**: 正常计算 (normal)

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的2倍

### 11. 参考信息
- **参考实现**: PyTorch torch.nn.functional.layer_norm
- **论文**: Layer Normalization (https://arxiv.org/abs/1607.06450)
- **类似算子**: batch_norm, rms_norm, instance_norm

### 12. 应用场景
- **目标模型**: Transformer, BERT, GPT, LLaMA
- **使用位置**: Transformer Block 中的归一化层（Pre-Norm 或 Post-Norm）

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0 | 功能 | P0 | eps=1e-5, normalized_shape=768 | x:[2, 512, 768], gamma:[768], beta:[768] | y:[2, 512, 768] | BERT-Base 典型配置 |
| 性能_P0 | 性能 | P0 | eps=1e-5, normalized_shape=4096 | x:[8, 1024, 4096], gamma:[4096], beta:[4096] | y:[8, 1024, 4096] | LLaMA 大模型配置 |
| 动态shape_1 | 功能 | P1 | eps=1e-5, normalized_shape=768 | x:[1, 128, 768], gamma:[768], beta:[768] | y:[1, 128, 768] | 小 batch 短序列 |
| 动态shape_2 | 功能 | P1 | eps=1e-5, normalized_shape=768 | x:[16, 2048, 768], gamma:[768], beta:[768] | y:[16, 2048, 768] | 大 batch 长序列 |

---
*生成时间: 2026-03-28T00:00:00Z*
*确认状态: 已确认*
