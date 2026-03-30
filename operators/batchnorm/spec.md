## 算子需求规范

### 1. 基础信息
- **算子名称**: batchnorm
- **算子分类**: normalization
- **数学公式**: $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$
- **功能描述**: Batch Normalization 对输入张量在通道维度上进行归一化处理。训练时使用当前 batch 的均值和方差，推理时使用 running mean 和 running variance。广泛应用于 CNN 等模型中，加速训练收敛并提高模型稳定性。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| eps 参数 | ✓ 需要 | ✓ 高 | 数值稳定性常数，防止除零 | P0 |
| dynamic_axis | ✓ 需要 | ✓ 高 | 支持 batch, seq_len 动态轴 | P0 |
| gamma (weight) | ✓ 需要 | ✓ 高 | 可学习缩放参数 | P0 |
| beta (bias) | ✓ 需要 | ✓ 高 | 可学习偏移参数 | P0 |
| training_mode | ✓ 需要 | ✓ 高 | 训练时计算 batch 统计量 | P1 |
| inference_mode | ✓ 需要 | ✓ 高 | 推理时使用 running 统计量 | P0 |
| running_mean | ✓ 需要 | ✓ 高 | 推理模式需要的均值统计量 | P1 |
| running_var | ✓ 需要 | ✓ 高 | 推理模式需要的方差统计量 | P1 |
| momentum | ✗ 暂缓 | ⚠ 中 | running 统计量更新系数 | P2 |

### 3. 算法描述

```
Algorithm: Batch Normalization (Forward)
────────────────────────────────────
输入: x ∈ R^{batch×seq_len×channels×*spatial}
      gamma, beta ∈ R^{channels}
      running_mean, running_var ∈ R^{channels} (推理模式)
      eps (scalar)
      training (bool)
输出: y ∈ R^{batch×seq_len×channels×*spatial}

1. if training:
     1.1 在通道维度上计算均值:
         mean = reduce_mean(x, axis=[0,1,*spatial], keepdims=True)
         // shape: [1, 1, channels, 1, ...]

     1.2 在通道维度上计算方差:
         var = reduce_mean((x - mean)^2, axis=[0,1,*spatial], keepdims=True)
         // shape: [1, 1, channels, 1, ...]

   else (inference):
     1.3 使用 running 统计量:
         mean = running_mean.reshape([1, 1, channels, 1, ...])
         var = running_var.reshape([1, 1, channels, 1, ...])

2. 归一化:
   x_norm = (x - mean) / sqrt(var + eps)
   // 广播到完整 shape

3. 仿射变换:
   y = x_norm * gamma + beta
   // gamma, beta 广播到完整 shape

4. return y
```

### 4. 数据流图

```
         输入 x                  gamma                beta
    ┌──────────────┐      ┌────────────┐      ┌────────────┐
    │[b,s,c,h,w]   │      │    [c]     │      │    [c]     │
    │   float32    │      │  float32   │      │  float32   │
    └──────┬───────┘      └─────┬──────┘      └─────┬──────┘
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  mean(x,c轴) │            │                   │
    │   [1,1,c,1]  │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ x - mean(x)  │            │                   │
    │  [b,s,c,h,w] │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │  var(x,c轴)  │            │                   │
    │   [1,1,c,1]  │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │sqrt(var+eps) │            │                   │
    │   [1,1,c,1]  │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ▼                    │                   │
    ┌──────────────┐            │                   │
    │ (x-mean)/σ   │            │                   │
    │  [b,s,c,h,w] │            │                   │
    └──────┬───────┘            │                   │
           │                    │                   │
           ├────────────────────┘                   │
           ▼                                        │
    ┌──────────────┐                                │
    │  * gamma     │                                │
    │  [b,s,c,h,w] │                                │
    └──────┬───────┘                                │
           │                                        │
           ├────────────────────────────────────────┘
           ▼
    ┌──────────────┐
    │  + beta      │
    │  [b,s,c,h,w] │
    └──────┬───────┘
           │
           ▼
       输出 y
    ┌──────────────┐
    │  [b,s,c,h,w] │
    │   float32    │
    └──────────────┘

动态轴: b (batch), s (seq_len)
归一化轴: c (channels)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [batch, seq_len, channels, H, W] | float32/float16/bfloat16 | batch, seq_len | 输入张量 |
| gamma (weight) | [channels] | float32/float16/bfloat16 | 无 | 可学习缩放参数 |
| beta (bias) | [channels] | float32/float16/bfloat16 | 无 | 可学习偏移参数 |
| running_mean | [channels] | float32 | 无 | 推理模式均值 (P1) |
| running_var | [channels] | float32 | 无 | 推理模式方差 (P1) |
| eps | scalar | float32 | 无 | 数值稳定性常数，默认 1e-5 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [batch, seq_len, channels, H, W] | float32/float16/bfloat16 | batch, seq_len | 归一化输出 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认 |
| float16 | ✓ | 0.01 | 0.01 | 混合精度场景 |
| bfloat16 | ✓ | 0.01 | 0.01 | 混合精度场景 |

### 7. 精度要求
- **atol**: 0.001 (float32), 0.01 (float16/bfloat16)
- **rtol**: 0.001 (float32), 0.01 (float16/bfloat16)

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
- **参考实现**: PyTorch torch.nn.functional.batch_norm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d
- **论文**: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (https://arxiv.org/abs/1502.03167)
- **类似算子**: layer_norm, rms_norm, instance_norm, group_norm

### 12. 应用场景
- **目标模型**: ResNet, VGG, CNN 类模型
- **使用位置**: Convolution 层后的归一化层

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0 | 功能 | P0 | eps=1e-5, channels=64 | x:[2, 32, 64, 28, 28], gamma:[64], beta:[64] | y:[2, 32, 64, 28, 28] | ResNet 典型配置 |
| 性能_P0 | 性能 | P0 | eps=1e-5, channels=256 | x:[8, 64, 256, 56, 56], gamma:[256], beta:[256] | y:[8, 64, 256, 56, 56] | 大 batch 性能测试 |
| 动态shape_1 | 功能 | P1 | eps=1e-5, channels=128 | x:[1, 16, 128, 14, 14], gamma:[128], beta:[128] | y:[1, 16, 128, 14, 14] | 小 batch 短序列 |
| 动态shape_2 | 功能 | P1 | eps=1e-5, channels=512 | x:[16, 128, 512, 7, 7], gamma:[512], beta:[512] | y:[16, 128, 512, 7, 7] | 大 batch 长序列 |
| 1D_BN | 功能 | P1 | eps=1e-5, channels=768 | x:[2, 512, 768], gamma:[768], beta:[768] | y:[2, 512, 768] | BatchNorm1d 配置 |

---
*生成时间: 2026-03-29T00:00:00Z*
*确认状态: 已确认*
