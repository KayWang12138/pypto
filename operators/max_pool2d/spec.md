## 算子需求规范

### 1. 基础信息
- **算子名称**: max_pool2d
- **算子分类**: pooling
- **数学公式**:
  $$\text{output}[n, c, h, w] = \max_{i \in [0, kH), j \in [0, kW)} \text{input}[n, c, s_h \cdot h + i - p_H, s_w \cdot w + j - p_W]$$

  其中：
  - $kH, kW$ 为 kernel_size 的高度和宽度
  - $s_h, s_w$ 为 stride 的高度和宽度
  - $p_H, p_W$ 为 padding 的高度和宽度偏移
- **功能描述**: max_pool2d 是对输入 tensor 在空间维度（H 和 W）上进行最大池化操作。对于输入 tensor 的每个滑动窗口，输出窗口内的最大值。常用于 CNN 网络中下采样特征图。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 动态轴支持 | ✓ 需要 | ✓ 高 | batch、H、W 维度支持动态 shape | P0 |
| kernel_size | ✓ 需要 | ✓ 高 | 支持 int 或 tuple 形式 | P0 |
| stride | ✓ 需要 | ✓ 高 | 默认等于 kernel_size | P0 |
| padding | ✓ 需要 | ✓ 高 | 支持 int 或 tuple 形式 | P0 |
| dilation | ✓ 需要 | ✓ 高 | 空洞池化，默认为 1 | P1 |
| ceil_mode | ✓ 需要 | ✓ 高 | 输出大小计算模式 | P1 |
| return_indices | ✗ 不需要 | ✓ 高 | 暂不支持返回索引 | P2 |

### 3. 算法描述

```
Algorithm: max_pool2d (Forward)
────────────────────────────────────
输入: X ∈ R^{N×C×H_in×W_in}, kernel_size (kH, kW), stride (sH, sW), padding (pH, pW), dilation (dH, dW), ceil_mode
输出: Y ∈ R^{N×C×H_out×W_out}

1. 计算输出大小:
   if ceil_mode:
     H_out = ceil((H_in + 2*pH - dH*(kH-1) - 1) / sH + 1)
     W_out = ceil((W_in + 2*pW - dW*(kW-1) - 1) / sW + 1)
   else:
     H_out = floor((H_in + 2*pH - dH*(kH-1) - 1) / sH + 1)
     W_out = floor((W_in + 2*pW - dW*(kW-1) - 1) / sW + 1)

2. 对每个输出位置 (n, c, h_out, w_out):
   2.1 计算输入窗口起始位置:
       h_start = h_out * sH - pH
       w_start = w_out * sW - pW

   2.2 遍历窗口内位置:
       max_val = -inf
       for i in [0, kH):
         for j in [0, kW):
           h_in = h_start + i * dH
           w_in = w_start + j * dW
           if h_in, w_in 在有效范围内:
             max_val = max(max_val, X[n, c, h_in, w_in])

   2.3 Y[n, c, h_out, w_out] = max_val

3. return Y
```

### 4. 数据流图

```
                    输入 input
           ┌─────────────────────┐
           │  [N, C, H_in, W_in]  │
           │   float16/float32    │
           └──────────┬───────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    ┌───────────┐           ┌───────────┐
    │   padding  │           │  sliding  │
    │ (optional) │           │  window   │
    └─────┬─────┘           └─────┬─────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌───────────────┐
              │   max reduce  │
              │  per window   │
              └───────┬───────┘
                      │
                      ▼
               ┌──────────────┐
               │ 输出 output   │
               │[N,C,H_out,W_out]│
               │ float16/float32 │
               └──────────────┘

参数:
- kernel_size: (kH, kW) 池化窗口
- stride: (sH, sW) 步长
- padding: (pH, pW) 填充
- dilation: (dH, dW) 空洞率
- ceil_mode: 输出大小计算模式

动态轴: N, H_in, W_in (支持运行时变化)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| input | [N, C, H_in, W_in] 或 [C, H_in, W_in] | float16, float32 | N, H_in, W_in | 输入特征图，N 可选 |

**参数规格**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| kernel_size | int 或 (int, int) | 必填 | 池化窗口大小 |
| stride | int 或 (int, int) | kernel_size | 池化步长 |
| padding | int 或 (int, int, int, int) | 0 | 填充大小 |
| dilation | int 或 (int, int) | 1 | 空洞率 |
| ceil_mode | bool | False | 是否使用 ceil 模式计算输出大小 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| output | [N, C, H_out, W_out] 或 [C, H_out, W_out] | 与输入相同 | N, H_out, W_out | 池化输出 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认精度 |
| float16 | ✓ | 0.005 | 0.005 | 需要特殊处理防止精度损失 |

### 7. 精度要求
- **atol**: 0.001 (float32), 0.005 (float16)
- **rtol**: 0.001 (float32), 0.005 (float16)
- **对齐目标**: PyTorch F.max_pool2d

### 8. 动态轴说明
- **动态轴**: N (batch), H_in (高度), W_in (宽度)
- **轴含义**:
  - N: batch 维度，表示样本数量
  - H_in: 输入特征图高度
  - W_in: 输入特征图宽度
- **取值范围**:
  - N: [1, INT32_MAX]
  - H_in: [kernel_size, INT32_MAX]
  - W_in: [kernel_size, INT32_MAX]
  - C: 通常静态，建议范围 [1, 2048]

### 9. 边界条件处理
- **零值**: 正常计算（零值参与 max 比较）
- **极值**: 正常计算
- **NaN/Inf**: 正常计算（遵循 IEEE 754 规范，max(NaN, x) = NaN）
- **越界访问**: padding 区域视为 -inf（不参与 max 计算）

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍
- **内存优化**: 考虑使用原地操作减少内存占用

### 11. 参考信息
- **参考实现**: PyTorch torch.nn.functional.max_pool2d
- **论文**: -
- **类似算子**: avg_pool2d, max_pool1d, max_pool3d

### 12. 应用场景
- **目标模型**: ResNet, VGG, MobileNet, EfficientNet 等 CNN 网络
- **使用位置**: 下采样层，通常在卷积层之后

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| resnet_pool | 性能 | P0 | kernel=3, stride=2, padding=1 | [32, 64, 112, 112] | [32, 64, 56, 56] | ResNet 典型池化层 |
| vgg_pool | 性能 | P0 | kernel=2, stride=2, padding=0 | [32, 128, 56, 56] | [32, 128, 28, 28] | VGG 典型池化层 |
| dynamic_batch | 功能 | P0 | kernel=3, stride=2, padding=1 | [?, 64, 224, 224] | [?, 64, 112, 112] | 动态 batch 测试 |
| dynamic_hw | 功能 | P0 | kernel=2, stride=2, padding=0 | [16, 32, ?, ?] | [16, 32, H/2, W/2] | 动态 H/W 测试 |
| float16_test | 功能 | P1 | kernel=3, stride=2, padding=1 | [16, 64, 56, 56] (fp16) | [16, 64, 28, 28] (fp16) | float16 精度验证 |
| dilation_test | 功能 | P1 | kernel=3, stride=1, padding=2, dilation=2 | [8, 32, 32, 32] | [8, 32, 32, 32] | 空洞池化测试 |
| ceil_mode_test | 功能 | P1 | kernel=2, stride=2, padding=0, ceil_mode=True | [8, 32, 7, 7] | [8, 32, 4, 4] | ceil_mode 测试 |

---
*生成时间: 2026-03-28*
*确认状态: 自动生成（非交互模式）*
