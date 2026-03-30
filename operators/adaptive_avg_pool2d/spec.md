## 算子需求规范

### 1. 基础信息
- **算子名称**: adaptive_avg_pool2d
- **算子分类**: pooling
- **数学公式**: $$output[b,c,h,w] = \frac{1}{(h_{end} - h_{start}) \times (w_{end} - w_{start})} \sum_{i=h_{start}}^{h_{end}-1} \sum_{j=w_{start}}^{w_{end}-1} input[b,c,i,j]$$

  其中池化窗口位置动态计算：
  - $h_{start} = \lfloor h \times iH / oH \rfloor$
  - $h_{end} = \lceil (h+1) \times iH / oH \rceil$
  - $w_{start} = \lfloor w \times iW / oW \rfloor$
  - $w_{end} = \lceil (w+1) \times iW / oW \rceil$

- **功能描述**: 自适应平均池化，将任意尺寸的输入池化到指定的输出尺寸。池化窗口的大小和位置根据输入和输出尺寸动态计算，支持非对齐的池化窗口。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 动态轴支持 | ✓ 需要 | ✓ 高 | N, H, W 作为动态轴 | P0 |
| 输出尺寸可配置 | ✓ 需要 | ✓ 高 | output_size 参数运行时计算池化窗口 | P0 |
| 单值输出尺寸 | ✓ 需要 | ✓ 高 | output_size 为单个 int 时 oH = oW | P1 |
| 3D/4D/5D 输入支持 | ✓ 需要 | ✓ 高 | 支持 [H,W], [C,H,W], [N,C,H,W] | P1 |
| 非对齐池化窗口 | ✓ 需要 | ✓ 高 | 窗口大小和起始位置动态计算 | P0 |

### 3. 算法描述

```
Algorithm: Adaptive Average Pooling 2D
────────────────────────────────────────
输入: input [N, C, H, W], output_size (oH, oW)
输出: output [N, C, oH, oW]

1. 解析 output_size:
   - 若为单个 int: oH = oW = output_size
   - 若为 tuple: (oH, oW)

2. for each batch n in [0, N):
     for each channel c in [0, C):
       for each output position (oh, ow) in [0, oH) x [0, oW):
         2.1 计算输入窗口范围:
             h_start = floor(oh * H / oH)
             h_end = ceil((oh + 1) * H / oH)
             w_start = floor(ow * W / oW)
             w_end = ceil((ow + 1) * W / oW)

         2.2 计算窗口内的平均值:
             win_h = h_end - h_start
             win_w = w_end - w_start
             sum_val = 0
             for i in [h_start, h_end):
               for j in [w_start, w_end):
                 sum_val += input[n, c, i, j]
             output[n, c, oh, ow] = sum_val / (win_h * win_w)

3. return output
```

### 4. 数据流图

```
    输入 input                              输出 output
┌──────────────────┐                   ┌──────────────────┐
│  [N, C, H, W]    │                   │  [N, C, oH, oW]  │
│   float32        │  adaptive_avg_    │   float32        │
│                  │  pool2d           │                  │
└────────┬─────────┘                   └────────┬─────────┘
         │                                      ▲
         │    ┌────────────────────────────┐    │
         └───▶│ 参数: output_size (oH, oW) │────┘
              │                            │
              │ 动态计算每个输出位置对应的  │
              │ 输入窗口范围和窗口大小      │
              │                            │
              │ h_start = floor(h*H/oH)    │
              │ h_end = ceil((h+1)*H/oH)   │
              └────────────────────────────┘

动态轴: N, H, W
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| input | [N, C, H, W] | float32 | N, H, W | 输入张量，支持 3D/4D/5D |
| output_size | (oH, oW) 或 int | int | - | 输出尺寸，单个 int 表示 oH=oW |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| output | [N, C, oH, oW] | float32 | N, oH, oW | 输出张量，维度与输入一致 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认 |
| float16 | ✓ | 0.001 | 0.001 | 可选 |
| bfloat16 | ✓ | 0.01 | 0.01 | 可选 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: N (batch), H (height), W (width)
- **轴含义**:
  - N: batch size
  - H: 输入特征图高度
  - W: 输入特征图宽度
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算（窗口平均值可能为 0）
- **极值**: 正常计算
- **NaN/Inf**: 正常传播

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

### 11. 参考信息
- **参考实现**: PyTorch torch.nn.functional.adaptive_avg_pool2d
- **论文**: 无
- **类似算子**: adaptive_max_pool2d, avg_pool2d

### 12. 应用场景
- **目标模型**: ResNet, MobileNet, CNN 分类网络
- **使用位置**: 全局平均池化层、特征图尺寸对齐

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | output_size=(7,7) | [16, 256, 14, 14] | [16, 256, 7, 7] | 典型 CNN 特征图下采样 |
| 功能_P0 | 功能 | P0 | output_size=(1,1) | [8, 512, 7, 7] | [8, 512, 1, 1] | 全局平均池化 |
| 动态轴_P0 | 功能 | P0 | output_size=(14,14) | [N, 64, H, W] | [N, 64, 14, 14] | 动态轴验证 |
| 单值尺寸_P1 | 功能 | P1 | output_size=8 | [4, 128, 16, 16] | [4, 128, 8, 8] | 单值输出尺寸 |
| 非对齐_P1 | 功能 | P1 | output_size=(5,5) | [2, 64, 7, 7] | [2, 64, 5, 5] | 非对齐池化窗口 |

---
*生成时间: 2026-03-29*
*确认状态: 自动确认（非交互模式）*
