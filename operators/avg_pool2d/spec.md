## 算子需求规范

### 1. 基础信息
- **算子名称**: avg_pool2d
- **算子分类**: reduction  <!-- 池化操作属于 reduction 类 -->
- **数学公式**: $$\text{output}[n, c, oh, ow] = \frac{1}{k_h \times k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \text{input}[n, c, oh \times s_h + i, ow \times s_w + j]$$
- **功能描述**: 对输入张量应用 2D 平均池化操作，支持 SAME 和 VALID 两种填充模式。该算子对输入特征图应用滑动窗口进行平均计算，常用于卷积神经网络中的下采样操作。

### 2. 关键特性
<!-- 简单算子，无需复杂特性拆解 -->

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 动态轴支持 | ✓ 需要 | ✓ 高 | batch_size 和 channels 维度支持动态 shape | P0 |
| SAME padding | ✓ 需要 | ✓ 高 | 输出尺寸向上取整，自动计算 padding | P0 |
| VALID padding | ✓ 需要 | ✓ 高 | 不填充，输出尺寸按公式计算 | P0 |

### 3. 算法描述
<!-- 简单算子，公式足以描述计算逻辑，无需算法描述 -->

### 4. 数据流图

```
                    输入 x
            ┌──────────────────┐
            │ [b, c, in_h, in_w] │
            │     float32        │
            └─────────┬──────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │    2D Avg Pooling   │
            │ kernel: (k_h, k_w)  │
            │ stride: (s_h, s_w)  │
            │ padding: SAME/VALID │
            └─────────┬───────────┘
                      │
                      ▼
                    输出 y
            ┌────────────────────┐
            │ [b, c, out_h, out_w] │
            │      float32         │
            └──────────────────────┘

动态轴: b (batch_size), c (channels)

SAME padding 输出尺寸:
  out_h = ceil(in_h / s_h)
  out_w = ceil(in_w / s_w)

VALID padding 输出尺寸:
  out_h = ceil((in_h - k_h + 1) / s_h)
  out_w = ceil((in_w - k_w + 1) / s_w)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [batch_size, channels, in_h, in_w] | float32 | batch_size, channels | 输入特征图 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [batch_size, channels, out_h, out_w] | float32 | batch_size, channels | 输出特征图 |

**属性参数**:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| kernel_size | Tuple[int, int] | 无（必须） | 池化窗口大小 (k_h, k_w) |
| stride | Tuple[int, int] | kernel_size | 步长 (s_h, s_w) |
| padding_mode | str | 'SAME' | 填充模式: 'SAME' 或 'VALID' |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认，主要支持类型 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: batch_size, channels
- **轴含义**:
  - batch_size: 批次大小，表示一次处理的样本数量
  - channels: 通道数，表示特征图的深度
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算（padding 区域填充 0）
- **极值**: 正常计算
- **NaN/Inf**: 正常计算（不特殊处理）

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

### 11. 参考信息
- **参考实现**: models/experimental/vector/AvgPool2d/avg_pool2d.py
- **论文**: 无
- **类似算子**: tf.nn.avg_pool2d, torch.nn.functional.avg_pool2d

### 12. 应用场景
- **目标模型**: 通用 CNN 网络
- **使用位置**: 卷积层后的下采样层

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| SAME_P0 | 性能 | P0 | kernel=(2,2), stride=(2,2), padding=SAME | [2, 3, 6, 6] | [2, 3, 3, 3] | SAME 模式核心场景 |
| VALID_P0 | 功能 | P0 | kernel=(3,3), stride=(2,2), padding=VALID | [4, 8, 12, 12] | [4, 8, 5, 5] | VALID 模式功能验证 |
| SAME_large | 性能 | P1 | kernel=(3,3), stride=(2,2), padding=SAME | [8, 64, 56, 56] | [8, 64, 28, 28] | 大尺寸特征图性能测试 |
| VALID_stride1 | 功能 | P2 | kernel=(2,2), stride=(1,1), padding=VALID | [2, 16, 8, 8] | [2, 16, 7, 7] | stride=1 边界场景 |

---
*生成时间: 2026-03-29*
*确认状态: 已确认（非交互模式自动确认）*
