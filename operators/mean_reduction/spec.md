## 算子需求规范

### 1. 基础信息
- **算子名称**: mean_reduction
- **算子分类**: reduction
- **数学公式**: $y = \text{mean}(x, \text{dim}) = \frac{\sum_{i} x_i}{N}$
- **功能描述**: 沿指定轴计算张量均值的归约操作，类似于 PyTorch 的 `torch.mean()` 或 `tensor.mean()`。支持动态轴（batch、seq 等维度）和 keepdim 参数。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 动态轴支持 | 需要 | 高 | b (batch), s (seq) 支持 runtime 动态 shape | P0 |
| 单轴归约 | 需要 | 高 | 沿单个指定轴计算均值 | P0 |
| keepdim 参数 | 需要 | 高 | 是否保持归约后的维度 | P1 |
| 多轴归约 | 不需要 | 中 | 暂不支持多轴同时归约 | P3 |
| dtype 参数 | 不需要 | 低 | 输出类型指定，暂不支持 | P3 |

### 3. 算法描述
<!-- 简单算子，公式足以描述计算逻辑，无需算法描述 -->

### 4. 数据流图

```
    输入 x                         输出 y
┌──────────────────┐         ┌──────────────────┐
│  [b, s, n, d]    │         │  [b, n, d]       │  (dim=1, keepdim=False)
│   float32        │ ──────▶ │   float32        │
└──────────────────┘         └──────────────────┘
                                   或
                             ┌──────────────────┐
                             │  [b, 1, n, d]    │  (dim=1, keepdim=True)
                             │   float32        │
                             └──────────────────┘

公式: y = mean(x, dim) = sum(x_i) / N  (沿指定轴dim归约)
动态轴: b (batch), s (seq)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [b, s, n, d] 或任意 shape | float32 | b, s | 输入张量，b=batch, s=seq_len, n=hidden_size 等 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | 去掉 dim 维度的 shape (keepdim=False) 或保持维度 (keepdim=True) | float32 | b (若 dim != 0) | 归约后的均值张量 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | Yes | 0.001 | 0.001 | 默认支持 |
| float16 | Yes | 0.01 | 0.01 | 可选支持 |
| bfloat16 | Yes | 0.01 | 0.01 | 可选支持 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: b (batch), s (seq_len)
- **轴含义**:
  - b: batch 维度，表示批次大小
  - s: sequence 维度，表示序列长度
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算 (mean of zeros = 0)
- **极值**: 正常计算 (注意 float 精度)
- **NaN/Inf**: 正常传播 (输入含 NaN/Inf 则输出可能含 NaN/Inf)

### 10. 性能要求
- **性能目标**: 无特殊要求，首跑精度成功即可

### 11. 参考信息
- **参考实现**: PyTorch `torch.mean(input, dim, keepdim=False)` - https://pytorch.org/docs/stable/generated/torch.mean.html
- **论文**: N/A
- **类似算子**: sum_reduction, max_reduction, min_reduction

### 12. 应用场景
- **目标模型**: 通用 Transformer 模型
- **使用位置**: 层归一化前的均值计算、池化层、特征聚合等

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | dim=1, keepdim=False | [b, s, 4096] | [b, 4096] | 核心性能场景，沿 seq 维度归约 |
| 功能_P0 | 功能 | P0 | dim=1, keepdim=True | [b, s, 4096] | [b, 1, 4096] | keepdim 功能验证 |
| 功能_P1 | 功能 | P1 | dim=0, keepdim=False | [b, s, 4096] | [s, 4096] | 沿 batch 维度归约 |
| 动态_P0 | 功能 | P0 | dim=1, keepdim=False | [b, s, d] (动态) | [b, d] | 动态 shape 测试 |

---
*生成时间: 2026-03-28T22:05:30Z*
*确认状态: 自动确认 (非交互模式)*
