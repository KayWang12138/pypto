## 算子需求规范

### 1. 基础信息
- **算子名称**: silu
- **算子分类**: element-wise  <!-- activation -->
- **数学公式**: $y = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
- **功能描述**: SiLU (Sigmoid Linear Unit) 激活函数，也称为 Swish。将输入逐元素乘以 sigmoid 值，在 LLM 中广泛使用（如 LLaMA、Mistral）。

### 2. 关键特性
<!-- 简单算子，无特殊特性 -->

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| - | - | - | 简单 element-wise 激活函数 | - |

### 3. 算法描述
<!-- 简单算子，公式足以描述计算逻辑 -->

逐元素计算：`y[i] = x[i] * sigmoid(x[i])`

### 4. 数据流图

```
    输入 x                    输出 y
+--------------+         +--------------+
|  [b, s, n, d] | ------->|  [b, s, n, d] |
|   float32     |  silu   |   float32     |
+--------------+         +--------------+

公式: y = x * sigmoid(x) = x / (1 + exp(-x))
动态轴: b, s (batch, seq)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [b, s, n, d] 或任意 shape | float16 / float32 / bfloat16 | 所有维度 | 输入张量 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | 与输入相同 | 与输入相同 | 与输入相同 | 输出张量，y = x * sigmoid(x) |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | Y | 0.001 | 0.001 | 默认 |
| float16 | Y | 0.001 | 0.001 | 常用 |
| bfloat16 | Y | 0.001 | 0.001 | 常用 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: b, s (batch, seq)
- **轴含义**: b = batch size, s = sequence length, n = num_heads/hidden dim split, d = head_dim
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算，0 * sigmoid(0) = 0 * 0.5 = 0
- **极值**: 正常计算，IEEE 754 标准处理
  - x -> +inf: y -> +inf * 1 = +inf
  - x -> -inf: y -> -inf * 0 = 0 (数值上趋近于0)
- **NaN/Inf**: 按 IEEE 754 标准传播

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的2倍

### 11. 参考信息
- **参考实现**: PyTorch `torch.nn.functional.silu` / `torch.ops.aten.silu`
- **论文**: "Searching for Activation Functions" (Ramachandran et al., 2017)
- **类似算子**: sigmoid, hardsigmoid, hardswish, gelu

### 12. 应用场景
- **目标模型**: LLaMA, Mistral, Swin Transformer, YOLO v7/v8
- **使用位置**: MLP 激活层、卷积后激活

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | dtype=float16 | [1, 4096, 4096] | [1, 4096, 4096] | LLaMA MLP 典型规模 |
| 功能_P0 | 功能 | P0 | dtype=float32 | [2, 1024, 512] | [2, 1024, 512] | 功能验证基础配置 |
| 功能_P1 | 功能 | P1 | dtype=bfloat16 | [4, 2048, 1024] | [4, 2048, 1024] | BF16 精度验证 |
| 边界_P0 | 边界 | P0 | dtype=float32 | [1, 1, 1] | [1, 1, 1] | 最小 shape 验证 |

---
*生成时间: 2026-03-28*
*确认状态: 已确认（自动模式）*
