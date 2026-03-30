## 算子需求规范

### 1. 基础信息
- **算子名称**: linear
- **算子分类**: matmul
- **数学公式**: $output = input \cdot weight^T + bias$
- **功能描述**: 线性层（全连接层）算子，对输入张量进行线性变换。支持任意维度的输入张量，对最后一维进行线性变换，保持其他维度不变。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 动态 Shape (batch, seq_len) | ✓ 需要 | ✓ 高 | 输入前 N-1 维可动态变化 | P0 |
| bias_add 融合 | ✓ 需要 | ✓ 高 | 可选择融合或分开实现 | P1 |
| 多 dtype 支持 | ✗ 不需要 | ⚠ 中 | 仅支持 float32 | P2 |
| tiling 策略 | ? 待定 | ⚠ 中 | 根据性能需求决定 | P2 |

### 3. 算法描述
<!-- 简单矩阵乘法算子，公式已足够描述计算逻辑，无需算法描述 -->

### 4. 数据流图

```
     输入 input         权重 weight       偏置 bias (可选)
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │[..., in_feat]│   │[out, in_feat]│   │   [out]      │
 │   float32    │   │   float32    │   │  float32     │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                  │                   │
        │                  ▼                   │
        │           ┌────────────┐             │
        │           │  weight^T  │             │
        │           └──────┬─────┘             │
        │                  │                   │
        ▼                  ▼                   │
     ┌─────────────────────────┐               │
     │     input @ weight^T    │               │
     └───────────┬─────────────┘               │
                 │                             │
                 ▼                             ▼
          ┌─────────────────────────────────────┐
          │              + (bias)               │
          └──────────────┬──────────────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ 输出 output   │
                  │[..., out_feat]│
                  │   float32    │
                  └──────────────┘

公式: output = input @ weight.T + bias
动态轴: batch, seq_len (input的前N-1维)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| input | [..., in_features] | float32 | batch, seq_len | 输入张量，最后一维为输入特征数 |
| weight | [out_features, in_features] | float32 | 无 | 权重矩阵 |
| bias | [out_features] | float32 | 无 | 偏置向量（可选） |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| output | [..., out_features] | float32 | batch, seq_len | 输出张量，最后一维为输出特征数 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认支持 |
| float16 | ✗ | - | - | 暂不支持 |
| bfloat16 | ✗ | - | - | 暂不支持 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: batch, seq_len
- **轴含义**:
  - batch: 批次维度，表示样本数量
  - seq_len: 序列长度维度，表示序列中 token 数量
- **取值范围**: [1, INT32_MAX]

### 9. 边界条件处理
- **零值**: 正常计算
- **极值**: 正常计算
- **NaN/Inf**: 正常计算

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的2倍

### 11. 参考信息
- **参考实现**: PyTorch torch.nn.functional.linear
- **论文**: 无
- **类似算子**: matmul, addmm

### 12. 应用场景
- **目标模型**: Transformer 类模型（BERT、GPT 等）
- **使用位置**: 全连接层、MLP 层、输出层

**典型配置**（建议至少提供一个，用于下游 golden 验证和设计方案生成）:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0_2D | 性能 | P0 | in_features=4096, out_features=4096, has_bias=True | input: [batch, 4096], weight: [4096, 4096], bias: [4096] | output: [batch, 4096] | 核心 2D 性能场景 |
| 性能_P0_3D | 性能 | P0 | in_features=1024, out_features=4096, has_bias=True | input: [batch, seq_len, 1024], weight: [4096, 1024], bias: [4096] | output: [batch, seq_len, 4096] | 核心 3D 性能场景（含动态轴） |
| 功能_P0_no_bias | 功能 | P0 | in_features=512, out_features=512, has_bias=False | input: [batch, 512], weight: [512, 512] | output: [batch, 512] | 无偏置功能验证 |
| 功能_P1_4D | 功能 | P1 | in_features=256, out_features=512, has_bias=True | input: [batch, heads, seq_len, 256], weight: [512, 256], bias: [512] | output: [batch, heads, seq_len, 512] | 4D 输入功能验证 |

---
*生成时间: 2026-03-28*
*确认状态: 自动确认（非交互模式）*
