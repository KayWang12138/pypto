## 算子需求规范

### 1. 基础信息
- **算子名称**: rms_norm
- **算子分类**: normalization
- **数学公式**: $y = x \cdot \gamma / \sqrt{\text{mean}(x^2) + \epsilon}$
- **功能描述**: 均方根归一化（Root Mean Square Normalization）。对输入张量在指定维度上计算均方根（RMS），然后用 RMS 进行归一化，最后乘以可学习的缩放参数。与 Layer Norm 相比，RMS Norm 不进行均值中心化，计算更简单，在 Transformer 架构（如 LLaMA、GPT-NeoX）中广泛使用。

### 2. 关键特性
<!-- 简单算子，无需填写关键特性表 -->

### 3. 算法描述
<!-- 公式已完整表述计算逻辑，无需算法描述 -->

### 4. 数据流图

```
    输入 x                    weight γ               eps
┌──────────────┐         ┌──────────────┐         ┌───┐
│  [b, s, d]   │         │    [d]       │         │1e-6│
└──────┬───────┘         └──────┬───────┘         └─┬─┘
       │                        │                   │
       │    ┌───────────────────┼───────────────────┘
       ▼    ▼                   ▼
    ┌─────────────┐
    │   x^2       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ mean(x^2)   │  ← 在最后一个维度上求均值
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ + eps       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   sqrt()    │  ← RMS
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ x / RMS     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   * γ       │
    └──────┬──────┘
           │
           ▼
    ┌──────────────┐
    │  输出 y       │
    │  [b, s, d]   │
    └──────────────┘

公式: y = x * γ / sqrt(mean(x^2) + ε)
动态轴: b (batch), s (seq_len)
```

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| x | [b, s, d] | float32 / bfloat16 | b, s | 输入张量，b为batch维度，s为序列长度，d为隐藏层维度 |
| weight | [d] | float32 / bfloat16 | 无 | 可学习的缩放参数γ，shape与归一化维度相同 |
| eps | scalar | float | 无 | 防止除零的小常数，默认值1e-6 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| y | [b, s, d] | float32 / bfloat16 | b, s | 归一化后的输出张量，shape与输入x相同 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认 |
| bfloat16 | ✓ | 0.01 | 0.01 | 推荐 |

### 7. 精度要求
- **atol**: 0.001 (float32) / 0.01 (bfloat16)
- **rtol**: 0.001 (float32) / 0.01 (bfloat16)

### 8. 动态轴说明
- **动态轴**: b (batch), s (seq_len)
- **轴含义**:
  - b: batch 维度，表示批次大小
  - s: seq_len 维度，表示序列长度
  - d: hidden_size 维度，固定大小
- **取值范围**:
  - b: [1, INT32_MAX]
  - s: [1, INT32_MAX]
  - d: 固定值，通常为 4096, 5120, 6656, 8192 等常见隐藏层大小

### 9. 边界条件处理
- **零值**: 正常计算（eps 防止除零）
- **极值**: 正常计算
- **NaN/Inf**: 正常计算，输入含 NaN/Inf 时输出可能含 NaN/Inf

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的2倍

### 11. 参考信息
- **参考实现**:
  - PyTorch: 无内置实现，参考 `torch.nn.LayerNorm`
  - HuggingFace: `transformers.models.llama.modeling_llama.LlamaRMSNorm`
  - NVIDIA: `apex.normalization.FusedRMSNorm`
- **论文**: Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
- **类似算子**: layer_norm, instance_norm, group_norm

### 12. 应用场景
- **目标模型**: LLaMA, GPT-NeoX, LLaMA2, LLaMA3 等 Transformer 架构
- **使用位置**: 每个 Transformer Block 的 Attention 和 FFN 层之前

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| LLaMA-7B | 性能 | P0 | eps=1e-6, d=4096 | x:[b,s,4096], weight:[4096] | y:[b,s,4096] | LLaMA-7B 模型配置 |
| LLaMA-13B | 性能 | P0 | eps=1e-6, d=5120 | x:[b,s,5120], weight:[5120] | y:[b,s,5120] | LLaMA-13B 模型配置 |
| LLaMA-70B | 性能 | P1 | eps=1e-6, d=8192 | x:[b,s,8192], weight:[8192] | y:[b,s,8192] | LLaMA-70B 模型配置 |
| 功能验证 | 功能 | P0 | eps=1e-6, d=64 | x:[2,128,64], weight:[64] | y:[2,128,64] | 小规模功能验证 |
| 动态轴测试 | 功能 | P0 | eps=1e-6, d=256 | x:[b,s,256], weight:[256] | y:[b,s,256] | 动态 batch 和 seq_len |

---
*生成时间: 2026-03-28T00:00:00Z*
*确认状态: 已确认（非交互模式自动生成）*
