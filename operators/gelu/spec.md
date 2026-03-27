# GELU 算子规格

## 基本信息

| 项目 | 值 |
|------|-----|
| 算子名称 | gelu |
| 分类 | activation |
| 公式 | GELU(x) = x * Φ(x) |
| 近似公式 | GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))) |

## 描述

GELU (Gaussian Error Linear Unit) 是一种常用的激活函数，结合了 ReLU 的非线性特性和 Dropout 的正则化效果。在 Transformer 模型（如 BERT、GPT）中广泛使用。

## 数据流图

```
    输入 x                    输出 y
┌──────────────┐         ┌──────────────┐
│  [b, s, n, d] │ ──────▶ │  [b, s, n, d] │
│   float32     │  GELU   │   float32     │
└──────────────┘         └──────────────┘

公式: GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
近似: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

动态轴: b, s (batch, seq_len)
```

## 输入规格

| 名称 | Shape | Dtype | 动态轴 | 描述 |
|------|-------|-------|--------|------|
| x | [batch, seq_len, ...] | float16, float32 | batch, seq_len | 输入张量 |

## 输出规格

| 名称 | Shape | Dtype | 动态轴 | 描述 |
|------|-------|-------|--------|------|
| y | [batch, seq_len, ...] | float16, float32 | batch, seq_len | 输出张量，shape 与输入相同 |

## 精度要求

| Dtype | atol | rtol |
|-------|------|------|
| float32 | 0.001 | 0.001 |
| float16 | 0.01 | 0.01 |

## 边界条件处理

| 场景 | 处理方式 |
|------|----------|
| 零值 | 正常计算 |
| 正/负无穷 | 正常计算（erf(±∞) = ±1） |
| NaN | 传播 NaN |

## 实现版本

使用 **近似版本**（tanh 实现），原因：
1. PyPTO 可能不支持 erf 函数
2. tanh 近似在大多数场景下精度足够
3. 计算效率更高

## 性能目标

首跑精度成功性能的 2 倍

## 参考实现

- PyTorch: `torch.nn.functional.gelu(x, approximate='tanh')`
- HuggingFace Activation Functions: https://huggingface.co/docs/transformers/main/en/model_doc/gelu

## 典型配置

| 配置名称 | 类型 | 优先级 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------------|------------|------|
| small | 性能 | P0 | [1, 128, 768] | [1, 128, 768] | BERT-base 场景 |
| medium | 性能 | P0 | [4, 512, 1024] | [4, 512, 1024] | GPT-2 场景 |
| large | 性能 | P1 | [8, 1024, 4096] | [8, 1024, 4096] | LLaMA 场景 |

---

*生成时间: 2026-03-27*
*来源: autodev 自动生成*
