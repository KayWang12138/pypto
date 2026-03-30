# scaled_dot_product_attention 箐

## 概述

本算子实现 Transformer 架构中的缩放点积注意力机制（Scaled Dot-Product Attention），是注意力机制的核心计算单元。

### 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 功能特性

| 特性 | 支持状态 | 说明 |
|------|----------|------|
| 缩放因子 | 支持 | 默认 1/sqrt(d)，可自定义 |
| 注意力掩码 | 支持 | 支持 float 类型掩码 |
| 因果掩码 | 支持 | is_causal=True |
| Dropout | 不支持 | 当前版本未实现 |
| 动态轴 | 支持 | batch, num_heads, seq_len |
| 数据类型 | 支持 | FP32, BF16 |

| is_causal | 支持 | 启用因果掩码 |
| attn_mask | 支持 | 支持 float 类型掩码 |
| scale | 支持 | 传入 scale参数 |
| dropout_p | 不支持 | Dropout 概率，| 数值稳定性 | 需要 | 数值稳定的 softmax 实现 |

| GQA | 不支持 | Grouped Query Attention |
| enable_gqa | 不需要 | 实验性功能 |

## 目录结构

```
operators/scaled_dot_product_attention/
├── spec.md                                  # 需求规范
├── api_report.md                            # API 探索报告
├── design.md                                # 设计文档
├── scaled_dot_product_attention_golden.py        # Golden 参考实现
├── scaled_dot_product_attention_impl.py          # 算子核心实现代码
├── test_scaled_dot_product_attention.py          # 测试代码
├── output/                         # 运行输出(自动生成)
└── README.md                               # 实现说明
```

[!IMPORTANT]
    - 本实现基于静态 shape 配置，    - **不支持动态 shape**: 动态 shape 需要使用 `pypto.loop` 进行切分处理
    - **不支持 dropout**: Dropout 当前不支持，需要自定义方案
    - **softmax 仅支持 FP32**: 输入会被转换为 FP32 进行计算

    - **transpose 4D 仅支持 (2,3)**: 4D transpose 只支持 (2,3) 轴交换

    - **bool 掩码不支持**: 仅支持 float 类型掩码