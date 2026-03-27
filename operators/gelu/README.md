# GELU 算子

GELU (Gaussian Error Linear Unit) 激活函数，广泛用于 Transformer 模型。

## 公式

```
GELU(x) = x * Φ(x)
       ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

其中 Φ 是标准正态分布的累积分布函数。

## 文件结构

```
operators/gelu/
├── spec.md           # 算子规格
├── design.md         # 设计方案
├── gelu_golden.py    # PyTorch golden 参考
├── gelu_impl.py      # PyPTO 实现
├── test_gelu.py      # 测试文件
└── README.md         # 本文档
```

## 使用方法

```python
import pypto
from gelu_impl import gelu

# 创建输入
x = pypto.tensor([batch, seq_len, hidden_dim], pypto.DT_FP32, "x")

# 执行 GELU
y = gelu(x)
```

## 支持的数据类型

| Dtype | 精度要求 |
|-------|----------|
| float32 | atol=0.001, rtol=0.001 |
| float16 | atol=0.01, rtol=0.01 |

## 典型配置

| 配置 | Shape | 场景 |
|------|-------|------|
| BERT-base | [1, 128, 768] | BERT 推理 |
| GPT-2 | [4, 512, 1024] | GPT-2 训练 |
| LLaMA | [8, 1024, 4096] | LLaMA 推理 |

## 测试

```bash
cd operators/gelu
python test_gelu.py
```

## 参考

- PyTorch: `torch.nn.functional.gelu(x, approximate='tanh')`
- 论文: "Gaussian Error Linear Units (GELUs)" - https://arxiv.org/abs/1606.08415

---

*生成时间: 2026-03-27*
*来源: pypto-op-autodev*
