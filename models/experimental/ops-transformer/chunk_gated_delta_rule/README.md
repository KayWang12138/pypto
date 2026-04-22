# chunk_gated_delta_rule

## 算子概述

基于 chunk 的门控 Delta Rule 前向传播算子，用于线性注意力机制（Linear Attention）中的隐藏状态递推计算。

## 数学公式

对于每个 chunk t (t = 0, 1, ..., NT-1):
```
h[t] = h[t-1]  (累积状态，初始为 h0 或零)
v_new[t] = v[t] - w[t] · h[t]  (残差计算)
若 USE_G:
    g_last = g[t_valid - 1]  (chunk 内最后一个有效 token 的 gate)
    v_new[t] = v_new[t] · exp(g_last - g[t])  (门控缩放)
    h[t] = h[t] · exp(g_last)  (状态衰减)
h[t+1] = h[t] + k[t]^T · v_new[t]  (状态更新)
```

## 目录结构

```
custom/chunk_gated_delta_rule/
├── SPEC.md                           # 算子需求规范
├── API_REPORT.md                     # API 探索报告
├── DESIGN.md                         # 设计方案
├── chunk_gated_delta_rule_golden.py  # Golden 参考实现
├── chunk_gated_delta_rule_impl.py    # PyPTO 实现
├── test_chunk_gated_delta_rule.py    # 测试入口
└── README.md                         # 说明文档
```

## 输入输出规格

**输入**:
- k: [B, T, Hg, K], float16 - Key 向量，GQA 模式
- w: [B, T, H, K], float16 - 门控权重
- v: [B, T, H, V], float16 - Value 向量
- g: [B, T, H], float32 - 门控向量 (可选)
- h0: [B, H, K, V], float16 - 初始状态 (可选)
- cu_seqlens: [N+1], int32 - 变长序列边界 (可选)

**输出**:
- h: [B, NT, H, K, V], float16 - 每个 chunk 的隐藏状态
- v_new: [B, T, H, V], float16 - 更新后的 value
- ht: [B, H, K, V], float16 - 最终隐藏状态 (可选)

**固定常量**:
- K = 128 (Key dimension)
- V = 128 (Value dimension)
- BT = 64 (Chunk size)

## 运行方式

```bash
# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export TILE_FWK_DEVICE_ID=0

# 运行测试
python test_chunk_gated_delta_rule.py

# 运行单个测试
python test_chunk_gated_delta_rule.py chunk_gated_delta_rule::test_fixed_p0
```

## 测试规格

- **定长模式**: B=1, T=2048, H=8, Hg=4, K=128, V=128
  - use_g=True/False, use_initial_state=True/False (4 组)
- **变长模式**:
  - seqlens=[512,512,512,512]
  - seqlens=[128,256,512,1024,128]
  - seqlens=[2048]
  - seqlens=[1024,1024]

## 精度要求

- rtol: 0.05
- atol: 0.05

## 已知限制

1. 当前实现使用固定配置 H=8, Hg=4，不支持动态 head 数配置
2. 编译过程中存在 PyPTO API 约束问题（expand_clone 广播限制）

## 验证入口

```python
from chunk_gated_delta_rule_impl import chunk_gated_delta_rule_wrapper
from chunk_gated_delta_rule_golden import chunk_gated_delta_rule_golden

# 生成输入数据
k = torch.randn(1, 2048, 4, 128, dtype=torch.float16)
w = torch.randn(1, 2048, 8, 128, dtype=torch.float16)
v = torch.randn(1, 2048, 8, 128, dtype=torch.float16)
g = torch.randn(1, 2048, 8, dtype=torch.float32)
h0 = torch.randn(1, 8, 128, 128, dtype=torch.float16)

# PyPTO 实现
h, v_new, ht = chunk_gated_delta_rule_wrapper(k, w, v, g, h0)

# Golden 参考
h_ref, v_new_ref, ht_ref = chunk_gated_delta_rule_golden(k, w, v, g, h0)

# 精度对比
torch.testing.assert_close(h, h_ref, rtol=0.05, atol=0.05)
```

---
*生成时间: 2026-04-21*