# FusedInferAttentionScore

## 算子概述

适配增量&全量推理场景的 FlashAttention 算子，支持全量计算和增量计算（Sq=1）两种场景。

## 数学公式

```
Attention(Q, K, V) = Softmax(Q @ K^T / sqrt(d)) @ V
```

## 输入输出规格

### 输入
| 参数 | Shape | 数据类型 | 说明 |
|------|-------|----------|------|
| query | [B, N, Sq, D] | BF16 | Query 矩阵 |
| key | [B, N, Skv, D] | BF16 | Key 矩阵 |
| value | [B, N, Skv, D] | BF16 | Value 矩阵 |
| scale | scalar | float | 缩放系数 (通常 1/√D) |

### 输出
| 参数 | Shape | 数据类型 | 说明 |
|------|-------|----------|------|
| attention_out | [B, N, Sq, D] | BF16 | Attention 输出 |

## 编译运行

```bash
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux

python3 custom/fused_infer_attention_score/fused_infer_attention_score.py
```

## 测试结果

| 场景 | Shape | max diff | 状态 |
|------|-------|----------|------|
| 全量推理 | B=2, N=8, Sq=16, Skv=16, D=64 | 0.015625 | ✓ |
| 增量推理 | B=1, N=8, Sq=1, Skv=128, D=64 | 0.001953 | ✓ |

## 已知限制

1. 当前仅支持 BF16 数据类型
2. 不支持量化（INT8/INT4）
3. 不支持 atten_mask、position embedding 等扩展功能