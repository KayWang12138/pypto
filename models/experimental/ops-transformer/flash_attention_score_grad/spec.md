# FlashAttentionScoreGrad 算子需求规格

## 1. 算子概述

| 属性 | 值 |
|------|-----|
| 算子名称 | flash_attention_score_grad |
| 分类 | attention |
| 功能 | Flash Attention Score 的反向计算，用于训练场景 |

## 2. 数学公式

### 正向计算（参考）
$$S = \text{Mask}(QK^T \cdot scale)$$
$$P = \text{Softmax}(S)$$
$$Y = P \cdot V$$

### 反向计算公式
$$dV = P^T \cdot dY$$
$$dQ = (dS \cdot K) \cdot scale$$
$$dK = (dS^T \cdot Q) \cdot scale$$

其中 $dS$ 通过 softmax 梯度计算：
$$dS = P \cdot (dP - \text{sum}(P \cdot dP, \text{dim}=-1, \text{keepdim=True}))$$
$$dP = dY \cdot V^T$$

> 注：等价形式 $dS = P \cdot (dP - \text{sum}(dY \cdot Y, \text{dim}=-1, \text{keepdim=True}))$，其中 $Y = P \cdot V$

## 3. 算法描述

```
Algorithm: FlashAttentionScoreGrad
────────────────────────────────────────
输入: Q, K, V, dY ∈ R^{B×N×S×D}, scale
输出: dQ, dK, dV

1. 计算 attention scores: S = Q @ K^T * scale
2. 计算 softmax: P = softmax(S, dim=-1)
3. 计算 dP: dP = dY @ V^T
4. 计算 softmax 梯度: dS = P * (dP - sum(P * dP, dim=-1, keepdim=True))
5. 计算 dV: dV = P^T @ dY
6. 计算 dQ: dQ = dS @ K * scale
7. 计算 dK: dK = dS^T @ Q * scale
```

## 4. 数据流图

```
    Q[B,N,Sq,D]    K[B,N,Skv,D]    V[B,N,Skv,D]    dY[B,N,Sq,D]
         │              │               │               │
         └──────┬───────┘               │               │
                ▼                       │               │
           ┌─────────┐                  │               │
           │ Q @ K^T │                  │               │
           └────┬────┘                  │               │
                ▼ * scale               │               │
           ┌─────────┐                  │               │
           │ softmax │                  │               │
           └────┬────┘                  │               │
                │                       │               │
                P                       │               │
                │                       │               │
                ├───────────────────────┤               │
                │                       │               │
                ▼                       ▼               │
           ┌─────────┐            ┌─────────┐          │
           │ P^T @ dY│            │ dY @ V^T│          │
           └────┬────┘            └────┬────┘          │
                │                      │               │
                ▼                      ▼               │
               dV                      dP              │
                                      │               │
                                      ▼               │
                           ┌──────────────────┐       │
                           │ P * (dP - sum()) │       │
                           └────────┬─────────┘       │
                                    │                 │
                                    dS                │
                                    │                 │
                        ┌───────────┴───────────┐     │
                        ▼                       ▼     │
                   ┌─────────┐             ┌─────────┐│
                   │dS @ K   │             │dS^T @ Q ││
                   └────┬────┘             └────┬────┘│
                        ▼ * scale               ▼ *scale
                       dQ                      dK
```

## 5. 输入输出规格

### 输入

| 名称 | Shape | 数据类型 | 说明 |
|------|-------|----------|------|
| query | [B, N, Sq, D] | BF16 | Query 张量 |
| key | [B, N, Skv, D] | BF16 | Key 张量 |
| value | [B, N, Skv, D] | BF16 | Value 张量 |
| dy | [B, N, Sq, D] | BF16 | 输出梯度 |
| scale_value | scalar | float | 缩放因子，默认 1/√D |

### 输出

| 名称 | Shape | 数据类型 | 说明 |
|------|-------|----------|------|
| dq_out | [B, N, Sq, D] | BF16 | Query 梯度 |
| dk_out | [B, N, Skv, D] | BF16 | Key 梯度 |
| dv_out | [B, N, Skv, D] | BF16 | Value 梯度 |

> **注意**：当前实现仅支持 BF16 数据类型，FP16/FP32 待后续支持。

## 6. 典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0 | 功能 | P0 | B=2,N=8,Sq=16,Skv=16,D=64 | Q,K,V,dY: [2,8,16,64] | dQ,dK,dV: [2,8,16,64] | 基础功能验证 |
| 性能_P0 | 性能 | P0 | B=1,N=8,Sq=256,Skv=256,D=128 | Q,K,V,dY: [1,8,256,128] | dQ,dK,dV: [1,8,256,128] | 性能验证 |

## 7. 精度要求

| 数据类型 | atol | rtol |
|----------|------|------|
| BFLOAT16 | 0.01 | 0.01 |

## 8. 约束说明

1. 输入 query、key、value、dy 的数据类型必须一致
2. 输入 query、key、value、dy 的 input_layout 必须一致
3. D（Head Dim）取值范围：1~512
4. Sq 和 Skv 可以不同

## 9. 生成时间

2026-03-19