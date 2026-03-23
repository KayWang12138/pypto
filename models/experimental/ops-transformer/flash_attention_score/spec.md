## 算子需求规范

### 1. 基础信息
- **算子名称**: flash_attention_score
- **算子分类**: attention
- **数学公式**: 
  - 基础版本: attention_out = Softmax(scale * (query @ key^T)) @ value
  - 使用Online Softmax算法实现，输出包括attention_out、softmax_max_out和softmax_sum_out
- **功能描述**: 使用FlashAttention算法实现self-attention（自注意力）的计算。采用Online Softmax算法进行数值稳定计算，输出注意力结果、softmax最大值和softmax求和结果。

### 2. 算法描述

```
Algorithm: Flash Attention Score (Forward with Online Softmax)
────────────────────────────────────
输入: query [B, N, Sq, D], key [B, N, Skv, D], value [B, N, Skv, D]
      scale_value
输出: attention_out [B, N, Sq, D], softmax_max_out [B, N, Sq, 1], softmax_sum_out [B, N, Sq, 1]

1. 初始化输出:
   attention_out = zeros([B, N, Sq, D])
   softmax_max_out = full([B, N, Sq, 1], -inf)
   softmax_sum_out = zeros([B, N, Sq, 1])

2. 外层循环 (S1_LOOP - 按query分块):
   for s1_idx in range(Sq // TILE_S1):
       s1_start = s1_idx * TILE_S1
       q_block = query[:, :, s1_start:s1_start+TILE_S1, :]
       
       m_running = full([TILE_S1, 1], -inf)  # 当前最大值
       l_running = zeros([TILE_S1, 1])       # 当前指数和
       acc_running = zeros([TILE_S1, D])     # 当前累加结果
       
       3. 内层循环 (S2_LOOP - 按key/value分块):
          for s2_idx in range(Skv // TILE_S2):
              s2_start = s2_idx * TILE_S2
              k_block = key[:, :, s2_start:s2_start+TILE_S2, :]
              v_block = value[:, :, s2_start:s2_start+TILE_S2, :]
              
              4. 计算注意力分数:
                 scores = q_block @ k_block^T
                 scores_scaled = scores * scale_value
              
              5. Online Softmax更新:
                 m_new = max(m_running, max(scores_scaled, dim=-1))
                 scores_shifted = scores_scaled - m_new
                 exp_scores = exp(scores_shifted)
                 
                 # 更新指数和
                 m_diff = m_running - m_new
                 exp_m_diff = exp(m_diff)
                 l_new = l_running * exp_m_diff + sum(exp_scores, dim=-1)
                 
                 # 更新累加结果
                 acc_running = acc_running * exp_m_diff + exp_scores @ v_block
                 
                 m_running = m_new
                 l_running = l_new
       
       6. 计算最终输出:
          attention_out[:, :, s1_start:s1_start+TILE_S1, :] = acc_running / l_running
          softmax_max_out[:, :, s1_start:s1_start+TILE_S1, :] = m_running
          softmax_sum_out[:, :, s1_start:s1_start+TILE_S1, :] = l_running

7. 返回 attention_out, softmax_max_out, softmax_sum_out
```

### 3. 数据流图

```
   query        key         value
     │           │            │
     │           │            │
     ▼           ▼            ▼
  ┌─────┐    ┌─────┐      ┌─────┐
  │Q_block│    │K_block│      │V_block│
  └──┬──┘    └──┬──┘      └──┬──┘
     │           │            │
     └─────┬─────┘            │
           ▼                  │
      Q @ K^T                 │
           │                  │
           ▼                  │
       * scale_value          │
           │                  │
           ▼                  │
    ┌─────────────┐            │
    │ Online      │◄───────────┤
    │ Softmax     │   V_block  │
    └──────┬──────┘            │
           │                   │
     ┌─────┴─────┐             │
     │           │             │
     ▼           ▼             │
 softmax_max  softmax_sum      │
     │                         │
     ▼                         │
 attn_weights @ V_block        │
     │                         │
     ▼                         │
 attention_out ◄────────────────┘
```

### 4. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| query | [B, N, Sq, D] | BFLOAT16 | B, Sq | 查询张量 |
| key | [B, N, Skv, D] | BFLOAT16 | B, Skv | 键张量 |
| value | [B, N, Skv, D] | BFLOAT16 | B, Skv | 值张量 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| attentionOut | [B, N, Sq, D] | BFLOAT16 | B, Sq | 注意力输出 |
| softmaxMaxOut | [B, N, Sq, 1] | FLOAT32 | B, Sq | Softmax的Max中间结果 |
| softmaxSumOut | [B, N, Sq, 1] | FLOAT32 | B, Sq | Softmax的Sum中间结果 |

**属性规格**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| scaleValue | FLOAT | 1.0 / sqrt(D) | 缩放系数，通常为1/sqrt(head_dim) |

### 5. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| bfloat16 | ✓ | 0.01 | 0.01 | 推荐类型 |

### 6. 精度要求
- **atol**: 0.01 (bfloat16)
- **rtol**: 0.01 (bfloat16)

### 7. 动态轴说明
- **动态轴**: B (batch), Sq (query序列长度), Skv (key/value序列长度)
- **轴含义**: 
  - B: batch size
  - Sq: query序列长度
  - Skv: key/value序列长度
  - N: 注意力头数
  - D: 每个头的维度
- **取值范围**: 
  - B: 1~2M
  - N: 1~256
  - S: 1~1M
  - D: 1~768

### 8. 边界条件处理
- **零值**: 正常计算
- **极值**: 通过Online Softmax算法保证数值稳定性
- **NaN/Inf**: Online Softmax算法避免数值溢出

### 9. 性能要求
- **性能目标**: 在NPU上达到与Ascend C算子相当的性能
- **分块策略**: TILE_S1=16, TILE_S2=16

### 10. 参考信息
- **参考实现**: Flash Attention论文
- **类似算子**: Scaled Dot-Product Attention

### 11. 应用场景
- **目标模型**: Transformer架构的大模型
- **使用位置**: Self-Attention层

**典型配置**（建议至少提供一个，用于下游 golden 验证和设计方案生成）:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0 | 功能 | P0 | B=2,N=8,Sq=16,Skv=16,D=64 | Q/K/V:[2,8,16,64] | Out:[2,8,16,64] | 基础功能验证 |
| 性能_P0 | 性能 | P0 | B=8,N=8,Sq=512,Skv=512,D=64 | Q/K/V:[8,8,512,64] | Out:[8,8,512,64] | 中等规模性能测试 |
| 性能_P1 | 性能 | P1 | B=1,N=8,Sq=1024,Skv=1024,D=64 | Q/K/V:[1,8,1024,64] | Out:[1,8,1024,64] | 大序列长度性能测试 |

---
*生成时间: 2026-03-19*
*确认状态: 已确认*
