---
schema_version: 1
op_name: causal_conv1d
supported_dtypes: [float16]
p0_shapes: [[2048, 2048]]
tolerance: {atol: 0.01, rtol: 0.01}
dynamic_axes: ['total_len', 'batch', 'dim', 'num_cache_lines']
dynamic_axes_ranges: {total_len: [1, 8192], batch: [1, 256], dim: [256, 4096], num_cache_lines: [1, 1024]}
shape_constraints: {width: [3, 6], state_len: '>= width-1'}
default_params: {'width': 4, 'activation': 'silu'}
perf_target: null
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: causal_conv1d
- **算子分类**: custom  <!-- 序列卷积，状态管理，多模式支持 -->

### 1.1 功能描述

因果卷积算子，用于 Mamba/SSM 模型的序列处理。支持两种运行模式：
- **Prefill 模式 (FN VARLEN)**: 处理完整序列，支持变长序列（packed layout）
- **Decode 模式 (UPDATE)**: 处理单个或多个 token（投机解码），状态缓存滚动更新

核心计算为 1D 因果卷积，每个 token 的输出由其自身和前 width-1 个历史 token 加权求和，可选 silu 激活函数。

### 1.2 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| width | int | 4 | 卷积窗口大小，支持 3-6 |
| activation | str | 'silu' | 激活函数，可选 'silu' 或 None |

### 1.3 数学公式

$$
y[t] = \text{activation}\left(\text{bias} + \sum_{i=0}^{\text{width}-1} w[i] \cdot x[t-\text{width}+1+i]\right)
$$

其中：
- $x[t]$ 为当前 token，$x[t-k]$ 为历史 token（从 conv_state 缓存读取）
- $w[i]$ 为卷积权重
- silu 激活函数: $\text{silu}(x) = x / (1 + e^{-x})$

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| Prefill 模式 (FN VARLEN) | ✓ 需要 | ✓ 高 | 处理完整序列，变长支持 | P0 |
| Decode 模式 (UPDATE) | ✓ 需要 | ✓ 高 | 单 token 和投机解码支持 | P0 |
| 状态缓存管理 (conv_state) | ✓ 需要 | ✓ 高 | 存储 width-1 个历史 token | P0 |
| 激活函数融合 (silu) | ✓ 需要 | ✓ 高 | silu/swish 可选融合 | P0 |
| 变长序列支持 (cu_seqlens) | ✓ 需要 | ✓ 高 | packed layout 支持 | P0 |
| Bias 支持 | ✗ 不需要 | ⚠ 中 | P1 版本暂不支持 | P1 |
| cache_indices 间接索引 | ✗ 不需要 | ⚠ 中 | P2 版本暂不支持 | P2 |
| initial_state_mode | ✗ 不需要 | ⚠ 中 | P2 版本暂不支持 | P2 |

### 3. 算法描述

#### Prefill 模式算法

```
Algorithm: causal_conv1d_prefill (FN VARLEN)
─────────────────────────────────────────────
输入: x[total_len, dim], weight[width, dim], conv_state[num_cache, state_len, dim], cu_seqlens[batch+1]
输出: y[total_len, dim], conv_state 更新

Grid = batch_size * dim_num * seqlen_num (按 dim 和 seqlen 分块)
block_M = 64, block_D = 512 (或 256)

for each (batch_id, seq_block, dim_block) in Grid:
    1. 计算序列边界: seq_start = cu_seqlens[batch_id], seq_end = cu_seqlens[batch_id+1]
    2. 加载权重 w[0..width-1] 到 UB
    
    3. 初始化历史缓冲:
       if seq_block == 0:
           hist[0..width-2] = conv_state[batch_id, 0..width-2, d_offset]  # 从缓存加载
       else:
           hist[0..width-2] = x[seq_start + (seq_block*block_M - hist_len + h), d_offset]  # 从历史 token 加载
    
    4. for t_idx in range(block_M):  # 处理 block 内的 token
        t = seq_block * block_M + t_idx
        if t >= seqlen: break
        
        x_cur = x[seq_start + t, d_offset]
        
        # 卷积计算
        acc = bias (或 0)
        for w_idx in range(width-1):
            acc += w[w_idx] * hist[w_idx]
        acc += w[width-1] * x_cur
        
        # silu 激活
        if activation:
            out = acc / (1 + exp(-acc))
        else:
            out = acc
        
        y[seq_start + t, d_offset] = out
        
        # 滚动历史
        for h in range(hist_len - 1):
            hist[h] = hist[h + 1]
        hist[hist_len - 1] = x_cur
    
    5. if seq_block == last_block:  # 最后一个 block，更新 conv_state
        for pos in range(hist_len):
            conv_state[batch_id, pos, d_offset] = x[seq_start + seqlen - hist_len + pos, d_offset]
```

#### Decode 模式算法

```
Algorithm: causal_conv1d_decode (UPDATE)
─────────────────────────────────────────
输入: x[batch, seqlen, dim], weight[width, dim], conv_state[num_cache, state_len, dim]
输出: y[batch, seqlen, dim] 或 y[batch, dim] (seqlen=1), conv_state 更新

Grid = dim_num (按 dim 分块)
block_D = 512 (或 256)

for each dim_block in Grid:
    d_offset = dim_block * block_D
    加载权重 w[0..width-1] 到 UB
    
    for b_idx in range(batch):
        state_offset = seqlen - 1
        
        # 从 conv_state 加载历史
        hist[0] = conv_state[b_idx, state_offset + 0, d_offset]
        hist[1] = conv_state[b_idx, state_offset + 1, d_offset]
        hist[2] = conv_state[b_idx, state_offset + 2, d_offset]
        (若 width > 3, 加载更多)
        
        for t_idx in range(seqlen):
            x_cur = x[b_idx, t_idx, d_offset]
            
            # 卷积计算
            acc = 0
            for w_idx in range(width-1):
                acc += w[w_idx] * hist[w_idx]
            acc += w[width-1] * x_cur
            
            # silu 激活
            if activation:
                out = acc / (1 + exp(-acc))
            else:
                out = acc
            
            y[b_idx, t_idx, d_offset] = out
            
            # 滚动历史
            for h in range(hist_len - 1):
                hist[h] = hist[h + 1]
            hist[hist_len - 1] = x_cur
        
        # 更新 conv_state
        conv_state[b_idx, 0, d_offset] = conv_state[b_idx, state_offset + 1, d_offset]
        conv_state[b_idx, 1, d_offset] = conv_state[b_idx, state_offset + 2, d_offset]
        for t_idx in range(seqlen):
            conv_state[b_idx, 2 + t_idx, d_offset] = x[b_idx, t_idx, d_offset]
```

### 4. 数据流图

#### Prefill 模式

```
输入 x (packed)                 输入 weight               输入 conv_state
[total_len, dim]                [width, dim]              [num_cache, state_len, dim]
       │                              │                           │
       │                              │                           │
       ▼                              ▼                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    对每个 batch + token_block + dim_block             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. 加载历史 hist[0..width-2] 从 conv_state 或历史 token          │  │
│  │ 2. 加载当前 x_cur                                               │  │
│  │ 3. 加载权重 w[0..width-1]                                       │  │
│  │ 4. acc = Σ w[i] * hist[i] + w[n] * x_cur                        │  │
│  │ 5. 若 activation: out = silu(acc)                               │  │
│  │ 6. 输出 y[t], 滚动更新 hist                                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
输出 y                                  状态更新
[total_len, dim]                        conv_state 存储最后 width-1 token
```

#### Decode 模式

```
输入 x                         输入 conv_state
[batch, seqlen, dim]           [num_cache, state_len, dim]
       │                              │
       │                              │
       ▼                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    对每个 batch + dim_block                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ 1. 从 conv_state[state_offset...] 加载历史                      │  │
│  │ 2. 对每个 seqlen token:                                         │  │
│  │    - acc = Σ w[i] * hist[i] + w[n] * x_cur                      │  │
│  │    - 若 activation: out = silu(acc)                             │  │
│  │    - 滚动更新 hist                                               │  │
│  │ 3. 滚动更新 conv_state                                           │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
输出 y
[batch, dim] (seqlen=1) 或 [batch, seqlen, dim] (seqlen>1)
```

### 5. 输入输出规格

#### Prefill 模式输入规格

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| x | [total_len, dim] | float16 | total_len, dim | ✓ 高 | packed layout 输入序列 |
| weight | [width, dim] | float16 | dim | ✓ 高 | 卷积权重，kernel format |
| conv_state | [num_cache_lines, state_len, dim] | float16 | num_cache_lines, dim | ✓ 高 | 状态缓存，state_len >= width-1 |
| cu_seqlens | [batch_size + 1] | int32 | - | ✓ 高 | 变长序列边界 |

#### Decode 模式输入规格

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| x | [batch, seqlen, dim] 或 [dim] | float16 | batch, seqlen, dim | ✓ 高 | 单 token 或投机解码序列 |
| weight | [width, dim] | float16 | dim | ✓ 高 | 卷积权重 |
| conv_state | [num_cache_lines, state_len, dim] | float16 | num_cache_lines, dim | ✓ 高 | 状态缓存 |

#### 输出规格

| 模式 | 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|------|-------|-------|--------|--------|------|
| Prefill | y | [total_len, dim] | float16 | total_len, dim | ✓ 高 | 卷积输出 |
| Decode (seqlen=1) | y | [batch, dim] | float16 | batch, dim | ✓ 高 | 单 token 输出 |
| Decode (seqlen>1) | y | [batch, seqlen, dim] | float16 | batch, seqlen, dim | ✓ 高 | 投机解码输出 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float16 | ✓ | 0.01 | 0.01 | 输入输出 dtype，内部计算用 float32 |
| float32 | ✓ | 0.001 | 0.001 | 内部累加精度 |

### 7. 精度要求
- **atol**: 0.01
- **rtol**: 0.01

### 8. 动态轴说明
- **动态轴**: ['total_len', 'batch', 'dim', 'num_cache_lines']
- **轴含义**: 
  - total_len: packed 序列总长度
  - batch: 批次大小
  - dim: 特征维度
  - num_cache_lines: 缓存行数
- **取值范围**: 
  - total_len: [1, 8192]
  - batch: [1, 256]
  - dim: [256, 4096]
  - num_cache_lines: [1, 1024]

### 9. 边界条件处理
- **零值**: 正常计算
- **极值**: 正常计算，silu 有数值稳定性
- **NaN/Inf**: 正常计算，不特殊处理

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

### 11. 参考信息
- **参考实现**: vLLM mamba ops causal_conv1d, tilelang 实现
- **论文**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **类似算子**: torch.nn.Conv1d (但需要因果 padding)

### 12. 应用场景
- **目标模型**: Mamba, Jamba, SSM 模型
- **使用位置**: 序列处理层，替代 attention 的卷积层

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| Prefill_P0 | 性能 | P0 | width=4, activation=silu, seqlen=2048, dim=2048, batch=1 | x[2048,2048], weight[4,2048], conv_state[1,3,2048] | y[2048,2048] | 核心 Prefill 场景 |
| Decode_P0 | 性能 | P0 | width=4, activation=silu, seqlen=1, dim=2048, batch=1 | x[1,1,2048], weight[4,2048], conv_state[1,3,2048] | y[1,2048] | 核心 Decode 场景 |
| Prefill_Varlen_P0 | 功能 | P0 | width=4, seqlens=[512,512,512,512], dim=2048 | x[2048,2048], weight[4,2048] | y[2048,2048] | 变长序列功能验证 |

---
*生成时间: 2026-04-21*
*确认状态: 已确认*
*置信度说明: ✓ 高（自身知识库/框架知识） / ⚠ 中（外部材料提取，需确认）*