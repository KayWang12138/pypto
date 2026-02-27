# GLM-4.5 Incremental Flash Attention (IFA) 实现原理

> 本文档详细分析 GLM-4.5 模型中 Incremental Flash Attention (IFA) 的 PyPTO 实现，包括 Tile 分块策略、Flash Attention 算法原理、Paged KV Cache 管理以及性能优化技术。

---

## 目录

- [一、概述](#一概述)
- [二、数学原理](#二数学原理)
- [三、Paged Attention 机制](#三paged-attention-机制)
- [四、Flash Attention 算法](#四flash-attention-算法)
- [五、Tile 分块策略](#五tile-分块策略)
- [六、核心代码分析](#六核心代码分析)
- [七、数据流与循环结构](#七数据流与循环结构)
- [八、性能优化技术](#八性能优化技术)
- [九、完整执行流程](#九完整执行流程)

---

## 一、概述

### 1.1 功能定位

GLM-4.5 IFA（Incremental Flash Attention）是专为**大模型推理场景**设计的高性能注意力机制实现，主要解决以下挑战：

| 挑战 | 解决方案 |
|-----|---------|
| **内存碎片化** | Paged KV Cache 管理 |
| **显存利用率低** | 基于块的灵活分配 |
| **推理吞吐受限** | Flash Attention + Tile 分块计算 |

### 1.2 核心特性

```
┌─────────────────────────────────────────────────────────────┐
│                  GLM-4.5 IFA 核心特性                        │
├─────────────────────────────────────────────────────────────┤
│  • Paged Attention：分页式 KV Cache 管理                    │
│  • Flash Attention：在线 Softmax，避免存储完整注意力矩阵     │
│  • Tile 分块计算：硬件感知的数据切分，优化内存访问           │
│  • 动态 Shape 支持：处理变长序列和动态 Batch                 │
│  • GQA 支持：Grouped Query Attention，KV 头数可小于 Q 头数  │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 输入输出规格

```
输入张量：
├── query:           [num_tokens, num_head, head_size]     (BF16)
├── key_cache:       [num_blocks, block_size, kv_head_num, head_size] (BF16)
├── value_cache:     [num_blocks, block_size, kv_head_num, head_size] (BF16)
├── block_tables:    [batch_size, max_num_blocks_per_query] (INT32)
└── actual_seqs:     [batch_size]                          (INT32)

输出张量：
└── attn_res:        [num_tokens, num_head, head_size]     (BF16)
```

---

## 二、数学原理

### 2.1 标准 Attention 公式

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：
- $Q \in \mathbb{R}^{N \times d}$：Query 矩阵
- $K \in \mathbb{R}^{M \times d}$：Key 矩阵
- $V \in \mathbb{R}^{M \times d}$：Value 矩阵
- $d_k$：头维度

### 2.2 Flash Attention 的 Online Softmax

传统方法需要存储完整的 $N \times M$ 注意力矩阵，Flash Attention 通过**分块计算 + 在线归一化**避免这一问题。

**核心思想**：将 Softmax 计算分解为可增量更新的形式。

定义：
$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{current}})
$$

$$
d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + d_{\text{current}} \cdot e^{m_{\text{current}} - m_{\text{new}}}
$$

$$
O_{\text{new}} = O_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + O_{\text{current}} \cdot e^{m_{\text{current}} - m_{\text{new}}}
$$

最终输出：
$$
O_{\text{final}} = \frac{O}{d}
$$

---

## 三、Paged Attention 机制

### 3.1 设计动机

传统 KV Cache 管理方式存在以下问题：

```
传统方式：
┌──────────────────────────────────────────────────┐
│  连续内存分配 → 固定大小 → 内存碎片化            │
│  序列长度变化 → 频繁重分配 → 性能开销            │
│  Batch 动态变化 → 对齐填充 → 显存浪费            │
└──────────────────────────────────────────────────┘
```

Paged Attention 借鉴操作系统**虚拟内存分页**思想：

```
Paged 方式：
┌──────────────────────────────────────────────────┐
│  块级管理 → 按需分配 → 无碎片                   │
│  非连续存储 → 灵活映射 → 高效利用               │
│  Block Table → 间接寻址 → 动态管理              │
└──────────────────────────────────────────────────┘
```

### 3.2 内存布局

```
Block Pool（物理内存）：
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ Block 0│ Block 1│ Block 2│ Block 3│ Block 4│  ...   │
│ [128,d]│ [128,d]│ [128,d]│ [128,d]│ [128,d]│        │
└────────┴────────┴────────┴────────┴────────┴────────┘

Block Table（逻辑映射）：
┌─────────────────────────────────────┐
│ Batch 0: [2, 0, 3, -1, -1, ...]    │  → 序列使用 Block 2, 0, 3
│ Batch 1: [1, 4, -1, -1, -1, ...]   │  → 序列使用 Block 1, 4
│ ...                                 │
└─────────────────────────────────────┘

实际序列长度：
┌─────────────────────────────────────┐
│ actual_seqs = [384, 256, ...]      │  → Batch 0 实际 384 tokens
└─────────────────────────────────────┘
```

### 3.3 Block Table 生成

```python
def gen_block_table(actual_seq_len, block_size, block_table_shape):
    """
    生成块映射表
    
    Args:
        actual_seq_len: 每个序列的实际长度
        block_size: 每个块的大小（如 128）
        block_table_shape: [batch_size, max_blocks_per_query]
    
    Returns:
        block_table: 块索引映射表，-1 表示未使用
    """
    # 计算每个序列需要的块数
    block_num_per_batch = [ceil(seq / block_size) for seq in actual_seq_len]
    
    # 随机分配块索引（模拟非连续分配）
    block_idx_list = random_permutation(total_blocks)
    
    # 填充 block_table
    for batch_idx, num_blocks in enumerate(block_num_per_batch):
        for j in range(num_blocks):
            block_table[batch_idx][j] = block_idx_list[idx++]
```

---

## 四、Flash Attention 算法

### 4.1 算法流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flash Attention 计算流程                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │  Tile 1 │ -> │  Tile 2 │ -> │  Tile 3 │ -> │  Tile N │      │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘      │
│       │              │              │              │            │
│       v              v              v              v            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Q x K^T │    │ Q x K^T │    │ Q x K^T │    │ Q x K^T │      │
│  │ Scale   │    │ Scale   │    │ Scale   │    │ Scale   │      │
│  │ Max     │    │ Max     │    │ Max     │    │ Max     │      │
│  │ Exp     │    │ Exp     │    │ Exp     │    │ Exp     │      │
│  │ Sum     │    │ Sum     │    │ Sum     │    │ Sum     │      │
│  │ x V     │    │ x V     │    │ x V     │    │ x V     │      │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘      │
│       │              │              │              │            │
│       v              v              v              v            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Update  │ -> │ Update  │ -> │ Update  │ -> │ Final   │      │
│  │ m, d, O │    │ m, d, O │    │ m, d, O │    │ O / d   │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 在线 Softmax 更新公式

**首次迭代（s2_idx == 0）**：

```python
# 计算注意力分数
sij = Q @ K^T                    # [g_tile, s2_tile]
sij_scale = sij * scale          # 缩放

# 计算当前块的 max 和 exp
mij = max(sij_scale, dim=-1)     # [g_tile, 1]
pij = exp(sij_scale - mij)       # [g_tile, s2_tile]

# 累积 sum 和输出
sum_update = sum(pij, dim=-1)    # [g_tile, 1]
max_update = mij                 # [g_tile, 1]

# 计算 output
oi_tmp = pij @ V                 # [g_tile, d]
oi_update = oi_tmp
```

**后续迭代（s2_idx > 0）**：

```python
# 计算当前块
sij = Q @ K^T
sij_scale = sij * scale
mij = max(sij_scale, dim=-1)
pij = exp(sij_scale - mij)

# 更新 max
m_new = max(max_update, mij)     # 新的最大值

# 更新 sum（需要校正旧值）
correction = exp(max_update - m_new)
sum_update = sum_update * correction + sum(pij)

# 更新 max
max_update = m_new

# 更新 output
oi_tmp = pij @ V
oi_update = oi_update * correction + oi_tmp
```

**最终输出（s2_idx == last）**：

```python
# 归一化
oi_final = oi_update / sum_update
```

---

## 五、Tile 分块策略

### 5.1 Tile 配置结构

```python
@dataclass
class AttentionTileConfig:
    g_tile: int              # Query 组大小（多头分组）
    s2_tile: int             # KV 序列长度 tile
    c1_tile_shape: list      # QK^T 矩阵乘法的 Cube tile
    v1_tile_shape: list      # Softmax 前的 Vector tile
    c2_tile_shape: list      # PV 矩阵乘法的 Cube tile
    v2_tile_shape: list      # 最终输出的 Vector tile
```

### 5.2 默认配置示例

```python
def get_qwen_common_config():
    # 模型参数
    b = 8           # batch size
    s1 = 1          # query 序列长度（解码阶段为 1）
    s2 = 16384      # KV 序列长度
    q_d = 128       # 头维度
    nq = 12         # Query 头数
    nkv = 1         # KV 头数（GQA）
    block_size = 128
    
    # Tile 配置
    cube_tile = 128
    m_tile = 128
    s2_tile = 512
    
    tile_cfg = AttentionTileConfig(
        g_tile=nq,                          # 12 个 query 头一组
        s2_tile=s2_tile,                    # KV 序列按 512 分块
        c1_tile_shape=[[128, 128], [128, 128], [128, 128]],  # QK^T
        v1_tile_shape=[128, 512],            # Softmax
        c2_tile_shape=[[128, 128], [128, 128], [128, 128]],  # PV
        v2_tile_shape=[128, 128]             # Output
    )
```

### 5.3 Tile 分块图示

```
Q 矩阵分块：
┌────────────────────────────────────┐
│  Query: [b*s1, nq, d]              │
│                                    │
│  ┌──────┬──────┬──────┐           │
│  │ Q_g0 │ Q_g1 │ Q_g2 │ ...       │  g_tile = nq
│  │[1,12,│[1,12,│[1,12,│           │  每个 tile 包含所有 query 头
│  │ 128] │ 128] │ 128] │           │
│  └──────┴──────┴──────┘           │
└────────────────────────────────────┘

K/V 矩阵分块：
┌────────────────────────────────────┐
│  Key/Value: [blocks, 128, nkv, d]  │
│                                    │
│  按 s2_tile = 512 切分：           │
│  ┌────────┬────────┬────────┐     │
│  │ K_s2_0 │ K_s2_1 │ K_s2_2 │ ... │
│  │[512,d] │[512,d] │[512,d] │     │
│  └────────┴────────┴────────┘     │
│                                    │
│  每个 s2_tile 由 4 个 block 组成   │
│  (512 / block_size = 512/128 = 4)  │
└────────────────────────────────────┘
```

### 5.4 Cube Tile 配置详解

对于矩阵乘法 $(M, K) \times (K, N) = (M, N)$：

```
QK^T 计算：
┌─────────────────────────────────────────────────────┐
│  Q: [g_tile, d]    K: [s2_tile, d]                 │
│                                                     │
│  set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])
│                                                     │
│  QK^T = Q @ K^T  →  [g_tile, s2_tile]              │
│                                                     │
│  配置: [[128, 128], [128, 128], [128, 128]]         │
│        M维度      K维度      N维度                  │
└─────────────────────────────────────────────────────┘

PV 计算：
┌─────────────────────────────────────────────────────┐
│  P: [g_tile, s2_tile]    V: [s2_tile, d]           │
│                                                     │
│  PV = P @ V  →  [g_tile, d]                        │
│                                                     │
│  配置: [[128, 128], [128, 128], [128, 128]]         │
└─────────────────────────────────────────────────────┘
```

---

## 六、核心代码分析

### 6.1 JIT Kernel 结构

```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_num_initial": 128,      # 子图初始数量
        "stitch_function_outcast_memory": 1024,  # 外部内存
        "stitch_function_inner_memory": 1024     # 内部内存
    },
    pass_options={
        "pg_upper_bound": 1536,                  # 子图大小上界
        "cube_l1_reuse_setting": {0: 4}          # Q 常驻 L1，4 次 matmul 复用
    }
)
def ifa_func_kernel(q, k, v, block_table, kv_act_seqs, atten_out):
    ...
```

### 6.2 输入 Shape 解析

```python
# 从输入获取动态 shape
shape_q = q.shape
shape_k = k.shape

bs_scalar = shape_q[0]        # num_tokens = b * s1
nq = shape_q[1]               # Query 头数
block_num_scalar = shape_k[0] # 总块数
block_size = shape_k[1]       # 块大小 (128)
nkv = shape_k[2]              # KV 头数
dn = shape_k[3]               # 头维度 (128)
b_scalar = kv_act_seqs.shape[0]  # batch size

# 计算派生值
s1_scalar = bs_scalar // b_scalar   # query 序列长度
group = nq // nkv                    # GQA 分组大小
g_loop = group // g_tile             # 组循环次数
```

### 6.3 Tensor Reshape

```python
# 将 3D/4D tensor reshape 为 2D 以便矩阵乘法
k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
q_2d_shape = (b_scalar * s1_scalar * nq, dn)

k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
```

### 6.4 KV Cache 组装

```python
# 从 block_table 获取块索引，组装连续的 K/V
kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")

for i in range(block_num):  # block_num = s2_tile // block_size
    block_idx = block_table[b_idx, idx + i]
    block_idx_valid = block_idx.max(0)  # 处理 -1 情况
    
    kj_assemble[i*block_size:(i+1)*block_size, 0:] = \
        pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
```

### 6.5 QK^T 计算

```python
# 获取当前 query tile
qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])

# QK^T 矩阵乘法
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
```

### 6.6 在线 Softmax 实现

```python
if pypto.is_loop_begin(s2_idx):
    # 首次迭代
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    max_update[:] = tilda_mij
    
else:
    # 后续迭代 - 需要校正
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    max_new = pypto.maximum(max_update, tilda_mij)
    tsub = pypto.sub(sij_scale, max_new)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
    sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    
    # 校正因子
    tsub2 = pypto.sub(max_update, max_new)
    max_update[:] = max_new
    update_mul = pypto.exp(tsub2)
    sum_update[:] = sum_update * update_mul + sum_local
```

### 6.7 PV 计算与输出

```python
# PV 矩阵乘法
pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

# 更新累积输出
pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
if pypto.is_loop_begin(s2_idx):
    oi_update[:] = oi_tmp
else:
    oi_update[:] = oi_update * update_mul + oi_tmp

# 最终输出
if pypto.is_loop_end(s2_idx):
    oi_final = pypto.div(oi_update, sum_update)
    oi_final_3d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, dn]), dtype)
    pypto.assemble(oi_final_3d, oi_ofs, atten_out)
```

---

## 七、数据流与循环结构

### 7.1 多层循环结构

```
┌─────────────────────────────────────────────────────────────────┐
│                        循环嵌套结构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for b_idx in pypto.loop(b_scalar):         # Batch 维度       │
│      for s1_idx in pypto.loop(s1_scalar):   # Query 序列维度   │
│          cur_seq = actual_seq - (s1-1-s1_idx)  # 当前 KV 长度  │
│          s2_loop = ceil(cur_seq / s2_tile)                     │
│          for n2_idx in pypto.loop(nkv):     # KV 头维度        │
│              for g_idx in pypto.loop(g_loop): # Query 头分组   │
│                  for s2_idx in pypto.loop(s2_loop):  # KV分块  │
│                      # Flash Attention 核心计算                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 数据流图

```
┌──────────────────────────────────────────────────────────────────┐
│                         IFA 数据流                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │  Query  │     │  Key Cache   │     │ Value Cache  │          │
│  │[bs,nq,d]│     │[blocks,128,  │     │[blocks,128,  │          │
│  │         │     │ nkv,d]       │     │ nkv,d]       │          │
│  └────┬────┘     └──────┬───────┘     └──────┬───────┘          │
│       │                 │                    │                   │
│       v                 v                    v                   │
│  ┌─────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │ Reshape │     │ Block Table  │     │ Block Table  │          │
│  │  to 2D  │     │   Lookup     │     │   Lookup     │          │
│  └────┬────┘     └──────┬───────┘     └──────┬───────┘          │
│       │                 │                    │                   │
│       v                 v                    │                   │
│  ┌─────────┐     ┌──────────────┐           │                   │
│  │  Q tile │────>│   K tile     │           │                   │
│  │[g,d]    │     │  [s2_tile,d] │           │                   │
│  └─────────┘     └──────┬───────┘           │                   │
│                         │                    │                   │
│                         v                    │                   │
│                  ┌──────────────┐            │                   │
│                  │    QK^T      │            │                   │
│                  │  [g,s2_tile] │            │                   │
│                  └──────┬───────┘            │                   │
│                         │                    │                   │
│                         v                    │                   │
│                  ┌──────────────┐            │                   │
│                  │ Online       │            │                   │
│                  │ Softmax      │            │                   │
│                  │ [g,s2_tile]  │            │                   │
│                  └──────┬───────┘            │                   │
│                         │                    │                   │
│                         v                    v                   │
│                  ┌──────────────┐     ┌──────────────┐          │
│                  │    P tile    │────>│   V tile     │          │
│                  │  [g,s2_tile] │     │[s2_tile,d]   │          │
│                  └──────────────┘     └──────┬───────┘          │
│                                              │                   │
│                                              v                   │
│                                       ┌──────────────┐          │
│                                       │    PV        │          │
│                                       │   [g,d]      │          │
│                                       └──────┬───────┘          │
│                                              │                   │
│                                              v                   │
│                                       ┌──────────────┐          │
│                                       │  Output      │          │
│                                       │  [bs,nq,d]   │          │
│                                       └──────────────┘          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 八、性能优化技术

### 8.1 内存优化

| 技术 | 描述 |
|-----|------|
| **Flash Attention** | 避免存储完整注意力矩阵，内存从 O(N×M) 降到 O(N) |
| **Paged KV Cache** | 按需分配，避免预分配浪费 |
| **Tile 分块** | 数据局部性优化，减少内存访问 |

### 8.2 计算优化

```python
# JIT 配置优化
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_num_initial": 128,      # 控制子图切分
        "stitch_function_outcast_memory": 1024,  # 外部内存配置
        "stitch_function_inner_memory": 1024     # 内部内存配置
    },
    pass_options={
        "pg_upper_bound": 1536,                  # 子图大小限制
        "cube_l1_reuse_setting": {0: 4}          # Q 在 L1 中复用 4 次
    }
)
```

### 8.3 L1 缓存复用

```python
# cube_l1_reuse_setting: {0: 4}
# 
# 含义：第 0 组矩阵乘法（QK^T）的 Q 矩阵在 L1 中保持，
# 可以被后续 4 次 matmul 复用
#
# 4 次 matmul 对应：
# - 同一个 Q tile 对不同的 K/V tile 计算
# - 避免重复加载 Q 数据
```

### 8.4 循环展开策略

```python
for s2_idx in pypto.loop(s2_loop, 
                         name="LOOP_s2", 
                         idx_name="s2_idx", 
                         unroll_list=[8, 4, 2, 1]):
    ...
```

**unroll_list 含义**：
- 尝试按 8 倍展开
- 如果不满足条件，尝试 4 倍
- 依次递减到 1（不展开）

### 8.5 子图切分优化

```python
pypto.set_pass_options(sg_set_scope=1)   # 设置子图作用域
# ... 计算操作 ...
pypto.set_pass_options(sg_set_scope=-1)  # 结束作用域
```

**作用**：控制哪些操作被归类到同一个子图，影响：
- 内存重用
- 调度优化
- 融合机会

---

## 九、完整执行流程

### 9.1 函数调用链

```
main()
  │
  ├──> test_ifa()
  │      │
  │      ├──> get_qwen_common_config()  # 获取配置
  │      │
  │      └──> IFA(atten_cfg)
  │             │
  │             ├──> gen_block_table()          # 生成块映射
  │             ├──> kv_cache_concat_bsnd()     # 转换 KV 格式（参考实现）
  │             │
  │             └──> attention()                # 主入口
  │                    │
  │                    ├──> check_args()        # 参数校验
  │                    └──> ifa_func()()        # JIT kernel
  │                           │
  │                           └──> ifa_func_kernel()  # 核心计算
```

### 9.2 PyPTO 与 PyTorch 对比验证

```python
# PyTorch 参考实现
for i in range(b):
    for j in range(s1):
        for n2_idx in range(nkv):
            kv_seq_len = kv_cache_actual_seq[i].item()
            seq_len = kv_seq_len - s1 + 1 + j
            
            q_bs = q[i * s1 + j]
            k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
            v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
            
            # 标准 attention 计算
            qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))
            qk_ele_res = qk_bmm_res * softmax_scale
            softmax_res, _, _ = softmax(qk_ele_res, True)
            bmm2_res = torch.matmul(softmax_res, v_bs)
            
            attention_output[i * s1 + j] = bmm2_res

# 精度验证
assert_allclose(np.array(attention_output.cpu().flatten().tolist()),
                np.array(out_torch.cpu().flatten().tolist()),
                rtol=0.0078125, atol=0.0001)
```

### 9.3 关键参数配置总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    推荐配置参数                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  模型参数：                                                     │
│  ├── batch_size:     1 ~ 32                                    │
│  ├── seq_len (s1):   1 (解码阶段)                              │
│  ├── kv_seq_len:     最大 16384                                │
│  ├── num_head (nq):  12                                        │
│  ├── kv_head (nkv):  1 (GQA)                                   │
│  ├── head_size:      128                                       │
│  └── block_size:     128                                       │
│                                                                 │
│  Tile 参数：                                                    │
│  ├── g_tile:         num_head (所有头一起计算)                 │
│  ├── s2_tile:        512                                       │
│  ├── cube_tile:      [128, 128, 128]                           │
│  └── vec_tile:       [128, 512] / [128, 128]                   │
│                                                                 │
│  性能参数：                                                     │
│  ├── pg_upper_bound:           1536                            │
│  ├── cube_l1_reuse_setting:    {0: 4}                          │
│  ├── stitch_function_inner:    1024                            │
│  └── unroll_list:              [8, 4, 2, 1]                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录：代码结构总览

```
glm_attention.py
├── 数据结构
│   ├── AttentionTileConfig     # Tile 配置
│   └── AttentionConfig         # Attention 参数配置
│
├── 辅助函数
│   ├── check_args()            # 参数校验
│   ├── get_qwen_common_config() # 获取默认配置
│   ├── gen_block_table()       # 生成块映射表
│   ├── kv_cache_concat_bsnd()  # KV Cache 格式转换
│   ├── get_special_array()     # 生成测试数据
│   └── softmax()               # PyTorch Softmax 参考
│
├── 核心实现
│   ├── ifa_func()              # JIT Kernel 工厂函数
│   │   └── ifa_func_kernel()   # Flash Attention 核心实现
│   │
│   └── attention()             # 对外接口（@allow_in_graph）
│
└── 测试
    ├── IFA()                   # 完整测试流程
    └── test_ifa()              # pytest 测试入口
```

---

**文档版本**：1.0  
**适用模型**：GLM-4.5  
**PyPTO 版本**：0.1.0+  
**最后更新**：2026年2月
