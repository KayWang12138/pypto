# PFA 详细迭代过程与寻址计算解读

## 文档说明

本文档在 `pfa_graph.md` 和 `pfa_instrucction.md` 的基础上，使用**具体数值例子**详细说明：
- 大矩阵如何拆分成小矩阵进行计算
- 每次迭代时的寻址计算过程
- 不同 tile 分块的具体数据流

**前置知识**: 建议先阅读 `pfa_graph.md` 了解整体数据布局和循环结构。

---

## 1. 配置参数与数据形状（示例场景）

### 1.1 测试配置

```python
# PFA 测试配置 (glm_attention_ifa_pfa_opt_v1.py:162-196)
b = 8           # batch size
s1 = 128        # query 序列长度
s2 = s1 = 128   # KV 序列长度
nq = 12         # query heads
nkv = 1         # KV heads (GQA)
d = 128         # head dimension
block_size = 128
s2_tile = 128

# Tile 配置
g_tile = nq = 12        # 一次处理所有 12 个 query heads
cube_tile = 128
m_tile = 128
```

### 1.2 输入张量形状

| 张量 | 原始形状 | Reshape 后 (2D) | 说明 |
|------|---------|----------------|------|
| **Query** | `[b*s1, nq, d]` = `[1024, 12, 128]` | `q_2d: [b*s1*nq, d]` = `[12288, 128]` | 1024 tokens × 12 heads = 12288 |
| **Key** | `[num_blocks, block_size, nkv, d]` = `[8, 128, 1, 128]` | `k_2d: [num_blocks*block_size, nkv*d]` = `[1024, 128]` | 8 blocks × 128 tokens/block = 1024 |
| **Value** | `[8, 128, 1, 128]` | `v_2d: [1024, 128]` | 同 Key |
| **block_table** | `[b, max_blocks_per_query]` = `[8, 1]` | - | 每个batch映射1个block |
| **Output** | `[1024, 12, 128]` | - | 与 Query 形状相同 |

---

## 2. Reshape 操作详解

### 2.1 Query Reshape: [1024, 12, 128] → [12288, 128]

**目的**: 将 3D tensor 展平为 2D，便于通过行索引直接定位到具体的 token 和 head。

```python
# 代码位置: Line 548, 552
q_2d_shape = (b_scalar * s1_scalar * nq, dn) = (8 * 128 * 12, 128) = (12288, 128)
q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
```

**Reshape 映射关系**:

```
原始 Q [1024, 12, 128]           Reshape 后 q_2d [12288, 128]
┌──────────────────────────┐     ┌──────────────────────────────────┐
│ token 0                  │     │ row 0: token 0, head 0  [0:128]  │
│ ├─ head 0:  [0:128]      │     │ row 1: token 0, head 1  [0:128]  │
│ ├─ head 1:  [0:128]      │     │ ...                              │
│ └─ ... head 11: [0:128]  │ ──> │ row 11: token 0, head 11 [0:128] │
│ token 1                  │     │ row 12: token 1, head 0  [0:128] │
│ └─ ...                   │     │ row 13: token 1, head 1  [0:128] │
│ ...                      │     │ ...                              │
│ token 1023               │     │ row 12287: token 1023, head 11   │
└──────────────────────────┘     └──────────────────────────────────┘

映射公式:
  q_2d_row = token_idx * nq + head_idx
  
  例如: token 100, head 5
    → q_2d_row = 100 * 12 + 5 = 1205
```

### 2.2 Key/Value Reshape: [8, 128, 1, 128] → [1024, 128]

```python
# 代码位置: Line 547, 550-551
k_2d_shape = (block_num_scalar * block_size, n2_sym * dn) = (8 * 128, 1 * 128) = (1024, 128)
k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
```

**Reshape 映射关系**:

```
原始 K [8, 128, 1, 128]           Reshape 后 k_2d [1024, 128]
┌──────────────────────────┐     ┌──────────────────────────────────┐
│ block 0                  │     │ row 0: block 0, token 0  [0:128] │
│ ├─ token 0:   [0:128]    │     │ row 1: block 0, token 1  [0:128] │
│ ├─ token 1:   [0:128]    │     │ ...                              │
│ └─ ... token 127: [0:128]│ ──> │ row 127: block 0, token 127      │
│ block 1                  │     │ row 128: block 1, token 0        │
│ └─ ...                   │     │ row 129: block 1, token 1        │
│ ...                      │     │ ...                              │
│ block 7                  │     │ row 1023: block 7, token 127     │
└──────────────────────────┘     └──────────────────────────────────┘

映射公式:
  k_2d_row = block_idx * block_size + token_offset_in_block
  
  例如: block 3, token 50
    → k_2d_row = 3 * 128 + 50 = 434
```

---

## 3. Block Table 机制与 KV 组装

### 3.1 Block Table 的作用

**问题**: KV cache 在内存中是非连续存储的（PagedAttention 机制），每个 batch 的 KV 可能分散在不同 block 中。

**解决**: `block_table` 记录每个 batch 的 KV 存储在哪些 block 中。

```
block_table [8, 1] 示例:
┌─────────┬────────────┐
│ batch 0 │ [block_3]  │  ──> batch 0 的 KV 存储在 block 3
├─────────┼────────────┤
│ batch 1 │ [block_7]  │  ──> batch 1 的 KV 存储在 block 7
├─────────┼────────────┤
│ batch 2 │ [block_1]  │  ──> batch 2 的 KV 存储在 block 1
├─────────┼────────────┤
│ ...     │ ...        │
└─────────┴────────────┘

KV Cache 物理布局 (非连续):
┌──────────────────────────────────────────┐
│ block_0 │ block_1 │ block_2 │ block_3 │ ... │ block_7 │
│ (空闲)  │batch_2  │ (空闲)  │batch_0  │     │batch_1  │
└──────────────────────────────────────────┘
            ▲                    ▲                 ▲
            │                    │                 │
     block_table[2]       block_table[0]    block_table[1]
```

### 3.2 KV Block 组装代码详解

**代码位置**: Line 586-592

```python
kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")  # [128, 128]
for i in range(block_num):  # block_num = s2_tile // block_size = 128 // 128 = 1
    block_idx = block_table[b_idx, idx + i]      # 从 block_table 获取 block 编号
    block_idx_valid = block_idx.max(0)           # 处理 -1 (无效block)
    kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
        pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
```

**具体例子**: 假设 `b_idx=0, s2_idx=0`

```
步骤1: 初始化 kj_assemble [128, 128]
       ┌──────────────────────────┐
       │ [0:128, 0:128] = 未初始化 │
       └──────────────────────────┘

步骤2: block_num = s2_tile // block_size = 128 // 128 = 1
       只需要循环 1 次

步骤3: i=0 时
       a) block_idx = block_table[0, 0] = 3
          (batch 0 的 KV 在 block 3)
       
       b) block_idx_valid = max(3, 0) = 3
          (如果不是 -1，保持原值)
       
       c) 从 k_2d 中提取 block 3 的数据
          - k_2d 起始行 = 3 * 128 = 384
          - 提取 k_2d[384:512, :] 共 128 行
          - 写入 kj_assemble[0:128, :]
       
       d) kj_assemble 更新后:
          ┌──────────────────────────┐
          │ [0:128, 0:128] = 来自block_3的数据 │
          └──────────────────────────┘

最终 kj_assemble [128, 128] 包含了 batch 0 的完整 KV 数据
```

**关键理解**:
- `block_idx * block_size` 计算出该 block 在 `k_2d` 中的起始行号
- `pypto.view(k_2d, [128, 128], [起始行, 0])` 提取该 block 的 128 行数据
- `kj_assemble[i*block_size:(i+1)*block_size, :]` 将提取的数据写入组装缓冲区

---

## 4. 五层循环迭代的寻址计算

### 4.1 循环结构回顾

```python
for b_idx in range(b):              # 0 ~ 7
    for s1_idx in range(s1):        # 0 ~ 127
        cur_seq = s1_idx + 1        # 因果注意力: 1 ~ 128
        s2_loop = ceil(cur_seq / 128)  # 总是 1 (因为 cur_seq <= 128)
        
        for n2_idx in range(nkv):   # 0 ~ 0 (只有1个KV head)
            for g_idx in range(g_loop):  # 0 ~ 0 (g_tile=12, g_loop=1)
                for s2_idx in range(s2_loop):  # 0 ~ 0
                    # 核心计算...
```

### 4.2 关键偏移量计算（逐项分解）

**代码位置**: Line 573-580

```python
idx = s2_idx * block_num                    # block 索引
bs_ofs = b_idx * s1_scalar + s1_idx         # batch-sequence 偏移
n1g_ofs = n2_idx * group + g_idx * g_tile   # head 组偏移
actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)  # 有效 tile 大小
oi_ofs = [bs_ofs, n1g_ofs, 0]               # 输出偏移
```

#### 偏移量计算详解（具体例子）

**例子 1**: `b_idx=2, s1_idx=50, s2_idx=0`

```
步骤1: idx = s2_idx * block_num = 0 * 1 = 0
       (从 block_table 的第 0 个 block 开始读取)

步骤2: bs_ofs = b_idx * s1 + s1_idx = 2 * 128 + 50 = 306
       含义: 在展平的 token 序列中，这是第 306 个 token
       
       token 分布:
       - batch 0: token 0 ~ 127
       - batch 1: token 128 ~ 255
       - batch 2: token 256 ~ 383  ← 我们在这里
           - s1_idx=50 → token 256+50 = 306

步骤3: n1g_ofs = n2_idx * group + g_idx * g_tile = 0 * 12 + 0 * 12 = 0
       含义: 从第 0 个 query head 开始，处理 12 个 heads (g_tile=12)

步骤4: cur_seq = s1_idx + 1 = 50 + 1 = 51
       actual_s2_tile = min(51, 128) = 51
       含义: 当前 query 位置是 50，只能看到 K[0:51] 和 V[0:51]
       
步骤5: oi_ofs = [306, 0, 0]
       含义: 结果写入 atten_out[306, 0:12, 0:128]
```

**例子 2**: `b_idx=5, s1_idx=100, s2_idx=0`

```
idx = 0 * 1 = 0
bs_ofs = 5 * 128 + 100 = 740
n1g_ofs = 0 * 12 + 0 * 12 = 0
cur_seq = 100 + 1 = 101
actual_s2_tile = min(101, 128) = 101
oi_ofs = [740, 0, 0]
```

---

## 5. Query 定位与提取（详细示例）

### 5.1 Query 定位公式

```python
# 代码位置: Line 583
qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
```

**参数说明**:
- `q_2d`: [12288, 128] - 展平后的 query tensor
- `[g_tile, dn]` = [12, 128] - 提取的形状（12个heads，每个128维）
- `[bs_ofs * nq + n1g_ofs, 0]` - 起始位置

**为什么是 `bs_ofs * nq + n1g_ofs`?**

```
q_2d 的行索引排列:
  row 0-11:    token 0 的 12 个 heads
  row 12-23:   token 1 的 12 个 heads
  row 24-35:   token 2 的 12 个 heads
  ...

所以:
  每个token占用 nq=12 行
  
  token 的起始行 = token_idx * 12
                = bs_ofs * 12
```

### 5.2 具体例子：提取 token 306 的 Query

**场景**: `b_idx=2, s1_idx=50, n1g_ofs=0`

```
步骤1: 计算起始行
       start_row = bs_ofs * nq + n1g_ofs = 306 * 12 + 0 = 3672

步骤2: 提取范围
       q_2d[3672:3684, :]  # 提取 12 行，每行 128 维
       
       对应的 heads:
         - row 3672: token 306, head 0  [0:128]
         - row 3673: token 306, head 1  [0:128]
         - row 3674: token 306, head 2  [0:128]
         - ...
         - row 3683: token 306, head 11 [0:128]

步骤3: 得到 qi [12, 128]
       ┌──────────────────────────────┐
       │ head 0:  q[0, 0:128]         │
       │ head 1:  q[1, 0:128]         │
       │ ...                          │
       │ head 11: q[11, 0:128]        │
       └──────────────────────────────┘
```

**可视化 q_2d 中的位置**:

```
q_2d [12288, 128]
┌──────────────────────────────────────────┐
│ row 0-3671: 跳过                         │
├──────────────────────────────────────────┤
│ row 3672: token 306, head 0  ← 提取开始  │
│ row 3673: token 306, head 1              │
│ row 3674: token 306, head 2              │
│ ...                                      │
│ row 3683: token 306, head 11 ← 提取结束  │
├──────────────────────────────────────────┤
│ row 3684-12287: 跳过                     │
└──────────────────────────────────────────┘
```

---

## 6. KV 定位与组装（详细示例）

### 6.1 因果注意力对 KV 的影响

**关键**: `actual_s2_tile` 限制了有效的 KV 数量。

**例子**: `b_idx=2, s1_idx=50, block_table[2]=1`

```
步骤1: cur_seq = s1_idx + 1 = 51
       当前 query 只能看到 51 个 KV tokens (位置 0~50)

步骤2: block_idx = block_table[2, 0] = 1
       batch 2 的 KV 存储在 block 1

步骤3: 提取 block 1 的数据到 kj_assemble
       k_2d 起始行 = 1 * 128 = 128
       提取 k_2d[128:256, :] → kj_assemble[0:128, :]

步骤4: 但因为 cur_seq=51，只有前 51 个有效
       kj_assemble[0:51, :]   ← 有效数据
       kj_assemble[51:128, :] ← 无效数据，不参与计算

步骤5: 使用 valid_shape 标记
       kj_assemble = pypto.view(kj_assemble, [128, 128], [0, 0], 
                                valid_shape=[51, 128])
```

**可视化 kj_assemble 的有效区域**:

```
kj_assemble [128, 128]
┌─────────────────────────────────────────┐
│ row 0-50:   有效 KV (对应位置 0~50)      │
│             来自 k_2d[128:179, :]       │
├─────────────────────────────────────────┤
│ row 51-127: 无效数据                     │
│             (不参与实际计算)             │
└─────────────────────────────────────────┘
```

### 6.2 Block 组装的详细步骤

**场景**: `b_idx=0, block_table[0]=3`

```
k_2d [1024, 128]
┌──────────────────────────────────────────┐
│ block 0: row 0-127                       │
├──────────────────────────────────────────┤
│ block 1: row 128-255                     │
├──────────────────────────────────────────┤
│ block 2: row 256-383                     │
├──────────────────────────────────────────┤
│ block 3: row 384-511  ← 提取这里的数据   │
│          这是 batch 0 的 KV              │
├──────────────────────────────────────────┤
│ ...                                      │
└──────────────────────────────────────────┘

组装步骤:
  1) block_idx = block_table[0, 0] = 3
  2) 起始行 = 3 * 128 = 384
  3) 提取 k_2d[384:512, :] → kj_assemble[0:128, :]

kj_assemble [128, 128]
┌──────────────────────────────────────────┐
│ row 0-127: 来自 k_2d[384:511, :]         │
│            即 block 3 的完整数据          │
└──────────────────────────────────────────┘
```

---

## 7. 矩阵运算的数据流（完整示例）

### 7.1 完整计算流程：`b_idx=2, s1_idx=50`

**配置**:
- batch 2, query 位置 50
- `block_table[2] = 1` (batch 2 的 KV 在 block 1)
- `cur_seq = 51`, `actual_s2_tile = 51`

**步骤 1: 提取 Query**

```python
bs_ofs = 2 * 128 + 50 = 306
n1g_ofs = 0

qi = pypto.view(q_2d, [12, 128], [306 * 12 + 0, 0])
   = q_2d[3672:3684, :]  # 提取 12 行

qi 形状: [12, 128]
┌────────────────────────────┐
│ token 306 的 12 个 heads   │
│ 每个head 128 维            │
└────────────────────────────┘
```

**步骤 2: 组装 Key**

```python
idx = 0 * 1 = 0
block_idx = block_table[2, 0] = 1
block_idx_valid = max(1, 0) = 1

kj_assemble[0:128, :] = k_2d[1*128 : 2*128, :]
                      = k_2d[128:256, :]  # 提取 block 1

kj_assemble = pypto.view(kj_assemble, [128, 128], [0, 0], 
                         valid_shape=[51, 128])

kj_assemble 形状: [128, 128] (但只有前 51 行有效)
┌─────────────────────────────────────┐
│ row 0-50:  有效 K (位置 0~50)       │
│ row 51-127: 无效 (不参与计算)       │
└─────────────────────────────────────┘
```

**步骤 3: QK^T 计算 (Cube 操作)**

```python
# 代码位置: Line 595
sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, 
                   a_trans=False, b_trans=True)

矩阵乘法:
  qi [12, 128] × kj_assemble^T [128, 128] = sij [12, 128]
  
但由于 valid_shape=[51, 128]，实际计算:
  qi [12, 128] × kj_assemble[0:51, :]^T [51, 128] = sij [12, 51]
  
sij = pypto.view(sij, [12, 128], [0, 0], valid_shape=[12, 51])

sij 形状: [12, 128] (但只有前 51 列有效)
┌──────────────────────────────────────┐
│ [12, 51] 有效注意力分数              │
│ 每行: 一个 head 对 51 个 K 的得分    │
│ 列 51-127: 无效                      │
└──────────────────────────────────────┘
```

**步骤 4: Online Softmax**

```python
# 代码位置: Line 602-608
sij_scale = pypto.mul(sij, softmax_scale)  # [12, 51] 有效
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)  # [12, 1]
tsub = pypto.sub(sij_scale, tilda_mij)  # [12, 51]
tilda_pij = pypto.exp(tsub)  # [12, 51]
sum_update = pypto.sum(tilda_pij, dim=-1, keepdim=True)  # [12, 1]

数据流:
  sij [12, 51] 
    → scale → [12, 51]
    → exp(max - x) → [12, 51]
    → sum → [12, 1]

sum_update 形状: [12, 1]
┌──────────────────┐
│ head 0: sum_0    │
│ head 1: sum_1    │
│ ...              │
│ head 11: sum_11  │
└──────────────────┘
```

**步骤 5: 组装 Value**

```python
# 代码位置: Line 610-617
vj_assemble = pypto.tensor([128, 128], v_2d.dtype, "vj_assemble")
block_idx = block_table[2, 0] = 1
vj_assemble[0:128, :] = v_2d[128:256, :]  # 提取 block 1

vj_assemble = pypto.view(vj_assemble, [128, 128], [0, 0], 
                         valid_shape=[51, 128])

vj_assemble 形状: [128, 128] (但只有前 51 行有效)
┌─────────────────────────────────────┐
│ row 0-50:  有效 V (位置 0~50)       │
│ row 51-127: 无效                    │
└─────────────────────────────────────┘
```

**步骤 6: PV 计算 (Cube 操作)**

```python
# 代码位置: Line 620
oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

矩阵乘法:
  tilda_pij [12, 51] × vj_assemble [51, 128] = oi_tmp [12, 128]
  
  每个head: 对51个V进行加权求和

oi_tmp 形状: [12, 128]
┌────────────────────────────────────┐
│ head 0:  weighted_sum_0 [128]      │
│ head 1:  weighted_sum_1 [128]      │
│ ...                                │
│ head 11: weighted_sum_11 [128]     │
└────────────────────────────────────┘
```

**步骤 7: 写入输出**

```python
# 代码位置: Line 653-656
oi_final = pypto.div(oi_update, sum_update)  # [12, 128]
oi_final_3d = pypto.cast(pypto.reshape(oi_final, [1, 12, 128]), dtype)
pypto.assemble(oi_final_3d, [306, 0, 0], atten_out)

写入位置: atten_out[306, 0:12, 0:128]

atten_out [1024, 12, 128]
┌──────────────────────────────────────┐
│ token 0-305: 跳过                    │
├──────────────────────────────────────┤
│ token 306:                           │
│   head 0:  output[306, 0, 0:128]     │
│   head 1:  output[306, 1, 0:128]     │
│   ...                                │
│   head 11: output[306, 11, 0:128]    │
├──────────────────────────────────────┤
│ token 307-1023: 跳过或后续处理       │
└──────────────────────────────────────┘
```

---

## 8. 多次 s2_idx 迭代的场景

### 8.1 何时需要多次 s2_idx 迭代？

当 `s2_tile < cur_seq` 时，需要多次迭代。

**例子**: `s1=256, s2_tile=128`

```
s1_idx | cur_seq | s2_loop | s2_idx 迭代次数
───────┼─────────┼─────────┼────────────────
   0   |    1    |    1    |  1 次
  50   |   51    |    1    |  1 次
 127   |  128    |    1    |  1 次
 128   |  129    |    2    |  2 次  ← 需要2次
 200   |  201    |    2    |  2 次
 255   |  256    |    2    |  2 次
```

### 8.2 两次迭代的详细过程

**场景**: `s1=256, s1_idx=200, s2_tile=128`

```
cur_seq = 200 + 1 = 201
s2_loop = ceil(201 / 128) = 2

迭代 1: s2_idx = 0
  - 提取 K[0:128, :], V[0:128, :]
  - actual_s2_tile = min(201, 128) = 128
  - 计算 QK^T[12, 128], PV[12, 128]
  - 更新 max_update, sum_update, oi_update

迭代 2: s2_idx = 1
  - 提取 K[128:201, :], V[128:201, :]  (只有 73 个)
  - actual_s2_tile = min(201-128, 128) = 73
  - 计算 QK^T[12, 73], PV[12, 128]
  - 校正旧的 max, sum, output
  - 累加到 oi_update

最终:
  oi_final = oi_update / sum_update
```

### 8.3 Online Softmax 增量更新

**代码位置**: Line 621-650

```python
# 第二次及后续迭代
if not pypto.is_loop_begin(s2_idx):
    # 计算新的 max
    max_new = pypto.maximum(max_update, tilda_mij)
    
    # 计算校正因子
    tsub2 = pypto.sub(max_update, max_new)
    update_mul = pypto.exp(tsub2)  # [12, 1]
    
    # 校正旧的累积值
    sum_update = sum_update * update_mul + sum_local
    oi_update = oi_update * update_mul + oi_tmp
    
    # 更新 max
    max_update = max_new
```

**数值例子**:

```
假设:
  第一次迭代: max_1 = 10, sum_1 = 5.0, output_1 = [2.0, 3.0, ...]
  第二次迭代: max_2 = 12, sum_2 = 3.0, output_2 = [1.5, 2.5, ...]

计算:
  max_new = max(10, 12) = 12
  update_mul = exp(10 - 12) = exp(-2) = 0.135
  
  sum_new = 5.0 * 0.135 + 3.0 = 3.675
  output_new = [2.0*0.135+1.5, 3.0*0.135+2.5, ...]
             = [1.77, 2.905, ...]
```

**可视化增量更新**:

```
┌──────────────────────────────────────────────────────┐
│ 第一次迭代 (s2_idx=0):                               │
│                                                       │
│   K[0:128] ──┐                                       │
│   V[0:128] ──┼──> max_1=10, sum_1=5.0, output_1     │
│   Q[200]    ─┘                                       │
└──────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│ 第二次迭代 (s2_idx=1):                               │
│                                                       │
│   K[128:201] ──┐                                     │
│   V[128:201] ──┼──> max_2=12, sum_2=3.0, output_2   │
│   Q[200]      ─┘                                     │
│                                                       │
│   校正:                                               │
│     max_new = max(10, 12) = 12                       │
│     update_mul = exp(10-12) = 0.135                  │
│     sum_new = 5.0*0.135 + 3.0 = 3.675                │
│     output_new = output_1*0.135 + output_2           │
└──────────────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│ 最终归一化:                                          │
│   final_output = output_new / sum_new                │
│                = [1.77/3.675, 2.905/3.675, ...]      │
└──────────────────────────────────────────────────────┘
```

---

## 9. 完整数据流图（从输入到输出）

### 9.1 单次迭代的完整数据流

**场景**: `b=8, s1=128, b_idx=2, s1_idx=50`

```
输入张量:
┌────────────────────────────────────────────────────────┐
│ q [1024, 12, 128]                                      │
│ k [8, 128, 1, 128]                                     │
│ v [8, 128, 1, 128]                                     │
│ block_table [8, 1]                                     │
└────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ Reshape:                                               │
│   q_2d [12288, 128]                                    │
│   k_2d [1024, 128]                                     │
│   v_2d [1024, 128]                                     │
└────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ 循环: b_idx=2, s1_idx=50                               │
│   计算: bs_ofs=306, n1g_ofs=0                          │
│   计算: cur_seq=51, actual_s2_tile=51                  │
└────────────────────────────────────────────────────────┘
           │
           ├─────────────────────────────────┐
           ▼                                 ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ 提取 Query:              │    │ 组装 K/V:                │
│   qi = q_2d[3672:3684,:] │    │   block_idx = table[2,0]=1│
│   qi [12, 128]           │    │   k_2d[128:256,:] → k_assm│
│                          │    │   v_2d[128:256,:] → v_assm│
│                          │    │   valid: [51, 128]        │
└──────────────────────────┘    └──────────────────────────┘
           │                                 │
           └─────────────┬───────────────────┘
                         ▼
           ┌─────────────────────────────────┐
           │ QK^T (Cube):                     │
           │   qi [12,128] × k_assm^T [128,51]│
           │   = sij [12, 51]                 │
           └─────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────────┐
           │ Online Softmax (Vector):         │
           │   scale → max → exp → sum        │
           │   sum_update [12, 1]             │
           │   max_update [12, 1]             │
           │   tilda_pij [12, 51]             │
           └─────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────────┐
           │ PV (Cube):                       │
           │   tilda_pij [12,51] × v_assm[51,128]│
           │   = oi_tmp [12, 128]             │
           └─────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────────┐
           │ 归一化:                          │
           │   oi_final = oi_tmp / sum_update│
           │   oi_final [12, 128]             │
           └─────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────────┐
           │ 写入输出:                        │
           │   atten_out[306, 0:12, 0:128]   │
           │   = oi_final                     │
           └─────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ 输出张量:                                              │
│   atten_out [1024, 12, 128]                           │
│   token 306 的 12 个 heads 的注意力输出已写入         │
└────────────────────────────────────────────────────────┘
```

---

## 10. 总结与关键要点

### 10.1 寻址计算公式汇总

| 操作 | 公式 | 说明 |
|------|------|------|
| **Token 偏移** | `bs_ofs = b_idx * s1 + s1_idx` | 定位到具体 token |
| **Head 偏移** | `n1g_ofs = n2_idx * group + g_idx * g_tile` | 定位到具体 head 组 |
| **Query 起始行** | `q_row = bs_ofs * nq + n1g_ofs` | q_2d 中的起始行 |
| **KV Block 起始行** | `k_row = block_idx * block_size` | k_2d 中的起始行 |
| **有效 KV 数量** | `actual_s2_tile = min(cur_seq - s2_idx*s2_tile, s2_tile)` | 因果掩码限制 |
| **输出位置** | `atten_out[bs_ofs, n1g_ofs:n1g_ofs+g_tile, :]` | 写入位置 |

### 10.2 关键理解要点

1. **Reshape 目的**: 将多维 tensor 展平为 2D，通过行索引直接定位
2. **Block Table**: 实现 KV cache 的非连续内存管理
3. **因果注意力**: `cur_seq = s1_idx + 1` 是核心，限制每个 query 的可见范围
4. **Online Softmax**: 支持分块计算，增量更新 max 和 sum
5. **valid_shape**: 标记有效数据区域，避免计算无效位置

### 10.3 性能优化要点

1. **Tile 大小对齐**: `s2_tile = block_size = 128`，避免非对齐访问
2. **Cube L1 复用**: Q 矩阵常驻 L1，减少全局内存访问
3. **循环展开**: `unroll_list=[8, 4, 2, 1]` 减少循环开销
4. **动态 Shape**: 支持不同序列长度，无需重新编译

---

## 附录：常见问题解答

### Q1: 为什么 q_2d 的行数是 12288？

**A**: `q_2d` 是展平后的形式：
```
原始: [1024 tokens, 12 heads, 128 dims]
展平: [1024 * 12, 128] = [12288, 128]

每个 token 的 12 个 heads 被展平为 12 行
```

### Q2: actual_s2_tile 如何影响计算？

**A**: `actual_s2_tile` 限制了参与计算的有效 KV 数量：

```python
# 因果注意力：位置 50 只能看到 51 个 KV
actual_s2_tile = min(51, 128) = 51

# 矩阵乘法时，虽然 kj_assemble 是 [128, 128]
# 但通过 valid_shape=[51, 128]，只有前 51 行参与实际计算
sij = pypto.view(sij, [12, 128], [0, 0], valid_shape=[12, 51])
```

### Q3: 为什么需要 Online Softmax？

**A**: 标准 Softmax 需要先计算完整的 max 和 sum，无法分块。Online Softmax 通过增量更新，支持分块计算，是 Flash Attention 的核心。

### Q4: block_idx_valid = block_idx.max(0) 的作用？

**A**: `block_table` 中 `-1` 表示无效 block。`max(0)` 将 `-1` 转为 `0`，避免索引越界。虽然读取了 block 0 的数据，但因为 `actual_s2_tile=0`，不会参与实际计算。

---

**文档版本**: v1.0  
**最后更新**: 2026-03-04  
**对应代码**: `glm_attention_ifa_pfa_opt_v1.py`
