# GLM-4.5 Attention PFA/IFA 优化版本 v2 完整分析

## 目录

1. [概述](#1-概述)
2. [代码结构](#2-代码结构)
3. [IFA 与 PFA 核心区别](#3-ifa-与-pfa-核心区别)
4. [数据结构与配置](#4-数据结构与配置)
5. [分块策略](#5-分块策略)
6. [计算策略](#6-计算策略)
7. [性能优化策略](#7-性能优化策略)
8. [功能实现详解](#8-功能实现详解)
9. [测试验证](#9-测试验证)
10. [v2 版本优化要点](#10-v2-版本优化要点)

---

## 1. 概述

### 1.1 模块功能

`glm_attention_ifa_pfa_opt_v2.py` 实现了 GLM-4.5 模型的两种 Flash Attention 机制：

| 模式 | 全称 | 用途 | Query 长度 |
|------|------|------|------------|
| **IFA** | Incremental Flash Attention | Decode 阶段（增量解码） | s1 = 1 |
| **PFA** | Prompt Flash Attention | Prefill 阶段（提示填充） | s1 >= 1 |

### 1.2 核心算法

基于 **Online Softmax**（Flash Attention）算法实现：
- 分块计算注意力，避免存储完整 N×N 注意力矩阵
- 增量更新 softmax 归一化因子
- 支持 Paged Attention（分页式 KV Cache 管理）
- 支持因果注意力（causal attention）用于 PFA

### 1.3 技术栈

```
PyTorch + torch_npu + pypto (华为昇腾 AI 处理器算子开发框架)
```

### 1.4 v2 版本特性

v2 版本相比 v1 版本的主要改进：
- ✅ 更激进的编译器优化配置
- ✅ 启用 Cube 和 Vector 的 nbuffer 模式
- ✅ 优化的 L1 缓存复用策略
- ✅ 更清晰的 IFA/PFA 代码分离

---

## 2. 代码结构

```
glm_attention_ifa_pfa_opt_v2.py (923 行)
├── 环境配置 (30-60)
│   └── 自动设置 PTO_TILE_LIB_CODE_PATH 和 TILE_FWK_DEVICE_ID
├── 参数校验 (67-87)
│   └── check_args() - 统一参数校验
├── 配置数据结构 (93-119)
│   ├── AttentionTileConfig
│   └── AttentionConfig
├── 配置生成函数 (125-196)
│   ├── get_ifa_config() - IFA 专用配置
│   └── get_pfa_config() - PFA 专用配置
├── 辅助函数 (203-292)
│   ├── gen_block_table() - 生成 block 映射表
│   ├── kv_cache_concat_bsnd() - KV cache 格式转换
│   └── softmax() - PyTorch 参考实现
├── IFA Kernel 实现 (299-465)
│   └── ifa_func() - IFA 专用 kernel
├── PFA Kernel 实现 (472-643)
│   └── pfa_func() - PFA 专用 kernel（v2 优化版）
├── 对外接口 (651-701)
│   ├── attention_ifa() - IFA 接口
│   └── attention_pfa() - PFA 接口
└── 测试函数 (708-923)
    ├── run_ifa_test() - IFA 测试
    ├── run_pfa_test() - PFA 测试
    ├── test_ifa() - pytest 测试入口
    └── test_pfa() - pytest 测试入口
```

---

## 3. IFA 与 PFA 核心区别

### 3.1 对比表格

| 特性 | IFA (Decode) | PFA (Prefill) |
|------|--------------|---------------|
| **Query 长度 (s1)** | 1（单 token） | >= 1（完整 prompt） |
| **KV 来源** | 从 block_table 读取历史 KV cache | 当前输入，按 block_table 组织 |
| **KV 长度计算** | `cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)` | `cur_seq = s1_idx + 1`（因果掩码） |
| **注意力类型** | 双向（Query 可见所有历史 KV） | 因果（Query[i] 只见 KV[0:i+1]） |
| **使用阶段** | 自回归解码生成 | 首次处理完整 prompt |
| **典型场景** | 文本生成、对话推理 | Prompt 预处理、上下文理解 |
| **pg_upper_bound** | 1536 | 10000（v2 优化） |
| **nbuffer 模式** | 未启用 | 启用（v2 优化） |

### 3.2 核心区别详解

#### 3.2.1 Query 长度差异

```python
# IFA 配置 (glm_attention_ifa_pfa_opt_v2.py:125-159)
def get_ifa_config(device="cpu"):
    s1 = 1  # 每次只处理一个 token
    s2 = 16384  # KV cache 可达 16K
    nq = 12
    nkv = 1

# PFA 配置 (glm_attention_ifa_pfa_opt_v2.py:162-196)
def get_pfa_config(device="cpu"):
    s1 = 128  # 处理完整 prompt（如 128 tokens）
    s2 = s1   # KV 长度等于 query 长度
    nq = 12
    nkv = 1
```

#### 3.2.2 因果注意力实现

**IFA** (glm_attention_ifa_pfa_opt_v2.py:371):
```python
# IFA: Query 可见历史所有 KV
cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
# cur_seq 表示当前 query 可以看到的 KV 序列长度
```

**PFA** (glm_attention_ifa_pfa_opt_v2.py:558):
```python
# PFA: 因果注意力 - 位置 s1_idx 的 query 只能看到位置 0 到 s1_idx 的 KV
# 关键: cur_seq = s1_idx + 1（包含当前位置）
cur_seq = s1_idx + 1  # 这是因果掩码的核心实现
s2_loop = (cur_seq + s2_tile - 1) // s2_tile
```

因果注意力示意图：

```
PFA 因果注意力掩码 (s1 = 4):
位置:    0   1   2   3
Q[0] -> [✓] [✗] [✗] [✗]  cur_seq = 1
Q[1] -> [✓] [✓] [✗] [✗]  cur_seq = 2
Q[2] -> [✓] [✓] [✓] [✗]  cur_seq = 3
Q[3] -> [✓] [✓] [✓] [✓]  cur_seq = 4
```

#### 3.2.3 实际有效 KV 长度计算

```python
# 两者共用此逻辑，但 cur_seq 的含义不同
actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
```

- **IFA**: `cur_seq` 来自 `kv_act_seqs`（历史累积长度）
- **PFA**: `cur_seq` 来自 `s1_idx + 1`（当前位置 + 1）

---

## 4. 数据结构与配置

### 4.1 AttentionConfig

```python
@dataclass
class AttentionConfig:
    b: int                      # batch size
    s1: int                     # query 序列长度
    s2: int                     # kv 序列长度
    n1: int                     # query head 数量
    n2: int                     # kv head 数量 (GQA)
    q_d: int                    # query head 维度
    kv_d: int                   # kv head 维度
    block_size: int = 128       # 每个 block 的 token 数
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0  # 缩放因子 = d^-0.5
    kv_layout: str = "PA_BSND"  # Paged Attention 布局
    actual_seq: torch.Tensor = None  # 实际序列长度
    block_table_batch: int = 0
    kv_num_blocks: int = 0
```

### 4.2 AttentionTileConfig

```python
@dataclass
class AttentionTileConfig:
    g_tile: int         # 头分块大小 (通常 = nq)
    s2_tile: int        # KV 序列分块大小 (= 128 或 512)
    c1_tile_shape: list # Cube 1 (QK^T) 的 tile 形状
    v1_tile_shape: list # Vector 1 的 tile 形状
    c2_tile_shape: list # Cube 2 (PV) 的 tile 形状
    v2_tile_shape: list # Vector 2 的 tile 形状
```

### 4.3 典型配置示例

```python
# IFA 默认配置
b=8, s1=1, s2=16384, nq=12, nkv=1, q_d=128, block_size=128, s2_tile=512

# PFA 默认配置
b=8, s1=128, s2=128, nq=12, nkv=1, q_d=128, block_size=128, s2_tile=128
```

---

## 5. 分块策略

### 5.1 Paged Attention 分页机制

```
block_table 结构:
┌─────────────────────────────────────────┐
│ batch 0: [block_3, block_7, block_1, -1]│
│ batch 1: [block_2, block_5, block_9, ...]│
│ ...                                      │
└─────────────────────────────────────────┘
        ↓ 映射
KV Cache (PA_BSND 格式):
┌──────────┐
│ block_0  │ [block_size, nkv, d]
│ block_1  │
│ block_2  │
│ ...      │
│ block_N  │
└──────────┘
```

### 5.2 Block 组装逻辑

**K/V 组装** (glm_attention_ifa_pfa_opt_v2.py:391-397, 580-590):

```python
# 从 block_table 组装 K/V
kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
for i in range(block_num):
    block_idx = block_table[b_idx, idx + i]
    block_idx_valid = block_idx.max(0)  # 处理 -1 的无效 block
    kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
        pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
    vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
        pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
```

### 5.3 Tile 分块参数

#### IFA Tile 配置

```python
# IFA (glm_attention_ifa_pfa_opt_v2.py:149-158)
cube_tile = 128
m_tile = 128
s2_tile = 512  # IFA 使用更大的 tile

c1_tile_shape = [[128, 128], [128, 128], [128, 128]]
v1_tile_shape = [128, 512]
c2_tile_shape = [[128, 128], [128, 128], [128, 128]]
v2_tile_shape = [128, 128]
```

#### PFA Tile 配置

```python
# PFA (glm_attention_ifa_pfa_opt_v2.py:186-195)
cube_tile = 128
m_tile = 128
s2_tile = 128  # PFA 使用较小的 tile

c1_tile_shape = [[128, 128], [128, 128], [128, 128]]
v1_tile_shape = [128, 128]
c2_tile_shape = [[128, 128], [128, 128], [128, 128]]
v2_tile_shape = [128, 128]
```

### 5.4 分块策略对比

| 参数 | IFA | PFA | 说明 |
|------|-----|-----|------|
| s2_tile | 512 | 128 | IFA 处理更长 KV |
| s2_loop | s2 / 512 | (s1_idx+1) / 128 | KV 分块迭代次数 |
| block_num | 4 | 1 | s2_tile / block_size |

---

## 6. 计算策略

### 6.1 Flash Attention 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Flash Attention 流程                      │
├─────────────────────────────────────────────────────────────┤
│  1. QK^T 分块计算                                            │
│     sij = matmul(qi, kj_assemble^T)                         │
│                                                              │
│  2. Online Softmax 更新                                      │
│     if first_tile:                                          │
│         max_new = max(sij * scale)                          │
│         sum_new = sum(exp(sij * scale - max_new))           │
│         oi = exp(...) @ V                                    │
│     else:                                                    │
│         max_new = max(max_old, max(sij * scale))            │
│         update_mul = exp(max_old - max_new)                 │
│         sum_new = sum_old * update_mul + sum_local          │
│         oi = oi_old * update_mul + exp(...) @ V             │
│                                                              │
│  3. 最终归一化                                               │
│     output = oi / sum                                        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Online Softmax 数学原理

标准 Softmax:
```
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

Online Softmax 增量更新:
```python
# 假设已处理块 max=m1, sum=s1, output=o1
# 新块 max=m2, sum=s2, output=o2

m_new = max(m1, m2)
update_mul = exp(m1 - m_new)  # 校正因子

s_new = s1 * update_mul + s2
o_new = o1 * update_mul + o2
```

### 6.3 计算循环结构

```python
# 五层嵌套循环 (glm_attention_ifa_pfa_opt_v2.py:555-642)
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):           # 1. Batch 维度
    for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):     # 2. Query 序列维度
        cur_seq = s1_idx + 1  # 因果注意力
        s2_loop = (cur_seq + s2_tile - 1) // s2_tile
        
        for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):    # 3. KV Head 维度
            for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"): # 4. Query Head 组维度
                # 初始化累加器
                oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                
                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx"):  # 5. KV 分块维度
                    # QK^T -> Online Softmax -> PV
```

### 6.4 矩阵运算分解

| 阶段 | 操作 | 形状 | 说明 |
|------|------|------|------|
| C1 | `Q @ K^T` | [g_tile, s2_tile] | 注意力分数 |
| V1 | `scale * sij` | [g_tile, s2_tile] | 缩放 |
| V1 | `exp(sij - max)` | [g_tile, s2_tile] | 数值稳定指数 |
| C2 | `P @ V` | [g_tile, dn] | 加权求和 |
| V2 | `oi / sum` | [g_tile, dn] | 归一化输出 |

---

## 7. 性能优化策略

### 7.1 IFA JIT 编译优化

```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 1024,
        "stitch_function_inner_memory": 1024
    },
    pass_options={
        "pg_upper_bound": 1536,
        "cube_l1_reuse_setting": {0: 4}
    },
    debug_options={"runtime_debug_mode": 1}
)
```

### 7.2 PFA JIT 编译优化（v2 版本重点）

```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128
    }, 
    pass_options={
        "sg_set_scope": -1,
        "pg_upper_bound": 10000,  # v2: 大幅提升
        "cube_l1_reuse_mode": 1,  # v2: 启用 L1 复用模式
        "cube_l1_reuse_setting": {-1: 8},  # v2: 全局复用
        "cube_nbuffer_mode": 1,  # v2: 启用 Cube nbuffer
        "cube_nbuffer_setting": {-1: 8},  # v2: 8 buffer
        "vec_nbuffer_mode": 1,  # v2: 启用 Vector nbuffer
        "vec_nbuffer_setting": {}
    },
    debug_options={
        "runtime_debug_mode": 1,
        "compile_debug_mode": 0
    }
)
```

### 7.3 Cube L1 复用策略

**IFA**:
```python
"cube_l1_reuse_setting": {0: 4}
```
- **0**: 第一个 matmul 操作（QK^T）
- **4**: 复用 4 次（对应 4 个 s2_tile 迭代）
- **效果**: Q 矩阵常驻 L1 Cache，减少全局内存访问

**PFA (v2)**:
```python
"cube_l1_reuse_mode": 1,
"cube_l1_reuse_setting": {-1: 8}
```
- **-1**: 全局复用（所有 matmul 操作）
- **8**: 复用 8 次
- **效果**: 更激进的 L1 缓存复用策略

### 7.4 NBuffer 优化（v2 新增）

```python
# Cube nbuffer
"cube_nbuffer_mode": 1,
"cube_nbuffer_setting": {-1: 8}  # 8 个 buffer

# Vector nbuffer
"vec_nbuffer_mode": 1,
"vec_nbuffer_setting": {}  # 使用默认设置
```

**效果**:
- 多缓冲区并行加载和计算
- 减少内存访问延迟
- 提高计算单元利用率

### 7.5 动态 Shape 支持

```python
# 动态维度声明 (glm_attention_ifa_pfa_opt_v2.py:307-310, 488-490)
q_shape = (pypto.frontend.dynamic("qshape"), q_shape[1], q_shape[2])
kv_shape = (pypto.frontend.dynamic("kvshape"), kv_shape[1], kv_shape[2], kv_shape[3])
bs = pypto.frontend.dynamic("bs")
```

### 7.6 循环展开优化

```python
# Batch 循环展开 (glm_attention_ifa_pfa_opt_v2.py:555)
for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx", unroll_list=[4, 2, 1]):

# Query 序列循环展开 (glm_attention_ifa_pfa_opt_v2.py:556)
for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx", unroll_list=[8, 4, 2, 1]):

# KV 分块循环展开 (IFA: Line 380)
for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
```

- 优先尝试展开 8 次，不行则尝试 4, 2, 1
- 减少循环开销，提高指令级并行

### 7.7 向量化/Cube 指令设置

```python
# Vector 单元 tile 设置
pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])  # [128, 128]

# Cube 单元 tile 设置
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])  # [128,128], [128,128], [128,128]
```

### 7.8 Pass 优化选项

```python
# 分段计算优化 (glm_attention_ifa_pfa_opt_v2.py:608-623)
# Scope 1: 计算新的 softmax 统计值
pypto.set_pass_options(sg_set_scope=1)
sij_scale = pypto.mul(sij, softmax_scale)
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
max_new = pypto.maximum(max_update, tilda_mij)
tsub = pypto.sub(sij_scale, max_new)
tilda_pij = pypto.exp(tsub)
tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
pypto.set_pass_options(sg_set_scope=-1)

# Scope 2: 更新累加器
pypto.set_pass_options(sg_set_scope=2)
tsub2 = pypto.sub(max_update, max_new)
max_update[:] = max_new
update_mul = pypto.exp(tsub2)
sum_update[:] = sum_update * update_mul + sum_local
pypto.set_pass_options(sg_set_scope=-1)
```

### 7.9 性能优化效果预估

| 优化项 | IFA 预期收益 | PFA v2 预期收益 |
|--------|------------|----------------|
| Cube L1 复用 | 减少 50%+ 全局内存访问 | 减少 60%+ 全局内存访问 |
| NBuffer 模式 | N/A | 20-30% 吞吐提升 |
| 循环展开 | 10-20% 吞吐提升 | 10-20% 吞吐提升 |
| Online Softmax | O(N) 内存 vs O(N²) | O(N) 内存 vs O(N²) |
| Paged Attention | 动态内存管理，减少碎片 | 动态内存管理，减少碎片 |

---

## 8. 功能实现详解

### 8.1 attention_ifa 接口

```python
@allow_in_graph
def attention_ifa(
    query: torch.Tensor,        # [b*s1, nq, d] - 当前 query
    key_cache: torch.Tensor,    # [num_blocks, block_size, nkv, d] - K cache
    value_cache: torch.Tensor,  # [num_blocks, block_size, nkv, d] - V cache
    block_tables: torch.Tensor, # [b, max_blocks] - block 映射表
    actual_seqs: torch.Tensor,  # [b] - 实际序列长度
    attn_res: torch.Tensor      # [b*s1, nq, d] - 输出
) -> None:
    """IFA - Decode 阶段接口"""
```

### 8.2 attention_pfa 接口

```python
@allow_in_graph
def attention_pfa(
    query: torch.Tensor,        # [b*s1, nq, d] - 完整 prompt query
    key_cache: torch.Tensor,    # [num_blocks, block_size, nkv, d]
    value_cache: torch.Tensor,  # [num_blocks, block_size, nkv, d]
    block_tables: torch.Tensor, # [b, max_blocks]
    query_seqs: torch.Tensor,   # [b] - query 序列长度
    attn_res: torch.Tensor      # [b*s1, nq, d]
) -> None:
    """PFA - Prefill 阶段接口（支持因果注意力）"""
```

### 8.3 输入张量格式要求

| 张量 | 维度 | 格式 | 数据类型 |
|------|------|------|----------|
| query | 3D | ND | BF16 |
| key_cache | 4D | ND | BF16 |
| value_cache | 4D | ND | BF16 |
| block_tables | 2D | ND | INT32 |
| actual_seqs / query_seqs | 1D | ND | INT32 |
| attn_res | 3D | ND | BF16 |

### 8.4 Block Table 生成

```python
def gen_block_table(actual_seq_len, block_size, block_table_shape):
    # 1. 计算每个 batch 需要的 block 数量
    block_num_per_batch = []
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
    
    # 2. 生成随机排列的 block 索引（模拟非连续内存）
    block_idx_list = torch.arange(0, block_num)
    block_idx_list = block_idx_list[torch.randperm(block_num)]
    
    # 3. 填充 block_table
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    # ... 填充逻辑 ...
```

### 8.5 KV Cache 格式转换

```python
def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, atten_config):
    """
    将 Paged KV Cache (PA 格式) 转换为连续的 BSND 格式
    用于 PyTorch 参考实现验证
    """
    for b_idx in range(b):
        for block_idx in block_list:
            if block_idx == -1:
                break
            # 按顺序组装连续的 KV cache
            k_cache[b_idx, start:end, :, :] = kv_cache[block_idx]
```

---

## 9. 测试验证

### 9.1 IFA 测试流程

```python
def run_ifa_test(atten_cfg):
    # 1. 生成测试数据
    q = torch.empty(q_shape).uniform_(-1, 1).to(device)
    k, v = ...  # KV cache
    
    # 2. 生成 block_table
    block_table = gen_block_table(...)
    
    # 3. PyTorch 参考实现
    for i in range(b):
        for j in range(s1):
            qk = torch.matmul(q, k.T) * scale
            attn = softmax(qk)
            output = torch.matmul(attn, v)
    
    # 4. 执行 IFA kernel
    attention_ifa(q, k, v, block_table, actual_seqs, out)
    
    # 5. 精度验证
    assert_allclose(ref, out, rtol=0.0078125, atol=0.0001)
```

### 9.2 PFA 测试流程（含因果注意力验证）

```python
def run_pfa_test(atten_cfg):
    # 1-2. 同 IFA
    
    # 3. PyTorch 参考实现（因果注意力）
    for i in range(b):
        seq_len = query_seq_len[i].item()
        for j in range(s1):
            # 关键：cur_kv_len = j + 1（因果掩码）
            cur_kv_len = j + 1
            k_bs = k_cache_bsnd[i, :cur_kv_len, ...]
            v_bs = v_cache_bsnd[i, :cur_kv_len, ...]
            # ... 计算 ...
    
    # 4. 执行 PFA kernel
    attention_pfa(q, k, v, block_table, query_seqs, out)
    
    # 5. 精度验证（PFA 使用更宽松的容差）
    assert_allclose(ref, out, rtol=0.05, atol=0.005)
```

### 9.3 精度容差说明

| 模式 | rtol | atol | 说明 |
|------|------|------|------|
| IFA | 0.0078125 | 0.0001 | 1/128，BF16 精度 |
| PFA | 0.05 | 0.005 | 更宽松，因 valid_shape 机制有精度损失 |

---

## 10. v2 版本优化要点

### 10.1 主要优化项

| 优化项 | v1 版本 | v2 版本 | 提升 |
|--------|---------|---------|------|
| pg_upper_bound | 1536 | 10000 | +550% |
| cube_l1_reuse_mode | 未启用 | 1（启用） | 新增 |
| cube_nbuffer_mode | 未启用 | 1（启用） | 新增 |
| vec_nbuffer_mode | 未启用 | 1（启用） | 新增 |

### 10.2 代码组织优化

1. **统一参数校验**: 使用单一 `check_args()` 函数
2. **清晰配置分离**: IFA 和 PFA 使用独立配置函数
3. **环境自动配置**: 自动设置必需的环境变量

### 10.3 性能优化总结

v2 版本通过以下优化实现了显著性能提升：

1. **更激进的编译器配置**:
   - `pg_upper_bound: 10000` 允许更大的子图，减少 kernel 启动开销
   - 更好的算子融合机会

2. **NBuffer 优化**:
   - Cube 和 Vector 单元都启用多缓冲模式
   - 实现计算与内存访问的流水线并行

3. **L1 缓存复用**:
   - 全局 L1 复用策略（`{-1: 8}`）
   - 减少全局内存访问次数

4. **循环展开优化**:
   - Batch 和 Query 序列维度都启用展开
   - 减少循环控制开销

---

## 附录

### A. 文件依赖关系

```
glm_attention_ifa_pfa_opt_v2.py
├── torch, torch_npu
├── pypto (昇腾算子开发框架)
├── numpy
└── utils.get_format
```

### B. 关键参数默认值

| 参数 | IFA 默认值 | PFA 默认值 |
|------|-----------|-----------|
| b | 8 | 8 |
| s1 | 1 | 128 |
| s2 | 16384 | 128 |
| nq | 12 | 12 |
| nkv | 1 | 1 |
| q_d | 128 | 128 |
| block_size | 128 | 128 |
| s2_tile | 512 | 128 |

### C. 性能调优建议

1. **s2_tile 对齐**: IFA 保持 s2_tile = 4 * block_size = 512，PFA 保持 s2_tile = block_size = 128
2. **g_tile 选择**: 通常设置为 nq，充分利用头并行
3. **cube_l1_reuse**: IFA 使用局部复用 {0: 4}，PFA 使用全局复用 {-1: 8}
4. **nbuffer 模式**: PFA 启用 nbuffer，IFA 保持默认
5. **内存规划**: 合理设置 stitch_function_*_memory 参数

---

**文档版本**: v2.0  
**最后更新**: 2026-03-05  
**对应代码**: `glm_attention_ifa_pfa_opt_v2.py`
