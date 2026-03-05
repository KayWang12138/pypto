# PFA Kernel 性能优化指南

> 本文档详细分析 GLM-4.5 PFA (Prompt Flash Attention) 算子的性能优化方向，基于代码分析和 PyPTO 官方文档。

---

## 目录

- [1. 概述](#1-概述)
- [2. 当前实现分析](#2-当前实现分析)
- [3. Tile 配置优化](#3-tile-配置优化)
- [4. JIT 编译选项优化](#4-jit-编译选项优化)
- [5. 循环展开优化](#5-循环展开优化)
- [6. 并行循环优化](#6-并行循环优化)
- [7. KV 组装优化](#7-kv-组装优化)
- [8. Online Softmax 融合优化](#8-online-softmax-融合优化)
- [9. L2 Cache 命中率优化](#9-l2-cache-命中率优化)
- [10. 调度策略优化](#10-调度策略优化)
- [11. 综合优化配置示例](#11-综合优化配置示例)
- [12. 优化验证流程](#12-优化验证流程)

---

## 1. 概述

### 1.1 PFA 算子特点

PFA (Prompt Flash Attention) 用于大模型推理的 **Prefill 阶段**，处理完整 prompt：

| 特性 | 值 |
|------|-----|
| Query 长度 (s1) | >= 1（完整 prompt） |
| KV 来源 | 当前输入，按 block_table 组织 |
| 注意力类型 | 因果注意力（Q[i] 只见 KV[0:i+1]） |
| 典型场景 | Prompt 预处理、上下文理解 |

### 1.2 性能优化目标

- **减少内存访问**：通过 Tile 配置和 L1/L2 缓存优化
- **提高计算并行度**：通过循环展开和多核并行
- **降低循环开销**：通过合理的分块策略
- **优化数据局部性**：通过 L2 亲和调度

---

## 2. 当前实现分析

### 2.1 当前 Tile 配置

```python
# glm_attention_ifa_pfa.py:175-184
cube_tile = 128
m_tile = 128
s2_tile = 128

tile_cfg = AttentionTileConfig(
    g_tile=nq,                                    # 12
    s2_tile=s2_tile,                              # 128
    c1_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
    v1_tile_shape=[64, s2_tile],                  # [64, 128]
    c2_tile_shape=[[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
    v2_tile_shape=[64, cube_tile]                 # [64, 128]
)
```

### 2.2 当前 JIT 配置

```python
# glm_attention_ifa_pfa.py:481-493
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 2048,
        "stitch_function_inner_memory": 2048
    },
    pass_options={
        "pg_upper_bound": 1536,
        "cube_l1_reuse_setting": {0: 4}
    }
)
```

### 2.3 当前循环结构

```python
# 五层嵌套循环
for b_idx in pypto.loop(b_scalar):           # Batch 维度
    for s1_idx in pypto.loop(s1_scalar):     # Query 序列维度
        cur_seq = s1_idx + 1                  # 因果注意力
        s2_loop = (cur_seq + s2_tile - 1) // s2_tile
        
        for n2_idx in range(n2_sym):          # KV Head (静态循环)
            for g_idx in range(g_loop):       # Query Head 组 (静态循环)
                for s2_idx in pypto.loop(s2_loop, unroll_list=[16, 8, 1]):
                    # Flash Attention 核心计算
```

---

## 3. Tile 配置优化

### 3.1 优化原则

根据 `matmul-performance-guide.md` 的指导：

1. **增大算数强度**：切分越大，重复搬运越少
2. **16KB 原则**：Vector 计算的 Tile 块控制在 16KB-64KB
3. **分型格式对齐**：外轴 16 元素对齐，内轴 32 字节对齐
4. **Buffer 空间约束**：L0C_SIZE = 131072 (FP32)

### 3.2 具体优化建议

| 参数 | 当前值 | 优化建议 | 原因 |
|------|--------|----------|------|
| `s2_tile` | 128 | **256 或 512** | 减少 s2_loop 迭代次数，降低循环开销 |
| `v1_tile` | [64, 128] | **[128, 256]** | 增大 vector tile，提高带宽利用率 |
| `c1_tile K轴` | [128, 128] | **[64, 256]** | 增大 K 轴 L1 切分，减少重复载入 |
| `v2_tile` | [64, 128] | **[128, 128]** | 与 cube tile 对齐 |

### 3.3 优化后配置

```python
def get_pfa_tile_config_optimized():
    """PFA 优化 Tile 配置"""
    cube_tile = 128
    m_tile = 128
    s2_tile = 256  # 增大 KV 分块

    tile_cfg = AttentionTileConfig(
        g_tile=nq,
        s2_tile=s2_tile,
        # QK^T: [g_tile, d] @ [s2_tile, d]^T = [g_tile, s2_tile]
        # M维度=L0:128, L1:128, K维度=L0:64, L1:256, N维度=L0:128, L1:128
        c1_tile_shape=[[128, 128], [64, 256], [128, 128]],
        v1_tile_shape=[128, s2_tile],  # [128, 256]
        # PV: [g_tile, s2_tile] @ [s2_tile, d] = [g_tile, d]
        c2_tile_shape=[[128, 128], [64, 256], [128, 128]],
        v2_tile_shape=[128, 128]
    )
    return tile_cfg
```

### 3.4 BF16/FP16 推荐 Tile 配置

```python
# 方案1：平衡配置
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256], 
                           enable_multi_data_load=True)

# 方案2：大 K 轴配置
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128], 
                           enable_multi_data_load=True)

# 方案3：紧凑配置
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128], 
                           enable_multi_data_load=True)
```

---

## 4. JIT 编译选项优化

### 4.1 runtime_options 优化

| 参数 | 当前值 | 优化建议 | 原因 |
|------|--------|----------|------|
| `stitch_function_num_initial` | 128 | 保持 | 控制子图切分粒度 |
| `stitch_function_outcast_memory` | 2048 | **4096** | 更大的外部内存允许更多子图融合 |
| `stitch_function_inner_memory` | 2048 | **4096** | 更大的内部内存支持更复杂的子图 |

### 4.2 pass_options 优化

| 参数 | 当前值 | 优化建议 | 原因 |
|------|--------|----------|------|
| `pg_upper_bound` | 1536 | **2048** | 允许更大的子图，减少调度开销 |
| `cube_l1_reuse_setting` | `{0: 4}` | **动态计算** | Q 矩阵复用次数应匹配实际 s2_loop |

### 4.3 cube_l1_reuse_setting 动态配置

```python
def get_pfa_jit_config(s1, s2_tile):
    """根据实际参数动态计算 JIT 配置"""
    # s2_loop = sum((i + s2_tile) // s2_tile for i in range(s1))
    # 对于因果注意力，平均 s2_loop 约为 s1 / (2 * s2_tile)
    avg_s2_loop = (s1 + s2_tile - 1) // (2 * s2_tile)
    
    return {
        "runtime_options": {
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 4096,
            "stitch_function_inner_memory": 4096
        },
        "pass_options": {
            "pg_upper_bound": 2048,
            "cube_l1_reuse_setting": {0: max(2, avg_s2_loop)},
            "cube_l1_reuse_mode": 1  # 开启全局 L1 复用
        }
    }
```

### 4.4 L1 复用原理

```
cube_l1_reuse_setting: {0: N}

含义：第 0 组 matmul (QK^T) 的 Q 矩阵在 L1 中保持，可被后续 N 次 matmul 复用

┌─────────────────────────────────────────────────┐
│  Q tile (常驻 L1)                               │
│  ┌─────────┐                                    │
│  │ [g, d]  │ ──> matmul_1 ──> QK^T_tile_1      │
│  │         │ ──> matmul_2 ──> QK^T_tile_2      │
│  │         │ ──> ...                            │
│  │         │ ──> matmul_N ──> QK^T_tile_N      │
│  └─────────┘                                    │
│  避免重复加载 Q 数据，减少 50%+ GM 访问         │
└─────────────────────────────────────────────────┘
```

---

## 5. 循环展开优化

### 5.1 当前配置

```python
# 行556
for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", 
                         unroll_list=[16, 8, 1]):
```

### 5.2 优化建议

**方案1：稳健配置**（与 IFA 对齐）
```python
unroll_list=[8, 4, 2, 1]
```

**方案2：短序列优化**（s1=128 场景）
```python
# PFA 因果注意力下，每个 s1_idx 的 s2_loop = s1_idx + 1
# 平均 s2_loop 较小，可以尝试更小的展开因子
unroll_list=[4, 2, 1]
```

**方案3：长序列优化**（s1=1024+ 场景）
```python
unroll_list=[16, 8, 4, 2, 1]
```

### 5.3 展开策略选择

| 场景 | 推荐 unroll_list | 原因 |
|------|------------------|------|
| s1 <= 128 | `[4, 2, 1]` | s2_loop 较小，小展开更有效 |
| s1 = 256~512 | `[8, 4, 2, 1]` | 平衡配置 |
| s1 >= 1024 | `[16, 8, 4, 2, 1]` | 大展开减少循环开销 |

---

## 6. 并行循环优化

### 6.1 当前问题

```python
# 行548-550：使用静态 Python 循环
for n2_idx in range(n2_sym):    # KV Head (nkv=1)
    for g_idx in range(g_loop):  # Query Head 组 (12/1=12)
```

静态 `range()` 循环不会被 PyPTO 框架识别，无法进行多核并行调度。

### 6.2 优化建议

使用 `pypto.loop` 替代静态循环：

```python
# 优化后：使用 pypto.loop 实现多核并行
for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
    for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
        # ...
```

### 6.3 预期收益

```
┌─────────────────────────────────────────────────────────┐
│                    并行化效果                            │
├─────────────────────────────────────────────────────────┤
│  原来 (range):                                          │
│  Core 0: [n2_0, g_0] -> [n2_0, g_1] -> ... (串行)      │
│  Core 1: idle                                           │
│  Core 2: idle                                           │
│  ...                                                    │
│                                                         │
│  优化后 (pypto.loop):                                   │
│  Core 0: [n2_0, g_0]                                    │
│  Core 1: [n2_0, g_1]   } 并行执行                       │
│  Core 2: [n2_0, g_2]                                    │
│  ...                                                    │
│  预期收益: 10-30% (取决于核数和负载均衡)               │
└─────────────────────────────────────────────────────────┘
```

---

## 7. KV 组装优化

### 7.1 当前实现

```python
# 行575-581
kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
for i in range(block_num):  # block_num = s2_tile // block_size = 1
    block_idx = block_table[b_idx, idx + i]
    block_idx_valid = block_idx.max(0)  # 处理 -1
    kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
        pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
```

### 7.2 优化方向

**方案A：增大 s2_tile 减少组装次数**

当 `s2_tile = 256, block_size = 128` 时：
- `block_num = 2`，需要 2 次组装
- 但 s2_loop 减半，总体减少循环开销

**方案B：预取下一轮 block**

```python
# 在 s2 循环开始前预取下一轮的 block 索引
# 减少循环内的间接访存延迟
next_idx = s2_idx + 1
if next_idx < s2_loop:
    # 预取下一轮 block_table
    for i in range(block_num):
        next_block_idx = block_table[b_idx, next_idx * block_num + i]
```

**方案C：连续内存布局优化**

如果可能，优化 block_table 的内存布局，使连续的 block 在物理内存中也连续：

```python
# 优化前：随机分布
block_table = [[3, 7, 1, -1], ...]

# 优化后：尽量连续
block_table = [[0, 1, 2, -1], ...]  # 连续 block
```

---

## 8. Online Softmax 融合优化

### 8.1 当前实现

使用 `pypto.set_pass_options(sg_set_scope=...)` 分段处理：

```python
# 行590-643
if pypto.is_loop_begin(s2_idx):
    # 首次迭代
    sij_scale = pypto.mul(sij, softmax_scale)
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    ...
else:
    pypto.set_pass_options(sg_set_scope=1)
    # 后续迭代 - 计算
    ...
    pypto.set_pass_options(sg_set_scope=-1)
    
    pypto.set_pass_options(sg_set_scope=2)
    # 后续迭代 - 更新
    ...
    pypto.set_pass_options(sg_set_scope=-1)
```

### 8.2 优化建议

**方案：减少 scope 切换，增加操作融合**

```python
# 优化：将更多操作放入同一 scope
pypto.set_pass_options(sg_set_scope=1)
# 融合：scale + amax + sub + exp + sum
sij_scale = pypto.mul(sij, softmax_scale)
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
tsub = pypto.sub(sij_scale, tilda_mij)
tilda_pij = pypto.exp(tsub)
sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
pypto.set_pass_options(sg_set_scope=-1)

# 单独处理更新逻辑
pypto.set_pass_options(sg_set_scope=2)
update_mul = pypto.exp(pypto.sub(max_update, max_new))
sum_update[:] = sum_update * update_mul + sum_local
oi_update[:] = oi_update * update_mul + oi_tmp
pypto.set_pass_options(sg_set_scope=-1)
```

### 8.3 融合效果

```
┌─────────────────────────────────────────────────────────┐
│              操作融合效果                                │
├─────────────────────────────────────────────────────────┤
│  未融合：                                               │
│  mul -> 写回 -> amax -> 写回 -> sub -> 写回 -> exp ...  │
│  (多次内存访问)                                         │
│                                                         │
│  融合后：                                               │
│  mul -> amax -> sub -> exp -> sum (流水线执行)          │
│  (中间结果保留在寄存器/片上缓存)                        │
│  预期收益: 5-15%                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 9. L2 Cache 命中率优化

### 9.1 理论基础

根据 `matmul-performance-guide.md`：

```
L2 命中率 = 1 - (1/nDim * aByte + 1/mDim * bByte) / (1/nL1 * aByte + 1/mL1 * bByte)

当 nDim * nL1 = mDim * mL1 时，L2 命中率最高
```

### 9.2 优化建议

对于 A2/A3 平台，24 核配置：

```python
# mL1 = 128, nL1 = 128 时
# 最优分核：mDim = nDim = sqrt(24) ≈ 5

# 在最外层循环增加 L2 友好的分块
m_view = mL1 * mDim  # 128 * 5 = 640
n_view = nL1 * nDim  # 128 * 5 = 640
```

### 9.3 代码示例

```python
# 参考 matmul-performance-guide.md 的 L2 优化示例
def pfa_with_l2_optimization():
    mL1 = 128
    nL1 = 128
    mDim = 5
    nDim = 5
    
    m_view = mL1 * mDim
    n_view = nL1 * nDim
    
    m_loop = (M + m_view - 1) // m_view
    n_loop = (N + n_view - 1) // n_view
    
    for m_idx in pypto.loop(m_loop):
        for n_idx in pypto.loop(n_loop):
            # 处理 m_view x n_view 的块
            # L2 命中率最优
```

---

## 10. 调度策略优化

### 10.1 可用调度策略

| 模式 | 名称 | 适用场景 |
|------|------|----------|
| 0 | 默认调度 | 通用场景 |
| 1 | L2 亲和调度 | 数据复用多的场景 |
| 2 | 公平调度 | 多线程、多核均衡场景 |

### 10.2 配置方法

```python
@pypto.frontend.jit(
    runtime_options={
        "device_sched_mode": 1  # L2 亲和调度
    }
)
```

### 10.3 建议测试顺序

1. **默认调度 (mode=0)**：作为基线
2. **L2 亲和调度 (mode=1)**：测试是否有提升
3. **公平调度 (mode=2)**：多核场景测试

---

## 11. 综合优化配置示例

### 11.1 完整优化配置

```python
def get_pfa_config_optimized(device="cpu"):
    """PFA 优化配置"""
    b = 8
    s1 = 128
    s2 = s1
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_size = 128
    block_table_batch = b
    kv_num_blocks = b * ((s1 + block_size - 1) // block_size)

    actual_seq_values = [s1] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(
        b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale,
        kv_layout=kv_layout, q_d=q_d, kv_d=q_d, block_size=block_size,
        block_table_batch=block_table_batch, kv_num_blocks=kv_num_blocks,
        actual_seq=actual_seq_tensor
    )
    atten_cfg.max_num_blocks_per_query = (s1 + block_size - 1) // block_size
    
    # 优化后的 Tile 配置
    cube_tile = 128
    m_tile = 128
    s2_tile = 256  # 增大 s2_tile
    
    tile_cfg = AttentionTileConfig(
        g_tile=nq,
        s2_tile=s2_tile,
        c1_tile_shape=[[128, 128], [64, 256], [128, 128]],  # 优化 K 轴
        v1_tile_shape=[128, s2_tile],  # [128, 256]
        c2_tile_shape=[[128, 128], [64, 256], [128, 128]],
        v2_tile_shape=[128, 128]
    )
    
    return atten_cfg, tile_cfg


def get_pfa_jit_config_optimized(s1, s2_tile):
    """优化后的 JIT 配置"""
    # 动态计算 L1 复用次数
    avg_s2_loop = (s1 + s2_tile - 1) // (2 * s2_tile)
    
    return {
        "runtime_options": {
            "stitch_function_num_initial": 128,
            "stitch_function_outcast_memory": 4096,  # 增大
            "stitch_function_inner_memory": 4096,    # 增大
            "device_sched_mode": 1  # L2 亲和调度
        },
        "pass_options": {
            "pg_upper_bound": 2048,  # 增大
            "cube_l1_reuse_setting": {0: max(2, avg_s2_loop)},
            "cube_l1_reuse_mode": 1
        },
        "debug_options": {
            "runtime_debug_mode": 1,
            "compile_debug_mode": 0
        }
    }
```

### 11.2 优化后的 Kernel 结构

```python
def pfa_func_optimized(q_shape, kv_shape, block_table_shape):
    """优化后的 PFA Kernel"""
    
    jit_config = get_pfa_jit_config_optimized(s1=128, s2_tile=256)
    
    @pypto.frontend.jit(**jit_config)
    def pfa_func_kernel(
        q: pypto.Tensor(q_shape, pypto.DT_BF16),
        k: pypto.Tensor(kv_shape, pypto.DT_BF16),
        v: pypto.Tensor(kv_shape, pypto.DT_BF16),
        block_table: pypto.Tensor(block_table_shape, pypto.DT_INT32),
        query_act_seqs: pypto.Tensor((bs,), pypto.DT_INT32),
        atten_out: pypto.Tensor(out_shape, pypto.DT_BF16)
    ):
        pypto.experimental.set_operation_options(combine_axis=True)
        
        atten_cfg, tile_cfg = get_pfa_config_optimized()
        softmax_scale = atten_cfg.softmax_scale
        
        # ... shape 解析 ...
        
        # 使用 pypto.loop 替代静态循环（并行化）
        for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
            for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
                cur_seq = s1_idx + 1
                s2_loop = (cur_seq + s2_tile - 1) // s2_tile
                
                # 并行化 KV Head 循环
                for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                    # 并行化 Query Head 组循环
                    for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                        # ...
                        
                        # 优化后的展开策略
                        for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", 
                                                 idx_name="s2_idx", 
                                                 unroll_list=[8, 4, 2, 1]):
                            # Flash Attention 核心计算
                            # ...
    
    return pfa_func_kernel
```

---

## 12. 优化验证流程

### 12.1 性能采集

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1}  # 开启性能数据采集
)
```

### 12.2 查看泳道图

1. 执行用例后，在 `output/output_时间戳/` 目录下生成 `merged_swimlane.json`
2. 使用 PyPTO Toolkit 插件查看泳道图
3. 分析性能瓶颈点

### 12.3 优化优先级

| 优先级 | 优化项 | 预期收益 | 实施难度 | 验证方法 |
|--------|--------|----------|----------|----------|
| **P0** | Tile 配置调整 | 10-20% | 低 | 泳道图对比 |
| **P0** | cube_l1_reuse_setting 动态配置 | 5-15% | 低 | GM 访问量对比 |
| **P1** | 使用 pypto.loop 替代静态循环 | 10-30% | 中 | 核利用率对比 |
| **P1** | L2 调度策略 | 5-10% | 低 | L2 命中率对比 |
| **P1** | 增大 stitch_function 内存配置 | 5-10% | 低 | 子图融合率对比 |
| **P2** | 循环展开参数调整 | 5-10% | 低 | 循环开销对比 |
| **P2** | K/V 组装优化 | 3-5% | 高 | 内存带宽对比 |

### 12.4 验证步骤

```bash
# 1. 基线测试
python test_pfa.py --config baseline

# 2. 应用单项优化
python test_pfa.py --config optimized_tile

# 3. 对比泳道图
# 使用 PyPTO Toolkit 对比 merged_swimlane.json

# 4. 迭代优化
# 根据瓶颈点继续调整
```

### 12.5 性能指标

| 指标 | 获取方法 | 优化目标 |
|------|----------|----------|
| 总耗时 | 泳道图末尾时间戳 | 最小化 |
| GM 访问量 | 泳道图 MTE2 统计 | 最小化 |
| L2 命中率 | 性能计数器 | 最大化 |
| 核利用率 | 泳道图核分布 | 均衡、高利用率 |
| 子图融合率 | 计算图节点数 | 最大化融合 |

### 12.6 推荐优化路径

```
Step 1: Tile 配置优化 (P0)
    ↓ 验证性能和精度
Step 2: JIT 配置优化 (P0)
    ↓ 验证性能和精度
Step 3: 并行循环优化 (P1)
    ↓ 验证性能和精度
Step 4: 调度策略优化 (P1)
    ↓ 验证性能和精度
Step 5: 其他优化项按需尝试
```

---

## 附录

### A. 参考文档

- `docs/tutorials/debug/performance.md` - 性能调优指南
- `docs/tutorials/debug/matmul-performance-guide.md` - Matmul 高性能编程指导
- `docs/tutorials/development/tiling.md` - Tiling 配置
- `docs/mylearning/pfa_instrucction.md` - PFA 实现原理
- `docs/mylearning/ifa_instrucction.md` - IFA 实现原理

### B. 配置参数速查表

#### Tile 配置约束

| 参数 | 约束条件 | 推荐值 (BF16) |
|------|----------|---------------|
| mL0 * nL0 * sizeof(float) | ≤ L0C_SIZE (131072) | 128 * 128 |
| kL0, nL0 | 32 字节对齐 | 64, 128, 256 |
| vec_tile | 16KB - 64KB, 尾轴 32B 对齐 | [128, 256] |

#### JIT 配置推荐值

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| pg_upper_bound | 1536-2048 | 子图大小上界 |
| stitch_function_inner_memory | 2048-4096 | 内部内存 (KB) |
| cube_l1_reuse_mode | 0 或 1 | L1 复用模式 |

### C. 常见问题

**Q1: 为什么增大 s2_tile 后性能反而下降？**

A: 可能原因：
- L0/L1 Buffer 空间不足
- block_num 增加导致组装开销增加
- s2_loop 过小，展开效果不佳

**Q2: cube_l1_reuse_setting 设置多少合适？**

A: 建议设置为实际 s2_loop 的平均值或略小，过大可能导致 L1 空间不足。

**Q3: 如何判断是否需要调整调度策略？**

A: 通过泳道图观察：
- 核间负载不均衡 → 尝试公平调度 (mode=2)
- L2 命中率低 → 尝试 L2 亲和调度 (mode=1)

---

*文档版本: 1.0*  
*适用模型: GLM-4.5*  
*PyPTO 版本: 0.1.0+*  
*最后更新: 2026年2月*
