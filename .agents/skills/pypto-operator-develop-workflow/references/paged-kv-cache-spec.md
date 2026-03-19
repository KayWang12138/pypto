# Paged KV Cache 模板规范

## 📋 概述

Paged KV Cache 是一种非连续内存访问模式，用于实现 Paged Attention。通过 block table 将逻辑上的连续 KV 序列映射到物理上非连续的内存块。

**核心思想**：将 KV cache 划分为固定大小的 block，使用 block table 记录逻辑序列到物理 block 的映射关系。

**适用场景**：
- 变长序列推理（不同请求的序列长度不同）
- 内存高效管理（按需分配，避免预分配浪费）
- 多轮对话（prefix caching）
- vLLM / Paged Attention 实现

---

## 🧮 数据结构

### Block Table

**作用**：记录逻辑序列到物理 block 的映射

**Shape**: `[batch_size, max_num_blocks_per_query]`

**示例**：
```python
# 3 个请求，每个请求最多 4 个 block
block_table = [
    [0, 1, -1, -1],    # 请求0: 使用物理 block 0, 1（序列长度 256，2个block）
    [2, 3, 4, -1],     # 请求1: 使用物理 block 2, 3, 4（序列长度 512，4个block）
    [5, -1, -1, -1]    # 请求2: 使用物理 block 5（序列长度 128，1个block）
]
```

**特殊值**：
- `-1`: 未使用的 slot（padding）

---

### KV Cache

**Shape**: `[num_blocks, block_size, num_heads, head_dim]`

**示例**：
```python
# 16 个 block，每个 block 128 个 token，8 个 head，每个 head 128 维
k_cache = torch.randn(16, 128, 8, 128, dtype=torch.bfloat16)
v_cache = torch.randn(16, 128, 8, 128, dtype=torch.bfloat16)
```

---

## 📐 模板结构

### 核心骨架

```python
@pypto.frontend.jit()
def paged_kv_cache_kernel(
    block_table: pypto.Tensor([], pypto.DT_INT32),
    k_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    output_k: pypto.Tensor([], pypto.DT_BF16),
    output_v: pypto.Tensor([], pypto.DT_BF16),
    batch_idx: int,
    tile_start_block: int,
    num_blocks: int,
    block_size: int,
    head_dim: int
):
    # 1. Reshape cache 为 2D（优化访问）
    k_2d = pypto.reshape(k_cache, [num_blocks * block_size, head_dim], inplace=True)
    v_2d = pypto.reshape(v_cache, [num_blocks * block_size, head_dim], inplace=True)
    
    # 2. 创建组装张量
    kj_assemble = pypto.tensor([s2_tile, head_dim], dtype, "kj_assemble")
    
    # 3. 逐块组装
    for i in range(num_blocks):
        block_idx = block_table[batch_idx, tile_start_block + i]
        block_idx_valid = block_idx.max(0)  # 处理 padding
        kj_assemble[i*block_size:(i+1)*block_size, :] = \
            pypto.view(k_2d, [block_size, head_dim], [block_idx_valid*block_size, 0])
    
    # 4. 设置 valid_shape
    kj_assemble = pypto.view(kj_assemble, [s2_tile, head_dim], [0, 0], valid_shape=[actual_size, head_dim])
    
    # 5. 输出
    output_k[:] = kj_assemble
```

---

## 🔑 关键 API

### 1. `pypto.reshape(tensor, new_shape, inplace=True)`

**作用**：改变张量形状，**必须使用 `inplace=True`** 避免内存拷贝

**用法**：
```python
# 将 4D cache reshape 为 2D
# [num_blocks, block_size, num_heads, head_dim] -> [num_blocks * block_size, num_heads * head_dim]
k_2d = pypto.reshape(k_cache, [num_blocks * block_size, head_dim], inplace=True)
```

---

### 2. `pypto.view(tensor, shape, offset, valid_shape=None)`

**作用**：创建张量视图，支持指定偏移量和有效形状

**参数**：
- `shape`: 输出形状 `[M, N]`
- `offset`: 起始偏移 `[row_offset, col_offset]`
- `valid_shape`: 实际有效数据的形状（用于变长序列）

**用法**：
```python
# 从 k_2d 的指定位置读取一个 block
block_data = pypto.view(k_2d, [block_size, head_dim], [block_idx * block_size, 0])

# 设置 valid_shape（实际序列长度可能小于 tile 大小）
kj_assemble = pypto.view(kj_assemble, [s2_tile, head_dim], [0, 0], valid_shape=[actual_seq_len, head_dim])
```

---

### 3. `block_idx.max(0)`

**作用**：处理 padding（将 `-1` 映射为 `0`）

**原因**：`block_table` 中 `-1` 表示未使用的 slot，直接访问会导致错误

**用法**：
```python
block_idx = block_table[batch_idx, tile_start + i]  # 可能为 -1
block_idx_valid = block_idx.max(0)  # -1 -> 0（读取 block 0 的数据，通常是 zeros）
```

---

## 📊 参数规范

### Block Table 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `block_table` | `DT_INT32` | Block 映射表 `[batch_size, max_blocks_per_query]` |
| `batch_idx` | `int` | 当前处理的 batch 索引 |
| `tile_start_block` | `int` | 当前 tile 的起始 block 索引（在 block_table 中） |
| `num_blocks` | `int` | 当前 tile 包含的 block 数量 |

---

### KV Cache 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `k_cache` | `DT_BF16` | Key cache `[num_blocks, block_size, num_heads, head_dim]` |
| `v_cache` | `DT_BF16` | Value cache `[num_blocks, block_size, num_heads, head_dim]` |
| `block_size` | `int` | 每个 block 的 token 数量（通常 128） |
| `head_dim` | `int` | 每个 head 的维度（通常 128） |

---

### 输出参数

| 参数 | Shape | 说明 |
|------|-------|------|
| `output_k` | `[tile_size, head_dim]` | 组装后的 Key 张量 |
| `output_v` | `[tile_size, head_dim]` | 组装后的 Value 张量 |

其中 `tile_size = num_blocks * block_size`

---

## ⚠️ 注意事项

### 1. Reshape 必须使用 inplace

**问题**：非 inplace reshape 会触发内存拷贝，降低性能

**解决**：
```python
# ✅ 正确
k_2d = pypto.reshape(k_cache, [num_blocks * block_size, head_dim], inplace=True)

# ❌ 错误（会拷贝）
k_2d = pypto.reshape(k_cache, [num_blocks * block_size, head_size])
```

---

### 2. 处理 Padding

**问题**：`block_table` 中 `-1` 表示 padding，直接访问会出错

**解决**：
```python
block_idx = block_table[batch_idx, tile_start + i]
block_idx_valid = block_idx.max(0)  # -1 -> 0
```

**注意**：block 0 应初始化为 zeros，避免读取垃圾数据

---

### 3. Valid Shape 设置

**问题**：实际序列长度可能小于 tile 大小，需要标记有效区域

**解决**：
```python
# 假设 tile_size = 512，但实际序列长度 = 256
actual_seq_len = 256
kj_assemble = pypto.view(
    kj_assemble, 
    [512, head_dim],  # tile 形状
    [0, 0],           # 起始位置
    valid_shape=[256, head_dim]  # 实际有效数据
)
```

---

### 4. Python for 循环（非 pypto.loop）

**说明**：Paged KV Cache 的 block 组装使用**普通 Python for 循环**，不是 `pypto.loop`

**原因**：`num_blocks` 通常较小（如 2-4），且需要动态索引 `block_table`

```python
# ✅ 正确（普通 for）
for i in range(num_blocks):
    block_idx = block_table[batch_idx, tile_start + i]
    # ...

# ❌ 错误（pypto.loop）
for i in pypto.loop(num_blocks):  # 不需要
    pass
```

---

## 📝 完整示例

```python
@pypto.frontend.jit()
def paged_kv_cache_kernel(
    block_table: pypto.Tensor([], pypto.DT_INT32),
    k_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v_cache: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    output_k: pypto.Tensor([], pypto.DT_BF16),
    output_v: pypto.Tensor([], pypto.DT_BF16),
    batch_idx: int,
    tile_start_block: int,
    num_blocks: int,
    block_size: int,
    head_dim: int
):
    num_blocks_scalar = k_cache.shape[0]
    s2_tile = num_blocks * block_size
    dtype = k_cache.dtype
    
    # Reshape cache 为 2D
    k_cache_2d = pypto.reshape(k_cache, [num_blocks_scalar * block_size, head_dim], inplace=True)
    v_cache_2d = pypto.reshape(v_cache, [num_blocks_scalar * block_size, head_dim], inplace=True)
    
    # 组装 Key Cache
    kj_assemble = pypto.tensor([s2_tile, head_dim], dtype, "kj_assemble")
    
    for i in range(num_blocks):
        # 从 block table 获取物理 block 索引
        block_idx = block_table[batch_idx, tile_start_block + i]
        
        # 处理 padding (-1 -> 0)
        block_idx_valid = block_idx.max(0)
        
        # 拷贝 block 数据到组装张量
        kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(k_cache_2d, [block_size, head_dim], [block_idx_valid * block_size, 0])
    
    # 设置 valid_shape
    kj_assemble = pypto.view(kj_assemble, [s2_tile, head_dim], [0, 0], valid_shape=[s2_tile, head_dim])
    
    # 组装 Value Cache
    vj_assemble = pypto.tensor([s2_tile, head_dim], dtype, "vj_assemble")
    
    for i in range(num_blocks):
        block_idx = block_table[batch_idx, tile_start_block + i]
        block_idx_valid = block_idx.max(0)
        vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
            pypto.view(v_cache_2d, [block_size, head_dim], [block_idx_valid * block_size, 0])
    
    vj_assemble = pypto.view(vj_assemble, [s2_tile, head_dim], [0, 0], valid_shape=[s2_tile, head_dim])
    
    # 输出
    output_k[:] = kj_assemble
    output_v[:] = vj_assemble
```

---

## 🛠️ 辅助函数

### 生成 Block Table

```python
def gen_block_table(actual_seq_len, block_size, block_table_shape):
    """
    生成 block 映射表
    
    Args:
        actual_seq_len: 实际序列长度 [batch_size]
        block_size: 每个 block 的大小
        block_table_shape: block table 形状 [batch_size, max_blocks_per_query]
    
    Returns:
        block_table: Block 映射表，未使用的 slot 为 -1
    
    Example:
        actual_seq_len = [256, 512, 128]
        block_size = 128
        block_table_shape = [3, 4]
        
        Output:
        [[0, 1, -1, -1],    # 256 tokens -> 2 blocks
         [2, 3, 4, -1],     # 512 tokens -> 4 blocks
         [5, -1, -1, -1]]   # 128 tokens -> 1 block
    """
    block_num_per_batch = []
    block_num = 0
    
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num += math.ceil(actual_seq / block_size)
    
    # 随机排列 block 索引
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]
    
    # 创建 block table
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    
    return block_table
```

---

## 🚀 性能优化建议

### 1. Reshape 优化

**必须使用 `inplace=True`**，避免内存拷贝：
```python
k_2d = pypto.reshape(k_cache, [num_blocks * block_size, head_dim], inplace=True)
```

---

### 2. Block Size 选择

**推荐**: `block_size = 128` 或 `64`

**原因**：
- 128 tokens × 128 dims × 2 bytes (BF16) = 32 KB per block
- 适合 L1/L0 buffer 大小

---

### 3. 批量处理

**优化**: 一次处理多个 block，减少 Python 循环开销

```python
# 如果 num_blocks 较大，考虑一次处理多个 block
for i in range(0, num_blocks, 2):  # 一次处理 2 个 block
    for j in range(2):
        if i + j < num_blocks:
            block_idx = block_table[batch_idx, tile_start + i + j]
            # ...
```

---

## 🔄 与 Online Softmax 结合

Paged KV Cache 通常与 Online Softmax 结合使用：

```python
@pypto.frontend.jit()
def paged_attention_kernel(
    q, block_table, k_cache, v_cache, output,
    batch_idx, num_tiles, block_size, head_dim, softmax_scale
):
    for tile_idx in pypto.loop(num_tiles, name="LOOP_TILE", idx_name="tile_idx"):
        # 1. 使用 Paged KV Cache 组装 K, V
        tile_start_block = tile_idx * tile_size // block_size
        num_blocks = tile_size // block_size
        
        kj = assemble_paged_kv(block_table, k_cache, batch_idx, tile_start_block, num_blocks)
        vj = assemble_paged_kv(block_table, v_cache, batch_idx, tile_start_block, num_blocks)
        
        # 2. 使用 Online Softmax 计算
        # ... (见 online-softmax-spec.md)
```

---

## 📚 参考资料

- **原理论文**: *Efficient Memory Management for Deep Learning Inference Servers* (vLLM)
- **PyPTO 实现**: `models/glm_v4_5/glm_attention.py:ifa_func_kernel` (lines 325-331, 353-360, 385-392)
- **模板代码**: `references/paged_kv_cache.py`
