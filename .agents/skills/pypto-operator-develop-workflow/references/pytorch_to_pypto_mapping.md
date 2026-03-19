# PyTorch 到 PyPTO API 映射规则指南

## 📌 概述

本文档明确 PyTorch 操作到 PyPTO DSL API 的映射关系，帮助开发者将 PyTorch Golden 代码转换为符合昇腾 NPU 架构的实现。

---

## 🔢 张量操作映射

### 1. 张量创建

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `torch.zeros(shape, dtype)` | `pypto.tensor(shape, dtype, "name")` | 创建零张量（PyPTO默认初始化为0） |
| `torch.empty(shape, dtype)` | `pypto.tensor(shape, dtype, "name")` | 创建未初始化张量 |
| `torch.randn(shape, dtype)` | **主机端准备** | PyPTO kernel内部不直接支持随机生成 |

**示例**:
```python
# PyTorch
accum = torch.zeros([batch, heads, dim], dtype=torch.float32)

# PyPTO
accum = pypto.tensor([batch, heads, dim], pypto.DT_FP32, "accum")
```

---

### 2. 数据类型转换

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `x.float()` | `pypto.cast(x, pypto.DT_FP32)` | 转换为 FP32 |
| `x.half()` | `pypto.cast(x, pypto.DT_FP16)` | 转换为 FP16 |
| `x.bfloat16()` | `pypto.cast(x, pypto.DT_BF16)` | 转换为 BF16 |
| `x.to(dtype)` | `pypto.cast(x, dtype)` | 通用类型转换 |

**示例**:
```python
# PyTorch
q_fp32 = q.float()

# PyPTO
q_fp32 = pypto.cast(q, pypto.DT_FP32)
```

---

### 3. 形状操作

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `x.reshape(shape)` | `pypto.reshape(x, shape)` 或 `pypto.reshape(x, shape, inplace=True)` | reshape |
| `x.view(shape)` | `pypto.view(x, shape, [offsets])` | view（需要指定offsets） |
| `x.transpose(dim0, dim1)` | `pypto.transpose(x, dim0, dim1)` | 转置 |
| `x.unsqueeze(dim)` | `pypto.reshape(x, [..., 1, ...])` | 增加维度 |

**重要差异**:
- PyPTO 的 `view` **必须** 指定 offsets（偏移量）
- PyPTO 的 `reshape` 支持 `inplace=True` 来避免内存拷贝

**示例**:
```python
# PyTorch
q_2d = q.reshape(batch * seq_len, heads * dim)
k_t = k.transpose(0, 1)

# PyPTO
q_2d = pypto.reshape(q, [batch * seq_len, heads * dim], inplace=True)
k_t = pypto.transpose(k, 0, 1)

# PyPTO view with offsets
q_tile = pypto.view(q_2d, [tile_size, dim], [offset, 0])
```

---

## 🧮 数学运算映射

### 1. 矩阵乘法

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `torch.matmul(a, b)` | `pypto.matmul(a, b, out_dtype)` | 矩阵乘法 |
| `a @ b` | `pypto.matmul(a, b, out_dtype)` | 矩阵乘法 |
| `torch.mm(a, b)` | `pypto.matmul(a, b, out_dtype)` | 2D矩阵乘法 |

**关键差异**:
- PyPTO 的 `matmul` **必须** 指定 `out_dtype`
- 需要先调用 `pypto.set_cube_tile_shapes()` 设置 Cube tiling

**示例**:
```python
# PyTorch
scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq, seq]

# PyPTO
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
scores = pypto.matmul(q, pypto.transpose(k, -2, -1), pypto.DT_FP32)
```

---

### 2. 逐元素运算

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `a + b` | `pypto.add(a, b)` | 加法 |
| `a - b` | `pypto.sub(a, b)` | 减法 |
| `a * b` | `pypto.mul(a, b)` | 乘法 |
| `a / b` | `pypto.div(a, b)` | 除法 |
| `torch.exp(x)` | `pypto.exp(x)` | 指数 |
| `torch.max(a, b)` | `pypto.maximum(a, b)` | 逐元素最大值 |
| `x.max()` | `pypto.max(x, dim)` | 标量最大值 |

**示例**:
```python
# PyTorch
scores_scaled = scores * scale
p = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)

# PyPTO
scores_scaled = pypto.mul(scores, scale)
m = pypto.amax(scores, dim=-1, keepdim=True)
p = pypto.exp(pypto.sub(scores, m))
```

---

### 3. 规约运算

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `x.sum(dim)` | `pypto.sum(x, dim, keepdim)` | 求和 |
| `x.max(dim)` | `pypto.amax(x, dim, keepdim)` | 沿维度最大值 |
| `x.min(dim)` | `pypto.amin(x, dim, keepdim)` | 沿维度最小值 |

**示例**:
```python
# PyTorch
sum_p = p.sum(dim=-1, keepdim=True)

# PyPTO
sum_p = pypto.sum(p, dim=-1, keepdim=True)
```

---

### 4. Softmax 映射

PyTorch 的 `torch.softmax()` 在 FlashAttention 中需要展开为 **Online Softmax**:

```python
# PyTorch (标准 softmax)
def softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=-1, keepdim=True)

# PyPTO (Online Softmax for FlashAttention)
# 参考 online_softmax.py 模板
for tile_idx in pypto.loop(num_tiles):
    sij = pypto.matmul(qi, kj_t, pypto.DT_FP32)
    sij_scaled = pypto.mul(sij, scale)
    
    if pypto.is_loop_begin(tile_idx):
        # 第一个tile: 初始化
        m = pypto.amax(sij_scaled, dim=-1, keepdim=True)
        p = pypto.exp(pypto.sub(sij_scaled, m))
        sum_p = pypto.sum(p, dim=-1, keepdim=True)
        out = pypto.matmul(p, vj, pypto.DT_FP32)
    else:
        # 后续tile: 增量更新
        m_new = pypto.maximum(m, pypto.amax(sij_scaled, dim=-1, keepdim=True))
        correction = pypto.exp(pypto.sub(m, m_new))
        p_new = pypto.exp(pypto.sub(sij_scaled, m_new))
        sum_local = pypto.sum(p_new, dim=-1, keepdim=True)
        sum_p = pypto.add(pypto.mul(sum_p, correction), sum_local)
        out = pypto.add(pypto.mul(out, correction), pypto.matmul(p_new, vj, pypto.DT_FP32))
        m = m_new
    
    if pypto.is_loop_end(tile_idx):
        out = pypto.div(out, sum_p)
```

---

## 🔄 循环和控制流映射

### 1. 循环结构

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `for i in range(n):` | `for i in pypto.loop(n, name="LOOP", idx_name="i"):` | NPU循环 |
| N/A | `pypto.loop(..., unroll_list=[8,4,2,1])` | 支持展开优化 |

**示例**:
```python
# PyTorch
for batch_idx in range(batch_size):
    for seq_idx in range(seq_len):
        # process

# PyPTO
for batch_idx in pypto.loop(batch_size, name="LOOP_b", idx_name="b_idx"):
    for seq_idx in pypto.loop(seq_len, name="LOOP_s", idx_name="s_idx"):
        # process
```

---

### 2. 条件分支

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `if condition:` | `if pypto.is_loop_begin(idx):` | 循环开始 |
| `if condition:` | `if pypto.is_loop_end(idx):` | 循环结束 |

**注意**: PyPTO kernel 内部的条件分支有限制，主要用于 `is_loop_begin` 和 `is_loop_end`

---

## 📦 内存管理映射

### 1. 索引和切片

| PyTorch | PyPTO | 说明 |
|---------|-------|------|
| `x[i]` | `x[i]` | 单元素索引 |
| `x[i:j]` | `x[i:j]` | 切片 |
| `x[:, i:j]` | `x[:, i:j]` | 多维切片 |
| `x[start:end, :] = y` | `x[start:end, :] = y` | 切片赋值 |

**示例**:
```python
# PyTorch
output[batch_idx] = result

# PyPTO
output[batch_idx] = result  # 或使用 pypto.assemble()
```

---

### 2. Paged KV Cache 访问

PyTorch 中直接索引，PyPTO 需要显式组装:

```python
# PyTorch
k_block = k_cache[block_idx]  # 直接索引
v_block = v_cache[block_idx]

# PyPTO (Paged KV Cache)
kj_assemble = pypto.tensor([tile_size, dim], dtype, "kj_assemble")
for i in range(num_blocks):
    block_idx = block_table[batch_idx, tile_start + i]
    block_idx_valid = block_idx.max(0)  # 处理padding
    kj_assemble[i*block_size:(i+1)*block_size, :] = \
        pypto.view(k_cache_2d, [block_size, dim], [block_idx_valid*block_size, 0])
```

---

## 🎯 FlashAttention 特定映射

### 完整的 FlashAttention 映射示例

```python
# PyTorch Golden
def flash_attention_torch(q, k, v, scale):
    batch, heads, seq_len, dim = q.shape
    
    # QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Softmax
    scores_max = scores.max(dim=-1, keepdim=True).values
    p = torch.exp(scores - scores_max)
    p_sum = p.sum(dim=-1, keepdim=True)
    p = p / p_sum
    
    # PV
    output = torch.matmul(p, v)
    return output

# PyPTO Kernel (Online Softmax + Paged KV)
@pypto.frontend.jit()
def flash_attention_pypto(q, k, v, block_table, kv_act_seqs, output, num_tiles, scale):
    # ... (参考 glm_attention.py 和 online_softmax.py 模板)
```

---

## ⚠️ 关键注意事项

1. **数据类型**: PyPTO 的 `matmul` **必须** 指定 `out_dtype`
2. **Tile 配置**: 矩阵乘法前**必须** 调用 `pypto.set_cube_tile_shapes()`
3. **View offsets**: PyPTO 的 `view` **必须** 指定 offsets
4. **循环命名**: PyPTO 循环**必须** 指定 `name` 和 `idx_name`
5. **Pass Options**: 复杂计算图需要使用 `pypto.set_pass_options()` 控制子图划分

---

## 📚 参考示例

- **完整 FlashAttention**: `models/glm_v4_5/glm_attention.py`
- **Online Softmax 模板**: `references/online_softmax.py`
- **Paged KV Cache 模板**: `references/paged_kv_cache.py`
