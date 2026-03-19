# Online Softmax 模板规范

## 📋 概述

Online Softmax 是一种增量式 Softmax 算法，支持分块计算长序列的注意力分数，避免一次性加载完整 KV cache。

**核心思想**：通过维护 `max`、`sum`、`output` 三个状态变量，逐步更新 softmax 结果。

**适用场景**：
- 长序列推理（序列长度超出 UB 容量）
- 流式推理（增量生成 token）
- Paged Attention 实现

---

## 🧮 算法原理

### 标准 Softmax

```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

### Online Softmax 增量更新

**状态变量**：
- `m`: 当前最大值 (max)
- `d`: 当前指数和 (sum)
- `o`: 当前输出累加 (output)

**更新公式**：
```
m_new = max(m_old, m_tile)
d_new = d_old * exp(m_old - m_new) + d_tile * exp(m_tile - m_new)
o_new = o_old * exp(m_old - m_new) + o_tile * exp(m_tile - m_new)
```

---

## 📐 模板结构

### 核心骨架

```python
@pypto.frontend.jit()
def online_softmax_kernel(q, k, v, output, num_tiles, softmax_scale):
    # 1. 初始化状态变量
    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
    
    # 2. 循环处理每个 tile
    for tile_idx in pypto.loop(num_tiles, name="LOOP_TILE", idx_name="tile_idx", unroll_list=[8, 4, 2, 1]):
        # 2.1 计算 QK^T
        sij = pypto.matmul(q, k, pypto.DT_FP32, a_trans=False, b_trans=True)
        
        # 2.2 第一个 tile: 初始化
        if pypto.is_loop_begin(tile_idx):
            # 初始化 max, sum, output
            pass
        
        # 2.3 后续 tiles: 增量更新
        else:
            # 更新 max, sum, output（带修正）
            pass
        
        # 2.4 最后一个 tile: 归一化
        if pypto.is_loop_end(tile_idx):
            output[:] = oi_update / sum_update
```

---

## 🔑 关键 API

### 1. `pypto.is_loop_begin(idx)`

**作用**：判断当前是否为循环的第一次迭代

**用法**：
```python
if pypto.is_loop_begin(tile_idx):
    # 初始化逻辑
    max_update[:] = mij
    sum_update[:] = pij_sum
    oi_update[:] = pij @ vj
```

---

### 2. `pypto.is_loop_end(idx)`

**作用**：判断当前是否为循环的最后一次迭代

**用法**：
```python
if pypto.is_loop_end(tile_idx):
    # 最终归一化
    output[:] = oi_update / sum_update
```

---

### 3. `pypto.set_pass_options(sg_set_scope=N)`

**作用**：控制子图划分，优化计算图执行顺序

**用法**：
```python
# 子图1: 计算当前 tile 的 max 和 exp
pypto.set_pass_options(sg_set_scope=1)
mij = pypto.amax(sij, dim=-1, keepdim=True)
pij = pypto.exp(sij - mij)
pypto.set_pass_options(sg_set_scope=-1)

# 子图2: 计算修正因子
pypto.set_pass_options(sg_set_scope=2)
correction = pypto.exp(max_update - max_new)
pypto.set_pass_options(sg_set_scope=-1)
```

---

## 📊 状态变量规范

### 必需状态变量

| 变量 | Shape | 数据类型 | 用途 |
|------|-------|---------|------|
| `max_update` | `[g_tile, 1]` | `DT_FP32` | 累积最大值 |
| `sum_update` | `[g_tile, 1]` | `DT_FP32` | 累积指数和 |
| `oi_update` | `[g_tile, dn]` | `DT_FP32` | 累积输出 |

**强制要求**：
- ✅ **必须使用 FP32** 进行累积，避免精度损失
- ✅ 必须在循环**外**创建（避免重复创建）

---

## 🎯 Tile 配置

### Cube Tile (矩阵乘法)

```python
cube_tile = [[128, 128], [128, 128], [128, 128]]
pypto.set_cube_tile_shapes(cube_tile[0], cube_tile[1], cube_tile[2])

# QK^T: [g_tile, dn] @ [dn, s2_tile] -> [g_tile, s2_tile]
sij = pypto.matmul(q, k, pypto.DT_FP32, a_trans=False, b_trans=True)

# PV: [g_tile, s2_tile] @ [s2_tile, dn] -> [g_tile, dn]
oi = pypto.matmul(pij, v, pypto.DT_FP32)
```

### Vector Tile (逐元素运算)

```python
vec_tile = [128, 512]
pypto.set_vec_tile_shapes(vec_tile[0], vec_tile[1])

# Softmax 计算
mij = pypto.amax(sij, dim=-1, keepdim=True)
pij = pypto.exp(sij - mij)
```

---

## ⚠️ 注意事项

### 1. 数值稳定性

**问题**：直接计算 `exp(x)` 可能溢出

**解决**：减去最大值 `exp(x - max(x))`

```python
# ✅ 正确
mij = pypto.amax(sij, dim=-1, keepdim=True)
pij = pypto.exp(sij - mij)

# ❌ 错误（可能溢出）
pij = pypto.exp(sij)
```

---

### 2. 修正因子计算

**关键**：更新累积值时必须应用修正因子

```python
# 计算新最大值
max_new = pypto.maximum(max_update, mij)

# 修正因子: exp(old_max - new_max)
correction = pypto.exp(max_update - max_new)

# 更新累积值（带修正）
sum_update[:] = sum_update * correction + sum_local
oi_update[:] = oi_update * correction + oi_local
```

---

### 3. 数据类型转换

**规则**：
- 计算（matmul, softmax）→ **FP32**
- 存储（输入/输出）→ **BF16/FP16**

```python
# 输入: BF16
sij_bf16 = pypto.matmul(q, k, pypto.DT_FP32)  # matmul 输出 FP32

# Softmax 计算: FP32
pij_fp32 = pypto.exp(sij_fp32 - mij)

# 矩阵乘法前转回 BF16（节省内存带宽）
pij_bf16 = pypto.cast(pij_fp32, pypto.DT_BF16)
oi_fp32 = pypto.matmul(pij_bf16, v, pypto.DT_FP32)

# 输出: BF16
output[:] = pypto.cast(oi_fp32, pypto.DT_BF16)
```

---

## 📝 完整示例

```python
@pypto.frontend.jit()
def online_softmax_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    output: pypto.Tensor([], pypto.DT_BF16),
    num_tiles: int,
    softmax_scale: float
):
    g_tile = q.shape[0]
    dn = v.shape[1]
    dtype = q.dtype
    
    # 初始化状态
    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
    
    # 循环处理
    for tile_idx in pypto.loop(num_tiles, name="LOOP_TILE", idx_name="tile_idx"):
        # QK^T
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
        sij = pypto.matmul(q, k, pypto.DT_FP32, a_trans=False, b_trans=True)
        
        pypto.set_vec_tile_shapes(128, 512)
        
        if pypto.is_loop_begin(tile_idx):
            # 初始化
            sij_scale = pypto.mul(sij, softmax_scale)
            mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
            pij = pypto.exp(sij_scale - mij)
            pij_bf16 = pypto.cast(pij, dtype)
            
            sum_update[:] = pypto.sum(pij, dim=-1, keepdim=True)
            max_update[:] = mij
            
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            oi_tmp = pypto.matmul(pij_bf16, v, pypto.DT_FP32)
            oi_update[:] = oi_tmp
        
        else:
            # 增量更新
            pypto.set_pass_options(sg_set_scope=1)
            
            sij_scale = pypto.mul(sij, softmax_scale)
            mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
            max_new = pypto.maximum(max_update, mij)
            
            pij = pypto.exp(sij_scale - max_new)
            pij_bf16 = pypto.cast(pij, dtype)
            sum_local = pypto.sum(pij, dim=-1, keepdim=True)
            
            pypto.set_pass_options(sg_set_scope=-1)
            
            # 修正
            pypto.set_pass_options(sg_set_scope=2)
            correction = pypto.exp(max_update - max_new)
            max_update[:] = max_new
            sum_update[:] = sum_update * correction + sum_local
            pypto.set_pass_options(sg_set_scope=-1)
            
            # 更新输出
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            oi_tmp = pypto.matmul(pij_bf16, v, pypto.DT_FP32)
            oi_update[:] = oi_update * correction + oi_tmp
        
        # 归一化
        if pypto.is_loop_end(tile_idx):
            oi_final = pypto.div(oi_update, sum_update)
            output[:] = pypto.cast(oi_final, dtype)
```

---

## 🚀 性能优化建议

### 1. 循环展开

```python
for tile_idx in pypto.loop(
    num_tiles, 
    name="LOOP_TILE", 
    idx_name="tile_idx",
    unroll_list=[8, 4, 2, 1]  # 优先尝试展开8次
):
    pass
```

### 2. L1 复用

```python
@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {0: 4}  # Q 常驻 L1，4次 matmul 合并
    }
)
```

### 3. 子图优化

使用 `sg_set_scope` 将相关操作放入同一子图，减少数据搬运。

---

## 📚 参考资料

- **原理论文**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
- **PyPTO 实现**: `models/glm_v4_5/glm_attention.py:ifa_func_kernel`
- **模板代码**: `references/online_softmax.py`
