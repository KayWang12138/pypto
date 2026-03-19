# PyPTO 算子开发约束与规范清单

## 📋 概述

本文档列出开发 PyPTO 算子时必须遵守的约束和规范，包括 UB 限制、循环展开规则、数据类型转换规范等。

---

## ✅ 开发前检查清单

### 1. 环境配置
- [ ] NPU 设备可用（`npu-smi info`）
- [ ] 环境变量 `TILE_FWK_DEVICE_ID` 已设置
- [ ] 环境变量 `PTO_TILE_LIB_CODE_PATH` 已设置
- [ ] PyPTO whl 包已编译安装

### 2. 算子设计
- [ ] 输入输出张量 shape 已明确
- [ ] 数据类型已确定（BF16/FP16/FP32）
- [ ] 数学公式已转换为 PyPTO API 组合
- [ ] 是否需要 Paged KV Cache
- [ ] 是否需要 Online Softmax

---

## 🔒 硬件约束

### 1. UB (Unified Buffer) 大小限制

**关键限制**: UB 是 NPU 上的高速缓存，大小有限（通常 ~1MB）

**规则**:
- ✅ 单个 tile 的数据**不能**超过 UB 大小
- ✅ 需要根据 UB 大小合理设置 tile shapes
- ✅ 使用 `pypto.set_vec_tile_shapes()` 和 `pypto.set_cube_tile_shapes()` 控制

**示例配置**:
```python
# Vector tile shapes (用于逐元素运算)
pypto.set_vec_tile_shapes(128, 512)  # [M, N] 形状

# Cube tile shapes (用于矩阵乘法)
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])  # [M, K], [K, N], [M, N]
```

**估算公式**:
```python
# 单个 tile 数据量（字节）
tile_size_bytes = M * N * dtype_size

# 例如: FP32 (4 bytes), [128, 512]
tile_size = 128 * 512 * 4 = 262144 bytes = 256 KB

# 确保多个临时 tensor 不超过 UB
total_ub_usage = tile1_size + tile2_size + ... < UB_SIZE
```

---

### 2. L1/L0 Buffer 限制

**Cube 操作**（矩阵乘法）使用 L1/L0 Buffer:
- L1 Buffer 用于存储输入矩阵分块
- L0 Buffer 用于存储累加结果

**配置示例** (from glm_attention.py):
```python
@pypto.frontend.jit(
    pass_options={
        "pg_upper_bound": 1536,  # 子图大小上界
        "cube_l1_reuse_setting": {0: 4}  # Q常驻，4次matmul合并
    }
)
```

---

## 🔄 循环展开规则

### 1. 循环命名规范

**强制要求**: 所有循环**必须** 指定 `name` 和 `idx_name`

```python
# ✅ 正确
for b_idx in pypto.loop(B, name="LOOP_b", idx_name="b_idx"):
    for s_idx in pypto.loop(S, name="LOOP_s", idx_name="s_idx"):
        pass

# ❌ 错误
for i in pypto.loop(B):  # 缺少 name 和 idx_name
    pass
```

---

### 2. 循环展开优化

**使用 `unroll_list` 参数**:

```python
# 推荐: 使用 unroll_list 自动展开
for s2_idx in pypto.loop(
    s2_loop, 
    name="LOOP_s2", 
    idx_name="s2_idx", 
    unroll_list=[8, 4, 2, 1]  # 优先尝试展开8次，不行则4次，依此类推
):
    pass
```

**unroll_list 选择原则**:
- 大循环（如 seq_len 维度）使用 `[8, 4, 2, 1]`
- 小循环（如 num_heads 维度）可以不展开或使用 `[4, 2, 1]`
- 展开次数越多，性能越好，但编译时间越长

---

### 3. 动态轴处理

**动态轴** (运行时确定的维度) 需要特殊处理:

```python
# 使用 pypto.DYNAMIC 标记动态维度
q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)

# 在 kernel 内部获取动态 shape
shape_q = q.shape
batch_size = shape_q[0]  # 动态获取
```

---

## 🔢 数据类型规范

### 1. 支持的数据类型

| PyPTO 类型 | C++ 对应 | 字节数 | 适用场景 |
|-----------|---------|-------|---------|
| `pypto.DT_FP32` | float | 4 | 累加、softmax 计算 |
| `pypto.DT_FP16` | half | 2 | 存储、低精度计算 |
| `pypto.DT_BF16` | bfloat16 | 2 | 存储、低精度计算（推荐） |
| `pypto.DT_INT32` | int32 | 4 | 索引、block_table |
| `pypto.DT_INT64` | int64 | 8 | 大规模索引 |

---

### 2. 类型转换规范

**规则**:
1. ✅ **计算时使用 FP32**（矩阵乘法、softmax、累加）
2. ✅ **存储时使用 BF16/FP16**（输入、输出、KV cache）
3. ✅ **显式调用 `pypto.cast()`**，不要依赖隐式转换

**示例**:
```python
# ✅ 正确
qi_fp32 = pypto.cast(qi, pypto.DT_FP32)  # 转换为 FP32 进行计算
scores = pypto.matmul(qi_fp32, kj_fp32, pypto.DT_FP32)  # 矩阵乘法输出 FP32
output_bf16 = pypto.cast(output_fp32, pypto.DT_BF16)  # 转回 BF16 输出

# ❌ 错误
scores = pypto.matmul(qi_bf16, kj_bf16, pypto.DT_BF16)  # BF16 精度不足
```

---

### 3. 精度容忍度

**BF16 精度损失**:
- 相对误差: ~1e-2 (1%)
- 绝对误差: ~1e-3

**验证时使用的容忍度** (from glm_attention.py):
```python
from numpy.testing import assert_allclose

# BF16 推荐容忍度
assert_allclose(
    golden_output, 
    pypto_output, 
    rtol=0.0078125,  # 1/128 ≈ 0.78%
    atol=0.0001
)
```

---

## 🎯 Tile 配置规范

### 1. Cube Tile Shapes (矩阵乘法)

**配置原则**:
- M, N, K 通常设为 64 或 128 的倍数
- 需要根据实际矩阵大小调整
- 参考 CANN 官方推荐配置

**示例** (from glm_attention.py):
```python
# QK^T: [g_tile, dn] @ [dn, s2_tile] -> [g_tile, s2_tile]
c1_tile = [[128, 128], [128, 128], [128, 128]]
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)

# PV: [g_tile, s2_tile] @ [s2_tile, dn] -> [g_tile, dn]
c2_tile = [[128, 128], [128, 128], [128, 128]]
pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
oi = pypto.matmul(pij, vj, pypto.DT_FP32)
```

---

### 2. Vector Tile Shapes (逐元素运算)

**配置原则**:
- 尽可能用满 UB
- 考虑多个临时 tensor 的总大小

**示例**:
```python
# 用于 softmax 计算
v1_tile = [128, 512]  # [M, N]
pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])

# 用于输出归一化
v2_tile = [128, 128]
pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
```

---

## 🧩 子图划分规范

### 1. Pass Options 配置

**用于控制计算图划分** (from glm_attention.py):

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},  # 最大拼接函数数
    pass_options={
        "pg_upper_bound": 1536,  # 子图大小上界（超过则不与其他子图合并）
        "cube_l1_reuse_setting": {0: 4}  # L1复用: 0表示第一组mmad，4代表4次matmul合并
    }
)
```

---

### 2. 子图作用域控制

**使用 `pypto.set_pass_options(sg_set_scope=N)`**:

```python
# sg_set_scope=1: 将接下来的操作放入子图1
pypto.set_pass_options(sg_set_scope=1)
sij_scale = pypto.mul(sij, softmax_scale)
mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
# ... 更多操作
pypto.set_pass_options(sg_set_scope=-1)  # 结束子图1

# sg_set_scope=2: 将接下来的操作放入子图2
pypto.set_pass_options(sg_set_scope=2)
# ... 其他操作
pypto.set_pass_options(sg_set_scope=-1)  # 结束子图2
```

**用途**: 优化计算图执行顺序，提高 NPU 利用率

---

## 📦 内存管理规范

### 1. 张量创建

**规则**:
- ✅ 使用 `pypto.tensor()` 创建临时张量
- ✅ 必须指定 `name` 参数（用于调试）
- ❌ 不要在循环内频繁创建大张量

**示例**:
```python
# ✅ 正确: 在循环外创建
accum = pypto.tensor([tile_size, dim], pypto.DT_FP32, "accum")
for i in pypto.loop(num_tiles):
    # 使用 accum

# ❌ 错误: 在循环内创建
for i in pypto.loop(num_tiles):
    temp = pypto.tensor([tile_size, dim], pypto.DT_FP32, "temp")  # 每次循环都创建
```

---

### 2. Inplace 操作

**优先使用 inplace 操作减少内存拷贝**:

```python
# ✅ 正确: inplace reshape
k_2d = pypto.reshape(k, [num_blocks * block_size, dim], inplace=True)

# ✅ 正确: 切片赋值
accum[:] = new_value

# ❌ 避免: 非 inplace 操作（会拷贝）
k_2d = pypto.reshape(k, [num_blocks * block_size, dim])  # 默认非 inplace
```

---

### 3. Paged KV Cache 访问模式

**标准模式** (from glm_attention.py):

```python
# 1. Reshape cache to 2D
k_2d = pypto.reshape(k_cache, [num_blocks * block_size, dim], inplace=True)

# 2. 创建组装张量
kj_assemble = pypto.tensor([s2_tile, dim], dtype, "kj_assemble")

# 3. 逐块组装
for i in range(num_blocks):
    block_idx = block_table[batch_idx, tile_start + i]
    block_idx_valid = block_idx.max(0)  # 处理 padding (-1 -> 0)
    kj_assemble[i*block_size:(i+1)*block_size, :] = \
        pypto.view(k_2d, [block_size, dim], [block_idx_valid*block_size, 0])

# 4. 设置 valid_shape
kj_assemble = pypto.view(kj_assemble, [s2_tile, dim], [0, 0], valid_shape=[actual_size, dim])
```

---

## 🚨 常见错误与解决方案

### 1. "UB size exceeded"

**原因**: Tile 太大，超过 UB 容量

**解决**:
```python
# 减小 tile shapes
pypto.set_vec_tile_shapes(64, 256)  # 原来是 [128, 512]
```

---

### 2. "Invalid Device"

**原因**: `TILE_FWK_DEVICE_ID` 未设置或设置错误

**解决**:
```bash
# 检查可用 NPU
npu-smi info

# 设置正确的 device ID
export TILE_FWK_DEVICE_ID=0  # 或其他可用 ID
```

---

### 3. "Unknown keyword argument(s): ['run_mode']"

**原因**: PyPTO kernel 调用时传递了不支持的关键字参数

**解决**:
```python
# ❌ 错误
kernel(inputs, run_mode="npu")

# ✅ 正确: 直接调用
kernel(*inputs)
```

---

### 4. 精度不达标

**原因**: 计算过程中精度损失累积

**解决**:
```python
# 确保关键计算使用 FP32
qi_fp32 = pypto.cast(qi, pypto.DT_FP32)
scores = pypto.matmul(qi_fp32, kj_fp32, pypto.DT_FP32)
```

---

## 📊 性能优化检查清单

### 1. 内存访问
- [ ] 优先使用 `inplace=True` 减少拷贝
- [ ] 合理设置 tile shapes 用满 UB
- [ ] Paged KV Cache 使用 2D reshape 优化访问

### 2. 计算优化
- [ ] 使用 `unroll_list` 展开循环
- [ ] 配置 `cube_l1_reuse_setting` 复用 L1
- [ ] 使用 `sg_set_scope` 优化子图划分

### 3. 数据类型
- [ ] 计算使用 FP32
- [ ] 存储使用 BF16/FP16
- [ ] 避免不必要的类型转换

---

## 📚 参考资料

- **官方文档**: `docs/api/` 目录下的 API 文档
- **示例代码**: `examples/` 目录下的官方示例
- **生产实现**: `models/glm_v4_5/glm_attention.py`
- **模板代码**: `references/online_softmax.py`, `references/paged_kv_cache.py`

---

## ✅ 提交前检查

- [ ] 代码编译无错误无警告
- [ ] 功能验证通过（与 PyTorch golden 对比）
- [ ] 精度验证通过（rtol/atol 满足要求）
- [ ] 性能验证通过（NPU 利用率合理）
- [ ] 文档完整（README.md 包含使用说明）
- [ ] 已知限制已记录
