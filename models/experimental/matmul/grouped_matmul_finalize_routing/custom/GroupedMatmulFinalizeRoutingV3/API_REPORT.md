# GroupedMatmulFinalizeRoutingV3 API 探索报告

## 1. 算子计算逻辑分解

基于 SPEC.md 中的数学公式，将 GroupedMatmulFinalizeRoutingV3 算子分解为以下原子操作：

### 1.1 计算流程概览

```
输入: x1(M,K), x2(E,K,N), scale(E,Ceil(K/64),N,2), pertokenScale(M,Ceil(K/64),2),
      groupList(E), rowIndex(M), sharedInput(bsdp,N), sharedInputWeight, sharedInputOffset
输出: out(batch,N)

步骤1: 分组矩阵乘法（GMM） - 对每个专家组执行 MXFP8 scaled_matmul
步骤2: 路由分配 - 按 rowIndex 进行 scatter add 操作
步骤3: 共享专家融合 - 加权融合共享专家输出（可选）
```

### 1.2 原子操作分解

| 步骤 | 原子操作 | 数学表达式 | 计算描述 |
|------|---------|------------|----------|
| **Step 1** | 分组矩阵乘法 | $y_i = (x_i \times weight_i) \times scale_i \times perTokenScale_i$ | MXFP8 量化矩阵乘法，按专家组分组 |
| **Step 1.1** | Tensor切片 | $x_i = x1[begin:end, :]$ | 根据 groupList 提取专家组对应的输入 |
| **Step 1.2** | Tensor切片 | $weight_i = x2[i]$ | 提取第 i 个专家的权重 |
| **Step 1.3** | Tensor切片 | $scale_i = scale[i]$ | 提取第 i 个专家的缩放因子 |
| **Step 1.4** | Tensor切片 | $pertoken\_scale_i = pertokenScale[begin:end]$ | 提取专家组对应的 token 级缩放因子 |
| **Step 1.5** | scaled_mm | $y_i = \text{scaled\_mm}(x_i, weight_i, scale_i, pertoken\_scale_i)$ | 执行 MXFP8 量化矩阵乘法 |
| **Step 2** | Scatter Add | $out[rowIndex[j], :] \mathrel{+}= intermediate[j, :]$ | 按路由索引将结果分配到输出 |
| **Step 3** | 加权融合 | $out[row,:] \mathrel{+}= sharedInputWeight \times sharedInput[j,:]$ | 融合共享专家输出 |

---

## 2. PyPTO API 映射表

### 2.1 核心计算 API

| 原子操作 | PyPTO API | API 功能 | 参数映射 | 产品支持 |
|---------|-----------|----------|----------|----------|
| **MXFP8 矩阵乘法** | `pypto.scaled_mm` | MXFP8/MXFP4 量化矩阵乘法 | `mat_a=x_i, mat_b=weight_i, out_dtype=DT_FP32, scale_a=scale_i, scale_b=...` | Ascend 950PR/950DT |
| **Tile Shape 设置** | `pypto.set_cube_tile_shapes` | 设置 cube 计算的 M/K/N 切分 | `m=[mL0,mL1], k=[kL0,kL1], n=[nL0,nL1]` | A2/A3 系列 |
| **Vector Tile 设置** | `pypto.set_vec_tile_shapes` | 设置 vector 计算的切分 | `*args=[v0,v1,v2,v3]` | A2/A3 系列 |

### 2.2 数据操作 API

| 原子操作 | PyPTO API | API 功能 | 使用方式 | 产品支持 |
|---------|-----------|----------|----------|----------|
| **Tensor切片** | `pypto.Tensor.__getitem__` | 通过索引/切片获取子 Tensor | `a[begin:end, :]` | 全产品 |
| **Tensor赋值** | `pypto.Tensor.__setitem__` | 通过索引/切片赋值 | `out[begin:end, :] = value` | 全产品 |
| **Scatter Add** | `pypto.index_add_` | 按 index 将 source 累加到 input | `pypto.index_add_(input, dim, index, source, alpha=1)` | 全产品 |
| **Scatter (替代)** | `pypto.scatter_` | 按 index 将 src 写入 input (支持 reduce='add') | `pypto.scatter_(input, dim, index, src, reduce='add')` | 全产品 |

### 2.3 控制流 API

| 原子操作 | PyPTO API | API 功能 | 使用方式 | 产品支持 |
|---------|-----------|----------|----------|----------|
| **循环结构** | `pypto.loop` | 定义循环操作 | `for i in pypto.loop(0, num_groups):` | A2/A3 系列 |
| **循环展开** | `pypto.loop_unroll` | 循环展开优化 | (可选优化) | A2/A3 系列 |

### 2.4 算术运算 API

| 原子操作 | PyPTO API | API 功能 | 使用方式 | 产品支持 |
|---------|-----------|----------|----------|----------|
| **加法** | `pypto.add` | 逐元素相加 | `pypto.add(input, other, alpha=1)` | 全产品 |
| **乘法** | `pypto.mul` | 逐元素相乘 | `pypto.mul(input, other)` | 全产品 |

### 2.5 初始化 API

| 原子操作 | PyPTO API | API 功能 | 使用方式 | 产品支持 |
|---------|-----------|----------|----------|----------|
| **初始化输出** | `pypto.zeros` | 创建全零 Tensor | `pypto.zeros(batch, n, dtype=pypto.DT_FP32)` | 全产品 |

---

## 3. API 详细约束清单

### 3.1 scaled_mm API 约束

#### 3.1.1 数据类型约束

| 参数 | 支持的数据类型 | 约束说明 |
|------|---------------|----------|
| `mat_a` | DT_FP8E5M2, DT_FP8E4M3 | 左矩阵必须使用 MXFP8 格式，左右矩阵数据类型需保持一致 |
| `mat_b` | DT_FP8E5M2, DT_FP8E4M3 | 右矩阵必须使用 MXFP8 格式，与左矩阵类型一致 |
| `scale_a` | DT_FP8E8M0 | 缩放因子使用 FLOAT8_E8M0 格式（纯指数格式） |
| `scale_b` | DT_FP8E8M0 | 缩放因子使用 FLOAT8_E8M0 格式 |
| `out_dtype` | DT_FP32, DT_FP16, DT_BF16 | 输出类型，MXFP8 场景推荐使用 DT_FP32 |

#### 3.1.2 Shape 约束

| 参数 | Shape 要求 | 约束说明 |
|------|-----------|----------|
| `mat_a` | [M, K] | K 维度需满足 64 元素对齐（NZ 格式需额外满足内轴 32 字节对齐，外轴 16 元素对齐） |
| `mat_b` | [K, N] 或 [N, K]（转置） | K 维度需满足 64 元素对齐 |
| `scale_a` | [M, K/64, 2] | 非转置时；转置时为 [K/64, M, 2] |
| `scale_b` | [K/64, N, 2] | 非转置时；转置时为 [N, K/64, 2] |

#### 3.1.3 MXFP8/MXFP4 特殊约束

| 约束项 | MXFP8 约束 | MXFP4 约束 |
|--------|-----------|-----------|
| K 维度 | 需满足 K % 64 == 0 | 必须为偶数，且 K ≠ 2 |
| N 维度 | 无特殊约束 | 非转置时 N 必须为偶数 |
| 数据类型 | FLOAT8_E4M3FN / FLOAT8_E5M2 | FLOAT4_E2M1 |

#### 3.1.4 前置条件约束

| 前置操作 | 约束说明 |
|---------|----------|
| `set_cube_tile_shapes` | **必须**在调用 scaled_mm 设置 M、K、N 轴切分大小 |
| `set_matrix_size` | 如果输入为 reshape 后的 NZ 格式，需设置原始 Shape 的 m,k,n 值 |

### 3.2 set_cube_tile_shapes 约束

#### 3.2.1 对齐约束

| Tile Shape | 对齐要求 | 公式 |
|-----------|---------|------|
| kL0, kL1 | 32 字节对齐 | kL0 * sizeof(dtype) % 32 == 0 |
| nL0, nL1 | 32 字节对齐 | nL0 * sizeof(dtype) % 32 == 0 |
| mL0 | ND 格式转置时需 32 字节对齐 | mL0 * sizeof(dtype) % 32 == 0 (仅转置场景) |

#### 3.2.2 值域约束

| 约束项 | 公式 |
|--------|------|
| mL0 <= mL1 | 0 < mL0 <= mL1 |
| mL1 % mL0 == 0 | mL1 是 mL0 的整数倍 |
| kL0 <= kL1 | 0 < kL0 <= kL1 |
| kL1 % kL0 == 0 | kL1 是 kL0 的整数倍 |
| nL0 <= nL1 | 0 < nL0 <= nL1 |
| nL1 % nL0 == 0 | nL1 是 nL0 的整数倍 |

#### 3.2.3 Buffer 空间约束

| Buffer | 空间约束公式 (FP16/BF16/FP32 输入) |
|--------|-----------------------------------|
| L0A | CeilAlign(mL0,16) * CeilAlign(kL0,16) * sizeof(aDtype) <= L0A_size |
| L0B | CeilAlign(nL0,16) * CeilAlign(kL0,16) * sizeof(bDtype) <= L0B_size |
| L0C | CeilAlign(mL0,16) * CeilAlign(nL0,16) * sizeof(DT_FP32) <= L0C_size |
| L1 | CeilAlign(mL1,16) * CeilAlign(kL1,16) * sizeof(aDtype) + CeilAlign(nL1,16) * CeilAlign(kL1,16) * sizeof(bDtype) <= L1_size |

> CeilAlign(value, align) = ((value + align - 1) // align) * align

### 3.3 index_add_ API 约束

#### 3.3.1 数据类型约束

| 参数 | 支持的数据类型 |
|------|---------------|
| `input` | DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT32 |
| `index` | DT_INT32, DT_INT64 |
| `source` | 与 input 类型一致 |

#### 3.3.2 Shape 约束

| 参数 | Shape 约束 |
|------|-----------|
| `input` | 2-4 维，不支持空 Tensor，Shape Size ≤ INT32_MAX |
| `index` | 1 维，Shape 大小与 source 的 dim 轴相同 |
| `source` | dim 轴 Shape 与 index 相同，其他维度与 input 相同 |

#### 3.3.3 Tiling 约束

| 约束项 | 说明 |
|--------|------|
| ViewShape[dim] | dim 轴不可切，需满足 viewshape[dim] >= max(input.shape[dim], source.shape[dim]) |
| TileShape | input、source 的 dim 轴以及 index 均不可切 |
| UB 内存 | 所有输入和输出的 TileShape 大小总和不能超过 UB 内存 |

### 3.4 Tensor 索引约束

#### 3.4.1 支持的索引类型

| 索引类型 | 是否支持 | 示例 |
|---------|----------|------|
| int / SymbolicScalar | ✓ | `a[0]`, `a[i]` |
| slice (start:end) | ✓ | `a[begin:end, :]` |
| Ellipsis (...) | ✓ | `a[..., 1:3]` |
| bool 类型索引 | ✗ | `a[True, False]` |
| Tensor 类型索引 | ✗ | `a[b]` (b 为 Tensor) |
| slice with step | ✗ | `a[1:2:2]` |

#### 3.4.2 切片约束

- **不支持 step 设置**：slice 的 step 默认固定为 1
- **不支持 Tensor 类型索引**：无法使用 Tensor 作为索引进行 gather/scatter

### 3.5 loop API 约束

| 约束项 | 说明 |
|--------|------|
| 参数类型 | start/stop/step 必须为 SymInt（符号整数） |
| 返回值 | 返回生成器，生成表示迭代值的符号整数 |
| 无特殊约束 | |

---

## 4. Tiling 需求分析

### 4.1 scaled_mm Tiling 配置

基于参考实现 gmm_mxfp8.py 的 Tiling 配置：

```python
# 参考配置（来自 gmm_mxfp8.py）
tile_config = {
    'ori_shape': [16, 512, 7168],      # M, K, N
    'm_tile_shape': [9, 9],            # mL0, mL1
    'k_tile_shape': [256, 256],        # kL0, kL1
    'n_tile_shape': [256, 256],        # nL0, nL1
    'vector_tile_shape': [1, 8, 256, 32],  # 用于 scale 处理
}
```

#### 4.1.1 Tiling 参数选择原则

| 参数 | 选择原则 | 推荐值范围 |
|------|---------|-----------|
| mL0 | 根据 groupList[i] 动态调整 | [1, 128] |
| mL1 | 根据 M 维度总大小 | [mL0, M] |
| kL0 | K 维度切分，需 32 字节对齐 | [64, 256] |
| kL1 | K 维度切分，需 32 字节对齐 | [kL0, K] |
| nL0 | N 维度切分，需 32 字节对齐 | [64, 256] |
| nL1 | N 维度切分，需 32 字节对齐 | [nL0, N] |

#### 4.1.2 Tiling 验证公式

```python
# 验证对齐约束
kL0 % 16 == 0  # FP8 每元素 1 字节，32 字节 = 32 元素，但文档要求 16 元素对齐
nL0 % 16 == 0
kL1 % kL0 == 0
nL1 % nL0 == 0

# 验证 Buffer 空间约束
L0A_usage = ceil_align(mL0, 16) * ceil_align(kL0, 16) * 1  # FP8 = 1 byte
L0B_usage = ceil_align(nL0, 16) * ceil_align(kL0, 16) * 1
L0C_usage = ceil_align(mL0, 16) * ceil_align(nL0, 16) * 4  # FP32 = 4 bytes
```

### 4.2 Scatter Add Tiling 配置

#### 4.2.1 index_add_ Tiling

```python
# TileShape 设置示例（用于 scatter add）
# 输入 input shape: [batch, n]
# 输入 source shape: [m, n]  (intermediate 结果)
# 输入 index shape: [m]  (rowIndex)
# dim = 0

# Tiling 约束：
# - dim 轴 (batch/m) 不可切，需全载
# - n 轴可以切分
pypto.set_vec_tile_shapes(1, n_tile)  # n_tile 用于切分 n 轴
```

#### 4.2.2 vector_tile_shape 配置

```python
# 用于 scale 处理的 vector tile shape
# 参考 gmm_mxfp8.py: [1, 8, 256, 32]
# shape 含义取决于 scale 的维度结构
pypto.set_vec_tile_shapes(1, 8, 256, 32)
```

---

## 5. 可行性判定

### 5.1 总体可行性：**可行**

### 5.2 分步可行性分析

| 计算步骤 | 可行性 | 支持产品 | 可用 API | 备注 |
|---------|--------|----------|----------|------|
| **Step 1: 分组矩阵乘法** | ✓ 可行 | Ascend 950PR/950DT | `scaled_mm` | MXFP8/MXFP4 场景原生支持 |
| **Step 1.1-1.4: Tensor切片** | ✓ 可行 | 全产品 | `Tensor.__getitem__` | 支持切片索引 |
| **Step 2: Scatter Add** | ✓ 可行 | 全产品 | `index_add_` 或循环+切片 | 两种实现方式可选 |
| **Step 3: 共享专家融合** | ✓ 可行 | 全产品 | `add`, `mul` | 标量加权融合 |
| **循环结构** | ✓ 可行 | A2/A3 系列 | `pypto.loop` | 支持动态循环次数 |
| **动态 Shape** | ✓ 可行 | Ascend 910B/910C | `pypto.DYNAMIC` | 支持 M 维度动态 |

### 5.3 产品支持矩阵

| 产品 | MXFP8/MXFP4 支持 | scaled_mm 支持 | index_add_ 支持 | loop 支持 |
|------|-----------------|----------------|-----------------|-----------|
| **Ascend 950PR/950DT** | ✓ | ✓ | ✓ | ✓ |
| **Atlas A3 训练/推理** | ✓ (伪量化) | ✓ | ✓ | ✓ |
| **Atlas A2 训练/推理** | ✓ (伪量化) | ✓ | ✓ | ✓ |

---

## 6. 实现方案设计

### 6.1 推荐实现方案（基于参考实现 gmm_mxfp8.py）

```python
@pypto.frontend.jit
def grouped_matmul_finalize_routing_kernel(
    x1: pypto.Tensor,                    # (M, K) MXFP8
    x2: pypto.Tensor,                    # (E, K, N) MXFP8
    scale: pypto.Tensor,                 # (E, Ceil(K/64), N, 2) E8M0
    pertoken_scale: pypto.Tensor,        # (M, Ceil(K/64), 2) E8M0
    group_list: list,                    # (E) 专家分组列表
    row_index: pypto.Tensor,             # (M) 路由索引
    shared_input: pypto.Tensor,          # (bsdp, N) 可选
    shared_input_weight: float,          # 融合系数
    shared_input_offset: int,            # 偏移
    out: pypto.Tensor,                   # (batch, N) 输出
    tile_config                          # Tiling 配置
) -> None:
    """
    GroupedMatmulFinalizeRoutingV3 融合算子实现
    
    计算流程：
    1. 分组矩阵乘法（GMM） - MXFP8 scaled_mm
    2. 路由分配（Scatter Add） - 按 rowIndex 分配
    3. 共享专家融合 - 加权融合共享专家输出
    """
    
    # 初始化中间结果 tensor
    intermediate = pypto.zeros(x1.shape[0], x2.shape[-1], dtype=pypto.DT_FP32)
    
    # Step 1: 分组矩阵乘法
    num_groups = x2.shape[0]
    begin = 0
    end = 0
    
    for i in pypto.loop(0, num_groups):
        begin = end
        end = end + group_list[i]
        
        # 提取当前专家组的输入/权重/scale
        x = x1[begin:end, :]
        weight = x2[i]
        scaled_x = scale[i]
        scaled_weight = pertoken_scale[begin:end]  # 注意：pertoken_scale 需按专家组切片
        
        # 设置 vector tile shapes for scale processing
        pypto.set_vec_tile_shapes(
            tile_config.vector_tile_shape[0],
            tile_config.vector_tile_shape[1],
            tile_config.vector_tile_shape[2],
            tile_config.vector_tile_shape[3]
        )
        
        # 设置 cube tile shapes for scaled_mm
        pypto.set_cube_tile_shapes(
            tile_config.m_tile_shape,
            tile_config.k_tile_shape,
            tile_config.n_tile_shape
        )
        
        # 执行 MXFP8 scaled_matmul
        intermediate[begin:end, :] = pypto.scaled_mm(
            x, weight, pypto.DT_FP32, scaled_x, scaled_weight
        )
    
    # Step 2: 路由分配（Scatter Add）
    # 方案A：使用 index_add_ API（推荐）
    # 注意：index_add_ 需要 index 为 INT32/INT64 tensor
    # row_index 需要转换为正确的格式
    
    pypto.set_vec_tile_shapes(1, tile_config.n_tile_shape[0])
    out = pypto.index_add_(out, 0, row_index, intermediate, alpha=1.0)
    
    # 方案B：使用循环 + 切片实现 scatter add（备用）
    # for j in pypto.loop(0, x1.shape[0]):
    #     target_row = row_index[j]  # 需要获取索引值（SymbolicScalar）
    #     out[target_row, :] = pypto.add(out[target_row, :], intermediate[j, :])
    
    # Step 3: 共享专家融合（可选）
    if shared_input is not None:
        for j in pypto.loop(0, shared_input.shape[0]):
            target_row = shared_input_offset + j
            # 加权融合：out += sharedInputWeight * sharedInput
            weighted_shared = pypto.mul(shared_input[j, :], shared_input_weight)
            out[target_row, :] = pypto.add(out[target_row, :], weighted_shared)
```

### 6.2 Scatter Add 实现方案对比

| 方案 | API | 优点 | 缺点 | 推荐度 |
|------|-----|------|------|--------|
| **方案A** | `index_add_` | 原生支持 scatter add，性能更好 | 需满足 Tiling 约束 | ★★★★★ |
| **方案B** | `scatter_` (reduce='add') | 支持累加模式 | 需满足更复杂的约束 | ★★★☆☆ |
| **方案C** | 循环 + 切片 + add | 灵活性高，约束少 | 性能较差，循环开销 | ★★☆☆☆ |

> **推荐使用方案A（index_add_）**：该 API 专为 scatter add 场景设计，性能最优。

---

## 7. 潜在问题与替代方案

### 7.1 Tensor 索引限制

| 问题 | 说明 | 替代方案 |
|------|------|----------|
| **不支持 Tensor 类型索引** | 无法使用 `a[b]` (b 为 Tensor) 进行 gather | 使用 `pypto.gather` API 或 `Tensor.__getitem__` 的特殊语法 `a[dim:index_tensor]` |
| **不支持 bool 类型索引** | 无法使用布尔掩码筛选 | 使用 `pypto.where` API 实现条件选择 |

### 7.2 Scatter Add 的挑战

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| **rowIndex 类型转换** | rowIndex 为 INT64，需确保与 index_add_ 的 index 类型匹配 | 确保 rowIndex tensor 为 DT_INT64 或 DT_INT32 |
| **UB 内存限制** | index_add_ 所有输入输出需满足 UB 内存约束 | 合理设置 TileShape，确保总大小不超过 UB |
| **动态 rowIndex** | rowIndex 值在编译时未知 | 使用 `pypto.DYNAMIC` 标记，或确保 rowIndex 值在合理范围内 |

### 7.3 MXFP8 缩放因子处理

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| **pertokenScale 切片** | pertokenScale 的 shape 为 (M, Ceil(K/64), 2)，需按专家组切片 | 使用 `pertoken_scale[begin:end]` 切片，保持 shape 结构 |
| **Scale 广播机制** | scaled_mm 内部自动处理 scale 的广播 | 无需手动广播，scaled_mm API 内部处理 |

---

## 8. 性能优化建议

### 8.1 Tiling 优化策略

| 优化项 | 建议 | 说明 |
|--------|------|------|
| **mL0 选择** | 根据 groupList 平均值设置 | 避免切分过于细碎，提高 L0 利用率 |
| **kL0/nL0 选择** | 优先使用较大值 (256) | 减少 L0 切分次数，提高计算效率 |
| **多核切 K** | 可选 enable_split_k=True | 当 K 维度较大时启用，提高多核利用率 |
| **vector_tile_shape** | 根据 scale shape 设置 | 优化 scale 数据搬运效率 |

### 8.2 循环优化策略

| 优化项 | 建议 | 说明 |
|--------|------|------|
| **loop_unroll** | 小循环可使用 loop_unroll | 减少循环控制开销 |
| **submit_before_loop** | 开启 submit_before_loop=True | 在循环开始前提交计算，减少同步开销 |
| **专家数量限制** | E ≤ 1024 | 遵循硬件约束 |

### 8.3 内存优化策略

| 优化项 | 建议 | 说明 |
|--------|------|------|
| **intermediate tensor** | 在 GM 上分配 | 避免 UB 内存溢出 |
| **输出初始化** | 使用 `pypto.zeros` | 在 GM 上初始化，减少数据搬移 |
| **共享专家融合** | 原地操作 | 使用 `add` API 原地更新输出 |

---

## 9. 测试验证建议

### 9.1 精度验证配置

```python
# 推荐测试配置
test_configs = [
    # MXFP8 基础配置
    {'m': 16, 'k': 512, 'n': 7168, 'e': 2, 'group_list': [7, 9]},
    # MXFP4 配置
    {'m': 16, 'k': 1024, 'n': 4096, 'e': 8, 'group_list': [2, 2, 2, 2, 2, 2, 2, 2]},
    # 大专家数量
    {'m': 128, 'k': 512, 'n': 7168, 'e': 16, 'group_list': [...]},
    # 空 tensor 场景
    {'m': 0, 'k': 512, 'n': 7168, 'e': 2},  # M=0 空输入
]

# 精度标准
rtol = 1e-3
atol = 1e-3
```

### 9.2 性能验证配置

| 验证项 | 指标 | 目标值 |
|--------|------|--------|
| **计算时间** | scaled_mm 平均耗时 | 参考 gmm_mxfp8.py 基准 |
| **Scatter Add 时间** | index_add_ 耗时 | < 10% 总计算时间 |
| **总耗时** | 算子总执行时间 | 对比拆分算子方案 |
| **内存占用** | workspace 大小 | 合理范围内 |

---

## 10. 总结

### 10.1 可行性总结

| 维度 | 结论 | 说明 |
|------|------|------|
| **API 完整性** | ✓ 完整 | 所有原子操作均有对应 PyPTO API |
| **产品支持** | ✓ 支持 | Ascend 950PR/950DT 原生支持 MXFP8 |
| **性能可行性** | ✓ 可优化 | 参考 gmm_mxfp8.py 的成熟实现 |
| **精度可行性** | ✓ 可验证 | 使用 golden 参考对比验证 |

### 10.2 关键风险点

| 风险项 | 风险等级 | 缓解措施 |
|--------|----------|----------|
| **Scatter Add 约束** | 中 | 使用 index_add_ API，合理设置 TileShape |
| **MXFP8 缩放因子处理** | 低 | scaled_mm API 内部自动处理 |
| **动态 Shape 支持** | 中 | 使用 pypto.DYNAMIC 标记动态维度 |
| **循环性能** | 低 | 使用 loop_unroll 优化小循环 |

### 10.3 后续工作建议

1. **Stage 3 (Golden)**: 生成 `{op}_golden.py`，参考 gmm_mxfp8.py 的 golden 实现
2. **Stage 4 (Design)**: 生成 `DESIGN.md`，细化 Tiling 策略和 loop 结构
3. **Stage 5 (Implement)**: 基于 gmm_mxfp8.py 实现完整算子，扩展 scatter add 和共享专家融合

---

**文档版本**: v1.0
**生成日期**: 2026-04-14
**参考实现**: /mnt/workspace/gitCode/cann/pypto/models/experimental/matmul/gmm_mxfp8.py
**产品支持**: Ascend 950PR/Ascend 950DT (MXFP8/MXFP4), Atlas A2/A3 (伪量化)