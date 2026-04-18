# GroupedMatmulFinalizeRoutingV3 算子规格文档

## 1. 算子名称

**GroupedMatmulFinalizeRoutingV3**

## 2. 算子功能描述

GroupedMatmulFinalizeRoutingV3 是一个 MoE（Mixture of Experts）场景下的融合算子，实现了分组矩阵乘法与路由最终化的融合计算。该算子将原本需要多个独立算子（GroupedMatmul + MoeFinalizeRouting）的操作合并为一个高效算子，减少了中间结果的存储和传输开销。

算子包含三个主要计算步骤：

1. **分组矩阵乘法（Grouped MatMul, GMM）**：对每个专家组进行独立的矩阵乘法计算，支持 MXFP8/MXFP4 量化格式
2. **路由专家与专家输出分配（Routing & Expert Assignment）**：按照 rowIndex 将每个 token 的计算结果分配到对应的输出位置
3. **共享专家输出融合（Shared Expert Fusion）**：将共享专家的输出与 MoE 专家的结果进行加权融合

适用场景：
- MoE（Mixture of Experts）模型推理
- 大规模专家模型的路由计算
- 需要高效率量化推理的场景（MXFP8/MXFP4）

产品支持：
- Ascend 950PR/Ascend 950DT：支持 MXFP8/MXFP4 量化场景
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持伪量化场景
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持伪量化场景

## 3. 数学公式

### 3.1 分组矩阵乘法（GMM）

对于每个专家组 $i$，执行带量化缩放的矩阵乘法：

$$
y_i = (x_i \times weight_i) \times scale_i \times perTokenScale_i
$$

其中：
- $x_i$ 是该专家对应的输入 token 矩阵
- $weight_i$ 是该专家对应的权重矩阵
- $scale_i$ 是 MXFP8/MXFP4 量化缩放因子（FLOAT8_E8M0 格式）
- $perTokenScale_i$ 是每个 token 的缩放因子（FLOAT8_E8M0 格式）

### 3.2 路由专家与专家输出分配

对于每个 token $j$，将专家输出按照 rowIndex 进行分配：

$$
y[rowIndex[i], :] = y[rowIndex[i], :] + y_{i(j)}[j - start_{i(j)}]
$$

其中：
- $i(j)$ 是 token $j$ 被分配到的专家索引
- $y_{i(j)}[j - start_{i(j)}]$ 是该 token 在对应专家下的计算结果
- $start_{i(j)}$ 是专家 $i(j)$ 处理的起始 token 位置

### 3.3 共享专家输出融合

将共享专家的输出与 MoE 专家的结果进行加权融合：

$$
y[rowIndex[i], :] = y[rowIndex[i], :] + sharedInputWeight \times sharedInput[j, :]
$$

其中：
- $sharedInputWeight$ 是共享专家融合系数（标量）
- $sharedInput[j, :]$ 是共享专家对 token $j$ 的输出

### 3.4 最终输出

综合上述三步，最终输出为：

$$
y[rowIndex[i], :] = \sum_{i \in \mathcal{E}[j]} y_i [j - start_i] + sharedInputWeight \times sharedInput[j, :]
$$

其中 $\mathcal{E}[j]$ 表示分配给 token $j$ 的专家集合。

## 4. 输入输出规格

### 4.1 输入张量（MXFP8 量化场景 - Ascend 950PR/950DT）

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | Shape | 约束说明 |
|--------|-----------|----------|----------|-------|----------|
| **x1** | 输入 | FLOAT8_E4M3FN / FLOAT8_E5M2 | ND | (m, k) | m ∈ [1, 16×1024×8]，支持 M=0 的空 Tensor。不支持 INT8 类型。 |
| **x2** | 输入 | FLOAT8_E4M3FN / FLOAT8_E5M2 | ND | (e, k, n) 或 (e, n, k) | e ∈ [1, 1024]。转置情况下 shape 为 (e, n, k)。不支持 INT4 类型。 |
| **scaleOptional** | 输入（必选） | FLOAT8_E8M0 | ND | (e, n, Ceil(k/64), 2) 或 (e, Ceil(k/64), n, 2) | 转置属性必须与 x2 保持一致。MXFP8/MXFP4 场景必须提供。 |
| **biasOptional** | 输入（可选） | BFLOAT16 或 null | ND | (e, n) | 可选参数，可为空。 |
| **pertokenScaleOptional** | 输入（必选） | FLOAT8_E8M0 | ND | (m, Ceil(k/64), 2) | MXFP8/MXFP4 场景必须提供。 |
| **groupListOptional** | 输入（必选） | INT64 | ND | (e) | 分组列表，e 与 x2 的第一维一致。 |
| **sharedInputOptional** | 输入（可选） | BFLOAT16 或 null | ND | (bsdp, n) | bsdp 代表 batchSize / dataParallelSize。可选参数。 |
| **logitOptional** | 输入（必选） | FLOAT32 | ND | (m) | MoE 专家对各个 token 的 logit 大小。MXFP8/MXFP4 场景必须提供。 |
| **rowIndexOptional** | 输入（必选） | INT64 | ND | (m) | 路由索引，用于 scatter add 操作。MXFP8/MXFP4 场景必须提供。 |

### 4.2 输出张量

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | Shape | 约束说明 |
|--------|-----------|----------|----------|-------|----------|
| **out** | 输出 | FLOAT32 | ND | (batch, n) | batch 和 sharedInputOffset 必须大于等于 0。 |

### 4.3 其他参数

| 参数名 | 数据类型 | 取值范围 | 说明 |
|--------|----------|----------|------|
| **dtype** | INT64 | 0（FLOAT32） | 计算的输出类型。目前仅支持 0（FLOAT32）。 |
| **sharedInputWeight** | FLOAT32 | 实数标量 | 共享专家融合系数，sharedInput 先与该参数乘，再与 MoE 专家结果累加。 |
| **sharedInputOffset** | INT64 | ≥ 0 | 共享专家输出在总输出中的偏移。 |
| **transposeX1** | BOOL | false | 左矩阵是否转置，仅支持 false。 |
| **transposeX2** | BOOL | true / false | 右矩阵是否转置。 |
| **groupListType** | INT64 | 0 或 1 | 分组模式：0 表示 cumsum 模式（前缀和），1 表示 count 模式。 |
| **tuningConfigOptional** | INT64数组 | 可选 | 调优参数。第一个元素表示各个专家处理的 token 数预期值。兼容历史版本，不使用时可为 nullptr。 |

### 4.4 MXFP4 场景额外约束

| 约束项 | 说明 |
|--------|------|
| K 维度 | 必须为偶数，且不能为 2。 |
| N 维度 | x2 非转置情况下，n 必须为偶数。 |
| 数据类型 | x1 和 x2 使用 FLOAT4_E2M1。 |
| scale | 使用 FLOAT8_E8M0。 |

## 5. 精度要求

### 5.1 精度验证标准

- **相对误差（rtol）**：≤ 1e-3
- **绝对误差（atol）**：≤ 1e-3

### 5.2 精度验证方法

精度验证采用以下流程：

1. **Golden 参考实现**：使用纯 PyTorch 实现，将 MXFP8 数据转换为 FP32 进行计算
2. **MXFP8 缩放因子处理**：
   - FLOAT8_E8M0 格式的 scale 需要广播到对应的 K 维度（32 倍扩展）
   - 对输入和权重分别应用对应的缩放因子
3. **分组矩阵乘法计算**：对每个专家组独立计算，结果拼接
4. **路由分配**：按照 rowIndex 进行 scatter add 操作
5. **共享专家融合**：按 sharedInputWeight 进行加权融合

### 5.3 MXFP8 量化说明

MXFP8（Microscaling FP8）是一种块级量化格式：
- **块大小**：32 个元素组成一个量化块
- **缩放因子格式**：FLOAT8_E8M0（纯指数格式，无尾数）
- **数据格式**：FLOAT8_E4M3FN 或 FLOAT8_E5M2
- **缩放因子 shape**：
  - 输入 x1: (m, Ceil(k/64), 2) - 每 64 个元素（2 个块）对应一个缩放因子
  - 权重 x2: (e, Ceil(k/64), n, 2) 或 (e, n, Ceil(k/64), 2) - 取决于是否转置

## 6. 典型配置

### 6.1 MXFP8 基础配置

```python
# 输入维度
m = 16          # token 数量
k = 512         # 输入特征维度
n = 7168        # 输出特征维度（中间层维度）
e = 2           # 专家数量

# 分组列表（count 模式）
group_list = [7, 9]  # 每个专家处理的 token 数

# 数据类型
x1_dtype = torch.float8_e4m3fn
x2_dtype = torch.float8_e4m3fn
scale_dtype = torch.float8_e8m0fnu  # FLOAT8_E8M0
pertoken_scale_dtype = torch.float8_e8m0fnu

# 路由参数
batch = 8       # 输出 batch 维度
sharedInputWeight = 1.0
sharedInputOffset = 0

# Shape 定义
x1_shape = (m, k)                         # (16, 512)
x2_shape = (e, k, n)                      # (2, 512, 7168)
scale_shape = (e, Ceil(k/64), n, 2)       # (2, 8, 7168, 2)
pertoken_scale_shape = (m, Ceil(k/64), 2) # (16, 8, 2)
group_list_shape = (e)                    # (2)
logit_shape = (m)                         # (16)
row_index_shape = (m)                     # (16)
shared_input_shape = (bsdp, n)            # bsdp = batch / e
out_shape = (batch, n)                    # (8, 7168)
```

### 6.2 MXFP8 参考测试配置（来自 gmm_mxfp8.py）

```python
# Tile 配置（用于 PyPTO 实现）
tile_config = {
    'ori_shape': [16, 512, 7168],
    'group_list': [7, 9],
    'tile_size': 256,
    'm_tile_shape': [9, 9],
    'k_tile_shape': [256, 256],
    'n_tile_shape': [256, 256],
    'vector_tile_shape': [1, 8, 256, 32],
    'a_trans': False,
    'b_trans': False,
}
```

### 6.3 MXFP4 配置示例

```python
# MXFP4 场景
k = 1024       # 必须为偶数
n = 4096       # 必须为偶数（非转置）
e = 8

x1_dtype = torch.float4_e2m1
x2_dtype = torch.float4_e2m1
scale_dtype = torch.float8_e8m0fnu
```

## 7. 算法详细描述

### 7.1 执行流程

```
输入: x1, x2, scale, pertokenScale, groupList, logit, rowIndex, sharedInput（可选）
输出: out

步骤 1: 分组矩阵乘法（GMM）
    for i in range(e):
        begin = sum(groupList[0:i])  # 或根据 groupListType 计算
        end = begin + groupList[i]
        
        # 提取当前专家的输入和权重
        x_i = x1[begin:end, :]       # shape: (groupList[i], k)
        weight_i = x2[i]              # shape: (k, n)
        scale_i = scale[i]            # shape: (Ceil(k/64), n, 2)
        pertoken_scale_i = pertokenScale[begin:end]  # shape: (groupList[i], Ceil(k/64), 2)
        
        # MXFP8 缩放矩阵乘法
        # 1. 将 scale 广播到 K 维度（32 倍扩展）
        # 2. 对输入和权重应用缩放
        # 3. 执行矩阵乘法
        y_i = scaled_matmul(x_i, weight_i, scale_i, pertoken_scale_i)
        
        # 存储中间结果
        intermediate[begin:end, :] = y_i

步骤 2: 路由分配（Routing）
    初始化 out = zeros(batch, n)
    
    for j in range(m):
        expert_idx = 根据 rowIndex 和 groupList 确定
        target_row = rowIndex[j]
        source_row = j
        
        # Scatter add 操作
        out[target_row, :] += intermediate[source_row, :]

步骤 3: 共享专家融合（可选）
    if sharedInput is not None:
        for j in range(bsdp):
            target_row = sharedInputOffset + j
            out[target_row, :] += sharedInputWeight * sharedInput[j, :]

输出: out
```

### 7.2 MXFP8 缩放机制详解

MXFP8 量化采用块级缩放策略：

1. **量化块结构**：
   - 每 32 个元素组成一个量化块
   - 每个 K 维度方向有 2 个连续的量化块（共 64 个元素）
   - 每 64 个元素共享一个 FLOAT8_E8M0 缩放因子（含 2 个子因子）

2. **缩放因子 shape 对齐**：
   ```
   输入 x1: (m, k) -> scale shape: (m, Ceil(k/64), 2)
   权重 x2: (e, k, n) -> scale shape: (e, Ceil(k/64), n, 2)
   
   广播后:
   x1_scale_broadcast: (m, k)  - 每个元素对应一个缩放值
   x2_scale_broadcast: (e, k, n) - 每个元素对应一个缩放值
   ```

3. **计算过程**：
   ```python
   # Golden 计算示例（参考 gmm_mxfp8.py）
   
   # 1. 调整 K 维度对齐（Ceil(k/32) % 2 != 0 时需要裁剪）
   if Ceil(k/32) % 2 != 0:
       scaled_x = scaled_x[:, :-1]
       scaled_weight = scaled_weight[:-1, :]
   
   # 2. 广播缩放因子（32 倍扩展）
   scaled_x_broadcast = repeat_interleave(scaled_x, 32, dim=-1)
   scaled_weight_broadcast = repeat_interleave(scaled_weight, 32, dim=-2)
   
   # 3. 对齐到实际 k 维度（padding）
   x_padded = pad(x, [0, k_padded - k])
   weight_padded = pad(weight, [0, 0, 0, k_padded - k])
   
   # 4. 应用缩放并计算
   x_fp32 = x_padded.to(float32) * scaled_x_broadcast
   weight_fp32 = weight_padded.to(float32) * scaled_weight_broadcast
   result = matmul(x_fp32, weight_fp32)
   ```

### 7.3 分组列表处理

支持两种分组模式（由 groupListType 参数控制）：

**模式 0（cumsum 模式）**：
```python
groupList = [3, 7, 10]  # 前缀和形式
# 专家 0 处理 token[0:3]
# 专家 1 处理 token[3:7]
# 专家 2 处理 token[7:10]
```

**模式 1（count 模式）**：
```python
groupList = [3, 4, 3]   # 计数形式
# 专家 0 处理 token[0:3]
# 专家 1 处理 token[3:7]
# 专家 2 处理 token[7:10]
```

### 7.4 空张量处理

算子支持以下空张量场景：
- x1 支持 M=0 的空 Tensor（shape 为 (0, k))
- x2 支持 N=0 的空 Tensor（shape 为 (e, k, 0) 或 (e, 0, k))

## 8. 约束与限制

### 8.1 必选/可选参数（MXFP8 场景）

| 参数 | 必选/可选 | 说明 |
|------|-----------|------|
| x1 | 必选 | 输入左矩阵 |
| x2 | 必选 | 输入权重矩阵 |
| scaleOptional | 必选 | MXFP8 缩放因子 |
| pertokenScaleOptional | 必选 | Token 级缩放因子 |
| groupListOptional | 必选 | 分组列表 |
| logitOptional | 必选 | MoE logit |
| rowIndexOptional | 必选 | 路由索引 |
| biasOptional | 可选 | 偏置，可为空 |
| sharedInputOptional | 可选 | 共享专家输出，可为空 |

**注意**：MXFP8/MXFP4 场景中，offsetOptional、antiquantScaleOptional、antiquantOffsetOptional 必须设置为空（nullptr）。

### 8.2 Shape 约束

1. **专家数量**：e ≤ 1024（MXFP8/MXFP4 场景）
2. **K 维度对齐**：
   - MXFP4: k 必须为偶数，且 k ≠ 2
   - MXFP8: k 支持 2048（参考文档）
3. **N 维度对齐**：
   - MXFP4（非转置）：n 必须为偶数
4. **缩放因子对齐**：
   - x2 与 scale 的转置属性必须保持一致
   - scale 的 K 维度为 Ceil(k/64)
5. **输出约束**：
   - batch ≥ 0
   - sharedInputOffset ≥ 0

### 8.3 数据类型约束

MXFP8 场景支持的类型组合：
```
x1: FLOAT8_E4M3FN 或 FLOAT8_E5M2
x2: FLOAT8_E4M3FN 或 FLOAT8_E5M2
scaleOptional: FLOAT8_E8M0
biasOptional: BFLOAT16 或 null
pertokenScaleOptional: FLOAT8_E8M0
groupListOptional: INT64
sharedInputOptional: BFLOAT16 或 null
logitOptional: FLOAT32
rowIndexOptional: INT64
out: FLOAT32
```

MXFP4 场景支持的类型组合：
```
x1: FLOAT4_E2M1
x2: FLOAT4_E2M1
scaleOptional: FLOAT8_E8M0
biasOptional: BFLOAT16 或 null
pertokenScaleOptional: FLOAT8_E8M0
groupListOptional: INT64
sharedInputOptional: BFLOAT16 或 null
logitOptional: FLOAT32
rowIndexOptional: INT64
out: FLOAT32
```

### 8.4 格式约束

所有张量仅支持 ND（Normal Dense）格式，不支持 NZ（Normal Z) 或其他特殊格式。

### 8.5 非连续张量

当前版本不支持非连续张量输入。

## 9. 实现参考

### 9.1 PyPTO 实现关键点

参考 `../gmm_mxfp8.py` 中的实现：

1. **使用 pypto.scaled_mm API**：
   ```python
   # 设置 cube tile shapes
   pypto.set_cube_tile_shapes(m_tile_shape, k_tile_shape, n_tile_shape)
   
   # 执行缩放矩阵乘法
   result = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)
   ```

2. **设置 vector tile shapes**：
   ```python
   pypto.set_vec_tile_shapes(v0, v1, v2, v3)
   ```

3. **分组处理循环**：
   ```python
   for i in range(num_groups):
       begin = end
       end = end + group_list[i]
       x = a[begin:end, :]
       weight = b[i]
       scaled_x = scaled_a[begin:end, :, :]
       scaled_weight = scaled_b[i]
       out[begin:end, :] = pypto.scaled_mm(...)
   ```

### 9.2 Golden 参考实现关键点

参考 `../gmm_mxfp8.py` 中的 `compute_golden_result` 和 `gen_golden` 函数：

1. **K 维度对齐处理**：
   ```python
   k_dim = x.shape[-1]
   if math.ceil(k_dim / 32) % 2 != 0:
       scaled_x_golden = scaled_x_golden[:, :-1]
       scaled_weight_golden = scaled_weight_golden[:-1, :]
   ```

2. **缩放因子广播**：
   ```python
   scaled_x_broadcast = torch.repeat_interleave(scaled_x_golden, repeats=32, dim=-1)
   scaled_weight_broadcast = torch.repeat_interleave(scaled_weight_golden, repeats=32, dim=-2)
   ```

3. **Padding 处理**：
   ```python
   x_pad_len = scaled_x_broadcast.shape[-1] - x.shape[-1]
   x_golden = torch.nn.functional.pad(x, [0, x_pad_len], mode='constant', value=0)
   ```

## 10. 验证方法

### 10.1 精度验证流程

1. 生成 MXFP8 格式的输入数据（x1, x2, scale, pertokenScale）
2. 计算 Golden 参考结果（使用 PyTorch FP32 计算）
3. 计算 PyPTO 实现结果（使用 NPU）
4. 对比结果，验证 rtol ≤ 1e-3, atol ≤ 1e-3

### 10.2 测试用例设计

基础测试用例应覆盖：
- 不同专家数量（e = 1, 2, 8, 16）
- 不同 token 数量（m = 16, 128, 1024）
- 不同维度（k = 512, 2048; n = 7168）
- 不同分组配置（均匀分组、非均匀分组）
- 转置/非转置场景
- 空 tensor 场景（M=0, N=0）
- 包含/不包含共享专家

## 11. 性能优化建议

1. **Tile Shape 选择**：根据实际 shape 选择合适的 tile shape，平衡内存占用和计算效率
2. **分组策略**：合理的 groupList 配置可提高专家计算的并行度
3. **调优参数**：使用 tuningConfigOptional 指定专家处理的 token 数预期值，可优化 tiling 切分

---

**文档版本**: v1.0
**参考文档**: aclnnGroupedMatmulFinalizeRoutingV3.md
**参考实现**: ../gmm_mxfp8.py
**适用产品**: Ascend 950PR/Ascend 950DT（MXFP8/MXFP4 场景）