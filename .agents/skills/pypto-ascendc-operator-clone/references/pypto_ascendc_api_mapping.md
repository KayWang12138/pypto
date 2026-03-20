# PyPTO API 与 AscendC API 映射参考

本文档提供PyPTO API与AscendC API的映射关系，帮助开发者快速找到对应的API。

**本文件内容仅供参考！使用过程中如有问题，以实际为准！PyPTO下的文档也可以查阅**

## 一、矩阵运算

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `Matmul<aType, bType, cType>` | `pypto.matmul(a, b, out_dtype=...)` | 矩阵乘法 |
| `BatchMatmul` | `pypto.matmul(a, b, out_dtype=...)` | 批量矩阵乘法 |
| `MMAD` | `pypto.matmul()` | 矩阵乘累加 |

**PyPTO matmul 示例**:
```python
# AscendC: Matmul<...> mm; mm.IterateBatch(...)
# PyPTO:
result = pypto.matmul(a, b, out_dtype=pypto.DT_FP32)
```

## 二、向量运算

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `Add` | `pypto.add(a, b)` | 加法 |
| `Sub` | `pypto.sub(a, b)` | 减法 |
| `Mul` | `pypto.mul(a, b)` | 乘法 |
| `Div` | `pypto.div(a, b)` | 除法 |
| `Muls` | `pypto.mul(a, scalar)` | 标量乘法 |
| `Adds` | `pypto.add(a, scalar)` | 标量加法 |
| `Exp` | `pypto.exp(a)` | 指数函数 |
| `Log` | `pypto.log(a)` | 对数函数 |
| `Sqrt` | `pypto.sqrt(a)` | 平方根 |
| `Pow` | `pypto.pow(a, exponent)` | 幂运算 |
| `Abs` | `pypto.abs(a)` | 绝对值 |
| `Neg` | `pypto.neg(a)` | 取负 |
| `Reciprocal` | `pypto.reciprocal(a)` | 倒数 |
| `Maximum` | `pypto.maximum(a, b)` | 逐元素最大值 |
| `Minimum` | `pypto.minimum(a, b)` | 逐元素最小值 |

## 三、归约运算

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `ReduceSum` | `pypto.sum(a, dim=..., keepdim=...)` | 求和归约 |
| `ReduceMax` | `pypto.amax(a, dim=..., keepdim=...)` | 最大值归约 |
| `ReduceMin` | `pypto.amin(a, dim=..., keepdim=...)` | 最小值归约 |
| `ReduceProd` | `pypto.prod(a, dim=..., keepdim=...)` | 乘积归约 |

**PyPTO 归约示例**:
```python
# AscendC: ReduceSum(sumUb, srcUb, reduceSize)
# PyPTO:
sum_result = pypto.sum(src, dim=-1, keepdim=True)
```

## 四、Softmax

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `Softmax` | `pypto.softmax(a, dim=...)` | 标准Softmax |
| `SoftmaxFlashV2` | `pypto.softmax(a, dim=...)` | PyPTO内部会优化 |

**PyPTO Softmax 示例**:
```python
# AscendC: SoftmaxFlashV2<...>(src, sum, max, dst, ...)
# PyPTO:
weights = pypto.softmax(scores, dim=-1)
```

## 五、形状操作

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `Reshape` | `pypto.reshape(a, shape)` | 改变形状 |
| `Transpose` | `pypto.transpose(a, dim0, dim1)` | 转置 |
| `Permute` | `pypto.transpose(a, dim0, dim1)` | 维度交换 |
| `View` | `pypto.view(a, shape, offset, valid_shape)` | 视图 |
| `Concat` | `pypto.concat(tensors, dim)` | 拼接 |
| `Expand` | `pypto.expand_clone(a, shape)` | 扩展 |
| `Squeeze` | `pypto.squeeze(a, dim)` | 压缩维度 |
| `Unsqueeze` | `pypto.unsqueeze(a, dim)` | 扩展维度 |

**PyPTO 转置示例**:
```python
# AscendC: DataCopy(transpose)
# PyPTO:
result = pypto.transpose(a, 2, 3)  # 交换第2和第3维度
```

## 六、数据拷贝与类型转换

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `DataCopy` | 自动处理 | 数据拷贝 |
| `Cast` | `pypto.cast(a, dtype)` | 类型转换 |
| `Duplicate` | `pypto.mul(a, 1.0)` 或直接使用 | 复制 |

**PyPTO 类型转换示例**:
```python
# AscendC: Cast(dst, src, RoundMode::CAST_ROUND, size)
# PyPTO:
result = pypto.cast(a, pypto.DT_FP32)  # 转为FP32
result = pypto.cast(a, pypto.DT_BF16)  # 转为BF16
```

## 七、条件选择

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `Select` | `pypto.where(condition, a, b)` | 条件选择 |
| `SelectWithBytesMask` | `pypto.where(mask, a, b)` | 掩码选择 |

**PyPTO 条件选择示例**:
```python
# AscendC: SelectWithBytesMask(dst, src, scalar, mask, tmp, shapeInfo)
# PyPTO:
result = pypto.where(mask, value_a, value_b)
```

## 八、数据类型

| AscendC类型 | PyPTO类型 | 说明 |
|-------------|-----------|------|
| `half` / `fp16_t` | `pypto.DT_FP16` | 16位浮点 |
| `bfloat16_t` | `pypto.DT_BF16` | BFloat16 |
| `float` | `pypto.DT_FP32` | 32位浮点 |
| `int8_t` | `pypto.DT_INT8` | 8位整数 |
| `uint8_t` | `pypto.DT_UINT8` | 无符号8位整数 |
| `int32_t` | `pypto.DT_INT32` | 32位整数 |

## 九、Tiling配置

| AscendC配置 | PyPTO配置 | 说明 |
|-------------|-----------|------|
| `TCubeTiling` | `pypto.set_cube_tile_shapes(...)` | Cube Tiling |
| `TilingKey` | 自动处理 | Tiling Key |
| `GetTilingData()` | 函数参数传递 | Tiling数据 |

**PyPTO Tiling 示例**:
```python
# AscendC: 需要复杂的Tiling配置
# PyPTO: 简化的Tiling设置
pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])  # Cube tile
pypto.set_vec_tile_shapes(1, 8, 16, D)  # Vector tile
```

## 十、特殊操作

| AscendC API | PyPTO API | 说明 |
|-------------|-----------|------|
| `SoftmaxFlashV2TilingFuncImpl` | `pypto.softmax()` | Softmax Tiling计算 |
| `GetBlockIdx()` | 自动并行处理 | 获取Block索引 |
| `GetSubBlockIdx()` | 自动并行处理 | 获取子Block索引 |
| `SetTail()` | 自动处理 | 设置尾块大小 |

## 十一、常用模式映射

### Attention计算

**AscendC实现**:
```cpp
// Q @ K^T
bmm1.SetTensorA(queryGm[offset]);
bmm1.SetTensorB(keyGm[offset], true);  // transpose
bmm1.IterateBatch(mm1Res[taskId]);

// Scale
Muls(score, score, scale, size);

// Softmax
SoftmaxFlashV2(weights, sum, max, score, ...);

// @ Value
bmm2.SetTensorA(weights);
bmm2.SetTensorB(valueGm[offset]);
bmm2.IterateBatch(output);
```

**PyPTO实现**:
```python
# Q @ K^T
scores = pypto.matmul(query, pypto.transpose(key, -2, -1), out_dtype=pypto.DT_FP32)

# Scale
scores = pypto.mul(scores, scale)

# Softmax
weights = pypto.softmax(scores, dim=-1)

# @ Value
output = pypto.matmul(weights, value, out_dtype=pypto.DT_FP32)
```

### Flash Attention分块计算

**注意**: PyPTO的softmax目前不支持手动的分块FlashSoftmax，但会自动优化。

对于需要手动分块的场景，可以使用`pypto.loop`:
```python
for idx in pypto.loop(0, num_blocks, 1):
    # 处理每个分块
    block_scores = ...  # 获取分块数据
    block_weights = pypto.softmax(block_scores, dim=-1)
    # 累加结果
```

## 十二、注意事项

1. **数据类型转换**: PyPTO的matmul支持`out_dtype`参数，可以指定输出类型
2. **转置操作**: PyPTO的transpose需要指定具体的维度交换
3. **Tiling配置**: PyPTO简化了Tiling配置，但仍需要合理设置tile大小
4. **并行处理**: PyPTO自动处理多核并行，不需要手动管理Block索引