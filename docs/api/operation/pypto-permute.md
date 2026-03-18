# pypto.permute

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

返回一个Tensor，该Tensor是输入Tensor的转置版本。根据指定的维度顺序重新排列输入张量的维度，该算子不改变张量中元素的总数和内容，只改变维度的排列方式。

## 函数原型

```python
permute(input: Tensor, const std::vector<int> &dims) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP16，DT_BF16，DT_INT16，DT_UINT16，DT_FP32，DT_INT32，DT_UINT32。<br> 不支持空Tensor；Shape仅支持2-5维；Shape Size不大于2147483647（即INT32_MAX）。<br> 算子对不同 Shape 支持不同，详见约束说明。 |
| dims    | 输入      | 维度顺序列表。必须是一个包含所有维度索引的排列，长度与输入张量的维度数相同。每个维度索引在 0 到 ndim-1 范围内且不重复。 |

## 返回值说明

返回一个与输入数据类型一致的Tensor，其维度顺序按照 dims 指定的顺序重新排列。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过 set_vec_tile_shapes 设置TileShape。TileShape的维度应与输入input一致。

示例：输入input shape为 [2, 3, 4]，目标排列为 [2, 0, 1]，TileShape可设置为 [2, 3, 4] 或根据切分需求设置合适的值。

```python
pypto.set_vec_tile_shapes(2, 3, 4)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3, 4], pypto.DT_FP32)
dims = [2, 0, 1]
y = pypto.permute(x, dims)
```

结果示例如下：

```python
输入数据x: [[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]]
输出数据y: [[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [0.5809,  0.4942]]
```

