# pypto.scatter

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    �?    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    �?    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    �?    |

## 功能说明

scatter_的non-inplace版本，将src的值写入input中。写入位置由index指定，返回一个新的Tensor，原input不会被修改。详细功能说明可参�?[pypto.scatter_](pypto-scatter_.md)�?
## 函数原型

```python
scatter(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

## 参数说明

| 参数�? | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 支持的类型为：Tensor�?<br> Tensor支持的数据类型为：DT_FP32、DT_FP16、DT_BF16�?<br> 不支持空Tensor；Shape仅支�?-4维；Shape Size不大�?147483647（即INT32_MAX）�?|
| dim     | 输入      | 指定用于索引的维度，支持input的维度范围内的任意维度�?<br> 合法的维度索�?，范围为�?input.dim �?input.dim - 1�?|
| index   | 输入      | input的一组索引�?<br> 支持的数据类型为：Tensor�?<br> Tensor支持的数据类型为：INT64、INT32�?<br> 支持的维度：和input保持一�?<br> 对于所有d != dim的维度，需满足要求：index.size(d) <= input.size(d) <br> 当src为Tensor时，所有维度都需满足：index.size(d) <= src.size(d) <br> 不支持空Tensor；Shape Size不大�?147483647（即INT32_MAX�?|
| src     | 输入      | src是更新的标量或Tensor�?<br> src为Element时，支持的数据类型为：DT_FP32、DT_FP16、DT_BF16，不支持输入INF/NAN <br> src为Tensor时，支持的数据类型为：DT_FP32、DT_FP16、DT_BF16，数据类型和 input 保持一致�?<br> |
| reduce  | 输入      | 要应用的归约操作，支�?'add' �?'multiply'，不传参时默认为直接替换 |

## 返回值说�?
返回更新后的Tensor，为non-inplace操作，原input不会被修改�?
## 约束说明

详细约束说明可参�?[pypto.scatter_](pypto-scatter_.md)�?
## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape�?
TileShape维度应和输出一致�?
```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([3, 5], pypto.DT_FP32)
y = pypto.tensor([2, 2], pypto.DT_INT64)
o = pypto.scatter(x, 0, y, 2.0)
```

结果示例如下�?
```
输入数据x:[[0 0 0 0 0],
           [0 0 0 0 0],
           [0 0 0 0 0]]
输入数据y:[[1 2],
           [0 1]]
输出数据o:[[2.0 0   0 0 0],
           [2.0 2.0 0 0 0],
           [0   2.0 0 0 0]]
```