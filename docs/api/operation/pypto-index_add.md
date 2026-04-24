# pypto.index_add

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    �?    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    �?    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    �?    |

## 功能说明

index_add_的non-inplace版本，将source的每一块数据乘以缩放因子alpha（默认为1）加到input的相应数据块上，返回一个新的Tensor，原input不会被修改。详细功能说明可参�?[pypto.index_add_](pypto-index_add_.md)�?
## 函数原型

```python
index_add(input: Tensor, dim: int, index: Tensor, source: Tensor, *, alpha: Union[int, float] = 1) -> Tensor
```

## 参数说明

| 参数�? | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数�?<br> 支持的类型为：Tensor�?<br> Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT8，DT_INT16，DT_INT32�?<br> 不支持空Tensor；Shape仅支�?-5维；Shape Size不大�?147483647（即INT32_MAX）�?|
| dim     | 输入      | int 类型，加法作用到 input 的维度； <br> 支持任意不超�?input 维数的值，详见约束说明�?|
| index   | 输入      | 源操作数，值代�?input 所�?dim 轴的索引�?<br> 支持的类型为：Tensor�?<br> Tensor支持的数据类型为：DT_INT32，DT_INT64�?<br> 不支持空 Tensor，Shape只支�?维，索引�?source �?dim 轴索引一一对应，Shape大小�?source 所�?dim 轴的Shape大小相同�?|
| source  | 输入      | 需要加�?input 的源操作数； <br> 支持的类型为：Tensor�?<br> Tensor的数据类�?�?input 相同�?<br> Shape支持2-5维，所�?dim 轴的Shape大小�?index 相同，其他维度的Shape大小�?input 相同�?|
| alpha   | 输入      | 标量，关键字参数�?<br> 表示累加时的缩放因子，默认为 1�?|

## 返回值说�?
返回更新后的Tensor，为non-inplace操作，原input不会被修改�?
## 约束说明

详细约束说明可参�?[pypto.index_add_](pypto-index_add_.md)�?
## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape�?
如输入input为[m, n, p]，dim�?，输入source为[m, t, p]，输入index为[t]，输出为[m, n, p]，TileShape设置为[m1, t1, p1]，则m1, t1, p1分别用于切分source�?m, t, p轴�?
```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### 接口调用示例

```python
x = pypto.tensor([2, 3], pypto.DT_INT32)        # shape (2, 3)
source = pypto.tensor([3, 3], pypto.DT_INT32)   # shape (3, 3)
index = pypto.tensor([3], pypto.DT_INT32)       # shape (3,)
dim = 0
y = pypto.index_add(x, dim, index, source, alpha=1)
```

结果示例如下�?
```python
输入数据 x:   [[0 0 0],
               [0 0 0]]
      source: [[1 1 1],
               [1 1 1],
               [1 1 1]]
      index:   [0 1 0]
输出数据 y:   [[2 2 2],
               [1 1 1]]               # shape (2, 3)
```