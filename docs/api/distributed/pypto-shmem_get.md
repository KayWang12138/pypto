# pypto.distributed.shmem_get

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

从输入的 shared memory tensor 中取出部分视图到本地。

## 函数原型

```python
shmem_get(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 源操作数，一个shared memory tensor。 |
| src_pe   | 输入      | shared memory tensor 所属的 pe。 <br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 0 <= src_pe < n_pes。|
| shape   | 输入      | 需要获取的视图大小。 <br> 参数类型为 list[int] 类型。 |
| offset   | 输入      | 需要获取的视图偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 需要保证 offsets 小于 shape。 |
| valid_shape   | 输入      | 用于指定需要获取的有效数据大小。 <br> 需要保证 valid_shape 小于 shape。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape 仅支持2维。 |

## 返回值说明

返回一个与 src 数据类型相同的 Tensor，形状为 shape 参数指定的后两维，如果指定了valid_shape，则实际形状为valid_shape的后两维。
假设 shape=[1, 128, 256], valid_shape=None, 则返回的 Tensor 形状为 [128, 256]。
假设 shape=[1, 128, 256], valid_shape=[1, 128, 128], 则返回的 Tensor 形状为 [128, 128]。

## 约束说明

无

## 调用示例

### TileShape设置示例

说明：调用该接口前，应通过set_vec_tile_shapes设置TileShape。TileShape维度应和输出一致。

示例1：输入的 shape 为 [1, m, n]，输出的 shape 为 [m, n], TileShape设置为 [m1, n1], 则 m1, n1 分别用于切分 m, n 轴。

```python
pypto.set_vec_tile_shapes(4, 8)
```

### 接口调用示例

-   示例1：从  pe = 1 的 shared memory tensor 的全部视图中获取数据并输出该数据，对应的输出数据 shape 为 [128, 256]。

```python
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 128, 256])
    pypto.set_vec_tile_shapes(128, 256)
    shmem_get_out = pypto.distributed.shmem_get(
        shmem_Tensor,
        1,
        pred=predToken,
    )
```

-   示例2：从  pe = 1 的 shared memory tensor 的部分视图中获取数据并输出该数据。该部分视图的 shape 为 [1, 128, 128]，offset 为 [0, 0, 0]，对应的输出数据 shape 为 [128, 128]，实际获取的数据有效大小为 [128, 64]。

```python
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 128, 256])
    pypto.set_vec_tile_shapes(128, 256)
    shmem_get_out = pypto.distributed.shmem_get(
        shmem_Tensor,
        1,
        [1, 128, 128],
        [0, 0, 0],
        valid_shape=[1, 128, 64],
        pred=predToken,
    )
```
