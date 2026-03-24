# pypto.distributed.shmem_store

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

以 offsets 指定的 shared memory tensor 索引位置为基准，将输入的 Tensor 赋值到 shared memory tensor 的对应区域。

## 函数原型

```python
shmem_store(
    src: Tensor,
    offsets: List[Union[int, SymbolicScalar]],
    dst: ShmemTensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    put_op: AtomicType = AtomicType.SET,
    pred: List[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 源操作数。 <br> 支持的数据类型为：DT_INT32、DT_FP16、DT_FP32、DT_BF16。 <br> 不支持空Tensor; Shape 仅支持2维; Shape Size 不大于2147483647（即INT32_MAX）。 <br> 支持的数据格式为 ND。 |
| offsets   | 输入      | 相对于 dst 的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 需要保证 offsets 小于 dst 的 shape。 |
| dst   | 输入      | 目的操作数，一个 shared memory tensor，其形状为 [1] + src.shape。 |
| dst_pe   | 输入      | shared memory tensor 所属的 pe。<br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 0 <= pe < n_pes。 |
| put_op   | 输入      | 数据传输时应用的原子操作类型。 <br> 支持的数据类型为: AtomicType.SET，AtomicType.ADD。 <br> 默认为AtomicType.SET类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor; Shape 仅支持2维。 |


## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

1.  pred 不能包含 src，即 src 不可出现在 pred 中。
2.  src 的 dtype 必须和 dst 的 dtype 一致。

## 调用示例

### TileShape设置示例

说明：调用该接口前，应通过 set_vec_tile_shapes 设置 TileShape。TileShape 维度应和输入一致。

示例1：输入的 shape 为 [m, n]，TileShape设置为 [m1, n1]，则 m1、n1 分别用于切分 m、n 轴。

```python
pypto.set_vec_tile_shapes(4, 8)
```

-   示例1：先创建一个 shared memory tensor，其形状为 [1] + 输入数据的形状。将输入数据赋值到 pe = 2 的 shared memory tensor 的指定区域，并与该视图原本的数据进行累加操作。注意，shared memory tensor 的 dtype 和 输入数据的 dtype 必须一致。

```python
    a = pypto.tensor([16, 32], pypto.DT_BF32, "tensor_a")
    b = pypto.tensor([32, 64], pypto.DT_BF32, "tensor_b")  
    matmul_out = pypto.matmul(a, b, pypto.DT_FP32)
    shmem_shape = [1] + matmul_out.shape
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP32, shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    store_out = pypto.experimental.shmem_store(
        matmul_out,
        [0, 0, 0],
        shmem_Tensor,
        2,
        put_op=pypto.AtomicType.ADD,
        pred=predToken,
    )
```

-   示例2：先创建一个 shared memory tensor，其形状为 [1] + 输入数据的形状。将输入数据赋值到 pe = 3 的 shared memory tensor 的指定区域，并覆盖该视图原本的数据。

```python
    a = pypto.tensor([16, 32], pypto.DT_BF32, "tensor_a")
    b = pypto.tensor([32, 64], pypto.DT_BF32, "tensor_b")  
    matmul_out = pypto.matmul(a, b, pypto.DT_FP32)
    shmem_shape = [1] + matmul_out.shape
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP32, shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    store_out = pypto.experimental.shmem_store(
        matmul_out,
        [0, 0, 0],
        shmem_Tensor,
        3,
        put_op=pypto.AtomicType.ADD,
        pred=predToken,
    )
```
