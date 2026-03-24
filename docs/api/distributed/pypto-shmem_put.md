# pypto.distributed.shmem_put

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

以 offsets 指定的 shared memory tensor 索引位置为基准，将输入的 Tensor 赋值到 shared memory tensor 的对应区域。

## 函数原型

```python
shmem_put(
    src: Tensor,
    offsets: list[Union[int, SymbolicScalar]],
    dst: ShmemTensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    put_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 源操作数。 <br> 支持的数据类型为：DT_INT32、DT_FP16、DT_FP32、DT_BF16。 <br> 不支持空 Tensor; Shape 仅支持2维; Shape Size 不大于 2147483647（即INT32_MAX）。 <br> 支持的数据格式为 ND。 |
| offsets   | 输入      | dst 的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 需要保证 offsets 小于 dst 的 dim。 |
| dst   | 输入      | 目的操作数，一个 shared memory tensor，其形状为 [1] + src.shape。 |
| dst_pe   | 输入      | shared memory tensor 所属的 pe。<br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 0 <= dst_pe < n_pes。 |
| put_op   | 输入      | 数据传输时应用的原子操作类型。 <br> 支持的数据类型为: AtomicType.SET、AtomicType.ADD。 <br> 默认为 AtomicType.SET 类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 对数据类型无要求。 <br> 不支持空 Tensor; Shape 仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

### TileShape设置示例

说明：调用该接口前，应通过set_vec_tile_shapes设置TileShape。TileShape维度应和输入一致。

-   示例1：输入的 shape 为 [m, n]，TileShape设置为 [m1, n1]，则 m1、n1 分别用于切分 m、n 轴。

    ```python
    pypto.set_vec_tile_shapes(4, 8)
    ```

### 接口调用示例

-   示例1：先创建一个 shared memory tensor，其形状为 [1] + 输入数据的形状。将输入数据赋值到 pe = 1 的 shared memory tensor 的指定区域，并与该视图原本的数据进行累加操作。

    ```python
    a = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
    b = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")  
    matmul_out = pypto.matmul(a, b, pypto.DT_FP16)
    shmem_shape = [1] + matmul_out.shape
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP32, shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    put_out = pypto.distributed.shmem_put(
        matmul_out,
        [0, 0, 0],
        shmem_Tensor,
        1,
        put_op=pypto.AtomicType.ADD,
        pred=[matmul_out],
    )
    ```

-   示例2：先创建一个 shared memory tensor，其形状为 [1] + 输入数据的形状。将输入数据赋值到 pe = 3 的 shared memory tensor 的指定区域，并覆盖该视图原本的数据。

    ```python
    a = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
    b = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")  
    matmul_out = pypto.matmul(a, b, pypto.DT_FP16)
    shmem_shape = [1] + matmul_out.shape
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP32, shmem_shape)
    pypto.set_vec_tile_shapes(16, 64)
    put_out = pypto.distributed.shmem_put(
        matmul_out,
        [0, 0, 0],
        shmem_Tensor,
        3,
        put_op=pypto.AtomicType.SET,
        pred=[matmul_out],
    )
    ```
