# pypto.distributed.shmem_put

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

异步地将本地 GM 数据发送到共享内存 GM (win区)。

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
| src   | 输入      | 位于本地 GM 的源张量。 <br> 支持的数据类型为：DT_INT32, DT_FP16, DT_FP32, DT_BF16。 <br> 不支持空 Tensor; Shape 仅支持2维; Shape Size不大于2147483647（即INT32_MAX）。 <br> 支持的数据格式为 ND。 |
| offsets   | 输入      | 共享内存张量在共享内存 GM (win区) 中的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 仅支持3维。 <br> 需要保证 offsets 小于 shape。 |
| dst   | 输入      | 位于共享内存 GM (win区) 的目标张量。 <br> 必须是通过 create_shmem_tensor 接口创建。 |
| dst_pe   | 输入      | 共享内存张量所在设备的编号。<br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 设备编号范围：0 <= dst_pe < n_pes。 |
| put_op   | 输入      | 数据传输时应用的原子操作类型。 <br> 支持的数据类型为: AtomicType.SET, AtomicType.ADD。 <br> 默认为 AtomicType.SET 类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空 Tensor; Shape 仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

```python
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP32, [1, 64, 128])
    matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    put_out = pypto.distributed.shmem_put(
        matmul_out,
        [0, 0, 0],
        shmem_Tensor,
        1,
        put_op=pypto.AtomicType.ADD,
        pred=[matmul_out],
    )
```
