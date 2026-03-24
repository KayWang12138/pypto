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
| src   | 输入      | 位于本地 GM 中的源张量。 <br> 支持的数据类型为：DT_INT32, DT_FP16, DT_FP32, DT_BF16。 <br> 不支持空Tensor; Shape仅支持2维; Shape Size不大于2147483647（即INT32_MAX）。 |
| offsets   | 输入      | 相对于共享内存 GM (win区) 的偏移。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 仅支持3维。 <br> 需要保证offsets小于shape。 |
| dst   | 输入      | 目标张量，位于共享内存 GM (win区)。 <br> 必须是通过create_shmem_tensor接口创建。 |
| dst_pe   | 输入      | 接受数据的目标设备编号。<br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 0 <= pe < n_pes。 |
| put_op   | 输入      | 在数据传输过程中应用的原子操作类型。 <br> 支持的数据类型为: AtomicType.SET, AtomicType.ADD。 <br> 默认为AtomicType.SET类型。 |
| pred   | 输入      | 用于控制该操作执行依赖关系。 <br> Tensor支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |


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
