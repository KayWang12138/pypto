# pypto.distributed.shmem_signal

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

将信号写入到共享内存 GM (win区) 的对应位置

## 函数原型

```python
shmem_signal(
    dst: ShmemTensor,
    dst_pe: Union[int, SymbolicScalar],
    signal: int,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    sig_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| dst   | 输入      | 共享内存GM (win区) 中的信号张量 。 <br> 必须是通过create_shmem_tensor或create_shmem_siganl接口创建。|
| dst_pe   | 输入      | 共享内存张量所属的设备编号, 0 <= pe < n_pes。 <br> 支持的数据类型为 int 或 SymbolicScalar 类型。 |
| signal   | 输入      | 写入到共享内存GM (win区)中的信号值。 <br> 支持的数据类型为：int类型。 |
| shape   | 输入      | 共享内存GM (win区)信号的形状。 <br> 参数类型为list[int]类型。 <br> Shape仅支持3维。 |
| offset   | 输入      | 信号张量在共享内存GM (win区)中的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 仅支持3维。 |
| notify_pe   | 输入      | 接收信号的目标设备编号，如果 notify_pe 为 None，则由 dst_pe 接收该信号。 <br> 如果 notify_pe=-1，则广播信号给所有设备。 <br> 支持 int 或 SymbolicScalar 类型的列表。 |
| sig_op   | 输入      | 在数据传输过程中应用的原子操作类型。 <br>支持的数据类型为: AtomicType.SET, AtomicType.ADD。 <br> 默认为AtomicType.SET类型。 |
| pred   | 输入      | 用于控制该操作执行依赖关系。 <br>Tensor支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

```python
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 64, 128])
    matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    put_out = pypto.distributed.shmem_put(
        matmul_out,
        [0, 0, 0],
        shmem_Tensor,
        1,
        put_op=pypto.AtomicType.ADD,
        pred=[matmul_out],
    )
    siganl_out = pypto.distributed.shmem_signal(
        shmem_Tensor,
        1,
        2,
        [1, 128, 256],
        [0, 0, 0],
        notify_pe=1,
        sig_op=pypto.AtomicType.ADD,
        pred=[put_out],
    )
```
