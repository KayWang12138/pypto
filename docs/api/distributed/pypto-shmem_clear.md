# pypto.distributed.shmem_clear

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在通信组内同步多个设备

## 函数原型

```python
shmem_clear(
    src: Tensor,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
    is_signal: bool = False,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要清除的共享内存GM (win区)区域，可以是数据张量或信号张量 。 <br> 支持的类型为pypto的数据类型，可选值：DT_INT32, DT_FP16, DT_FP32, DT_BF16。 <br> 不支持空Tensor。 <br> 必须是通过create_shmem_tensor或create_shmem_siganl接口创建的张量。|
| shape   | 输入      | 要清除的共享内存GM (win区)区域的形状，可以是数据或信号的形状。 <br> 参数类型为list[int]类型。 <br> 仅支持3维。 |
| offsets   | 输入      | 要清除的共享内存GM (win区)区域的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 仅支持3维。|
| pred   | 输入      | 用于控制该操作执行依赖关系。 <br>Tensor支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |
| is_signal   | 输入      | 要清除的区域是否存放信号。 <br>Tensor支持的数据类型为：bool类型。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

```python
    matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 64, 128])
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_data,
        [1, 128, 256],
        [0, 0, 0],
        pred=[matmul_out],
        is_signal=False,
    )
    data_clear_dummy = pypto.distributed.shmem_clear(
        shmem_signal,
        [1, 128, 256],
        [0, 0, 0],
        pred=[matmul_out],
        is_signal=True,
    )
```
