# pypto.distributed.shmem_clear_data

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

通过清除共享内存张量，同步通信组内的多个设备

## 函数原型

```python
shmem_clear_data(
    src: ShmemTensor,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要清除的共享内存 GM (win区) 张量 。 <br> 必须是通过 create_shmem_tensor 接口创建的张量。|
| shape   | 输入      | 要清除的共享内存 GM (win区) 张量的形状。 <br> 参数类型为 list[int] 类型。 <br> 仅支持3维。 |
| offsets   | 输入      | 要清除的共享内存 GM (win区) 张量的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 仅支持3维。 <br> 需要保证 offsets 小于 shape。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

```python
    matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    shape = [1, 128, 256]
    offset = [0, 0, 0]
    data_clear_dummy = pypto.distributed.shmem_clear_data(
        shmem_tensor,
        shape,
        offset,
        pred=[matmul_out],
    )
```