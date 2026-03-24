# pypto.distributed.shmem_get

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

异步地从共享内存 GM (win区) 获取数据到本地 GM。

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
| src   | 输入      | 位于共享内存 GM (win区) 的张量。 <br> 必须是通过 create_shmem_tensor 接口创建。 |
| src_pe   | 输入      | 共享内存张量所属的设备编号。 <br> 支持的数据类型为 int 或 SymbolicScalar 类型。 <br> 设备编号范围：0 <= src_pe < n_pes。|
| shape   | 输入      | 本地 GM 目标张量的形状。 <br> 参数类型为 list[int] 类型。 <br> Shape 仅支持3维。 |
| offset   | 输入      | 目标张量在本地 GM 中的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> Shape 仅支持3维。 <br> 需要保证 offsets 小于 shape。 |
| valid_shape   | 输入      | 用于指定需要获取的有效数据大小。 <br> 需要保证 valid_shape 小于 shape。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape 仅支持2维。 |

## 返回值说明

返回输出Tensor：位于本地 GM 中的目标张量，Tensor支持的数据类型为：DT_INT32, DT_FP16, DT_FP32, DT_BF16， Shape为2维。

## 约束说明

无

## 调用示例

```python
    wait_until_out = pypto.distributed.shmem_wait_until(
        shmem_Tensor,
        1,
        pypto.OpType.EQ,
        4,
        [1, 128, 256],
        [0, 0, 0],
        clear_signal=True,
        pred=predToken,
    )
    shmem_get_out = pypto.distributed.shmem_get(
        shmem_Tensor,
        1,
        [1, 128, 256],
        [0, 0, 0],
        valid_shape=[1, 128, 128],
        pred=[wait_until_out],
    )
```
