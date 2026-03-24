# pypto.distributed.shmem_clear_signal

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

清除当前 pe 对应的 shared memory tensor 的信号值

## 函数原型

```python
shmem_clear_signal(
    src: ShmemTensor,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    pred: list[Tensor] = None
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      |  要清除的 shared memory tensor 。 |
| shape   | 输入      | 需要清除的视图大小。 <br> 参数类型为 list[int] 类型。 |
| offsets   | 输入      | 需要清除的视图的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 需要保证 offsets 小于 shape。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

-   示例1：创建了一个 shape = [1, 128, 256] 的 shared memory tensor，清除当前 pe 对应的 shared memory tensor 的部分视图的信号值。该部分视图的 shape 为 [1, 128, 128], offset 为 [0, 0, 0]。

```python
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_signal(
        shmem_tensor,
        [1, 128, 128],
        [0, 0, 0],
        pred=predToken,
    )
```

-   示例2：创建了一个 shape = [1, 128, 256] 的 shared memory tensor，清除当前 pe 对应的 shared memory tensor 的全部视图的信号值。

```python
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 128, 256])
    data_clear_dummy = pypto.distributed.shmem_clear_signal(
        shmem_tensor,
        pred=predToken,
    )
```