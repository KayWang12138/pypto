# pypto.distributed.shmem_view

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在通信组内同步多个设备

## 函数原型

```python
shmem_view(
    input: ShmemTensor,
    shape: list[int] = None,
    offsets: list[Union[int, SymbolicScalar]] = None,
    *,
    valid_shape: Optional[list[Union[int, SymbolicScalar]]] = None,
) -> ShmemTensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      |  要提取局部视图的输入张量，必须是共享内存张量 。 <br> 必须是通过create_shmem_tensor接口创建的用于传输数据tensor。 |
| shape   | 输入      | 获取共享内存视图的形状。 <br> 仅支持3维。 |
| offsets   | 输入      | 获取每个维度相对于输入张量的偏移量，用于构造共享内存视图。 <br> 仅支持3维。 <br> 需要保证offsets小于input的shape。 |
| valid_shape  | 输入      | 取出指定示意块中有效数据大小。 <br> 仅支持3维。 <br> 需要保证valid_shape小于input的shape。 |

## 返回值说明

返回输出 ShmemTensor：从输入张量提取的局部共享内存视图，其大小为 shape。

## 约束说明

无

## 调用示例

```python
    x = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 64, 128])
    shape = [4, 4]
    offsets = [0, 4]
    valid_shape = [2, 4]
    y = pypto.distributed.shmem_view(x, shape, offsets, valid_shape=valid_shape)
```
