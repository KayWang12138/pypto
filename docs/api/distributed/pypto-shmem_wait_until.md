# pypto.distributed.shmem_wait_until

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在共享内存 GM (win区) 的对应位置等待信号写入

## 函数原型

```python
shmem_wait_until(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    cmp: OpType = OpType.EQ,
    cmp_value: int = 0,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    clear_signal: bool = False,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 共享内存 GM (win区) 中的信号张量 。 <br> 必须是通过 create_shmem_tensor 或 create_shmem_siganl 接口创建。 |
| src_pe  | 输入      | 共享内存张量所属的设备编号。 <br> 支持的数据类型为：int 或 SymbolicScalar。 <br> 设备编号范围：0 <= src_pe < n_pes。 |
| cmp   | 输入      | 用于条件判断的比较操作类型。 <br> 目前仅支持EQ（等于）类型。 |
| cmp_value   | 输入      | 要等待的目标数值。 <br> 支持的数据类型为 int 类型。 |
| shape   | 输入      | 共享内存 GM (win区) 信号张量的形状。 <br> 参数类型为 list[int] 类型。 <br> Shape 仅支持3维。 |
| offset   | 输入      | 信号张量在共享内存 GM (win区) 中的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> Shape 仅支持3维。 |
| clear_siganl   | 输入      | 是否在等待完成后重置信号（true/false）。 <br>支持的数据类型为: bool类型。 <br> 默认为false。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空 Tensor; Shape 仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

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
        clear_signal=False,
        pred=pred_token,
    )
```
