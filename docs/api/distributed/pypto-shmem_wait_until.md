# pypto.distributed.shmem_wait_until

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在共享内存 GM (win区) 的对应位置等待信号写入
等待一个信号到达当前 pe，该信号


通过操作某个 shared memory tensor，触发信号并将该信号发送到 target_pe。

## 函数原型

```python
shmem_wait_until(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    cmp_value: int = 0,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    cmp: OpType = OpType.EQ,
    clear_signal: bool = False,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 触发信号的 shared memory tensor。 <br> 必须是通过 create_shmem_tensor 或 create_shmem_siganl 接口创建。 |
| src_pe  | 输入      | shared memory tensor 所属的 pe。 <br> 支持的数据类型为：int 或 SymbolicScalar。 <br> 0 <= src_pe < n_pes。 |
| cmp   | 输入      | 用于条件判断的比较操作类型。 <br> 目前仅支持EQ（等于）类型。 |
| cmp_value   | 输入      | 要等待的目标数值。 <br> 支持的数据类型为 int 类型。 |
| shape   | 输入      | shared memory tensor 的形状。 <br> 参数类型为 list[int] 类型。 <br> Shape 仅支持3维。 |
| offset   | 输入      | 相对于 shared memory tensor 的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 |
| clear_siganl   | 输入      | 是否在等待完成后重置信号（true/false）。 <br>支持的数据类型为: bool类型。 <br> 默认为false。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空 Tensor; Shape 仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

在设置 TileShape 时，确保切块总数不超过1024。

## 调用示例

### TileShape设置示例

说明：调用该接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape 维度应和参数 shape 的后两维一致。

示例1：参数 shape 为 [1, m, n], TileShape设置为 [m1, n1], 则 m1, n1 分别用于切分 m, n 轴。

```python
pypto.set_vec_tile_shapes(4, 8)
```

### 接口调用示例

```python
    pypto.set_vec_tile_shapes(64， 64)
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
