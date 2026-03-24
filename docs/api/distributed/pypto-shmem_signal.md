# pypto.distributed.shmem_signal

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

通过操作某个 shared memory tensor，触发信号并将该信号发送到 target_pe。

## 函数原型

```python
shmem_signal(
    src: ShmemTensor,
    src_pe: Union[int, SymbolicScalar],
    signal: int,
    shape: list[int] = None,
    offset: list[Union[int, SymbolicScalar]] = None,
    *,
    target_pe: Union[int, SymbolicScalar],
    sig_op: AtomicType = AtomicType.SET,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src  | 输入      | 触发信号的 shared memory tensor。|
| src_pe   | 输入      | shared memory tensor 所属的 pe, 0 <= pe < n_pes。 <br> 支持的数据类型为 int 或 SymbolicScalar 类型。 |
| signal   | 输入      | 发送到 src 中的信号值。 <br> 支持的数据类型为：int类型。 |
| shape   | 输入      |  shared memory tensor 的形状。 <br> 参数类型为 list[int] 类型。 <br> Shape 仅支持3维。 |
| offset   | 输入      | 相对于 shared memory tensor 的偏移量。 <br> 支持 int 或 SymbolicScalar 类型的列表。 |
| target_pe   | 输入      | 接收信号的 pe，如果 target_pe 为 None，则由 dst_pe 接收该信号。 <br> 如果 notify_pe=-1，则广播信号给所有设备。 <br> 支持 int 或 SymbolicScalar 类型的列表。 |
| sig_op   | 输入      | 数据传输时应用的原子操作类型。 <br>支持的数据类型为: AtomicType.SET, AtomicType.ADD。 <br> 默认为AtomicType.SET类型。 |
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

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
    shmem_Tensor = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 64, 128])
    pypto.set_vec_tile_shapes(64， 64)
    siganl_out = pypto.distributed.shmem_signal(
        shmem_Tensor,
        1,
        2,
        [1, 64, 128],
        [0, 0, 0],
        target_pe=1,
        sig_op=pypto.AtomicType.ADD,
        pred=predToken,
    )
```
