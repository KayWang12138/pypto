# pypto.distributed.shmem_store

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

将本地 UB 数据存储到共享内存 GM (win区)。

## 函数原型

```python
shmem_store(
    src: Tensor,
    offsets: List[Union[int, SymbolicScalar]],
    dst: Tensor,
    dst_pe: Union[int, SymbolicScalar],
    *,
    pred: List[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 位于本地 UB 中的源张量。 <br> Tensor支持的数据类型为：DT_INT32, DT_FP16, DT_FP32, DT_BF16。 <br> 不支持空Tensor; Shape仅支持2维; Shape Size不大于2147483647（即INT32_MAX）。 |
| offsets   | 输入      | 相对于共享内存 GM (win区) 的偏移。 <br> 支持 int 或 SymbolicScalar 类型的列表。 <br> 仅支持3维。 |
| dst   | 输入      | 目标张量，位于共享内存GM (win区)。 <br> Tensor支持的数据类型为：DT_INT32, DT_FP16, DT_FP32, DT_BF16。 <br> 不支持空Tensor; Shape仅支持4维。 <br> 必须是通过create_shmem_tensor接口创建的用于传输数据tensor。 |
| dst_pe   | 输入      | 目标设备的pe。<br> 支持的数据类型为int或SymbolicScalar类型。 <br> 0 <= pe < n_pes。 |
| pred   | 输入      | 用于控制该操作执行依赖关系。 <br> Tensor支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape仅支持2维。 |


## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

1.  pred 不能包含 src, 即 src 不可出现在 pred 中。

## 调用示例

```python
    shmem_data, _ = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 64, 128])
    matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    store_out = pypto.experimental.shmem_store(
        matmul_out,
        [0, 0, 0],
        shmem_data,
        2,
        pred=[matmul_out],
    )
```
