# pypto.distributed.shmem_barrier_all

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在通信组内同步多个设备

## 函数原型

```python
shmem_barrier_all(
    src: ShmemTensor,
    pred: list[Tensor] = None,
) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| src   | 输入      | 共享内存 GM (win区) 中的 barrier 信号张量。 <br> 必须是通过 create_shmem_tensor 或 create_shmem_siganl 接口创建。|
| pred   | 输入      | 用于控制操作执行的依赖关系张量列表。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor; Shape 仅支持2维。 |

## 返回值说明

返回输出Tensor：用于表示操作完成的依赖关系。

## 约束说明

无

## 调用示例

```python
    matmul_out = pypto.matmul(A_tile, B_tile, pypto.DT_FP16)
    shmem_signal = pypto.distributed.create_shmem_signal("tp",8)
    barrier_out = pypto.distributed.shmem_barrier_all(shmem_signal, [matmul_out])
```
