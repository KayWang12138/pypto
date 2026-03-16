# pypto.distributed.create_shmem_tensor

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在共享内存GM (win区) 中创建对称张量

## 函数原型

```python
create_shmem_tensor(group_name: str, n_pes: int, dtype: DataType, shape: list[int]) -> tuple[Tensor, Tensor]
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| group_name   | 输入      | 集合通信操作所在通信域的名字，字符串长度: 1~128。<br> 支持的类型为：str类型。 |
| n_pes   | 输入      | 通信域中的进程总数，n_pes > 0。 <br> 同一个group_name下的创建shmem Tensor必须保证n_pes一致。 <br> 支持的类型为int类型。 |
| dtype   | 输入      |创建的shmem Tensor的数据类型。 <br> 支持的类型为pypto的数据类型，可选值：DT_INT32, DT_FP16, DT_FP32, DT_BF16。 |
| shape   | 输入      |创建的shmem Tensor的形状。 <br> 参数类型为list[int]类型。 <br> 仅支持3维。 <br> 运行时判断当前创建的shmem tensor是否超出win区大小，进行报错提示。 |

## 返回值说明

返回输出tuple[Tensor, Tensor]：
- 第一个tensor通信操作的共享内存张量, 用于传输数据, Tensor的数据类型和dtype一致, Shape为[n_pes] + shape
- 第一个tensor用于进程间同步的信号张量, Tensor的数据类型为DT_INT32, Shape为[n_pes, n_pes] + shape

## 约束说明

无

## 调用示例

```python
shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor("tp", 8, pypto.DT_FP16, [1, 64, 128])
```
