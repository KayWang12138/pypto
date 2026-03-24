# pypto.distributed.create_shmem_signal

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 推理系列产品 |    √     |
| Atlas A2 推理系列产品 |    √     |

## 功能说明

在共享内存GM (win区) 中创建一个信号张量

## 函数原型

```python
create_shmem_signal(group_name: str, n_pes: int) -> ShmemTensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| group_name   | 输入      | 集合通信操作所在通信域的名字，字符串长度: 1~128。<br> 支持的类型为：str类型。 |
| n_pes   | 输入      | 通信域中的进程总数，n_pes > 0。 <br> 同一个group_name下的创建shmem signal Tensor必须保证n_pes一致。 <br> 支持的类型为int类型。 |

## 返回值说明

返回一个ShmemTensor：用于协调进程执行顺序的信号张量，通常用于进程间同步。

## 约束说明

无

## 调用示例

```python
signal = pypto.distributed.create_shmem_tensor("tp",8)
```