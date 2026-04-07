# pypto.stateless_random_uniform_v2

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

生成指定shape的均匀分布随机数，其元素范围为$[0, 1)$。
$$
x_i \sim U(0, 1)
$$

## 函数原型

```python
stateless_random_uniform_v2(shape: Tensor, key: Tensor, counter: Tensor, alg: Tensor, dtype: DataType) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| shape   | 输入      | 输出Tensor的形状。 <br> 支持的数据类型为：DT_INT32, DT_INT64。                    |
| key     | 输入      | 随机数生成器的seed。 <br> 支持的数据类型为：DT_UINT64。                            |
| counter | 输入      | 随机数生成器的计数器。 <br> 支持的数据类型为：DT_UINT64。                           |
| alg     | 输入      | 随机数生成算法，当前仅支持值1（Philox算法）。 <br> 支持的数据类型为：DT_INT32。     |
| dtype   | 输入      | 输出Tensor的数据类型。 <br> 支持的数据类型为：DT_FP32, DT_FP16, DT_BF16。         |

## 返回值说明

返回一个指定shape、数据类型为dtype的Tensor，其元素服从均匀分布，元素范围为$[0, 1)$。

## 调用示例

```python
shape = torch.tensor([4, 4], dtype=torch.int32)
key = torch.tensor([1234], dtype=torch.uint64)
counter = torch.tensor([0, 1], dtype=torch.uint64)
alg = torch.tensor([1], dtype=torch.int32)

pto_shape_tensor = pypto.from_torch(shape)
pto_key_tensor = pypto.from_torch(key)
pto_counter_tensor = pypto.from_torch(counter)
pto_alg_tensor = pypto.from_torch(alg)
dtype = pypto.DT_FP32

y = pypto.stateless_random_uniform_v2(shape, key, counter, alg, dtype)
```

结果示例如下：

```python
输出数据y: [[0.1689806  0.9725481  0.90036285 0.16582811]
            [0.1454581  0.48029935 0.02495587 0.99239147]
            [0.02835405 0.10649502 0.45283175 0.87260246]
            [0.6877538  0.24809706 0.95886254 0.24039495]]
```
