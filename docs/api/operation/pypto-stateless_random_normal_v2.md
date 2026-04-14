# pypto.stateless_random_normal_v2

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

生成指定shape的正态分布（高斯分布）随机数，其元素服从均值为0，方差为1。
$$
x_i \sim N(0, 1)
$$

## 函数原型

```python
stateless_random_normal_v2(shape: List[int], key: List[int], counter: List[int], alg: List[int], dtype: DataType) -> Tensor
```

## 参数说明


| 参数名  | 输入/输出 | 说明                                                         |
|---------|-----------|------------------------------------------------------------|
| shape   | 输入      | 输出Tensor的形状。 <br> 长度支持1-5。                                 |
| key     | 输入      | 随机数生成器的seed。 <br> 长度仅支持为1。                                 |
| counter | 输入      | 随机数生成器的计数器。 <br> 长度仅支持为2。                                  |
| alg     | 输入      | 随机数生成算法，当前仅支持值1（Philox算法），3（auto_select，选择Philox算法）。 <br> 长度仅支持为1。 |
| dtype   | 输入      | 输出Tensor的数据类型。 <br> 支持的数据类型为：DT_FP32, DT_FP16, DT_BF16。    |

## 返回值说明

返回一个指定shape、数据类型为dtype的Tensor，其元素服从均值为0，方差为1的正态分布。

## 调用示例

```python
shape = [4, 4]
key = [1234]
counter = [0, 1]
alg = [1]
dtype = pypto.DT_FP32

y = pypto.stateless_random_normal_v2(shape, key, counter, alg, dtype)
```

结果示例如下：

```python
输出数据y: [[-0.32364845  1.8577391   0.39556974  0.2311697 ]
            [ 0.24243996 -1.9485782  -0.12983137  2.7137496 ]
            [ 1.6558666   2.0938187  -0.90338254  0.8765667 ]
            [ 0.86518306  0.01034508  0.2893259   0.01748212]]
```
