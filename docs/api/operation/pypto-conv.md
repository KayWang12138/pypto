# pypto.conv

## 产品支持情况

| 产品             | 是否支持 |
|:-----------------|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

实现输入input、weight完成卷积运算，支持bias参数，计算公式为：out = input @ weight + bias (@表示为卷积处理)

-   input 、weight、bias为源操作数；input 为左矩阵，weight为右矩阵，bias为完成卷积操作之后的res累加的输入数据输入
-   out 为目的操作数，存放卷积处理结果的矩阵

## 函数原型

```python
conv(input, weight, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

# 待完善