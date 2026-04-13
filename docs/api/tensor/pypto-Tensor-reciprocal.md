# Tensor.reciprocal

## 功能说明

计算张量的元素级倒数，即 `out = 1 / input`。

## 接口原型

```python
tensor.reciprocal(precision_type=pypto.RecipAlgorithm.DEFAULT) -> Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|:-----|:-----|:-----|
| precision_type | RecipAlgorithm, 可选 | 倒数操作的精度模式。默认值为 `RecipAlgorithm.DEFAULT`。<br>**HIGH_PRECISION**：使用更高精度的计算方式，减少精度损失。<br>**DEFAULT**：直接使用芯片指令进行计算。 |

## 返回值

| 类型 | 说明 |
|:-----|:-----|
| Tensor | 包含输入张量元素级倒数的新张量。 |

## 代码示例

### 示例 1：基本使用

```python
import pypto

x = pypto.tensor([4], pypto.DT_FP32)
y = x.reciprocal()

# Input x:  [-0.4595, -2.1219, -1.4314,  0.7298]
# Output y: [-2.1763, -0.4713, -0.6986,  1.3702]
```

### 示例 2：使用高精度模式

```python
import pypto

# 使用高精度模式进行 FP16 计算
x = pypto.tensor([4], pypto.DT_FP16)
y = x.reciprocal(pypto.RecipAlgorithm.HIGH_PRECISION)

# Input x:  [4]
# Output y: [0.25]
```

## 相关接口

- [Tensor.rsqrt](pypto-Tensor-rsqrt.md)：计算张量的元素级平方根的倒数。
- [Tensor.div](pypto-Tensor-div.md)：计算张量与另一个张量的元素级除法。