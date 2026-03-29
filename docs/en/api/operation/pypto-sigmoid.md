# pypto.sigmoid

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Applies the sigmoid activation function to each element of the input Tensor. The formula is:

$$
sigmoid(input) = \frac{1}{1 + e^{-input}}
$$

## Function Prototype

```python
sigmoid(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported data type: DT_FP32. <br> Empty Tensor not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor type. Its shape and data type are the same as the input Tensor, and its elements are the results of mapping the input elements through the sigmoid function to the interval \(0, 1\).

## Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sigmoid(x)
```

Example result:

```python
input data x: [-3.0, 0.0, 2.0, 5.0]
output data y: [0.0474, 0.5000, 0.8808, 0.9933]
```

