# pypto.softmax

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Applies the softmax function to the input Tensor along a specified dimension, normalizing the elements of that dimension into a probability distribution with values in \[0, 1\] (the sum of all elements equals 1). The formula is:

$$
\text{softmax}(input)_i = \frac{e^{input_i - \text{max}(input)}}{\sum_{j=1}^{k} e^{input_j - \text{max}(input)}}
$$

## Function Prototype

```python
softmax(input: Tensor, dim: int) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported data type: DT_FP32. <br> Empty Tensor not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |
| dim       | input        | Specifies the dimension for normalization. <br> Supports negative indexing (e.g., -1 refers to the last dimension). <br> Must be within the range [-input.dim, input.dim-1]. |

## Return Value

Returns a Tensor type. Its shape and data type are the same as the input Tensor. The sum of elements along the dimension specified by dim equals 1.

## Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.softmax(x, -1)
```

Example result:

```python
input data x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
output data y: [[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]]
```

