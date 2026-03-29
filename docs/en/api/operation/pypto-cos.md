# pypto.cos

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the cosine (trigonometric function cos\( \)) of each element in the input tensor, element-wise. Returns a tensor with the same shape as the input.

## Function Prototype

```python
cos(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported data types: DT_FP32. <br> Empty tensors are not supported; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns a tensor with the same shape and data type as the input, where each element is the cosine of the corresponding input element.

## Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.cos(x)
```

Example result:

```python
Input x: [0.0000, 0.7854, 1.5708, 2.3562]
Output y: [1.0000, 0.7071, 0.0000, -0.7071]
```
