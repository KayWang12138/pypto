# pypto.sin

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the sine value (trigonometric function sin\( \)) of each element in the input Tensor, element-wise.

## Function Prototype

```python
sin(input: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported data type: DT_FP32. <br> Empty Tensor not supported; Shape Size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor type. Its shape and data type are the same as the input Tensor, and its elements are the sine values of the corresponding elements in the input Tensor.

## Example

```python
x = pypto.tensor([4], pypto.DT_FP32)
y = pypto.sin(x)
```

Example result:

```python
input data x: [-0.5461, 0.1347, -2.7266, -0.2746]
output data y: [-0.5194, 0.1343, -0.4032, -0.2711]
```

