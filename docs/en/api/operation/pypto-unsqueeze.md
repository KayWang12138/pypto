# pypto.unsqueeze

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Inserts a new dimension into the input tensor.

## Function Prototype

```python
unsqueeze(input: Tensor, dim: int) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand.<br> Supported data types: all data types supported by PyPto.<br> Empty tensors are not supported; shape size must not exceed 2147483647 (INT32_MAX). |
| dim       | Input        | Specifies the position (index) at which to insert the new dimension.<br> Negative indices are supported.<br> Must be within the range [-input.dim - 1, input.dim]. |

## Return Value

Returns an output tensor with a new dimension of size 1 inserted at the specified `dim` position, sharing data and attributes with the input tensor.

## Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.unsqueeze(x, 0)
```

Example result:

```python
Input x: [[1, 2, 3],
            [4, 5, 6]]
Output y: [[[1, 2, 3],
             [4, 5, 6]]]
```
