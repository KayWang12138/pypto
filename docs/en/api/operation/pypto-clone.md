# pypto.clone

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Copies the input source data and returns it.

## Function Prototype

```python
clone(input: Tensor) -> Tensor:
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported data types: data types supported by PyPTO. <br> Empty Tensor not supported; shape size must not exceed 2147483647 (i.e., INT32_MAX). |

## Return Value

Returns a Tensor with the same shape and data type as the input.

## Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.clone(x)
```

Example result:

```python
input x: [[1, 2],
          [3, 4]]
output y: [[1, 2],
           [3, 4]]
```

