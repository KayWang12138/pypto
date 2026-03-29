# pypto.Tensor.dim

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Get the number of dimensions of the Tensor.

## Function Prototype

```python
dim(self) -> int
```

## Parameters

None

## Return Value

The number of dimensions of the Tensor.

## Constraints

This is a read-only property.

## Example

```python
t = pypto.tensor((2, 3, 4), pypto.DT_FP32)
out = t.dim
```

Example result:

```python
output data out: 3
```

