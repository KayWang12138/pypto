# pypto.Tensor.dtype

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Get the data type of the Tensor.

## Function Prototype

```python
dtype(self) -> DataType
```

## Parameters

None

## Return Value

Returns the data type of the Tensor.

## Constraints

None.

## Example

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
out = t.dtype
```

Example result:

```python
output data out: DataType.DT_FP32
```

