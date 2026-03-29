# pypto.Tensor.id

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Get the unique identifier of the Tensor.

## Function Prototype

```python
id(self) -> int
```

## Parameters

None

## Return Value

Returns the unique identifier of the Tensor.

## Constraints

This is a read-only property.

## Example

```python
t = pypto.tensor((4, 4), pypto.DT_FP32)
print(t.id)  # output the Tensor's ID
```

