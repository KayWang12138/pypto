# pypto.Element.dtype

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Gets the data type.

## Function Prototype

```python
def dtype(self) -> pypto.DataType
```

## Parameters

NA

## Return Value

Returns the Element type.

## Constraints

Read-only data.

## Example

```python
t = pypto.element(pypto.DT_FP32, 3)
t.dtype
```

