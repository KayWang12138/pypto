# pypto.Tensor.shape

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Get the Tensor shape.

## Function Prototype

```python
shape(self) -> List[SymInt]
```

## Parameters

None

## Return Value

Returns the shape list of the Tensor.

## Constraints

None.

## Example

```python
t = pypto.tensor((16, 32), pypto.DT_FP32)
out = t.shape
```

Example result:

```python
output data out: [16, 32]
```

