# pypto.Tensor.format

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Get the format of the Tensor.

## Function Prototype

```python
format(self) -> TileOpFormat
```

## Parameters

None

## Return Value

TileOpFormat: Returns the format of the Tensor.

## Constraints

This is a read-only property.

## Example

```python
t = pypto.tensor((4, 4), pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND)
print(t.format)
```

Example result:

```text
output: TileOpFormat.TILEOP_ND
```

