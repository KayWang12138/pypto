# pypto.Tensor.name

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Get or set the name of the Tensor.

## Function Prototype

```python
name(self) -> str
name(self, value: str) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| value     | input        | The name to set for the Tensor. |

## Return Value

The name of the Tensor.

## Constraints

None.

## Example

```python
t = pypto.tensor((2, 3), pypto.DT_FP32)
n1 = t.name
t.name = "my_tensor"
n2 = t.name
```

Example result:

```python
output n1: ""
output n2: "my_tensor"
```

