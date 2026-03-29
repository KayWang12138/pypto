# pypto.Element Constructor

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Creates an Element.

## Function Prototype

```python
def __init__(self, dtype, data) : ...
```

## Parameters


| Parameter | Input/Output | Description                  |
|-----------|--------------|------------------------------|
| dtype     | Input        | Data type, see <a href="../datatype/DataType.md">DataType</a> |
| value     | Input        | An integer or floating-point number |

## Return Value

Returns an Element.

## Constraints

None.

## Example

```python
t = pypto.Element(pypto.DT_FP32, 3)
```

