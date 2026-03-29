# pypto.bytes\_of

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Returns the size in bytes of a given data type.

## Function Prototype

```python
bytes_of(dtype: pypto.DataType) -> int
```

## Parameters


| Parameter | Input/Output | Description                              |
|-----------|--------------|------------------------------------------|
| dtype     | Input        | The data type whose byte size is to be queried. |

## Return Value

Returns the size in bytes of the given data type.

## Constraints

None.

## Example

```python
pypto.bytes_of(pypto.DT_FP32)
```

Example output:

```python
Output: 4
```

