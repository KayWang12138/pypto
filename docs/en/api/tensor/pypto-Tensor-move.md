# pypto.Tensor.move

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Move data from one Tensor to the current Tensor.

## Function Prototype

```python
move(self, other: 'Tensor') -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| other     | input        | The source Tensor whose data is to be moved. |

## Return Value

None

## Constraints

None.

## Example

```python
t1 = pypto.tensor((2, 3), pypto.DT_FP32)
t2 = pypto.tensor((2, 3), pypto.DT_FP32)
# Move t2's data to t1
t1.move(t2)
```

