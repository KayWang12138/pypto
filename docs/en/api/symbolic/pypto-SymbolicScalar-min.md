# pypto.SymbolicScalar.min

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the minimum of two symbolic scalars.

## Function Prototype

```python
min(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| other     | Input        | The symbolic scalar or integer to compare against. |

## Return Value

The minimum of the two values.

## Constraints

-   If both values are concrete, a concrete constant value is returned
-   If at least one value is not concrete, a symbolic expression is returned

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.min(s2)
out2 = s1.min(3)
s3 = pypto.SymbolicScalar(x)
out3 = s3.min(2)
```

Example output:

```python
Output data out1: 5
Output data out2: 3
Output data out3: RUNTIME_Min(x, 2)
```

