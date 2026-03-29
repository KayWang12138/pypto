# pypto.SymbolicScalar.max

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Computes the maximum of two symbolic scalars.

## Function Prototype

```python
max(self, other: 'SymbolicScalar | int') -> 'SymbolicScalar'
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| other     | Input        | The symbolic scalar or integer to compare against. |

## Return Value

The maximum of the two values.

## Constraints

-   If both values are concrete, a concrete constant value is returned
-   If at least one value is not concrete, a symbolic expression is returned

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar(5)
out1 = s1.max(s2)
out2 = s1.max(13)
s3 = pypto.SymbolicScalar("x")
out3 = s3.max(2)
```

Example output:

```python
Output data out1: 10
Output data out2: 13
Output data out3: RUNTIME_Max(x, 2)
```

