# pypto.SymbolicScalar.concrete

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Gets the concrete numeric value of the symbolic scalar.

## Function Prototype

```python
concrete(self) -> int
```

## Parameters

None

## Return Value

The concrete numeric value of the symbolic scalar.

## Constraints

-   This method can only be called when is\_concrete\(\) returns True
-   If the symbolic scalar is not concrete, a ValueError exception will be raised

## Example

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.concrete()

# Raises an exception if the symbolic scalar has no concrete value
s2 = pypto.SymbolicScalar("x")
out2 = s2.concrete()
```

Example output:

```python
Output data out1: 10
Output data out2: raises exception ValueError: Not concrete value
```

