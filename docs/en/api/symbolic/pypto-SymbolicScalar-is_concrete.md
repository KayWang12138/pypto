# pypto.SymbolicScalar.is\_concrete

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Determines whether the symbolic scalar has a concrete numeric value.

## Function Prototype

```python
is_concrete(self) -> bool
```

## Parameters

None

## Return Value

Returns True if a concrete value exists, otherwise returns False.

## Constraints

-   Constant values are always concrete
-   Some expressions may also have concrete values under certain conditions

## Example

```python
s1 = pypto.SymbolicScalar(10)
out1 = s1.is_concrete()
s2 = pypto.SymbolicScalar("x")
out2 = s2.is_concrete()
```

Example output:

```python
Output data out1: True
Output data out2: False
```

