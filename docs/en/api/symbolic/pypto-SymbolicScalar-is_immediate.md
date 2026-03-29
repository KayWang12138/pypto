# pypto.SymbolicScalar.is\_immediate

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Determines whether the symbolic scalar is an immediate value (a concrete numeric value).

## Function Prototype

```python
is_immediate(self) -> bool
```

## Parameters

None

## Return Value

Returns True if it is an immediate value, otherwise returns False.

## Constraints

An immediate value is a constant whose concrete value can be determined at compile time

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_immediate()
out2 = s2.is_immediate()
```

Example output:

```python
Output data out1: True
Output data out2: False
```

