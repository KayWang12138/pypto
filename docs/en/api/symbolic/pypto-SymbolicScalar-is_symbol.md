# pypto.SymbolicScalar.is\_symbol

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Determines whether the symbolic scalar is a symbol.

## Function Prototype

```python
is_symbol(self) -> bool
```

## Parameters

None

## Return Value

Returns True if it is a symbol, otherwise returns False.

## Constraints

A symbolic variable is a variable whose concrete value cannot be determined at compile time

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
out1 = s1.is_symbol()
out2 = s2.is_symbol()
```

Example output:

```python
Output data out1: False
Output data out2: True
```

