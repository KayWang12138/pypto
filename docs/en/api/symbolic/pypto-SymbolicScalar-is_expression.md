# pypto.SymbolicScalar.is\_expression

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Determines whether the symbolic scalar is an expression.

## Function Prototype

```python
is_expression(self) -> bool
```

## Parameters

None

## Return Value

Returns True if it is an expression, otherwise returns False.

## Constraints

An expression is formed by combining multiple symbols or constants through arithmetic operations

## Example

```python
s1 = pypto.SymbolicScalar(10)
s2 = pypto.SymbolicScalar("x")
s3 = s2 + 5
out1 = s1.is_expression()
out2 = s2.is_expression()
out3 = s3.is_expression()
```

Example output:

```python
Output data out1: False
Output data out2: False
Output data out3: True
```

