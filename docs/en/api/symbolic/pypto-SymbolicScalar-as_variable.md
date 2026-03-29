# pypto.SymbolicScalar.as\_variable

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Marks the symbolic scalar as an intermediate variable.

## Function Prototype

```python
as_variable(self) -> None
```

## Parameters

None

## Return Value

None

## Constraints

-   This is an in-place operation that modifies the internal state of the symbolic scalar
-   Typically used to optimize expressions by marking complex expressions as intermediate variables

## Example

```python
s = pypto.SymbolicScalar("x")
s.as_variable()  # Mark as intermediate variable
```

