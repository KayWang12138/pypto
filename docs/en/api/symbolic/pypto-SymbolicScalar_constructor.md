# pypto.SymbolicScalar Constructor

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Creates a new SymbolicScalar instance, supporting multiple construction methods.

## Function Prototype

```python
__init__(self,
         arg0: Union[int, str, 'SymbolicScalar'] = None,
         arg1: Union[int, None] = None
) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| self      | Output       | Reference to the instance object, passed automatically by Python. |
| arg0      | Input        | The value or name of the symbolic scalar. Can be:<br> - int: an integer value, creates a constant symbolic scalar<br> - str: a symbol name, creates a symbolic scalar<br> - SymbolicScalar: another symbolic scalar, used for copying |
| arg1      | Input        | The value of the symbolic scalar; only optionally used when arg0 is a string. |

## Return Value

None.

## Constraints

-   If arg0 is an integer, arg1 is ignored
-   If arg0 is a string and arg1 is an integer, a symbolic scalar with an initial value is created
-   If arg0 is a string and arg1 is None, a symbolic scalar without an initial value is created
-   If arg0 is a SymbolicScalar, its underlying implementation object is copied

## Example

```python
a = pypto.SymbolicScalar()
b = pypto.SymbolicScalar(10)
c = pypto.SymbolicScalar("x")
d = pypto.SymbolicScalar("x", 10)
```

