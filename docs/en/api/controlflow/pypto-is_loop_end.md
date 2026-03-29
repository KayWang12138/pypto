# pypto.is\_loop\_end

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Defines a loop operation, implementing the for loop functionality in Python.

## Function Prototype

```python
def is_loop_end(scalar: SymInt) -> SymbolicScalar
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| scalar    | Input        | The index of the current loop. |

## Return Value

Returns a symbolic scalar expression indicating whether this is the loop end (boolean value)

## Constraints

-   scalar must be a symbolic scalar returned by the loop iterator
-   If it is not a loop index, a ValueError exception will be raised
-   When the function is not decorated with @pypto.frontend.jit or @pypto.frontend.function, the conditional expression must be wrapped with pypto.cond

## Example

```python
# Without decorator, must wrap conditional expression with pypto.cond
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.cond(pypto.is_loop_end(idx)):
            ...

# With decorator, no need to wrap with pypto.cond
@pypto.frontend.jit
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.is_loop_end(idx):
            ...
```
