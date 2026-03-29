# pypto.is\_loop\_begin

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Determines whether the current iteration is the beginning of a loop.

## Function Prototype

```python
is_loop_begin(scalar: SymInt) -> SymbolicScalar
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| scalar    | Input        | The index of the current loop. |

## Return Value

Returns a symbolic scalar expression indicating whether this is the loop beginning (boolean value)

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
        if pypto.cond(pypto.is_loop_begin(idx)):
            ...

# With decorator, no need to wrap with pypto.cond
@pypto.frontend.jit
def kernel():
    ...
    for idx in pypto.loop(0, 10, 1):
        if pypto.is_loop_begin(idx):
            ...
```

