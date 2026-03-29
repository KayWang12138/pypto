# pypto.cond

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Defines an if conditional operation, implementing the if functionality in Python.

## Function Prototype

```python
cond(scalar: SymInt)
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| scalar    | Input        | A conditional expression that can be an integer or SymbolicScalar (symbolic scalar), used to evaluate whether the condition is true |

## Return Value

pypto\_impl.RecordIfBranch: Returns a conditional branch object for use with Python's if statement

## Constraints

-   Must be used in conjunction with Python's if, elif, and else statements
-   The conditional expression is recorded into the computation graph
-   Supports nested conditional statements
-   When the function is not decorated with @pypto.frontend.jit or @pypto.frontend.function, the conditional expression must be wrapped with pypto.cond

## Example

```python
# Without decorator, must wrap conditional expression with pypto.cond
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.cond(pypto.is_loop_end(s2_idx, bn_per_batch)):
            ...

# With decorator, no need to wrap with pypto.cond
@pypto.frontend.jit
def kernel():
    ...
    for s2_idx in pypto.loop(0, 10, 1, power_of_2(max_unroll_times), name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        if pypto.is_loop_end(s2_idx, bn_per_batch):
            ...
```

