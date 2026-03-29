# pypto.function

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Defines a PyPTO computation function. Operations required to build the computation graph can be added within this function.

## Function Prototype

```python
function(name: str, *args, **kwargs) -> Iterator
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| name      | Input        | The name of the function, used to identify the computation graph. |
| *args     | Input        | Used to obtain the passed-in Tensors. |

## Return Value

Returns a context manager for use in a with statement

## Constraints

None.

## Example

```python
with pypto.function("main", a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for _ in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        c[:] = a + b
```

