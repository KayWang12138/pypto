# pypto.pass\_verify\_print

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

When the accuracy debugging Verify feature is enabled, use this interface to save the results of a specified Tensor computation to a data file.


## Function Prototype

```python
pass_verify_print(*values, cond: Union[int, SymbolicScalar] = 1) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| *values   | Input        | Meaning: specifies the data or information to print. <br> Description: pypto.Tensor: prints tensor data; int/pypto.SymbolicScalar: prints the corresponding value; other Python objects: prints the corresponding string representation <br> Type: List[pypto.Tensor,int,pypto.SymbolicScalar,Object] <br> Value range: NA <br> Default value: NA |
| cond      | Input        | Meaning: specifies the condition that must be met to print data <br> Description: if the expression evaluates to 1: prints the specified data; if the expression evaluates to 0: does not print data <br> Type: Optional[int,pypto.SymbolicScalar] <br> Value range: 0, 1 <br> Default value: 1 |

## Return Value

None.

## Constraints

This function takes effect only after pypto.set_verify_options(enable_pass_verify=True) is set.

## Example

```python
verify_options = {
        "enable_pass_verify": True,
      }

@pypto.frontend.jit(verify_options=verify_options)
def user_kernel(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor):
    ...
    for idx in pypto.loop(10):
        t0 = pypto.tensor(...)
        t1 = pypto.tensor(...)
        t2 = pypto.SOME_OP1(t0, t1)
        pypto.pass_verify_print(t2)
        t3 = pypto.SOME_OP2(t0, t2)
        pypto.pass_verify_print(t3, cond=(idx == 5))
         ...
```

