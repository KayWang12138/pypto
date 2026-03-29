# pypto.pass\_verify\_save

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

When the accuracy debugging Verify feature is enabled, use this interface to save the intended computation results of a specified Tensor to a data file.

## Function Prototype

```python
pass_verify_save(
    tensor: Tensor,
    fname: Union[str, SymbolicScalar, int],
    cond: Union[int, SymbolicScalar] = 1,
    **kwargs: Union[int, SymbolicScalar, pypto_impl.SymbolicScalar],
) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| tensor    | Input        | Meaning: a pypto.Tensor within a pypto kernel function. <br> Description: the variable name of the Tensor. <br> Type: pypto.Tensor <br> Value range: NA |
| fname     | Input        | Meaning: filename template that defines the filename prefix for saving the tensor. The tensor's memory dump is saved to {fname}.data and the tensor's metadata (shape, dtype) is saved to {fname}.csv. The save path is: ${work_path}/output/output_*/tensor/ <br> Description: str: a simple filename prefix; a string containing "$NAME" to be matched: replaces $NAME with the value of NAME in kwargs, then uses the resulting string as the filename prefix. <br> Type: str <br> Value range: NA |
| cond      | Input        | Meaning: specifies the condition that must be met to print data <br> Description: if the expression evaluates to 1: prints the specified data; if the expression evaluates to 0: does not print data. <br> Type: Optional[int,pypto.SymbolicScalar] <br> Value range: 0, 1 <br> Default value: 1 |
| **kwargs  | Input        | Specifies the values of the placeholder strings in the fname parameter. |

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
        # Save t2 to file t2.*
        pypto.pass_verify_save(t2, 't2-fileprefix')
        t3 = pypto.SOME_OP2(t0, t2)
        # When idx==5, save t3 to file t3_debug_loop_5.*
        pypto.pass_verify_save(t3, "t3_debug_loop_$idx", cond=(idx == 5), idx=5)
         ...

```

