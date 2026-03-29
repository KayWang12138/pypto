# pypto.set\_verify\_golden\_data

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

This interface includes the following required features:

-   Feature 1: Sets the actual input and output lists used when the user executes an operator into the tool, so that the tool can perform simulation computations using the same inputs during coarse-grained verification
-   Feature 2: Sets the user's existing reference computation data (golden) into the tool, so the tool can compare the simulation results against the reference output after computing them, thereby coarsely verifying the correctness of the simulation results

## Function Prototype

```python
set_verify_golden_data(in_out_tensors=None, goldens=None)
```

## Parameters


| Parameter       | Input/Output | Description                                                                 |
|-----------------|--------------|-----------------------------------------------------------------------------|
| in_out_tensors  | Input        | Meaning: sets the actual input and output lists used when the user executes an operator into the detection tool, in positional correspondence. <br> Description: in jit invocation mode, this option does not need to be set <br> Type: List[Union(pypto.Tensor, torch.Tensor)] <br> Value range: NA <br> Default value: NA |
| goldens         | Input        | Meaning: sets the user's existing reference computation data (golden) outputs into the tool for comparison verification. <br> Description: this list has the same length and positional correspondence as the operator's input/output parameter list. Setting a position to None indicates that the data comparison at that position is skipped. <br> Type: List[Union(pypto.Tensor, torch.Tensor)] <br> * The torch.Tensor device attribute must be CPU; NPU is not supported. <br> Value range: NA <br> Default value: NA |

## Return Value

void: Set methods have no return value. The operation takes effect upon successful completion.

## Constraints

This function takes effect only after pypto.set_verify_options(enable_pass_verify=True) is set.

## Example

```python
set_verify_golden_data(goldens=[None, None, golden_out0])
set_verify_golden_data([real_in0, real_in1, real_out0], [None, None, golden_out0])
```

