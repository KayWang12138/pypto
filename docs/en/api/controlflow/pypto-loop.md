# pypto.loop

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Defines a loop operation, implementing the for loop functionality in Python.

## Function Prototype

```python
loop(stop: SymInt, /, **kwargs) -> Iterator[SymInt]
loop(start: SymInt, stop: SymInt, step: Optional[SymInt] = 1, /, **kwargs) -> Iterator[SymInt]
```

## Parameters


| Parameter         | Input/Output | Description                                                                 |
|-------------------|--------------|-----------------------------------------------------------------------------|
| start             | Input        | The starting value of the loop. |
| stop              | Input        | The termination value of the loop. |
| step              | Input        | The step size of each iteration. |
| **kwargs          | Input        | - name(str): The loop identifier name; defaults to f"loop_{loop_idx}".<br> - idx_name(str): The name of the loop index variable; defaults to f"loop_idx_{loop_idx}".<br> - submit_before_loop(bool): Whether to submit computation before the loop starts; defaults to False. When enabled, forcibly submits the currently accumulated computation tasks to AICore for execution before the loop begins. |

## Return Value

Returns a generator that sequentially yields symbolic integers representing the value of each iteration.

## Constraints

None.

## Example

```python
for _ in pypto.loop(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
   ...
```

