# pypto.loop\_unroll

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

pypto.loop\_unroll is a loop iterator function that supports loop unrolling. It works similarly to pypto.loop, with the addition of the unroll\_list parameter to support multiple unrolling strategies.

## Function Prototype

```python
loop_unroll(*args, **kwargs) -> Iterator[Tuple[SymInt, int]]
```

## Parameters


| Parameter         | Input/Output | Description                                                                 |
|-------------------|--------------|-----------------------------------------------------------------------------|
| *args             | Input        | Three optional parameters: loop start value (start), loop end value (stop), and loop step size (step). Supports the following three forms:<br> - Single-argument form: stop(SymInt), start defaults to 0, step defaults to 1. Equivalent to: loop_unroll(0, stop, 1)<br> - Two-argument form: start(SymInt), stop(SymInt), equivalent to loop_unroll(start, stop, 1)<br> - Three-argument form: start(SymInt), stop(SymInt), step(SymInt), equivalent to loop_unroll(start, stop, step) |
| **kwargs          | Input        | - name(str): The loop identifier name; defaults to f"loop_{loop_idx}".<br> - idx_name(str): The name of the loop index variable; defaults to f"loop_idx_{loop_idx}".<br> - unroll_list(List[int]): The set of loop unroll factors to apply; defaults to an empty set. The loop provides as many unrolling strategies as there are elements in this set. When the unroll factor is n, the loop step becomes step*n and each iteration executes n loop bodies. Each unroll factor generates a different code path.<br> - submit_before_loop(bool): Whether to submit computation before the loop starts; defaults to False. When enabled, forcibly submits the currently accumulated computation tasks to AICore for execution before the loop begins. |

## Return Value

Returns an iterator that produces a tuple (idx, unroll\_factor\) on each iteration, where idx is the current loop index value and unroll\_factor identifies the currently selected unrolling strategy.

## Constraints

-   The unroll factor list is sorted, deduplicated, and always includes 1
-   Unroll factors are sorted in descending order
-   Each unroll factor generates a sub-loop
-   Using loop_unroll with unroll_list configured in multiple nested loops will greatly increase the number of compiled graphs, impacting compilation performance

## Example

```python
for _ in pypto.loop_unroll(0, 10, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx", unroll_list=[1, 2, 4]):
   ...
```

