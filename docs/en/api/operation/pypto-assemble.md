# pypto.assemble

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Assigns the input Tensor input to the corresponding region of the output Tensor out, based on the index position in out specified by offsets.

## Function Prototype

```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor) -> None

assemble(inputs: List[Tuple[Tensor, List[Union[int, SymbolicScalar]]]], out: Tensor, parallel: bool = False) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | input        | Source operand. <br> Supported data types: data types supported by PyPto. <br> Empty Tensor not supported; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| inputs    | input        | A list of Tuples composed of source operands and output offsets. <br> Supported data type for each element: data types supported by PyPto. <br> Empty Tensor not supported; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| offsets   | input        | Offsets relative to the target output. <br> Must ensure offsets are less than the shape of out. |
| out       | input        | Destination operand. <br> Supported data types: data types supported by PyPto. <br> Empty Tensor not supported; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| parallel  | input        | Whether to execute in parallel. <br> Default value: False. |

## Return Value

No return value. Modifies out in-place.

## Constraints

None.

## Example

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
out = pypto.tensor([4, 4], pypto.DT_FP32)
offsets = [0, 0]
pypto.assemble(x, offsets, out)

y = pypto.tensor([2, 2], pypto.DT_FP32)
pypto.assemble([(x, offsets), (y, [2, 2])], out)
```

Example result:

```python
Output data x: [[1, 1]
                [1, 1]]
Input data out: [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
Output data out: [[1, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]
Output data out1: [[1, 1, 0, 0],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1]]
```

