# pypto.reshape

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Changes the shape of a Tensor, modifying the shape of the valid\_shape portion.

## Function Prototype

```python
reshape(input: Tensor,shape: List[int],*,valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, inplace: bool = False) -> Tensor
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| input       | input        | Source operand. <br> Supported data types: data types supported by PyPTO. <br> Empty Tensor not supported; Shape Size must not exceed INT32_MAX. |
| shape       | input        | Target shape. <br> Shape Size must not exceed INT32_MAX; when a dimension is -1, automatic inference is supported. |
| valid_shape | input        | The shape of the valid data in the output Tensor; valid_shape Size must not exceed INT32_MAX. |
| inplace     | input        | Whether this is an inplace operation; when set to True, no new address will be allocated for the output. |

## Return Value

Returns the output Tensor. The data type is the same as input, and the shape is the shape specified by the input parameter.

## Constraints

When inplace is True, the input and output must be the input and output of the current loop respectively; the output cannot serve as the output of the entire Function.

## Example

Example 1:

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.reshape(x, [4, 1], [2, 1])
z = pypto.add(y, 1.0)
```

Example result:

```python
input data x: [[1, 2],
               [3, 4]]
output data y: [[1],
                [2],
                [3],
                [4]]
output data z: [[2],
                [3],
                [3],
                [4]]
```

Example 2:

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
for _ in pypto.loop(1, name="reshape_inplace", idx_name="tmp_loop"):
    x_1 = x.reshape(x, [4], inplace=True)
for _ in pypto.loop(1, name="loop", idx_name="loop"):
    y = pypto.add(x_1, 1.0)
```

Example result:

```python
input data x: [[1, 2],
               [3, 4]]
output data y: [2, 3, 4, 5]
```

