# pypto.copysign

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Takes the magnitude of each element of `input` and the sign of the corresponding element of `other`, element-wise. The formula is:

$$
res_i = input_i * sign(other_i)
$$

## Function Prototype

```python
copysign(input: Tensor, other: Tensor) -> Tensor
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; shape supports 2–4 dimensions only, with broadcasting along a single dimension to match shapes; shape size must not exceed 2147483647 (INT32_MAX). |
| other     | Input        | Source operand. <br> Supported type: Tensor. <br> Supported tensor data types: DT_FP32, DT_FP16, DT_BF16. <br> Empty tensors are not supported; shape supports 2–4 dimensions only, with broadcasting along a single dimension to match shapes; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output tensor. The tensor's data type is the same as `input` and `other`, and the shape is the broadcast shape of `input` and `other`.

## Constraints

1.  `input` and `other` must have the same type.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

For the non-broadcast scenario, if input shape is [m, n] and other is [m, n], output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

For the broadcast scenario, if input shape is [m, n] and other is [m, 1], output is [m, n], and TileShape is set to [m1, n1], then m1 and n1 are used to split the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([3, 3], pypto.DT_FP32)
b = pypto.tensor([3, 3], pypto.DT_FP32)
out = pypto.copysign(a, b)
```

Example result:

```python
Input  x : [[1 -2  3],
           [4  5 -6],
           [-7 8  9]]
Input  y : [[-1 6 -8],
            [1 -1  0],
            [7 -8  9]]
Output out:[[-1 2 -3],
            [4 -5  6],
            [7 -8  9]]
```
