# pypto.expand\_exp\_dif

## Supported Products

| Product | Supported |
|:--------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series | √ |
| Atlas A2 Training Series / Atlas A2 Inference Series | √ |

## Description

Computes the natural exponential base e raised to the power of (input - other), where the value of the last or second-to-last axis of `other` is 1. Returns a Tensor with the same data type and shape as the input `input`.

$$
e^{(input - other)}
$$

## Function Prototype

```python
expand_exp_dif(input: Tensor, other: Tensor) -> Tensor
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| input | Input | Source operand. <br> Supported type: Tensor. <br> Supported Tensor data types: DT_FP16, DT_FP32. <br> Empty Tensors are not supported; shape supports only 2–4 dimensions, and broadcasting along a single dimension to the same shape is supported; shape size must not exceed 2147483647 (INT32_MAX). |
| other | Input | Source operand. <br> Supported type: Tensor. <br> Supported Tensor data types: DT_FP16, DT_FP32. <br> Empty Tensors are not supported; shape supports only 2–4 dimensions, and broadcasting along a single dimension to the same shape is supported; the value of the last or second-to-last axis must be 1; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

Returns the output Tensor. Its data type and shape are the same as `input`.

## Constraints

1. `input` and `other` must have the same data type.
2. The value of the last or second-to-last axis of `other` must be 1.

## TileShape Configuration Example

Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output.

For a non-broadcast scenario: if the input `input` shape is [m, n] and `other` is [m, n], the output is [m, n], and TileShape is set to [m1, n1], where m1 and n1 are used to split the m and n axes respectively.

For a broadcast scenario: if the input `input` shape is [m, n] and `other` is [m, 1], the output is [m, n], and TileShape is set to [m1, n1], where m1 and n1 are used to split the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

## Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([1, 3], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

Example result:

```python
Input data x:     [[1, 2, 3], [4, 5, 6]]
Input data y:     [[1, 2, 3]]
Output data out:  [[ 1.      ,  1.      ,  1.      ],
                   [20.085537, 20.085537, 20.085537]]
```

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.tensor([2, 1], pypto.DT_FP32)
out = pypto.expand_exp_dif(x, y)
```

Example result:

```python
Input data x:     [[1, 2, 3], [4, 5, 6]]
Input data y:     [[1], [2]]
Output data out:  [[ 1.       ,  2.718282 ,  7.3890557],
                   [ 7.3890557, 20.085537 , 54.59815  ]]
```
