# pypto.topk

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the top-k largest or smallest values and their corresponding indices along the last dimension.

If the input is a vector, finds the top-k largest or smallest values and their indices in the vector. If the input is a matrix, computes the top-k largest or smallest values and their indices in each row along the last dimension. As shown below, sorting a 2D matrix with shape \(4, 32\) with k=1, the output is \[\[32\] \[32\] \[32\] \[32\]\].

![](../figures/nz-reduce.png)

## Function Prototype

```python
topk(input: Tensor, k: int, dim: Optional[int]=None, largest: bool=True) -> Tuple[Tensor, Tensor]
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| input     | Input        | Source operand.<br> Supported type: Tensor.<br> Supported tensor data types: DT_FP32.<br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| k         | Input        | Number of elements to return.<br> Must satisfy: 1 <= k <= input.shape[dim]. |
| dim       | Input        | Specifies the dimension along which to sort.<br> Currently only sorting along the last dimension is supported, i.e., dim = -1 or dim = input.shape.size() - 1. |
| largest   | Input        | If True, returns the largest elements. If False, returns the smallest elements. |

## Return Value

Returns a named tuple (values, indices) containing the values and indices of the top-k largest or smallest elements in each row of `input` along the specified dimension `dim`.

## Constraints

1.  Only topk operation on the last axis is supported;
2.  The last axis of TileShape must be 32-byte aligned \(TileShape\[-1\]\*4 % 32 == 0\), and must be less than 22 KB \(TileShape\[-1\]\*4 < 22KB\);
3.  k <= TileShape\[-1\] && k <= input.shape\[-1\];

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the input `input` dimensions.

Example 1: If the input `input` shape is `[m, n, p]`, `dim` is 2, and `largest` is True, the output is `[m, n, k]`. Set TileShape to `[m1, n1, p1]`, where `m1`, `n1`, `p1` tile the `m`, `n`, `p` axes respectively. `p1` must be greater than or equal to `k`; the k axis does not support tiling and must be fully loaded.

```python
pypto.set_vec_tile_shapes(4, 16, 32)
```

### Interface Call Example

```python
x = pypto.tensor([2, 3], pypto.DT_FP32)
y = pypto.topk(x, 2, -1, True)
```

Example result:

```python
Input x: [[1.0 2.0 3.0],
            [1.0 2.0 3.0]]
Output y[0]: [[3.0 2.0],
               [3.0 2.0]]
Output y[1]: [[2, 1],
               [2, 1]]
```
