# pypto.where

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

`condition` is a boolean mask tensor. For any element position in the tensors, this operation performs element-wise selection based on the boolean mask tensor `condition`. The computation can be formally expressed as follows:

$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

`condition` must be a Tensor; `input` and `other` can be Tensor, float, or Element. Broadcasting rules are as follows (only single-axis broadcasting is supported):

1.  When `input`, `other`, and `condition` are all tensors, the shape of `result` is determined by broadcasting all three.

    Example: input:\[1,20,20\], other:\[20,1,20\], condition:\[20,20,1\], result:\[20,20,20\]

2.  When only `input` and `condition` are tensors, the shape of `result` is determined by broadcasting the two.

    Example: input:\[1,20,20\], condition:\[20,20,1\], result:\[20,20,20\]

3.  When only `other` and `condition` are tensors, the shape of `result` is determined by broadcasting the two.

    Example: other:\[20,1,20\], condition:\[20,20,1\], result:\[20,20,20\]

4.  When only `condition` is a tensor, the shape of `result` matches the shape of `condition`.

## Function Prototype

```python
where(
    condition: Tensor,
    input: Union[Tensor, float, Element],
    other: Union[Tensor, float, Element]
) -> Tensor
```

## Parameters


| Parameter   | Input/Output | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| condition   | Input        | Supported type: Tensor.<br> Supported tensor data types: DT_BOOL.<br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX).<br> Used as a condition to select elements from `input` or `other`. |
| input       | Input        | Supported types: float, Element, or Tensor.<br> When the type is float, it is automatically converted to Element type (float maps to DT_FP32). To use other data types, construct via Element.<br> Supported data types for Tensor and Element: DT_FP32, DT_FP16, DT_BF16.<br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| other       | Input        | Supported types: float, Element, or Tensor.<br> When the type is float, it is automatically converted to Element type (float maps to DT_FP32). To use other data types, construct via Element.<br> Supported data types for Tensor and Element: DT_FP32, DT_FP16, DT_BF16.<br> Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |

## Return Value

`result`: Tensor. The shape is determined by broadcasting the inputs; see the broadcasting scenarios above for details. The data type is consistent with `input` and `other`.

## Constraints

1. It is recommended to use Element rather than passing a float scalar; correctness is not guaranteed for fp16 scenarios when using float.

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Example 1: Non-broadcast scenario — `condition` is `[m, n]`, `input` is `[m, n]`, `other` is `[m, n]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

Example 2: Broadcast scenario — `condition` is `[m, 1]`, `input` is `[m, n]`, `other` is `[m, n]`, output is `[m, n]`. Set TileShape to `[m1, n1]`, where `m1` and `n1` tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
cond1 = pypto.tensor([4], pypto.DT_BOOL)
a1 = pypto.tensor([4], pypto.DT_FP32)
b1 = pypto.tensor([4], pypto.DT_FP32)
out1 = pypto.where(cond1, a1, b1)

# Using scalar inputs
out2 = pypto.where(cond1, 1, 0)

# Broadcasting example
cond2 = pypto.tensor([2, 2], pypto.DT_BOOL)
a2 = pypto.tensor([2], pypto.DT_FP32)
b2 = 0.0
out3 = pypto.where(cond2, a2, b2)
```

Example result:

```python
Input cond1: [True, False, True, False]
Input a1:    [1.0  2.0  3.0  4.0]
Input b1:    [10.0 20.0 30.0 40.0]
Output out1:  [1.0  20.0 3.0  40.0]

Output out2:  [1.0 0.0 1.0 0.0]

Input cond2 = [[True, False], [False, True]]
Input a2:      [1.0 2.0]
Output out3:   [[1.0 0.0],
                 [0.0 2.0]]
```
