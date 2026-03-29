# pypto.lrelu

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Applies the Leaky ReLU (Leaky Rectified Linear Unit) activation function element-wise to the input tensor. The formula is as follows:

$$
\text{res}_i =
\begin{cases}
\text{input}_i & \text{if } \text{input}_i \geq 0 \\
\text{negative\_slope} \cdot \text{input}_i & \text{if } \text{input}_i < 0
\end{cases}
$$

where `negative_slope` is the negative slope parameter, with a default value of `0.01`.


## Function Prototype

```python
lrelu(input: Tensor, negative_slope: float = 0.01) -> Tensor
```

## Parameters

| Parameter      | Input/Output | Description                                                                 |
|----------------|--------------|-----------------------------------------------------------------------------|
| input          | Input        | Source operand.<br>Supported type: Tensor.<br>Supported tensor data types: DT_FP16, DT_BF16, DT_FP32.<br>Empty tensors are not supported; shape supports only 2–4 dimensions; shape size must not exceed 2147483647 (INT32_MAX). |
| negative_slope | Input        | Slope coefficient for the negative region.<br>Supported types: float or Element type. Default value is `0.01` (float), which is automatically converted to the Element type (float maps to DT_FP32). To use other data types, construct via Element.<br>Must be a non-negative real number (≥ 0); special values such as `nan` and `inf` are not supported. |

## Return Value

Returns an output tensor with the same data type and shape as `input`.

## Constraints

1.  The data type of `input` must be DT_FP16, DT_BF16, or DT_FP32.
2.  `negative_slope` must be a non-negative float (≥ 0) and must not be `nan` or `inf`.
3.  It is recommended to use Element for `negative_slope` rather than passing a float scalar; correctness is not guaranteed for fp16 scenarios when using float.
4.  In-place operations are not supported (i.e., the output cannot share memory with the input).

## Example

### TileShape Configuration Example

Note: Before calling this operation interface, set the TileShape via `set_vec_tile_shapes`.

The TileShape dimensions must match the output dimensions.

Example 1: If the input `input` shape is `[m, n]` and the output is `[m, n]`, set TileShape to `[m1, n1]`, where `m1` and `n1` are used to tile the `m` and `n` axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
a = pypto.tensor([[-1.0, 0.0, 1.0]], pypto.DT_FP32)
out = pypto.lrelu(a)
```

Example result:

```python
Input a:   [[-1.0  0.0  1.0]]
Output out: [[-0.01  0.0   1.0]]
```
