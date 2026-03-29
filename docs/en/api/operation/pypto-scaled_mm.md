# pypto.scaled\_mm

## Supported Products

| Product          | Supported |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |

## Description

Performs an MX quantized matrix multiplication of mat_a and mat_b. The formula is: out = (mat_a * scale_a) @ (mat_b * scale_b)

-   mat_a, mat_b, scale_a, and scale_b are source operands. mat_a is the left matrix; mat_b is the right matrix; scale_a is the quantization parameter for the left matrix; scale_b is the quantization parameter for the right matrix.
-   out is the destination operand, the matrix that stores the result of the matrix multiplication.

## Function Prototype

```python
scaled_mm(mat_a, mat_b, out_dtype, scale_a, scale_b, *, a_trans = False, b_trans = False, scale_a_trans = False, scale_b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## Parameters


| Parameter         | Input/Output | Description                                                                 |
|-------------------|--------------|-----------------------------------------------------------------------------|
| mat_a             | input        | Represents the input left matrix. Empty Tensor not supported. <br> Supported data types: DT_FP8E5M2, DT_FP8E4M3; the data types of the left and right matrices must be consistent. <br> Supported matrix dimensions: 2D. <br> Supported input matrix formats: TILEOP_ND, TILEOP_NZ. <br> For TILEOP_ND (ND format): the outer axis range is [1, 2^31 - 1], and the inner axis range is [1, 65535]. <br> For TILEOP_NZ (NZ format): the shape dimensions must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. <br> In addition to format constraints, the shape dimensions must satisfy 64-element alignment on the K axis. <br> Inner/outer axis: when the input matrix mat_a is not transposed, the data layout is [M, K], where the outer axis is M and the inner axis is K; when transposed, the data layout is [K, M], where the outer axis is K and the inner axis is M. <br> When using the pypto.view interface, ensure that the shape dimensions passed to View also satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| mat_b             | input        | Represents the input right matrix. Empty Tensor not supported. <br> Supported data types: DT_FP8E5M2, DT_FP8E4M3; the data types of the left and right matrices must be consistent. <br> Supported matrix dimensions: 2D. <br> Supported input matrix formats: TILEOP_ND, TILEOP_NZ. <br> For TILEOP_ND (ND format): the outer axis range is [1, 2^31 - 1], and the inner axis range is [1, 65535]. <br> For TILEOP_NZ (NZ format): the shape dimensions must satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. <br> In addition to format constraints, the shape dimensions must satisfy 64-element alignment on the K axis. <br> Inner/outer axis: when the input matrix mat_b is not transposed, the data layout is [K, N], where the outer axis is K and the inner axis is N; when transposed, the data layout is [N, K], where the outer axis is N and the inner axis is K. <br> When using the pypto.view interface, ensure that the shape dimensions passed to View also satisfy 32-byte alignment on the inner axis and 16-element alignment on the outer axis. |
| out_dtype         | output       | Represents the output matrix data type; supports DT_FP32, DT_FP16, DT_BF16. |
| scale_a           | input        | Represents the quantization parameter for the input left matrix. Empty Tensor not supported. <br> Supported data type: DT_FP8E8M0. <br> Supported quantization parameter dimensions: 3D. <br> Input quantization parameter shape: when not transposed, the shape is [M, K/64, 2]; when transposed, the shape is [K/64, M, 2]. The M and K values equal those of the input matrix mat_a. <br> Supported format for input quantization parameters: TILEOP_ND. |
| scale_b           | input        | Represents the quantization parameter for the input right matrix. Empty Tensor not supported. <br> Supported data type: DT_FP8E8M0. <br> Supported quantization parameter dimensions: 3D. <br> Input quantization parameter shape: when not transposed, the shape is [K/64, N, 2]; when transposed, the shape is [N, K/64, 2]. The M and K values equal those of the input matrix mat_a. <br> Supported format for input quantization parameters: TILEOP_ND. |
| a_trans           | input        | Parameter a_trans indicates whether the input left matrix is transposed; default is False. |
| b_trans           | input        | Parameter b_trans indicates whether the input right matrix is transposed; default is False. |
| scale_a_trans     | input        | Parameter scale_a_trans indicates whether the quantization parameter of the input left matrix is transposed; default is False. |
| scale_b_trans     | input        | Parameter scale_b_trans indicates whether the quantization parameter of the input right matrix is transposed; default is False. |
| c_matrix_nz       | input        | Parameter c_matrix_nz indicates whether the output matrix uses NZ format; default is False. Currently only False is supported, meaning the output matrix only supports ND format. |
| extend_params     | input        | Supports bias and fixpipe dequantization; data type is dictionary format. Default is None. Currently only None is supported, meaning bias and fixpipe dequantization are not supported. |

## Return Value

Returns the output matrix (Tensor).

## Constraints

-   Before calling the matmul interface, set the tiling sizes for the M, N, and K axes via pypto.set\_cube\_tile\_shapes.
-   When the input to the matmul interface after calling pypto.reshape is in NZ format, the pypto.set\_matrix\_size interface must be called to set the original shape m, k, n values of the input to matmul before pypto.reshape.

## Example

```python
mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([64, 2, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([2, 32, 2], pypto.DT_FP8E8M0, "scale_b")
out1 = pypto.scaled_mm(a1, b1, pypto.DT_BF16, scale_a, scale_b)

mat_a = pypto.tensor([64, 128], pypto.DT_FP8E5M2, "mat_a")
mat_b = pypto.tensor([128, 32], pypto.DT_FP8E5M2, "mat_b")
scale_a = pypto.tensor([2, 64, 2], pypto.DT_FP8E8M0, "scale_a")
scale_b = pypto.tensor([32, 2, 2], pypto.DT_FP8E8M0, "scale_b")
out1 = pypto.scaled_mm(a1, b1, pypto.DT_BF16, scale_a, scale_b, scale_a_trans=True, scale_b_trans=True)
```

