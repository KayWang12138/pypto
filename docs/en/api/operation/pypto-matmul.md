# pypto.matmul

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Performs matrix multiplication of `input` and `mat2`. The formula is: out = input @ mat2

-   `input` and `mat2` are the source operands; `input` is the left matrix and `mat2` is the right matrix.
-   `out` is the destination operand, storing the result of the matrix multiplication.

## Function Prototype

```python
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

## Parameters

Table 1: API Parameters

| Parameter         | Input/Output | Description                                                                 |
|-------------------|--------------|-----------------------------------------------------------------------------|
| input             | Input        | The left input matrix. Empty tensors are not supported. <br> Supported data types: DT_INT8, DT_FP16, DT_BF16, DT_FP32, DT_HF8, DT_FP8E5M2, DT_FP8E4M3. When the data type of `input` is DT_FP8E5M2, `mat2` can be DT_FP8E5M2 or DT_FP8E4M3; when the data type of `input` is DT_FP8E4M3, `mat2` can be DT_FP8E5M2 or DT_FP8E4M3. For all other data types, the left and right matrices must have matching data types. <br> Supported matrix dimensions: 2D, 3D, or 4D; left and right matrix dimensions must match. <br> Supported input matrix formats: TILEOP_ND, TILEOP_NZ (DT_FP32 and DT_FP8E5M2 inputs do not support TILEOP_NZ format). <br> For TILEOP_ND (ND format): outer axis range [1, 2^31 - 1], inner axis range [1, 65535]. <br> For TILEOP_NZ (NZ format): inner axis must be 32-byte aligned (16-element aligned when output matrix data type is DT_INT32), and outer axis must be 16-element aligned. <br> Inner/outer axes: when `input` is not transposed, the data layout is [M, K], so the outer axis is M and the inner axis is K; when `input` is transposed, the data layout is [K, M], so the outer axis is K and the inner axis is M. <br> When using `pypto.view`, the shape passed to View must also satisfy inner-axis 32-byte alignment (16-element aligned for DT_INT32 output) and outer-axis 16-element alignment. <br> When the matrix dimension is 3D or 4D, the `pypto.view` scenario is not supported. |
| mat2              | Input        | The right input matrix. Empty tensors are not supported. <br> Supported data types: DT_INT8, DT_FP16, DT_BF16, DT_FP32, DT_HF8, DT_FP8E5M2, DT_FP8E4M. <br> Supported matrix dimensions: 2D, 3D, or 4D; left and right matrix dimensions must match. <br> Supported input matrix formats: TILEOP_ND, TILEOP_NZ (DT_FP32 and DT_FP8E5M2 inputs do not support TILEOP_NZ format). <br> For TILEOP_ND (ND format): outer axis range [1, 2^31 - 1], inner axis range [1, 65535]. <br> For TILEOP_NZ (NZ format): inner axis must be 32-byte aligned (16-element aligned when output matrix data type is DT_INT32), and outer axis must be 16-element aligned. <br> Inner/outer axes: when `mat2` is not transposed, the data layout is [K, N], so the outer axis is K and the inner axis is N; when `mat2` is transposed, the data layout is [N, K], so the outer axis is N and the inner axis is K. <br> When using `pypto.view`, the shape passed to View must also satisfy inner-axis 32-byte alignment (16-element aligned for DT_INT32 output) and outer-axis 16-element alignment. <br> When the matrix dimension is 3D or 4D, the `pypto.view` scenario is not supported. |
| out_dtype         | Output       | The output matrix data type. Supported: DT_FP32, DT_FP16, DT_BF16, DT_INT32. <br> - When input matrix data type is DT_FP16, `out_dtype` can be DT_FP32 or DT_FP16. <br> - When input matrix data type is DT_BF16, `out_dtype` can be DT_FP32 or DT_BF16. <br> - When input matrix data type is DT_INT8, `out_dtype` can only be DT_INT32. <br> - When input matrix data type is DT_FP32, `out_dtype` can only be DT_FP32. <br> - When input matrix data type is DT_FP8E5M2 or DT_FP8E4M3, `out_dtype` can be DT_FP16, DT_BF16, or DT_FP32. |
| a_trans           | Input        | Specifies whether the left input matrix is transposed. Default: False. |
| b_trans           | Input        | Specifies whether the right input matrix is transposed. Default: False. |
| c_matrix_nz       | Input        | Specifies whether the output matrix format is NZ. Default: False. Currently only False is supported, meaning the output matrix only supports ND format. |
| extend_params     | Input        | Supports bias, fixpipe dequantization, and TF32 rounding mode TransMode. See table below for details. <br> - Data type is dictionary format. <br> - This parameter and its internal parameters are all optional. |

Table 2: extend_params Parameter Description

| Parameter         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| scale             | Dequantization parameter for the output matrix in per-tensor quantization scenarios (using a single scale factor to map high-precision numbers to low-precision numbers). <br> Input is float type: 1 sign bit + 8 exponent bits + 10 mantissa bits participate in computation. |
| scale_tensor      | Dequantization matrix for the output matrix in per-channel quantization scenarios (computing an independent set of quantization parameters for each output channel). <br> `scale_tensor` input must be a uint64_t Tensor. During computation, uint64_t is converted to the lower 32 bits of float type, then 1 sign bit + 8 exponent bits + 10 mantissa bits participate in computation. <br> The first dimension of `scale_tensor` must be 1, and the N dimension must equal the N dimension of `mat2`. <br> `scale_tensor` only supports ND format. <br> Only 2D matrix dimensions are supported. |
| bias_tensor       | The bias matrix. <br> Input type: Tensor. <br> When the input left/right matrix data type is DT_FP16, the bias matrix data type can be DT_FP16 or DT_FP32. <br> When the input left/right matrix data type is DT_FP32, the bias matrix data type can only be DT_FP32. <br> When the input left/right matrix data type is DT_INT8, the bias matrix data type can only be DT_INT32. <br> When the input left/right matrix data type is DT_FP8E5M2 or DT_FP8E4M3, the bias matrix data type can only be DT_FP32. <br> `bias_tensor` only supports ND format. <br> The first dimension of `bias_tensor` must be 1, and the N dimension must equal the N dimension of `mat2`. <br> Bias does not support multi-core K-split. <br> Only 2D matrix dimensions are supported. |
| relu_type         | Specifies whether to apply a ReLU operation to the output matrix. <br> Input type: [ReLuType](../datatype/ReLuType.md). <br> Supports RELU and NO_RELU modes. <br> Only 2D matrix dimensions are supported. |
| trans_mode        | Specifies whether to enable TF32 computation and the TF32 rounding mode. <br> Input type: [TransMode](../datatype/TransMode.md). <br> CAST_NONE: Do not enable float-to-TF32 data type conversion. <br> CAST_RINT: Enable float-to-TF32 conversion; rounding rule: round to nearest integer, ties go to even. <br> CAST_ROUND: Enable float-to-TF32 conversion; rounding rule: round to nearest integer, ties away from zero. <br> Only applicable when both input and output matrix data types are DT_FP32. <br> Only 2D matrix dimensions are supported. |

## Return Value

Returns the output matrix (`out`) as a Tensor.

## Constraints

-   Atlas A2 Training Series/Atlas A2 Inference Series: DT_HF8, DT_FP8E5M2, and DT_FP8E4M3 are not supported; the `trans_mode` parameter in `extend_params` is not supported.
-   Atlas A3 Training Series/Atlas A3 Inference Series: DT_HF8, DT_FP8E5M2, and DT_FP8E4M3 are not supported; the `trans_mode` parameter in `extend_params` is not supported.
-   Before calling the `matmul` interface, set the tiling sizes on the M, N, and K axes via `pypto.set\_cube\_tile\_shapes`.
-   When the matrix dimension is 3D or 4D, call `pypto.set\_vec\_tile\_shapes` to set the vector TileShape. If not set, the interface internally sets a 2D vec\_tile\_shape with a value of 128×128.
-   When the input to `matmul` is in NZ format after calling `pypto.reshape`, call `pypto.set\_matrix\_size` to set the original shape's m, k, n values before `pypto.reshape`.
-   When the input matrix dimension to `matmul` is 3D/4D and the data format is NZ, call `pypto.set\_matrix\_size` to set the original shape's m, k, n values.

## Example

```python
a1 = pypto.tensor([16, 32], pypto.DT_BF16, "tensor_a")
b1 = pypto.tensor([32, 64], pypto.DT_BF16, "tensor_b")
out1 = pypto.matmul(a1, b1, pypto.DT_BF16)

a2 = pypto.tensor((2, 16, 32), pypto.DT_FP16, "tensor_a")
b2 = pypto.tensor((2, 32, 16), pypto.DT_FP16, "tensor_b")
out2 = pypto.matmul(a2, b2, pypto.DT_FP16)

a3 = pypto.tensor((1, 32, 64), pypto.DT_FP32, "tensor_a")
b3 = pypto.tensor((3, 64, 16), pypto.DT_FP32, "tensor_b")
out3 = pypto.matmul(a3, b3, pypto.DT_FP32)

a = pypto.tensor((16, 32), pypto.DT_FP16, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_FP16, "tensor_b")
bias = pypto.tensor((1, 64), pypto.DT_FP16, "tensor_bias")
extend_params = {'bias_tensor': bias}
pypto.matmul(a, b, pypto.DT_FP32, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
extend_params = {'scale': 0.2}
pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

 a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
 b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
 extend_params = {'scale': 0.2, 'relu_type': pypto.ReLuType.RELU}
 pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)

 a = pypto.tensor((16, 32), pypto.DT_INT8, "tensor_a")
 b = pypto.tensor((32, 64), pypto.DT_INT8, "tensor_b")
 scale_tensor = pypto.tensor((1, 64), pypto.DT_UINT64, "tensor_scale")
 extend_params = {'scale_tensor': scale_tensor, 'relu_type': pypto.ReLuType.RELU}
 pypto.matmul(a, b, pypto.DT_BF16, a_trans=False, b_trans=False, c_matrix_nz=False, extend_params=extend_params)
```
