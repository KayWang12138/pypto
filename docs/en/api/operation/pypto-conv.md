# pypto.conv

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    ×     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    ×     |

## Description

Performs a convolution operation on `input_conv` and `weight`, with optional `bias`. The formula is: out = input_conv @ weight + bias (where @ denotes convolution)

-   `input_conv`, `weight`, and `bias` are source operands; `input_conv` is the input feature map, `weight` is the kernel, and `bias` is the input bias.
-   `out` is the destination operand, storing the convolution result.
-   Quantization scenarios are not currently supported.
-   ReLU functionality is not currently supported.

## Function Prototype

```python
conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups=1, transposed=False, output_paddings=[], extend_params=None) -> Tensor
```

## Parameters

| Parameter         | Input/Output | Description                                                                 |
|-------------------|--------------|-----------------------------------------------------------------------------|
| input_conv        | Input        | Input feature map tensor. <br> Empty tensors are not supported. <br> Supported dimensions: 3D (1D conv), 4D (2D conv), 5D (3D conv). <br> Supported formats: NCL, NCHW, NCDHW. <br> Supported data types: DT_FP16, DT_BF16, DT_FP32. <br> Shape constraint: each dimension in the range [1, 1000000]. input_conv: cin = weight: cin * groups |
| weight            | Input        | Convolution kernel tensor. <br> Dimensions must match `input_conv` (3D/4D/5D). <br> Data type must match `input_conv`. <br> Shape constraint: each dimension in the range [1, 1000000]. |
| out_dtype         | Input        | Output tensor data type. <br> Supported: DT_FP16, DT_BF16, DT_FP32. <br> Must match `input_conv`; may be specified independently in fixpipe quantization scenarios. |
| strides           | Input        | Convolution strides, unidirectional parameter. <br> - 1D (1D conv) <br> - 2D (2D conv) <br> - 3D (3D conv) <br> Value range: [1, 63]. |
| paddings          | Input        | Convolution padding, bidirectional parameter. <br> - 2D (1D conv) <br> - 4D (2D conv) <br> - 6D (3D conv) <br> Value range: [0, 255]; padding per dimension must be < the corresponding kernel size. |
| dilations         | Input        | Dilation rate for dilated convolution, unidirectional parameter. <br> - 1D (1D conv) <br> - 2D (2D conv) <br> - 3D (3D conv) <br> Value range: [1, 63]. |
| groups            | Input        | Number of groups for grouped convolution, default 1. <br> Value range: [1, 65535]. <br> Cin and Cout must both be divisible by groups. |
| transposed        | Input        | Whether to perform transposed convolution (deconvolution), default False. <br> True is not currently supported. |
| output_paddings   | Input        | Output-side padding for transposed convolution, used only when transposed=True. <br> Not currently supported. |
| extend_params     | Input        | Dictionary of extended parameters; supports bias, scale, relu, and scale_tensor: <br> - bias_tensor: optional bias tensor with shape (C_out,); only ND format is supported; bias data type must match `input_conv`. <br> - scale: float, per-tensor scaling factor. <br> - scale_tensor: per-channel scaling tensor of type uint64 with shape [1, Cout]; ND format only. <br> - relu_type: activation type; supports RELU/NO_RELU, etc. |

## Return Value

Returns the output tensor from the convolution:
- 1D convolution output shape: (Batch, Cout, Wout)
- 2D convolution output shape: (Batch, Cout, Hout, Wout)
- 3D convolution output shape: (Batch, Cout, Dout, Hout, Wout)

Each output shape dimension is in the range [1, 1000000].

## Constraints

### 1. Shape Validity Constraints
- Input feature map (input_conv): Batch, Cin, Hin, Win, Din dimensions must be in the range [1, 1000000];
- Kernel (weight): Cout, Kh, Kw, Kd dimensions must be in the range [1, 1000000];
- Bias (bias_tensor): shape must equal [Cout], otherwise validation fails;
- Output feature map: H_out, W_out, D_out dimensions must be in the range [1, 1000000].

### 2. Attribute Parameter Validity Constraints
- Basic dimension matching constraints:
  - The number of dimensions in `strides` must match the convolution dimensions (length=2 for 2D conv, length=3 for 3D conv);
  - The number of dimensions in `dilations` must match the convolution dimensions (length=2 for 2D conv, length=3 for 3D conv);
  - The number of dimensions in `paddings` must be 2× the convolution dimensions (length=4 for 2D conv, length=6 for 3D conv);
- Numeric range constraints:
  - `strides` value range: [1, 63];
  - `dilations` value range: [1, 63];
  - `paddings` value range: [0, 255], with each dimension's padding < the corresponding kernel dimension size (e.g., padding_h < Kh, padding_w < Kw);
  - `groups` value range: [1, 65535];
- Kernel constraints:
  - Kh ≤ 255, Kw ≤ 255;
  - Kh × Kw × 32bytes/dtype ≤ 65535; dtype is the number of bits in the `input_conv` data type (e.g., FP16 is 16, FP32 is 32);
- Channel constraints:
  - Cin (input channels) must be divisible by groups;
  - Cout (output channels) must be divisible by groups;
  - CinFmap = CinWeight × groups.

### 3. Cache Space Constraints
- Before calling the `conv` interface, the convolution TileShape split sizes for L1/L0 levels must be configured via `pypto.set_conv_tile_shapes`.

### 4. Feature Support Constraints
- transposed=True (transposed convolution) is not supported and will raise a RuntimeError;
- input_conv/weight only support DT_FP16, DT_BF16, DT_FP32 data types; other types will raise a ValueError;
- The dimensions of `input_conv` and `weight` must match (e.g., if `input_conv` is 4D then `weight` must also be 4D), otherwise a RuntimeError is raised.

## Example

```python
# Basic 2D convolution example (dynamic axis splitting is not currently supported)
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")

# Subgraph merging is not currently supported and must be manually disabled
pypto.set_pass_options(
    cube_l1_reuse_setting={-1: 1},
    cube_nbuffer_setting={-1: 1},
)

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1])

# 2D convolution with bias and ReLU (ReLU is not currently supported)
input_conv = pypto.tensor((1, 32, 8, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 32, 1, 1), pypto.DT_FP16, "weight")
bias = pypto.tensor((32,), pypto.DT_FP16, "bias")
extend_params = {'bias_tensor': bias, 'relu_type': pypto.ReLuType.RELU}

# Subgraph merging is not currently supported and must be manually disabled
pypto.set_pass_options(
    cube_l1_reuse_setting={-1: 1},
    cube_nbuffer_setting={-1: 1},
)

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1],
                   paddings=[0, 0, 0, 0],
                   dilations=[1, 1],
                   extend_params=extend_params)

# 3D convolution example (dynamic axis splitting is not currently supported)
input_conv = pypto.tensor((1, 96, 2, 16, 16), pypto.DT_FP16, "input_conv")
weight = pypto.tensor((32, 96, 1, 1, 1), pypto.DT_FP16, "weight")

# Subgraph merging is not currently supported and must be manually disabled
pypto.set_pass_options(
    cube_l1_reuse_setting={-1: 1},
    cube_nbuffer_setting={-1: 1},
)

out = pypto.conv(input_conv, weight, pypto.DT_FP16,
                   strides=[1, 1, 1],
                   paddings=[0, 0, 0, 0, 0, 0],
                   dilations=[1, 1, 1])
```
