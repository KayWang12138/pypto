# pypto.cast

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Performs precision conversion based on the data types of the source and destination Tensors. If the destination operand is an integer type and the source value exceeds the representable range of that integer type, the result is the maximum or minimum value of the destination type.

Before understanding precision conversion rules, it is important to understand the floating-point representation format and binary rounding rules:

-   Floating-point representation format
    -   DT\_FP16 is 16 bits total, comprising 1 sign bit (S), 5 exponent bits (E), and 10 mantissa bits (M).

        When E is neither all zeros nor all ones, the represented value is:

        \(-1\)<sup>S</sup>  \* 2<sup>E - 15</sup>  \* \(1 + M\)

        When E is all zeros, the represented value is:

        \(-1\)<sup>S</sup>  \* 2<sup>-14</sup>  \* M

        When E is all ones: if M is all zeros, the represented value is ±inf (depending on the sign bit); if M is not all zeros, the represented value is nan.

        ![](../figures/pypto.cast.png)

        In the above figure, S=0, E=15, M = 2<sup>-1</sup>  + 2<sup>-2</sup>, so the represented value is 1.75.

    -   DT\_FP32 is 32 bits total, comprising 1 sign bit (S), 8 exponent bits (E), and 23 mantissa bits (M).

        When E is neither all zeros nor all ones, the represented value is:

        \(-1\)<sup>S</sup>  \* 2<sup>E - 127</sup>  \* \(1 + M\)

        When E is all zeros, the represented value is:

        \(-1\)<sup>S</sup>  \* 2<sup>-126</sup>  \* M

        When E is all ones: if M is all zeros, the represented value is ±inf (depending on the sign bit); if M is not all zeros, the represented value is nan.

        ![](../figures/pypto.cast-0.png)

        In the above figure, S = 0, E = 127, M = 2<sup>-1</sup>  + 2<sup>-2</sup>, so the represented value is 1.75.

    -   DT\_BF16 is 16 bits total, comprising 1 sign bit (S), 8 exponent bits (E), and 7 mantissa bits (M).

        When E is neither all zeros nor all ones, the represented value is:

        \(-1\)<sup>S</sup>  \* 2<sup>E - 127</sup>  \* \(1 + M\)

        When E is all zeros, the represented value is:

        \(-1\)<sup>S</sup>  \* 2<sup>-126</sup>  \* M

        When E is all ones: if M is all zeros, the represented value is ±inf (depending on the sign bit); if M is not all zeros, the represented value is nan.

        ![](../figures/pypto.cast-1.png)

        In the above figure, S = 0, E = 127, M = 2<sup>-1</sup>  + 2<sup>-2</sup>, so the represented value is 1.75.

-   Binary rounding rules are similar to decimal rounding. Details are as follows:

    ![](../figures/pypto.cast-2.png)

    -   In CAST\_RINT mode: if the first bit of the portion to be rounded is 0, no carry; if the first bit is 1 and subsequent bits are not all zeros, carry; if the first bit is 1 and subsequent bits are all zeros, no carry if the last bit of M is 0, carry if the last bit of M is 1.

    -   In CAST\_FLOOR mode: if S is 0, no carry; if S is 1, no carry when the portion to be rounded is all zeros, otherwise carry.
    -   In CAST\_CEIL mode: if S is 1, no carry; if S is 0, no carry when the portion to be rounded is all zeros; otherwise carry.
    -   In CAST\_ROUND mode: if the first bit of the portion to be rounded is 0, no carry; otherwise carry.
    -   In CAST\_TRUNC mode: always no carry.
    -   In CAST\_ODD mode: if the portion to be rounded is all zeros, no carry; if the portion to be rounded is not all zeros, no carry when the last bit of M is 1, carry when the last bit of M is 0.

## Function Prototype

```python
cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE,
     satmode: SaturationMode = SaturationMode.OFF) -> Tensor
```

## Parameters


| Parameter      | Input/Output | Description                                                                 |
|----------------|--------------|-----------------------------------------------------------------------------|
| input          | input        | Source operand. <br> Supported type: Tensor. <br> Tensor supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64. <br> Empty Tensor not supported; shape supports only 1–4 dimensions; shape size must not exceed 2147483647 (i.e., INT32_MAX). |
| dtype          | input        | The target data type after precision conversion. <br> Supported data types: DT_FP32, DT_FP16, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64. |
| CastMode       | input        | Source operand enum type used to control the precision conversion processing mode. For details, see: [CastMode](../datatype/CastMode.md). <br> Default is CAST_NONE; for conversions between common types, the framework automatically converts and aligns with torch. See Constraints for details. |
| SaturationMode | input        | Saturation mode enum type used to control overflow handling when converting floating-point numbers to integers. For details, see: [SaturationMode](../datatype/SaturationMode.md). <br> Default is OFF (truncation mode). When set to ON, values exceeding the target type range are clamped to the maximum or minimum value (saturating truncation). See Constraints for details. |

## Constraints

1.  If the destination operand is an integer type and the source value exceeds the representable range, the result is the maximum or minimum value of that integer type. For example, when converting DT\_FP16 to DT\_INT8, if the input is 130.0, the output will be 127 (the upper bound of DT\_INT8).
2.  The following conversions are supported:
    1.  DT\_FP16 to DT\_FP32\\DT\_INT32\\DT\_INT16\\DT\_INT8\\DT\_UINT8
    2.  DT\_BF16 to DT\_FP32\\DT\_INT32
    3.  DT\_INT32 to DT\_FP32\\DT\_INT16\\DT\_INT64
    4.  DT\_FP32 to DT\_BF16\\DT\_FP16\\DT\_INT16\\DT\_INT32\\DT\_INT64
    5.  DT\_UINT8 to DT\_FP16
    6.  DT\_INT8 to DT\_FP16
    7.  DT\_INT16 to DT\_FP32\\DT\_FP16
    8.  DT\_INT64 to DT\_FP32\\DT\_INT32

3.  Precision conversion processing modes (CastMode) are supported. The default modes are as follows:
    1.  DT\_FP32 -\> DT\_FP16\\DT\_BF16 : CAST\_RINT, aligned with Torch.
    2.  DT\_FP16\\DT\_BF16\\DT\_INT32 -\> DT\_FP32 : aligned with Torch.
    3.  DT\_FP32 -\> DT\_INT32 :  CAST\_TRUNC, see Constraint 1.
    4.  DT\_FP16 -\> DT\_INT8:  CAST\_TRUNC, see Constraint 1.
    5.  DT\_INT32-\> DT\_FP16 : aligned with Torch.

4.  When the source and destination types are the same, in some scenarios a no-op may be generated and precision is not guaranteed.

5.  Saturation mode defaults:
    1.  FP16→UINT8, FP16→INT8, FP32→INT16, FP16→INT16, INT64→INT32, INT32→INT16 conversions default to OFF mode (truncation/wrap-around) for PyTorch compatibility. When cast is used in quantization scenarios or other requirements, users can set these 6 conversion scenarios to ON mode (saturation) as needed.
    2.  Other conversions default to ON mode (saturation); OFF mode is not supported.

## Example

### TileShape Configuration Example

Before calling this operation interface, set the TileShape via set_vec_tile_shapes.

The TileShape dimensions should be consistent with the output.

For example, if input shape is [m, n] and output is [m, n], setting TileShape to [m1, n1] means m1 and n1 are used to tile the m and n axes respectively.

```python
pypto.set_vec_tile_shapes(4, 16)
```

### Interface Call Example

```python
x = pypto.tensor([2], pypto.DT_FP32)
y = pypto.cast(x, pypto.DT_FP16)
```

Example result:

```python
Input data x: [2.0, 3.0] # x.dtype: pypto.DT_FP32

Output data y: [2.0, 3.0] # y.dtype: pypto.DT_FP16
```

#### Using Saturation Mode (recommended for floating-point to integer conversion)

```python
# Example 1: FP16 to INT8, using saturation mode to prevent overflow
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
# Output: [127, -128, 50]

# Example 2: FP16 to INT8, using truncation mode
x = pypto.tensor([300.0, -300.0, 50.0], pypto.DT_FP16)
y = pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.OFF)
# Output: [44, -44, 50]
```
