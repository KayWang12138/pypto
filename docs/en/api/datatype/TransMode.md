# TransMode

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Ascend 950PR/Ascend 950DT |    √     |

## Description

In the scenario where both the input and output matrices of matmul use the FP32 data type, this setting controls whether TF32 computation is enabled and specifies the rounding mode when TF32 is enabled. When enabled, FP32 data is converted to TF32 during matrix multiplication. <br> TF32 uses 1 sign bit, 8 exponent bits, and 10 mantissa bits, totaling 19 bits for computation. The reduced number of mantissa bits decreases hardware computational load and accelerates computation, but also introduces precision loss. <br> The input data format remains FP32, and the TransMode parameter is used to determine the rounding mode. In general, CAST_ROUND (round to nearest integer, ties away from zero) mode is recommended.

## Prototype Definition

```python
class TransMode(enum.Enum):
     CAST_NONE = ...   # Disable conversion of float data type to TF32 data type
     CAST_RINT = ...   # Round to nearest integer, ties round to even
     CAST_ROUND = ...  # Round to nearest integer, ties round away from zero
```

