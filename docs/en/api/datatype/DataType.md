# DataType

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

DataType is an enumeration class in the PTO framework used to represent Tensor data types. It defines all supported data types, including integers, floating-point numbers, booleans, and more. As the core type identifier for Tensor operations, DataType is used to specify the storage format and computation precision of a Tensor.

## Prototype Definition

```python
class DataType(enum.Enum):
     ...  # Enumeration class definition containing all supported data types

 # Data type constant definitions
 DT_INT4 = ...     # 4-bit signed integer, occupies fractional byte memory
 DT_INT8 = ...     # 8-bit signed integer, occupies 1 byte of memory
 DT_INT16 = ...    # 16-bit signed integer, occupies 2 bytes of memory
 DT_INT32 = ...    # 32-bit signed integer, occupies 4 bytes of memory
 DT_INT64 = ...    # 64-bit signed integer, occupies 8 bytes of memory
 DT_FP8 = ...      # 8-bit floating-point, used for low-precision computation
 DT_FP16 = ...     # 16-bit half-precision floating-point, occupies 2 bytes of memory
 DT_FP32 = ...     # 32-bit single-precision floating-point, occupies 4 bytes of memory
 DT_BF16 = ...     # 16-bit Brain Float format, occupies 2 bytes of memory
 DT_HF4 = ...      # 4-bit Half Float format, occupies 1 byte of memory
 DT_HF8 = ...      # 8-bit Half Float format, occupies 1 byte of memory
 DT_FP8E4M3 = ...  # 8-bit floating-point, 4 exponent bits, 3 mantissa bits, occupies 1 byte of memory
 DT_FP8E5M2 = ...  # 8-bit floating-point, 5 exponent bits, 2 mantissa bits, occupies 1 byte of memory
 DT_FP8E8M0 = ...  # 8-bit floating-point, 8 exponent bits, 0 mantissa bits, occupies 1 byte of memory
 DT_UINT8 = ...    # 8-bit unsigned integer, occupies 1 byte of memory
 DT_UINT16 = ...   # 16-bit unsigned integer, occupies 2 bytes of memory
 DT_UINT32 = ...   # 32-bit unsigned integer, occupies 4 bytes of memory
 DT_UINT64 = ...   # 64-bit unsigned integer, occupies 8 bytes of memory
 DT_BOOL = ...     # Boolean type, occupies 1 byte of memory
 DT_DOUBLE = ...   # 64-bit double-precision floating-point, occupies 8 bytes of memory
```

## Constraints

-   Only Ascend 950PR/Ascend 950DT support DT_FP8E4M3, DT_FP8E5M2, and DT_FP8E8M0 types

