# CastMode

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

CastMode defines the rounding mode used during data type conversion, controlling how precision is handled during floating-point conversions to ensure accuracy of conversion results.

## Prototype Definition

```python
class CastMode(enum.Enum):
     CAST_NONE = ...   # No conversion mode: truncates directly without rounding
     CAST_RINT = ...   # Round to nearest integer, ties round to even
     CAST_ROUND = ...  # Round to nearest integer, ties round away from zero
     CAST_FLOOR = ...  # Round down, towards negative infinity
     CAST_CEIL = ...   # Round up, towards positive infinity
     CAST_TRUNC = ...  # Truncation rounding, towards zero
     CAST_ODD = ...    # Round to odd, Von Neumann rounding
```

