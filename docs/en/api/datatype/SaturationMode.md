# SaturationMode

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

SaturationMode defines how overflow is handled when converting floating-point numbers to integers, controlling the strategy applied when the source data exceeds the representable range of the target integer type, ensuring correctness and predictability of conversion results.

## Prototype Definition

```python
class SaturationMode(enum.Enum):
     OFF = ...   # Truncation mode (default): directly truncates the overflow portion, which may result in overflow
     ON = ...    # Saturation mode: values outside the range are clamped to the maximum or minimum value of the target type
```


