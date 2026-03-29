# ScatterMode

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

ScatterMode defines the reduce mode of the scatter function

## Prototype Definition

```python
class ScatterMode(enum.Enum):
     None = ...     # Data transfer only
     ADD = ...      # Addition mode
     MULTIPLY = ... # Multiplication mode
```

