# LogBaseType

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

LogBaseType defines the base type for logarithmic operations, used to specify the base of the logarithm function. It supports the commonly used natural logarithm, base-2 logarithm, and base-10 logarithm.

## Prototype Definition

```python
class LogBaseType(enum.Enum):
     LOG_E = ...   # Natural logarithm base, logarithm with base e (approximately 2.718)
     LOG_2 = ...   # Base-2 logarithm, commonly used in information theory and computer science
     LOG_10 = ...  # Base-10 logarithm, commonly used in scientific computation and engineering
```

