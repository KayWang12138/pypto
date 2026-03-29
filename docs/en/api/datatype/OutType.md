# OutType

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

OutType defines the output data type, used to specify the output format of certain operations (such as comparison operations), distinguishing between boolean output and bit-value output.

## Prototype Definition

```python
class OutType(enum.Enum):
     BOOL = ...  # Boolean output type: outputs True or False
     BIT = ...   # Bit output type: outputs a bit value of 0 or 1
```

