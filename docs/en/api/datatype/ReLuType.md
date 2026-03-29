# ReLuType

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

ReLuType defines the mode of the ReLU activation function, used to enable or disable the ReLU feature.

## Prototype Definition

```python
class ReLuType(enum.Enum):
     NO_RELU= ...  # Disable ReLU feature
     RELU= ...     # Enable ReLU feature
```

