# ReduceMode

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

ReduceMode defines the execution mode for reduction operations, used to specify the reduction computation method in multi-threaded or multi-device environments, ensuring correctness and performance of computation results.

## Prototype Definition

```python
class ReduceMode(enum.Enum):
     ATOMIC_ADD = ...  # Atomic addition reduction: uses atomic operations to ensure thread-safe data accumulation in multi-threaded environments
```

