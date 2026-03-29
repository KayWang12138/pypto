# CachePolicy

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

CachePolicy defines the cache policy for Tensors, used to control how data behaves across cache levels, optimizing memory access performance and reducing memory bandwidth consumption.

## Prototype Definition

```python
class CachePolicy(enum.Enum):
     PREFETCH = ...        # Prefetch policy: loads data into cache in advance to reduce access latency
     NONE_CACHEABLE = ...  # Non-cacheable policy: data is not stored in cache and is accessed directly from main memory
```

