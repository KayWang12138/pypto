# pypto.Tensor.get\_cache\_policy

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Check whether a specific cache policy is enabled.

## Function Prototype

```python
get_cache_policy(self, policy: CachePolicy) -> bool
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| policy    | input        | The cache policy type. |

## Return Value

Whether the cache policy is enabled.

## Constraints

None.

## Example

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
out = t.get_cache_policy(pypto.CachePolicy.PREFETCH)
```

Example result:

```python
output data out: False
```

