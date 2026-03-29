# pypto.Tensor.set\_cache\_policy

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Set the cache attribute of the Tensor.

## Function Prototype

```python
set_cache_policy(self, policy: CachePolicy, value: bool) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                                 |
|-----------|--------------|-----------------------------------------------------------------------------|
| policy    | input        | The cache policy type. Optional values:<br> - CachePolicy.NONE_CACHEABLE: The chip provides L2 Cache capability, but some operator characteristics mean that having an L2 Cache is actually worse than not having one. After configuring this, the tensor will not go through the L2 Cache. Common use cases include:<br>   - Constants similar to weights: if the operator only reads from the output once without reuse, there is no need to enter L2;<br>   - Output shape is too large: the memory first accessed by the downstream operator is not the last output result of the upstream operator; entering L2 instead triggers the downstream operator to write back to the output, degrading performance. |
| value     | input        | Whether to enable the cache policy. |

## Return Value

None

## Constraints

None.

## Example

```python
t = pypto.tensor((16, 16), pypto.DT_FP32)
t.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```

