# pypto.get\_conv\_tile\_shapes

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    ×     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    ×     |

## Description

Retrieves the TileShape sizes configured for convolution (conv) computation, as well as the enable state of the L0TileInfo switch.

## Function Prototype

```python
def get_conv_tile_shapes() -> Tuple[pypto_impl.TileL1Info, pypto_impl.TileL0Info, bool]
```

## Parameters

None.

## Return Value

Returns the TileShape sizes at L0 and L1 cache levels, and whether the L0TileInfo switch is enabled.

## Constraints

None.

## Example

```python
pypto.get_conv_tile_shapes()
```

