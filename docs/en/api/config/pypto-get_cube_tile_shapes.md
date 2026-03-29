# pypto.get\_cube\_tile\_shapes

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the TileShape sizes configured for cube computation, as well as the enable state of the multi-core split-K feature switch.

## Function Prototype

```python
get_cube_tile_shapes() -> Tuple[List[int], List[int], List[int], bool]:
```

## Parameters

void

## Return Value

Returns the TileShape sizes in the m, k, and n directions, as well as whether the multi-core split-K feature is enabled.

## Constraints

None.

## Example

```python
pypto.get_cube_tile_shapes()
```

