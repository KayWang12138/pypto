# pypto.get\_vec\_tile\_shapes

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Gets the TileShape sizes used in vector computation.

## Function Prototype

```python
get_vec_tile_shapes() -> List[int]
```

## Parameters

void

## Return Value

Returns the TileShape size for each dimension.

## Constraints

None.

## Example

```python
pypto.get_vec_tile_shapes()
```

