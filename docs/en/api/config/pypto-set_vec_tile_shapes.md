# pypto.set\_vec\_tile\_shapes

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Sets the TileShape sizes used in vector computation.

## Function Prototype

```python
set_vec_tile_shapes(*args: int) -> None
```

## Parameters


| Parameter | Input/Output | Description                                                     |
|-----------|--------------|-----------------------------------------------------------------|
| *args     | Input        | TileShape size for each dimension. A maximum of 4 inputs are allowed. |

## Return Value

void

## Constraints

TileShape must satisfy the following constraints:

Each dimension must be greater than 0.

Assuming TileShape is two-dimensional \{m, n\}, then:

-   （m \> 0）&& \(n \> 0\)

## Example

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)
```

