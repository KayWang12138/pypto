# TileOpFormat

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

TileOpFormat defines the tile operation format for Tensors, used to optimize memory access and computation efficiency in different computation modes. It primarily distinguishes between tile formats for dense and sparse computation.

## Prototype Definition

```python
class TileOpFormat(enum.Enum):
     TILEOP_ND = ...  # N-dimensional Tensor, supports standard multi-dimensional array operations
     TILEOP_NZ = ...  # Same as FRACTAL_NZ/NZ — a format obtained by applying padding (pad), splitting (reshape), and transposition (transpose) to the two lowest dimensions of a Tensor (all dimensions of a Tensor, with rightmost being lowest and leftmost being highest)
```

