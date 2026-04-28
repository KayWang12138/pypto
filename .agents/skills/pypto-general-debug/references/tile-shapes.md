# Tile Shape Configuration (DEBUG_GUIDEBOOK.md §9.15)

*Agent-learned patterns from GDR kernel development.*

**Issue:** Fixed tile shapes may not match actual tensor dimensions.

**For simple kernels without loops:**
```python
B, T, H, K = x.shape
pypto.set_vec_tile_shapes(B, H, T, K)
```

**For complex kernels with loops:**
```python
TILE_0 = 16   # first tile dimension
TILE_1 = 4    # second tile dimension
TILE_2 = 8    # third tile dimension
TILE_3 = 32   # fourth tile dimension
pypto.set_vec_tile_shapes(TILE_0, TILE_1, TILE_2, TILE_3)
```

**Rule:** Tile shape values should divide evenly into tensor dimensions for best performance.

## See also

- For matmul, both `set_vec_tile_shapes` AND `set_cube_tile_shapes` are required — see `matmul.md` "Both vec and cube tile shapes needed".
- For `F21004` / `REGISTER_COPY` invalid vec tile errors, see `error-codes.md` §4.
