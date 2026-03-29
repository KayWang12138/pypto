# pypto.set_conv_tile_shapes

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Ascend 950PR/Ascend 950DT |    √     |
| Atlas A3 Training Series/Atlas A3 Inference Series |    ×     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    ×     |

## Description

Sets the TileShape (tile shape) sizes for each dimension at the L1/L0 cache levels in convolution (conv) computation, and controls the enable state of the L0TileInfo configuration switch.

## Function Prototype

```python
def set_conv_tile_shapes(tile_l1_info: pypto_impl.TileL1Info, tile_l0_info: pypto_impl.TileL0Info = None) -> None
```

## Parameters

| Parameter       | Input/Output | Description                                   |
|--------------|-----------|--------------------------------------|
| tile_l1_info | Input      | TileShape configuration for convolution computation at the L1 cache level |
| tile_l0_info | Input      | TileShape configuration for convolution computation at the L0 cache level |

## Return Value

void

## Constraints

TileShape must satisfy the following constraints:

**Note: Both L1 and L0 have tileN, but they represent different things. tileL1Info.tileN represents the number of output channels; tileL0Info.tileN represents the size of n at the L0 level.**

- Alignment constraints:

    - Each dimension value in tileL1Info must satisfy the following range constraints:

        - 1 <= tileHin <= Hin (Hin is the actual height of the input feature map)

        - When wout % 16 = 0: 1 <= tileHout <= Hout (Hout is the actual height of the output feature map)

        - When wout % 16 != 0: tileHout = 1

        - 1 <= tileWin <= Win (Win is the actual width of the input feature map)

        - 1 <= tileWout <= CeilAlign(Wout, 16) (Wout is the actual width of the output feature map)

        - When tileHout > 1: tileWout == wout == tileW

        - 1 <= tileCinFmap <= Cin (Cin is the actual number of input feature map channels)

        - tileCinFmap * sizeof(dtype) % 32 == 0

        - 1 <= tileCinWeight <= Cin (Cin is the actual number of weight input channels)

        - tileCinWeight * sizeof(dtype) % 32 == 0

        - 1 <= tileN <= CeilAlign(Cout // groups, 16) (Cout is the actual number of output feature map channels)

        - tileN % 16 == 0

        - tileBatch = 1 (represents the batch count)

    - Each dimension value in tileL0Info must satisfy the following alignment constraints:

        - tileK `C0 <= tileK <= min(kAL1, kBL1)`

        - tileK `tileK % C0 == 0`

        - tileK `kAL1 % tilek == 0`

        - tilek `kBL1 % tilek == 0`

        - tileW must be 16-element aligned: `tileW % 16 == 0`

        - tileW `1 <= tileW <= tileWout`

        - tileH `1 <= tileH <= tileHout`

        - tileN (representing the size of n at L0 level) must be 16-element aligned: `tileN % 16 == 0`

        - tileN `1 <= tileN <= CeilAlign(tileL1Info.tileN, 16)`

        Where:

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

    - L0 and L1 dimension level constraints:

        - 1 <= tileL0Info.tileH <= tileL1Info.tileHout and tileL1Info.tileHout % tileL0Info.tileH == 0

        - 1 <= tileL0Info.tileW <= tileL1Info.tileWout and tileL1Info.tileWout % tileL0Info.tileW == 0

        - 1 <= tileL0Info.tileN <= tileL1Info.tileN

- Buffer space constraints:

    - L0A, L0B, L0C space constraints:

        ```
        CeilAlign(tileH * tileW, 16)* CeilAlign(tileK, C0) * sizeof(dtype) <= L0A_size

        CeilAlign(tileK, C0) * CeilAlign(tileN, 16) * sizeof(dtype) <= L0B_size

        CeilAlign(tileH * tileW, 16)* CeilAlign(tileN, 16) * sizeof(FP32) <= L0C_size
        ```

        Where:

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `L0A_size = 65536 bytes`

        - `L0B_size = 65536 bytes`

        - `L0C_size = 131072 bytes`

        - `ALIGN_SIZE_32 = 32`

    - L1 space constraints:

        ```
        CeilAlign(hinL1 * winL1 * kAL1 * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(nL1 * kBL1 * sizeof(dtype), ALIGN_SIZE_32) + CeilAlign(tileN * sizeof(dtype), ALIGN_SIZE_32) <= L1_size
        ```

        Where:

        - `hinL1 = min((tileHout - 1) * strideH + (Kh - 1) * dilationH + 1, Hin)` (Hin is the height of the input feature map)

        - `winL1 = min((tileWout - 1) * strideW + (Kw - 1) * dilationW + 1, Win)` (Win is the width of the input feature map)

        - `kAL1 = CeilAlign(tileCinFmap * kh * kw, C0)`

        - `kBL1 = CeilAlign(tileCinWeight * kh * kw, C0)`

        - `nL1 = tileN` (number of output channels)

        - `dtype is the data type of input_conv (input matrix)`

        - `C0 = ALIGN_SIZE_32 / sizeof(dtype)`

        - `ALIGN_SIZE_32 = 32`

        - `CeilAlign(value, align) {  return ((value + align - 1) // align) * align;}`

- Special scenario constraints:

    - When `tileL0Info` is not provided, a default `TileL0Info` instance is used automatically, and the L0TileInfo switch is automatically disabled. When a valid `tileL0Info` is provided, the L0TileInfo switch is automatically enabled.

    - The convolution kernel/channel dimension configuration must match the actual input/output channel counts and kernel size of the convolution operator to avoid tile sizes exceeding the operator's dimension range.

## Example

```python
# Construct L1 Tile configuration (ensure all values are within valid ranges)
l1_tile = pypto_impl.TileL1Info(
    tileHin=4,        # must satisfy 1 <= tileHin <= Hin
    tileHout=4,       # must satisfy 1 <= tileHout <= Hout
    tileWin=8,        # must satisfy 1 <= tileWin <= Win
    tileWout=8,       # must satisfy 1 <= tileWout <= Wout
    tileCinFmap=16,   # must satisfy 1 <= tileCinFmap <= Cin
    tileCinWeight=32, # must satisfy 1 <= tileCinWeight <= Cin
    tileN=16,         # must satisfy 1 <= tileN <= Cout
    tileBatch=1       # must satisfy tileBatch = 1
)

# Construct L0 Tile configuration (satisfying alignment constraints)
l0_tile = pypto_impl.TileL0Info(
    tileH=2,   # must satisfy tileH <= tileL1Info.tileHout and tileL1Info.tileHout % tileH == 0
    tileW=8,   # must satisfy tileW <= tileL1Info.tileWout and tileL1Info.tileWout % tileW == 0
    tileK=32,  # must satisfy tileK * sizeof(DT_FP16) % 32 == 0 (32*2=64, 64%32=0)
    tileN=16   # must satisfy tileN % 16 == 0
)

# Set convolution TileShape (enable L0TileInfo)
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile, tile_l0_info=l0_tile)

# Set only L1 TileShape (disable L0TileInfo)
pypto.set_conv_tile_shapes(tile_l1_info=l1_tile)
```
