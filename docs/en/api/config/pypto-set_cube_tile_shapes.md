# pypto.set\_cube\_tile\_shapes

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

This interface must be called before invoking `pypto.matmul` to set the tile sizes for matrix operations. For specific tile configuration guidance, refer to [Matmul High-Performance Programming](https://gitcode.com/cann/pypto/blob/master/docs/tutorials/debug/matmul_performance_guide.md).


## Function Prototype

```python
set_cube_tile_shapes(m: List[int], k: List[int], n: List[int], enable_split_k: bool = False) -> None
```

## Parameters


| Parameter            | Input/Output | Description                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| m                      | Input      | TileShape (tile shape) partition sizes for the m dimension at L0 and L1 cache levels, corresponding to mL0 and mL1 respectively |
| k                      | Input      | TileShape (tile shape) partition sizes for the k dimension at L0 and L1 cache levels, corresponding to kL0 and kL1 respectively |
| n                      | Input      | TileShape (tile shape) partition sizes for the n dimension at L0 and L1 cache levels, corresponding to nL0 and nL1 respectively |
| enable_split_k         | Input      | Setting True enables the multi-core split-K feature for matmul; False disables it. Default is False. |

## Return Value

void

## Constraints

TileShape must satisfy the following constraints:

-   Alignment constraints:
    -   kL0, kL1, nL0, and nL1 must all be 32-byte aligned (DT\_FP32 input scenarios require 16-element alignment). For example: when the input matrix data type is DT\_FP16, kL0 \* sizeof\(DT\_FP16\) % 32 == 0.
    -   When matrix A has format ND and is transposed (i.e., data layout is \(K, M\)), mL0 must satisfy 32-byte alignment.
    -   When matrices A and B have format NZ, the outer axis tile size must satisfy 16-element alignment, and the inner axis tile size must satisfy 32-byte alignment. For example, in the non-transposed scenario for matrix A, the outer axis is M and the inner axis is K, so mL0 and mL1 must satisfy 16-element alignment, and kL0 and kL1 must satisfy 32-byte alignment.
    -   0 < mL0 <= mL1 and mL1 % mL0 == 0
    -   0 < kL0 <= kL1 and kL1 % kL0 == 0
    -   0 < nL0 <= nL1 and nL1 % nL0 == 0

-   Buffer space constraints:
    -   L0A, L0B, L0C space constraints:
        -   When input dtype is DT\_FP16, DT\_BF16, or DT\_FP32

            CeilAlign\(mL0,16\)\* CeilAlign\(kL0,16\) \* sizeof\(aDtype\) <= L0A\_size

            CeilAlign\(nL0,16\) \* CeilAlign\(kL0,16\)\* sizeof\(bDtype\) <= L0B\_size

            CeilAlign\(mL0,16\)\* CeilAlign\(nL0,16\)\* sizeof\(cDtype\) <= L0C\_size

            where aDtype and bDtype are the input dtypes, and cDtype is DT\_FP32

        -   When input dtype is DT\_INT8

            CeilAlign\(mL0,32\)\* CeilAlign\(kL0,32\) \* sizeof\(aDtype\) <= L0A\_size

            CeilAlign\(nL0,32\) \* CeilAlign\(kL0,32\)\* sizeof\(bDtype\) <= L0B\_size

            CeilAlign\(mL0,32\)\* CeilAlign\(nL0,32\)\* sizeof\(cDtype\) <= L0C\_size

            where aDtype and bDtype are the input dtypes, and cDtype is DT\_INT32

    -   L1 space constraints:

        -   Input dtype is DT\_FP16, DT\_BF16, or DT\_FP32

            CeilAlign\(mL1,16\)\* CeilAlign\(kL1,16\) \* sizeof\(aDtype\) + CeilAlign\(nL1,16\) \* CeilAlign\(kL1,16\) \* sizeof\(bDtype\) <= L1\_size

            where aDtype and bDtype are the input dtypes

        -   Input dtype is DT\_INT8

            CeilAlign\(mL1,32\)\* CeilAlign\(kL1,32\) \* sizeof\(aDtype\) + CeilAlign\(nL1,32\) \* CeilAlign\(kL1,32\) \* sizeof\(bDtype\) <= L1\_size

            where aDtype and bDtype are the input dtypes

        The basic implementation of CeilAlign (element alignment) is:

        ```
        def ceil_align(value, align) {  return ((value + align - 1) // align) * align;}
        ```

Bias scenario constraints:

-   Bias space constraints:
    -   The BTBuffer size is 1 KB, and bias data is fully converted to fp32 upon reaching BTBuffer. The following constraint must be satisfied:

        nL0 \* 4 <= BTBuffer\_size

FixPipe scenario constraints:

-   FixBuffer space constraints:
    -   The FixBuffer size is 2 KB, and scaleTensor data is of type uint64\_t. The following constraint must be satisfied:

        nL0 \* 8 <= FixBuffer\_size

Output must satisfy the following constraints:

-   Format constraint: When the output is in NZ format, the inner axis (N axis) must be 32-byte aligned.

When the input matrix dimensions are 3D or 4D, the enable\_split\_k parameter only supports False; enabling multi-core split-K is not supported.

Supported data types for multi-core split-K scenarios:

-   When input matrix data type is DT\_FP16, out\_dtype can be DT\_FP32.
-   When input matrix data type is DT\_BF16, out\_dtype can be DT\_FP32.
-   When input matrix data type is DT\_INT8, out\_dtype can be DT\_INT32.
-   When input matrix data type is DT\_FP32, out\_dtype can be DT\_FP32.

## Example

```python
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128], True)
```

