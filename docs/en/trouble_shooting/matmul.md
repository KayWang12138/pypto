# MATMUL Component Error Codes

- **Range**: FC3XXX - FC5XXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for MATMUL sub-class operators.

## Error Code Definitions and Usage

The unified definitions for matmul-related error codes can be found in the `/framework/src/interface/utils/matmul_error.h` file.

## Troubleshooting Recommendations

Based on the different ErrorCodes in the logs, refer to the following troubleshooting recommendations:

### FC3000 ERR_PARAM_INVALID
1. **Check tensor basic information**: Confirm that the dimensions, data types, and formats of input/output tensors are all valid.
2. **Check dimension validity**: Confirm that matrix dimensions are positive integers, with no zero or negative dimensions.
3. **Check parameter completeness**: Confirm that all required parameters for the Matmul interface have been provided, with no missing or out-of-bounds values.
4. **Check log context**: Use MATMUL_LOGE logs to view parameter details and locate invalid fields.

### FC3001 ERR_PARAM_MISMATCH
1. **Check input matrix dimensions**: Confirm that the K dimension of matrix A matches the K dimension length of matrix B.
2. **Check data types**: The data types of matrices A, B, and C match the types required by the compute core.
3. **Check transpose configuration**: Confirm that the transposeA / transposeB configuration is consistent with the actual memory layout.
4. **Check log context**: Use MATMUL_LOGE logs to print shape information and locate mismatched items.

### FC3002 ERR_PARAM_UNSUPPORTED
1. **Check data formats**: Confirm that NPU-supported data formats are used (e.g., ND, FRACTAL_Z, etc.).
2. **Check data types**: Confirm that no low-precision/high-precision types unsupported by the current hardware are included.
3. **Check dimension combinations**: Confirm that batch, M/N/K dimensions do not exceed the hardware support limits.
4. **Check log context**: Review the unsupported parameter types and retry with compatible configurations.

### FC4000 ERR_CONFIG_TILE
1. **Check tile partition parameters**: Confirm that M/N/K tile sizes are within the hardware-supported range.
2. **Check tile validity**: Confirm that tile sizes are divisible by the corresponding dimensions, with no illegal zero values.
3. **Check tile strategy**: Confirm that officially recommended tile combinations are used, with no custom illegal tiles.
4. **Check log context**: Obtain invalid tile parameters from the log and correct the tile configuration.

### FC4001 ERR_CONFIG_ALIGNMENT
1. **Check tensor addresses**: Confirm that device addresses are aligned to 16B/32B/64B.
2. **Check tile sizes**: Confirm that tile sizes satisfy hardware alignment constraints.
3. **Check workspace memory**: Confirm that workspace memory is allocated by the unified memory manager.
4. **Check log context**: Review misaligned addresses/sizes and re-request using the memory allocation interface.

### FC4002 ERR_CONFIG_UNSUPPORTED
1. **Check configuration combinations**: Confirm that the tile configuration, data types, and formats are a supported combination.
2. **Check operator modes**: Confirm that incompatible computation modes and precision modes are not mixed.
3. **Check hardware compatibility**: Confirm that the current configuration matches the running NPU hardware model.
4. **Check log context**: Based on log hints, replace with a supported configuration combination.

### FC5000 ERR_RUNTIME_NULLPTR
1. **Check input/output tensors**: Confirm that the passed-in Tensor is non-null and has completed address allocation.
2. **Check context initialization**: Confirm that the matmul context and configuration handle have been properly created.
3. **Check function input parameters**: Confirm that the calling layer has not passed null pointers to the matmul interface.
4. **Check log context**: Locate the null pointer variable and check the upper-level initialization and assignment process.

### FC5001 ERR_RUNTIME_STATE
1. **Check initialization flow**: Confirm that the Matmul context has completed initialization before executing computation.
2. **Check state machine transitions**: Confirm that calls are made in the order: Initialize -> Configure -> Execute -> Release.
3. **Check resource status**: Confirm that dependent NPU resources and workspace have not been released prematurely.
4. **Check log context**: Review the abnormal status code and trace back the process call order.

### FC5002 ERR_RUNTIME_LOGIC
1. **Check execution branches**: Confirm that the computation flow has not entered an undefined/abnormal branch.
2. **Check intermediate results**: Confirm that temporary data and index values during computation meet expectations.
3. **Check invariant constraints**: Confirm that all preconditions for the core computation logic are satisfied.
4. **Check log context**: Use logs to locate abnormal paths and verify the computation logic.


## Troubleshooting Methods

### Troubleshooting Steps

1. **Log persistence**: Enable DEBUG logging and specify the log persistence path:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH=./my_log
```

2. **Check input parameters**: Refer to the corresponding operator or API documentation (e.g., `/docs/api/operation/pypto-matmul.md`, operator descriptions) to confirm whether input and output Tensors meet the Matmul constraint specifications, including Shape, Dtype, Format, etc.

3. **Check tile settings**: Before calling the Matmul operator, `set_cube_tile_shapes()` is executed to set the tile size. Check whether the configured TileShape size meets the tiling constraints. You can check via:

```py
pypto.set_cube_tile_shapes([32, 32], [16, 16], [32, 32])   #[mL0, mL1], [kL0, kL1], [nL0, nL1]
tile_shape_info = pypto.get_cube_tile_shapes()
print(tile_shape_info)
#Output: [32, 32], [16, 16], [32, 32]
```

## Typical Scenarios

Using ERR_CONFIG_ALIGNMENT as an example — this indicates that the input Tensor does not meet alignment constraints. Pay attention to reshape/view, dimension swapping (transpose scenarios), and whether the corresponding dimension alignment requirements have changed. For example, when the Format is TILEOP_NZ (NZ format), the Shape dimensions must satisfy inner-axis 32-byte alignment. When the input matrix is not transposed, the corresponding data layout is [M, K], where the outer axis is M and the inner axis is K. When the input matrix is transposed, the corresponding data layout is [K, M], where the outer axis is K and the inner axis is M.
