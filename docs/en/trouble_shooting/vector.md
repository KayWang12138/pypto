# VECTOR Component Error Codes

- **Range**: FC0XXX - FC2XXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for VECTOR sub-class operators.

## Error Code Definitions and Usage

The unified definitions for vector-related error codes can be found in the `/framework/src/interface/utils/vector_error.h` file.

## Troubleshooting Recommendations

Based on the different ErrorCodes in the logs, refer to the following troubleshooting recommendations:

### FC0000 ERR_PARAM_INVALID
1. **Check tensor basic information**: Confirm that the dimensions, data types, and formats of input/output tensors are all valid.
2. **Check dimension validity**: Confirm that vector dimensions are positive integers, with no zero or negative dimensions.
3. **Check parameter completeness**: Confirm that all required parameters for the Vector interface have been provided, with no missing or out-of-bounds values.
4. **Check log context**: Use the relevant logs to view parameter details and locate invalid fields.

### FC0001 ERR_PARAM_DTYPE_UNSUPPORTED
1. **Check data types**: Confirm that NPU-supported data types are used (e.g., FP32, FP16, INT32, etc.).
2. **Check type combinations**: Confirm that the data types of multiple tensors involved in the operation match and are all supported.
3. **Check operation support**: Confirm that the current operation supports the specified data type.
4. **Check log context**: Review the unsupported data type and retry with a compatible type.

### FC1000 ERR_CONFIG_TILE
1. **Check tile partition parameters**: Confirm that tile sizes are within the hardware-supported range.
2. **Check tile validity**: Confirm that tile sizes are divisible by the corresponding dimensions, with no illegal zero values.
3. **Check tile strategy**: Confirm that officially recommended tile combinations are used, with no custom illegal tiles.
4. **Check log context**: Obtain invalid tile parameters from the log and correct the tile configuration.

### FC1001 ERR_CONFIG_ALIGNMENT
1. **Check tensor addresses**: Confirm that device addresses are aligned to 16B/32B.
2. **Check tile sizes**: Confirm that tile sizes satisfy hardware alignment constraints.
3. **Check dimension alignment**: Confirm that vector dimensions meet the hardware-required alignment conditions (e.g., 32-byte alignment).
4. **Check log context**: Review misaligned addresses/sizes and adjust tensor shapes or re-request using the memory allocation interface.

### FC2000 ERR_RUNTIME_NULLPTR
1. **Check input/output tensors**: Confirm that the passed-in Tensor is non-null and has completed address allocation.
2. **Check context initialization**: Confirm that the context and configuration handle required for vector operations have been properly created.
3. **Check function input parameters**: Confirm that the calling layer has not passed null pointers to the vector interface.
4. **Check log context**: Locate the null pointer variable and check the upper-level initialization and assignment process.

### FC2001 ERR_RUNTIME_LOGIC
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

2. **Check input parameters**: Refer to the corresponding operator or API documentation to confirm whether input and output Tensors meet the Vector constraint specifications, including Shape, Dtype, Format, etc.

3. **Check tile settings**: Before calling the Vector operator, the relevant tile settings are executed. Check whether the configured TileShape size meets the tiling constraints.

## Typical Scenarios

Using ERR_CONFIG_ALIGNMENT as an example — this indicates that the input Tensor does not meet alignment constraints. Pay attention to reshape/view, dimension swapping (transpose scenarios), and whether the corresponding dimension alignment requirements have changed. For example, when performing vector operations, some operations may require the last dimension of the input tensor to satisfy 32-byte alignment. Ensure that the shape of the input data meets this requirement.
