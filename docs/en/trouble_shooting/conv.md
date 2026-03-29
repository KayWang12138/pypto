# CONV Component Error Codes

(To be supplemented)

- **Range**: FC6XXX - FC8XXX
- This document describes the error code definitions, scenario descriptions, and troubleshooting recommendations for CONV sub-class operators.

## Error Code Definitions and Usage

The unified definitions for related error codes can be found in the `framework/src/interface/utils/conv_error.h` file.
---

## Error Code Definitions and Scenario Descriptions

### 1. Operation (Operation illegal interception errors, `ConvError::Operation`, FC61xxx)

| Scene Enum | Error Code | Error Stage | Scenario Description |
|---------|------|----------------------------------|----------|
| `INPUT_INVALID` | **FC6101** | `conv.operation.checkinput` | Operation validates that input parameters are invalid (dimensions, shape, data type, etc.). |
| `OVER_BUFFER_LIMIT` | **FC6102** | `conv.operation.checkweight` | Operation validates that the buffer limit is exceeded. |
| `UNKNOWN` | **FC6199** | `conv.operation.reserved` | Reserved error code for unknown errors in the Operation stage. |

### 2. Tile Partitioning (Tile graph partitioning, `ConvError::ExpandFunction`, FC62xxx)

| Scene Enum | Error Code | Error Stage | Scenario Description |
|---------|------|----------|------|
| `EXPANDFUNC_TENSOR_OP_NULLPTR` | **FC6201** | `conv.expandfunc.tensor_nullptr` | Tile graph partitioning: null pointer error for tensor graph processing node. |
| `EXPANDFUNC_TENSOR_ATTR_GET_FAILED` | **FC6202** | `conv.expandfunc.get_attr` | Tile graph partitioning: failed to get tensor graph node attribute. |
| `EXPANDFUNC_TILE_OP_NULLPTR` | **FC6203** | `conv.expandfunc.tile_nullptr` | Tile graph partitioning: null pointer error for newly generated tile graph node. |
| `EXPANDFUNC_PARAMS_INVALID` | **FC6204** | `conv.expandfunc.params_check` | Tile graph partitioning: parameter mismatch error (dimensions, types, tile block configuration). |
| `EXPANDFUNC_INNER_STATUS_FAILED` | **FC6205** | `conv.expandfunc.check_status` | Tile graph partitioning: abnormal return value from internal functional function. |
| `UNKNOWN` | **FC6299** | `conv.operation.reserved` | Reserved error code for unknown errors in the ExpandFunc tile graph partitioning stage. |

### 3. CodeGen

| Scene Enum | Error Code | Error Stage | Scenario Description |
|---------|------|----------|------|
| `CODEGEN_GET_ATTR_FAILED` | **FC6301** | `conv.codegen.get_attr` | Codegen code generation: failed to get tensor graph node attribute. |
| `CODEGEN_CHECK_ATTR_INVALID` | **FC6302** | `conv.codegen.check_attr` | Codegen code generation: tensor graph node attribute validation is invalid. |
| `CODEGEN_CHECK_DIM_INVALID` | **FC6303** | `conv.codegen.check_dim` | Codegen code generation: shape/offset validation dim is invalid. |
| `UNKNOWN` | **FC6399** | `conv.operation.reserved` | Reserved error code for unknown errors in the Codegen code generation stage. |

### 4. TileOp

| Scene Enum | Error Code | Error Stage | Scenario Description |
|---------|------|----------|------|
| `TILEOP_TENSOR_FORMAT_FAILED` | **FC6401** | `conv.tileop.check_tensor_format` | TileOp: tensor hardware FORMAT validation failed. |
| `TILEOP_SHAPE_SIZE_FAILED` | **FC6402** | `conv.tileop.check_shape_size` | TileOp: shape size validation failed. |
| `TILEOP_STC_SHAPE_INVALID` | **FC6403** | `conv.tileop.check_stc_shape` | TileOp: static shape is invalid. |
| `TILEOP_INDEX_INVALID` | **FC6404** | `conv.tileop.check_index` | TileOp: index validation for getting shape/stride is invalid. |
| `UNKNOWN` | **FC6499** | `conv.operation.reserved` | Reserved error code for unknown TileOp errors. |

---

## Troubleshooting Recommendations

### Operation shape/TileShape Interception Compilation Errors
Refer to the constraint documentation for reference: `docs/api/config/pypto-set_conv_tile_shapes.md`


### Pass Graph Stage Interception Compilation Errors
1. Enable compilation debug mode, dump pass stage graphs, configure `debug_options={"compile_debug_mode": 1}`
```python
@pypto.frontend.jit(debug_options={"compile_debug_mode": 1})
def conv_kernel()
```

2. Re-run the problematic test case. A dump result with the corresponding timestamp will be generated under output. Based on the graph stage shown in the error log, use pto-toolkit to open and view the dump graph before the executing graph stage, and troubleshoot the Tile subgraph cut from the conv operation.
