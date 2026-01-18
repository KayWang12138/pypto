# IR Converter Tools

This directory contains tools to convert IR (Intermediate Representation) text format to NumPy and PyPTO implementations.

## Overview

The IR converter tools read IR text files (`.ir` format) and generate equivalent implementations in:
- **NumPy**: Pure NumPy code for reference and testing
- **PyPTO**: PyPTO code that can be executed with `@pypto.jit`

## Tools

### ir_to_numpy.py

Converts IR text format to NumPy code.

**Usage:**
```bash
python tools/ir_converter/ir_to_numpy.py examples/ir/example00_basic_operations.ir [output.py]
```

**Example:**
```python
# Input IR (example00_basic_operations.ir)
func.func @test_value(%input_3: tensor<[%b_1, 128], fp32>, ...) {
    statement.op {
        %mul1_res_12 = tensor.mul %loop_tile_11, %scale1_4 : ...
    }
}

# Output NumPy
import numpy as np

def test_value(input_3: np.ndarray, ...):
    """NumPy implementation converted from IR."""
    mul1_res_12 = loop_tile_11 * scale1_4
    return
```

### ir_to_pypto.py

Converts IR text format to PyPTO code.

**Usage:**
```bash
python tools/ir_converter/ir_to_pypto.py examples/ir/example00_basic_operations.ir [output.py]
```

**Example:**
```python
# Input IR (example00_basic_operations.ir)
func.func @test_value(%input_3: tensor<[%b_1, 128], fp32>, ...) {
    statement.op {
        %mul1_res_12 = tensor.mul %loop_tile_11, %scale1_4 : ...
    }
}

# Output PyPTO
import pypto

@pypto.jit
def test_value(input_3: pypto.Tensor, ...):
    """PyPTO implementation converted from IR."""
    mul1_res_12 = loop_tile_11 * scale1_4
    return
```

## Supported Operations

### Binary Operations
- `tensor.mul`, `tensor.add`, `tensor.sub`, `tensor.div`
- `tile.OP_MUL`, `tile.OP_ADD`, `tile.OP_SUB`, `tile.OP_DIV`
- `tensor.OP_SCALAR_ADD`, `tensor.OP_SCALAR_MUL`, etc.

### Unary Operations
- `tensor.neg`, `tensor.abs`, `tensor.exp`, `tensor.sqrt`, `tensor.ln`

### Special Operations
- `tensor.view` - View operation (simplified to assignment)
- `tensor.assemble` / `tile.assemble` - Assemble operation (simplified to assignment)

## Data Type Mapping

| IR Type | NumPy | PyPTO |
|---------|-------|-------|
| `fp32` | `np.float32` | `pypto.float32` |
| `fp64` | `np.float64` | `pypto.float64` |
| `int32` | `np.int32` | `pypto.int32` |
| `int64` | `np.int64` | `pypto.int64` |
| `bool` | `bool` | `bool` |

## Limitations

1. **Control Flow**: For loops and if statements are not yet fully converted
2. **View/Assemble**: Complex view and assemble operations are simplified
3. **Symbolic Dimensions**: Symbolic dimensions (like `%b_1`) are preserved as variable names
4. **Type Inference**: Some type information may be lost in conversion

## Examples

Convert all IR examples:

```bash
# Convert to NumPy
for ir_file in examples/ir/*.ir; do
    python tools/ir_converter/ir_to_numpy.py "$ir_file" "${ir_file%.ir}_numpy.py"
done

# Convert to PyPTO
for ir_file in examples/ir/*.ir; do
    python tools/ir_converter/ir_to_pypto.py "$ir_file" "${ir_file%.ir}_pypto.py"
done
```

## Future Enhancements

- Full support for for loops and if statements
- Better handling of view and assemble operations
- Support for more operation types
- Type inference and validation
- Error checking and validation
