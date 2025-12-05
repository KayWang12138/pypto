# PyPTO Examples

A comprehensive collection of examples demonstrating PyPTO capabilities, organized by difficulty level.

## Overview

This directory contains examples that show how to use PyPTO for various tasks:
- **Beginner**: Fundamental operations and concepts
- **Intermediate**: Neural network components and patterns
- **Advanced**: Complex architectures and optimization
- **Models**: Real-world model implementations

## Quick Start

1. **New to PyPTO?** Start with [Beginner Examples](beginner/README.md)
2. **Building neural networks?** Check out [Intermediate Examples](intermediate/README.md)
3. **Need advanced patterns?** See [Advanced Examples](advanced/README.md)
4. **Looking for real-world usage?** Explore [Model Examples](models/README.md)

## Directory Structure

```
examples/
├── beginner/              # For users new to PyPTO
│   └── 01_basic_operations/
├── intermediate/          # For users familiar with basics
│   ├── 01_layer_normalization/
│   ├── 02_custom_activation/
│   ├── 03_ffn_module/
│   └── 04_softmax/
├── advanced/              # For experienced users
│   ├── 01_attention/
│   └── 02_multi_function/
└── models/                # Real-world model implementations
    └── qwen3/
```

## Learning Paths

### Path 1: Quick Start (1-2 hours)
1. [Basic Operations](beginner/01_basic_operations/) - Learn fundamentals
2. [Layer Normalization](intermediate/01_layer_normalization/) - Understand normalization
3. [Custom Activation](intermediate/02_custom_activation/) - Build custom functions

### Path 2: Neural Networks (3-4 hours)
1. Complete Path 1
2. [FFN Module](intermediate/03_ffn_module/) - Complete feed-forward network
3. [Attention](advanced/01_attention/) - Attention mechanism
4. [Multi-Function Module](advanced/02_multi_function/) - Function composition

### Path 3: Advanced Usage (5-6 hours)
1. Complete Path 2
2. [Model Examples](models/qwen3/) - Real-world implementations
3. Experiment with your own architectures

## Example Categories

### 🟢 [Beginner Examples](beginner/README.md)

Perfect for getting started:
- [Basic Operations](beginner/01_basic_operations/) - Core PyPTO operations

### 🟡 [Intermediate Examples](intermediate/README.md)

Build on the basics:
- [Layer Normalization](intermediate/01_layer_normalization/) - Normalization techniques
- [Custom Activation](intermediate/02_custom_activation/) - Building custom activations
- [FFN Module](intermediate/03_ffn_module/) - Complete feed-forward network
- [Softmax](intermediate/04_softmax/) - Softmax implementation

### 🔴 [Advanced Examples](advanced/README.md)

Complex patterns and architectures:
- [Attention Mechanism](advanced/01_attention/) - Scaled dot-product attention
- [Multi-Function Module](advanced/02_multi_function/) - Function composition

### 🏭 [Model Examples](models/README.md)

Real-world implementations:
- [Qwen3](models/qwen3/) - Qwen3 model components

## Prerequisites

### System Requirements

- **NPU Device**: Ascend NPU (e.g., Atlas series)
- **CANN**: Ascend CANN toolkit (latest version)
- **Python**: 3.7+
- **PyTorch**: Compatible version with torch_npu

### Software Setup

1. **Install CANN**:
   ```bash
   source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
   ```

2. **Install PyPTO**:
   ```bash
   cd pypto-dev
   python3 build.py --build-whl
   pip install dist/pypto-*.whl
   ```

3. **Verify Setup**:
   ```bash
   npu-smi info  # Check NPU availability
   python3 -c "import pto; print('PyPTO installed')"
   ```

## Running Examples

### Prerequisites

```bash
# Source CANN environment
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# Set device ID (required for NPU examples)
export TILE_FWK_DEVICE_ID=0
```

### Basic Usage

All example scripts support running all examples or selecting specific ones:

```bash
# Run all examples
python3 examples/beginner/01_basic_operations/basic_operations.py

# Run specific example by ID
python3 examples/beginner/01_basic_operations/basic_operations.py 2  # Element-wise Operations

# List all available examples
python3 examples/beginner/01_basic_operations/basic_operations.py --list
```

### Running by Category

```bash
# Beginner examples
python3 examples/beginner/01_basic_operations/basic_operations.py

# Intermediate examples
python3 examples/intermediate/01_layer_normalization/layer_norm.py
python3 examples/intermediate/02_custom_activation/custom_activation.py
python3 examples/intermediate/03_ffn_module/ffn_module_example.py
python3 examples/intermediate/04_softmax/softmax.py

# Advanced examples
python3 examples/advanced/01_attention/attention.py
python3 examples/advanced/02_multi_function/multi_function_module.py

# Model examples
python3 examples/models/qwen3/add_rms_norm.py
```

## Example Comparison

| Example | Difficulty | Time | Category |
|---------|-----------|------|----------|
| [Basic Operations](beginner/01_basic_operations/) | ⭐ | 15-20 min | Beginner |
| [Layer Normalization](intermediate/01_layer_normalization/) | ⭐⭐ | 20-30 min | Intermediate |
| [Custom Activation](intermediate/02_custom_activation/) | ⭐⭐ | 20-30 min | Intermediate |
| [FFN Module](intermediate/03_ffn_module/) | ⭐⭐ | 30-45 min | Intermediate |
| [Attention](advanced/01_attention/) | ⭐⭐⭐ | 30-45 min | Advanced |
| [Multi-Function Module](advanced/02_multi_function/) | ⭐⭐⭐ | 30-45 min | Advanced |

## Common Patterns

### Pattern 1: Basic JIT Function

```python
x_torch = ...
y_torch = ...
# convert to pypto tensors
x = pypto.from_torch(x_torch)
y = pypto.from_torch(y_torch)
inputs = [x]
outputs = [y]
@pypto.jit
def my_function(inputs, outputs):
    x = inputs[0]
    y = outputs[0]
    pypto.set_vec_tile_shapes(32, 32)
    y[:] = pypto.operation(x)
```

### Pattern 2: Dynamic Shapes

```python
# create torch tensors
x_torch = ...
y_torch = ...
# convert to pypto tensors
# Mark batch dimension as dynamic
x = pypto.from_torch(x_torch, dynamic_axis=[0])
y = pypto.from_torch(y_torch)
inputs = [x]
outputs = [y]
@pypto.jit
def my_function(inputs, outputs):
    x = inputs[0]
    y = outputs[0]
    pypto.set_codegen_options(support_dynamic_unaligned=True)
    # ... rest of function
```

### Pattern 3: Function Composition

```python
# Function 1
@pypto.jit
def function1(inputs, outputs):
    # ...

# Function 2
@pypto.jit
def function2(inputs, outputs):
    # ...

# Compose
function1([x], [intermediate])
function2([intermediate], [out])
```

## Troubleshooting

### Common Issues

1. **"Device not found"**
   - Check NPU availability: `npu-smi info`
   - Verify device ID: `export TILE_FWK_DEVICE_ID=0`

2. **"Compilation failed"**
   - Check tensor shapes are compatible
   - Verify tiling configuration
   - Ensure CANN environment is sourced

3. **"Result mismatch"**
   - Check data types match
   - Verify operations are correct
   - Consider numerical precision (FP16/BF16)

4. **"Memory allocation failed"**
   - Reduce batch size or sequence length
   - Check available NPU memory

### Getting Help

- **Documentation**: See [User Guide](../docs/user/README.md)
- **Examples**: Each example has a detailed README
- **Troubleshooting**: See [Troubleshooting Guide](../docs/user/reference/troubleshooting.md)

## Additional Resources

### Documentation

- [Installation Guide](../docs/user/getting_started/installation.md)
- [Quick Start Guide](../docs/user/getting_started/quick_start.md)
- [Core Concepts](../docs/user/core_concepts/tensors.md)
- [Operations Guide](../docs/user/core_concepts/operations.md)
- [Functions & JIT](../docs/user/core_concepts/functions_jit.md)
- [Advanced Topics](../docs/user/advanced/advanced_topics.md)

## Contributing

Want to add an example? Here's what to include:

1. **Example Code**: Clear, well-commented code with tests
2. **Documentation**: README with overview, usage, and examples
3. **Update This README**: Add to appropriate category

## License

All examples are part of the CANN Open Software and are licensed under the CANN Open Software License Agreement Version 2.0.

---

**Happy Coding with PyPTO! 🚀**
