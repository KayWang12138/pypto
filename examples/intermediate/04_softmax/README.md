# Softmax Example

A softmax implementation demonstrating dynamic shapes, tiling, and loop-based processing.

## Overview

This example shows how to implement softmax from basic operations, including:
- Manual softmax computation
- Dynamic axis marking
- Tiling configuration
- Loop-based processing

## Features

- ✅ Softmax implementation
- ✅ Dynamic shape support
- ✅ Tiling configuration
- ✅ Verified against PyTorch

## Running the Example

### Prerequisites

```bash
# Source CANN environment
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# Set device ID (required for NPU examples)
export TILE_FWK_DEVICE_ID=0
```

### Basic Usage

```bash
# Run all examples
python3 examples/intermediate/04_softmax/softmax.py

# Run specific example by ID
python3 examples/intermediate/04_softmax/softmax.py 1

# List all available examples
python3 examples/intermediate/04_softmax/softmax.py --list
```

## Key Concepts

### Softmax Formula

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

### Implementation Steps

1. Find maximum (for numerical stability)
2. Subtract maximum
3. Compute exponentials
4. Sum exponentials
5. Divide by sum

## See Also

- [Basic Operations](../../beginner/01_basic_operations/) - Start here for basics
- [Custom Activation](../../intermediate/02_custom_activation/) - More custom operations
- [Operations Guide](../../../docs/user/core_concepts/operations.md) - Available operations

