# Loop Feature Example

These examples briefly illustrate:  
- Basic Loop Usage: including the input order of start, end, and step parameters
- Loop Begin and End Function: including the judgment of the loop function's begin and end
- Loop Unroll Function: including the interface of the unroll function
- Loop OP Position Rule: including the placement rule for operations: they can only be placed in the innermost loop
- Loop Compile Phase Print Feature: The print behavior within loops executes only during compilation, cannot print real variable values, and its execution count is related to the number of paths.

## Overview

These examples mainly remind users of the loop function's features to ensure correct usage.  


## Features

- ✅ Basic Loop Usage  
- ✅ Loop Begin and End Function  
- ✅ Loop Unroll Function  
- ✅ Loop OP Position Rule  
- ✅ Loop Compile Phase Print Feature

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
python3 examples/intermediate/05_loop/loop_feature.py

# Run specific example by ID
python3 examples/intermediate/05_loop/loop_feature.py 1

# List all available examples
python3 examples/intermediate/05_loop/loop_feature.py --list
```

## Key Concepts


### Implementation Steps

1. Define loop bounds: specify that the loop function sets parameters in the order of start, end, step
2. Insert loop begin and end marker: it can determine if currently at the start or end of loop execution
3. Place loop body operations following OP position rules
4. Apply loop unrolling if needed
5. Demonstrate the feature that loop's print executes during compilation

## See Also

- [Basic Operations](../../beginner/01_basic_operations/) - Start here for basics
- [Custom Activation](../../intermediate/02_custom_activation/) - More custom operations
- [Operations Guide](../../../docs/user/core_concepts/operations.md) - Available operations

