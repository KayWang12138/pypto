# Qwen3 Model Examples

Model-specific examples demonstrating PyPTO implementations for Qwen3 architecture components.

## Overview

This directory contains PyPTO implementations of key components from the Qwen3 model architecture, including:

- **Add RMSNorm**: Residual connection with RMS normalization
- **Attention Pre-processing**: Pre-attention computation
- **Attention**: Scaled dot-product attention mechanism
- **FFN MLP**: Feed-forward network with Mixture-of-Experts (MoE) support
- **Expert Selection**: Expert routing for MoE models
- **Token Table**: Token accumulation table management

These examples demonstrate real-world usage of PyPTO for production model components.

## Files

### `add_rms_norm.py`
Implements add + RMSNorm operation with dynamic unaligned support.

**Features**:
- Residual connection
- RMS normalization
- Dynamic batch size support
- Unaligned dimension handling

### `attention_pre.py`
Pre-attention processing for Qwen3 attention mechanism.

**Features**:
- Query, Key, Value preparation
- Attention configuration
- Tiling optimization

### `qwen3_attention.py`
Complete attention mechanism implementation.

**Features**:
- Scaled dot-product attention
- Multi-head attention support
- Dynamic sequence length support

### `ffn_mlp.py`
Feed-forward network with Mixture-of-Experts support.

**Features**:
- MoE FFN implementation
- Expert token handling
- Token accumulation tables

### `select_experts.py`
Expert selection for MoE routing.

**Features**:
- Expert routing logic
- Dynamic expert selection
- Token-to-expert mapping

### `token_table.py`
Token accumulation table management.

**Features**:
- Token table operations
- Expert token tracking
- Accumulation management

## Prerequisites

- PyPTO installed
- CANN environment configured
- NPU device available
- PyTorch and torch_npu installed
- Understanding of Qwen3 architecture

## Running the Examples

### Prerequisites

```bash
# Source CANN environment
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# Set device ID (required for NPU examples)
export TILE_FWK_DEVICE_ID=0
```

### Basic Usage

Each example script supports running all examples or selecting specific ones:

```bash
# Run all examples in a script
python3 examples/models/qwen3/add_rms_norm.py
python3 examples/models/qwen3/attention_pre.py
python3 examples/models/qwen3/qwen3_attention.py
python3 examples/models/qwen3/ffn_mlp.py
python3 examples/models/qwen3/select_experts.py
python3 examples/models/qwen3/token_table.py

# Run specific example by ID
python3 examples/models/qwen3/add_rms_norm.py 1
python3 examples/models/qwen3/attention_pre.py 1

# List all available examples
python3 examples/models/qwen3/add_rms_norm.py --list
```

### Available Examples

- **add_rms_norm.py**: Add + RMSNorm operation
- **attention_pre.py**: Attention pre-processing with RMSNorm and RoPE
- **qwen3_attention.py**: Flash Attention with Paged Attention format
- **ffn_mlp.py**: Qwen3 Feed-Forward Network with MoE support
- **select_experts.py**: Expert selection with top-K and renormalization
- **token_table.py**: Token accumulation table for MoE models

## Key Concepts

### Dynamic Unaligned Support

Many Qwen3 components use dynamic unaligned dimensions:

```python
pypto.set_codegen_option("support_dynamic_unaligned", True)
pypto.set_host_option("only_codegen", True)
```

This enables handling of variable batch sizes and sequence lengths that may not be aligned to tile boundaries.

### MoE (Mixture-of-Experts)

The Qwen3 model uses MoE architecture where:
- Multiple expert networks are available
- Each token is routed to specific experts
- Expert outputs are combined based on routing weights

### Token Accumulation Tables

For efficient MoE processing:
- Tokens are grouped by selected experts
- Accumulation tables track which tokens go to which experts
- Batch processing optimizes expert computation

## Integration Notes

These examples are designed for:
- **Production use**: Optimized for real model deployment
- **Performance**: Tuned tiling configurations
- **Flexibility**: Dynamic shape support for variable inputs

## See Also

- [Basic Operations](../../beginner/01_basic_operations/) - Start here for basics
- [Layer Normalization](../../intermediate/01_layer_normalization/) - Normalization details
- [Attention Example](../../advanced/01_attention/) - General attention mechanism
- [FFN Module](../../intermediate/03_ffn_module/) - Standard FFN implementation

## License

This code is part of the CANN Open Software and is licensed under the CANN Open Software License Agreement Version 1.0.

