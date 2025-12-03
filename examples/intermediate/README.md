# Intermediate Examples

These examples build on the basics and demonstrate more complex PyPTO patterns and neural network components.

## Overview

Intermediate examples show how to implement common neural network components and patterns using PyPTO. They assume familiarity with basic PyPTO operations.

## Examples

### [01. Layer Normalization](01_layer_normalization/)

Implement layer normalization variants:
- Standard LayerNorm (with mean and variance)
- RMSNorm (simpler variant)
- Static and dynamic batch size support

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 20-30 minutes  
**Prerequisites**: [Basic Operations](../beginner/01_basic_operations/)

### [02. Custom Activation Functions](02_custom_activation/)

Learn to compose operations to create custom activations:
- SiLU (Swish): `x * sigmoid(x)`
- GELU: `x * sigmoid(1.702 * x)`
- SwiGLU: `Swish(gate) * up`
- GeGLU: `GELU(gate) * up`

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 20-30 minutes  
**Prerequisites**: [Basic Operations](../beginner/01_basic_operations/)

### [03. FFN Module](03_ffn_module/)

Complete Feed-Forward Network module:
- Multiple activation functions (GELU, SwiGLU, ReLU)
- Static and dynamic shapes
- Configurable tiling
- Production-ready implementation

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 30-45 minutes  
**Prerequisites**: Layer Normalization, Custom Activations

### [04. Softmax](04_softmax/)

Softmax implementation:
- Manual softmax computation
- Dynamic axis marking
- Tiling configuration
- Loop-based processing

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 20-30 minutes  
**Prerequisites**: [Basic Operations](../beginner/01_basic_operations/)

## Learning Path

1. Complete [Beginner Examples](../beginner/README.md) first
2. Start with Layer Normalization or Custom Activations
3. Progress to FFN Module for a complete component
4. Move to [Advanced Examples](../advanced/README.md) when ready

## Next Steps

After completing these examples, you should:
- ✅ Understand normalization techniques
- ✅ Know how to build custom operations
- ✅ Be able to implement complete modules
- ✅ Be ready for advanced patterns

**Ready for more?** → [Advanced Examples](../advanced/README.md)

