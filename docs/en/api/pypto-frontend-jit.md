# pypto.frontend.jit

## Supported Products

| AI Processor Type | Supported |
|------------|:--------:|
| Ascend 910C | √ |
| Ascend 910B | √ |
| Ascend 310B | ☓ |
| Ascend 310P | ☓ |
| Ascend 910 | ☓ |

## Description

`pypto.frontend.jit` is the core decorator in the new frontend architecture. It just-in-time (JIT) compiles Python functions into efficient computation graphs for execution on NPU. The new frontend does not support return values; only in-place modification is supported. It supports passing torch tensors as well as variables of other types.

Main features:
- **In-place Modification**: Kernel functions pass computation results by modifying output tensors in-place; return values are not supported
- **Type Annotations**: Explicitly specify tensor shapes and data types in function signatures
- **Direct Invocation**: During testing, torch tensors and variables of other types can be passed directly without explicit conversion
- **Dynamic Shape Support**: Works with `pypto.DYNAMIC` to support dimensions that change at runtime
- **Multiple Run Modes**: Supports both NPU and SIM (simulator) run modes

## Function Prototype

```python
@pypto.frontend.jit(
    host_options=None,
    runtime_options=None,
    codegen_options=None,
    pass_options=None
)
def kernel_function(...):
    ...
```

## Parameters

| Parameter | Input/Output | Description |
|--------|----------|------|
| func | Input | The function decorated by frontend.jit. This is the kernel entry point that describes the computation process and is used to build the computation graph. |
| codegen_options | Input | Type: `dict[str, any]`. Used to set codegen configuration options. See [Parameters](./config/pypto-set_codegen_options.md) for parameter details. |
| host_options | Input | Type: `dict[str, any]`. Used to set host configuration options. See [Parameters](./config/pypto-set_host_options.md) for parameter details. |
| pass_options | Input | Type: `dict[str, any]`. Used to set Pass configuration options. See [Parameters](./config/pypto-set_pass_options.md) for parameter details. |
| runtime_options | Input | Type: `dict[str, any]`. Used to set runtime configuration options. See [runtime_options Parameters](./config/pypto-jit.md#runtime_options_detail) for parameter details. |

## Return Value

Returns the decorated function, which can be called directly for execution.

## Constraints

1. Tensor parameters must use type annotations to specify them as `pypto.Tensor` type
2. Dynamic dimensions must be annotated with `pypto.DYNAMIC` or `pypto.DYN` in parameter annotations
3. Tensor format is indicated with the format annotation. Non-explicit format annotation is supported (see example 1 for `a`); the default is `pypto.TileOpFormat.TILEOP_ND`.
   When format is explicitly annotated, better performance can be achieved. The passed-in torch tensor must match the format declared in `pypto.Tensor` to obtain optimal performance.
4. Tensor parameters must come before non-tensor parameters (such as `scalar` and `tiling`)
5. Non-tensor parameters support keyword arguments, positional arguments, and default values

**Notes on `pypto.Tensor[...]`**:
- It is recommended to use the `pypto.Tensor[[shape], dtype]` bracket syntax in kernel function declarations, which conforms to Python type annotation conventions
- The old parenthesis syntax `pypto.Tensor([shape], dtype)` is also supported for backward compatibility
- `key=value` keyword arguments inside brackets are not supported (Python syntax limitation); only positional passing or dictionaries can be used
- `pypto.Tensor[]` (empty arguments) is not supported

## Example

### Example 1: Basic Usage

```python
@pypto.frontend.jit
def add_kernel(
    a: pypto.Tensor([3], pypto.DT_FP32),
    b: pypto.Tensor([3], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_NZ),
    out: pypto.Tensor([3], pypto.DT_FP32)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


# Call directly with torch tensors
x = torch.randn(3, dtype=torch.float32, device='npu:0')
y = torch.randn(3, dtype=torch.float32, device='npu:0')
result = add_kernel(x, y)
```

### Example 2: Specifying Run Mode

```python
# NPU mode
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_npu(x: pypto.Tensor):
    ...

# Cost Model mode
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def kernel_sim(x: pypto.Tensor):
    ...
```
