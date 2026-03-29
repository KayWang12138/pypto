# pypto.frontend.function

## Supported Products

| AI Processor Type | Supported |
|------------|:--------:|
| Ascend 910C | √ |
| Ascend 910B | √ |
| Ascend 310B | ☓ |
| Ascend 310P | ☓ |
| Ascend 910 | ☓ |

## Description

`pypto.frontend.function` is used to define reusable computation subgraphs or function modules, allowing the same computation logic to be reused across multiple kernel functions. This feature is designed to improve code modularity and maintainability.

Intended features:
- **Code Reuse**: Encapsulate common computation patterns as functions
- **Modular Design**: Combine multiple small functions when building complex operators
- **Type Safety**: Explicitly specify input and output types in function signatures
- **Automatic Optimization**: The compiler can inline or optimize function calls

## Function Prototype

```python
@pypto.frontend.function
def reusable_function(
    arg1: pypto.Tensor,
    arg2: pypto.Tensor,
    ...
) -> pypto.Tensor:
    ...
```

## Parameters

No parameter configuration. The decorator is applied directly to the function.

## Return Value

Returns the decorated reusable function object.

## Constraints

1. Only function name calls are supported; method calls are not supported
2. Parameters: all required parameters must be provided (no default values), passed strictly in declaration order (no reordering or keyword arguments)
3. Recursive functions are not supported

## Comparison with pypto.frontend.jit

| Feature | pypto.frontend.jit | pypto.frontend.function |
|------|-------------------|------------------------|
| Purpose | Define executable kernel functions | Define reusable subfunctions |
| Compilation | Compiled into a complete computation graph | Inlined or optimized as a subgraph |
| Invocation | Can be called directly from outside | Can only be called inside JIT functions |
| Execution | Independent execution unit | Embedded and executed within the parent function |
