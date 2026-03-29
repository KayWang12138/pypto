# Compilation and Execution

PyPTO builds a compilable computation graph structure on NPU hardware through function definitions, and uses the `@pypto.frontend.jit` decorator to enable Just-In-Time (JIT) compilation, thereby fully leveraging the NPU's parallel computing capabilities and improving operator execution efficiency.

## Kernel Function Definition

Before performing JIT compilation, you need to define the kernel function, obtain input and output tensors, configure tiling information, and implement the computation logic.

-   Basic function definition:

    ```python
    def add_kernel(input: pypto.Tensor, out: pypto.Tensor):
        # Tiling setting
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = input + 1
    ```

-   Multi-input/output function definition:

    ```python
    def add_kernel(input0: pypto.Tensor, input1: pypto.Tensor, out: pypto.Tensor):
         # Tiling setting
         pypto.set_vec_tile_shapes(1, 4, 1, 64)
         out[:] = input0 + input1
    ```

## JIT Compilation

Once the kernel's computation flow and data flow have been written using PyPTO functions, you can add the `pypto.frontend.jit` decorator to mark the function as a JIT compilation target and trigger PyPTO's compilation pipeline.

```python
@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
     # Tiling setting
     pypto.set_vec_tile_shapes(1, 4, 1, 64)
     out[:] = input0 + input1
```

The JIT compilation process is as follows:

-   On the first call, the function executes in "recording mode": operations are recorded and optimized into a computation graph, and the compiler generates optimized code for the NPU and caches the binary file.
-   On subsequent calls, the function directly invokes the cached binary file for execution on the NPU, with no need to recompile.

For a complete example, refer to: [hello_world](../../../examples/00_hello_world/hello_world.py).


## Conditional Compilation

The JIT decorator supports parameter configuration, enabling different conditional compilation scenarios based on the configuration:

```python
@pypto.frontend.jit(
    host_options={},
    pass_options={},
    runtime_options={},
    verify_options={},
    debug_options={}
)
def advanced_function(input0, input1):
    # Implement custom computation logic
    pass
```

JIT configuration options are described below:

-   codegen\_options: Code generation settings.
-   host\_options: Host-side options.
-   pass\_options: Compiler pass transfer options.
-   runtime\_options: Runtime execution options.
-   verify\_options: Accuracy verification tool options.
-   debug\_options: Performance data collection feature configuration options.

In addition to enabling different configurations via the JIT decorator, you can also call `pypto.set_codegen_options`, `pypto.set_host_options`, `pypto.set_pass_options`, `pypto.set_runtime_options`, `pypto.set_verify_options`, and `pypto.set_debug_options` directly in code for configuration. For example:

```python
pypto.set_codegen_options(support_dynamic_aligned=True)
```

It is recommended to configure options via JIT parameters first, as JIT configuration options offer convenience and avoid placing code unrelated to data flow and computation inside the computation function.

## Defining Multiple JIT Functions

You can define multiple JIT functions and use them together:

```python
def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    if add1_flag:
        t3 = input0 + input1
        output[:] = t3 + val
    else:
        output[:] = input0 + input1

@pypto.frontend.jit
def add_kernel_true(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, True)


@pypto.frontend.jit
def add_kernel_false(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, False)


#Use both functions
def add_add1flag_false(input_data0, input_data1, val=0):
    output_data = torch.empty_like(input_data0)
    add_kernel_false(input_data0, input_data1, output_data, val)
    return output_data

def add_add1flag_true(input_data0, input_data1, val=0):
    output_data = torch.empty_like(input_data0)
    add_kernel_true(input_data0, input_data1, output_data, val)
    return output_data

add_add1flag_false(input_data0, input_data1, val)
add_add1flag_true(input_data0, input_data1, val)
```

For a complete example, refer to: [multi_jit.py](../../../examples/03_advanced/patterns/function/multi_jit.py)

