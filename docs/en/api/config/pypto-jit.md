# pypto.jit

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

The jit function is a tool for decorating and optimizing dynamic functions. It converts Python functions into efficient computation graphs using just-in-time compilation technology. The jit function can accept a function along with several optional configuration parameters, including codegen\_options, host\_options, pass\_options, and runtime\_options. When jit is invoked, it automatically manages the compilation and execution of the function. On the first run, jit compiles the function and generates an execution plan; subsequent calls reuse the compiled result to improve performance. In addition, jit supports a caching mechanism that determines whether recompilation is needed based on the shapes of the input and output tensors, thereby optimizing computational efficiency.

## Function Prototype

```python
def jit(dyn_func=None,
        *,
        codegen_options=None,
        host_options=None,
        pass_options=None,
        runtime_options=None)
```

## Parameters

### API Parameters

| Parameter            | Input/Output | Description                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| dyn_func          | Input      | The function decorated by jit. pypto.Tensor objects must be passed as tile parameters for building the computation graph. |
| codegen_options   | Input      | Type: dict[str, any]. Used to set codegen configuration options. See [Parameters](pypto-set_codegen_options.md) for details. |
| host_options      | Input      | Type: dict[str, any]. Used to set host configuration options. See [Parameters](pypto-set_host_options.md) for details. |
| pass_options      | Input      | Type: dict[str, any]. Used to set Pass configuration options. See [Parameters](pypto-set_pass_options.md) for details. |
| runtime_options   | Input      | Type: dict[str, any]. Used to set runtime configuration options. See [runtime_options Parameters](#runtime_options_detail) for details. |

### runtime_options Parameters <a id="runtime_options_detail"></a>

| Parameter                         | Description                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| device_sched_mode               | Meaning: Sets the scheduling mode of the computation subgraph. <br> Description: 0: Default scheduling mode — ready subgraphs are placed in a shared queue, and each scheduling thread competes to claim and dispatch subgraphs on a first-in-first-out basis. <br> 1: L2 cache affinity scheduling mode — the subgraph with the most recently satisfied dependency is prioritized for dispatch, achieving L2 cache reuse. <br> 2: Fair scheduling mode — when multiple threads on AICPU manage multiple AICORE devices, subgraph dispatch attempts to maintain fairness across threads; this mode incurs additional scheduling overhead. <br> 3: Enables both L2 cache affinity scheduling mode and fair scheduling mode simultaneously. <br> Type: int <br> Value range: 0, 1, 2, or 3 <br> Default: 0 <br> Affected pass scope: NA |
| stitch_function_max_num        | Meaning: Controls the maximum number of device task computation jobs submitted to the schedule AICPU per batch by the ctrlflow AICPU in the machine runtime. <br> Description: The value represents the maximum number of loops handled per stitch task. A larger value generally increases in-batch parallelism at the cost of greater workspace memory usage. <br> Note: This configuration replaces stitch_function_inner_memory, stitch_function_outcast_memory, and stitch_function_num_initial. <br> Type: int <br> Value range: 1–1024 <br> Default: 128 <br> Affected pass scope: NA |
| run_mode                       | Meaning: Sets the execution device for the computation subgraph. <br> Description: <br> 0: Execute on NPU <br> 1: Execute on simulator <br> Type: int <br> Value range: 0 or 1 <br> Default: Determined by whether the CANN environment variable is set. If the environment variable is set, execution is on NPU; otherwise, execution is on the simulator. <br> Affected pass scope: NA |
| valid_shape_optimize            | Meaning: Compilation optimization option for validshape in dynamic shape scenarios. When enabled, the main block (where shape equals validshape) in the dynamic axis loop is compiled with static shape, and the tail block is compiled with dynamic shape. <br> Description: <br> 0: Default value — disables the validshape compilation optimization option; all loop iterations are compiled with dynamic shape. <br> 1: Enables the validshape compilation optimization option. <br> Type: int <br> Value range: 0 or 1 <br> Default: 0 <br> Affected pass scope: NA |
| ready_on_host_tensors           | Meaning: A list of input tensor names of the kernel entry function that are ready on the Host side, in the format ["tensor1", "tensor2", ...]. <br> Description: If the computation logic of an operator has a value dependency on a certain input tensor (i.e., reads the tensor's values), and the device data for that tensor has been prepared in advance on the Host side, the CPU control flow can be dispatched earlier to improve performance. <br> Type: list of string <br> Default: empty list <br> Affected pass scope: NA |

## Return Value

None

## Constraints

The computation parameters passed to the decorated function must be of type pypto.Tensor.

## Example

Decoration without parameters

```python
@pypto.jit
def func(tensor1, tensor2, tensor3):
...
```

Decoration with configuration

```python
@pypto.jit(
    codegen_options={"support_dynamic_aligned": True}
)
def func(tensor1, tensor2, tensor3):
...
```

