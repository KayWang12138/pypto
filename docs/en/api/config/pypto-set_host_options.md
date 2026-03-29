# pypto.set\_host\_options

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

This interface is a core part of the dynamic runtime configuration management feature provided by the compilation framework. It transforms parameters that were previously statically configured in tile\_fwk\_config.json into dynamic, programmable instructions. Its primary function is to control the execution of the on-device deployment process.

## Function Prototype

```python
set_host_options(*, compile_stage: Optional[CompStage] = None,
                    compile_monitor_enable: Optional[bool] = None,
                    compile_timeout: Optional[int] = None,
                    compile_timeout_stage: Optional[int] = None,
                    compile_monitor_print_interval: Optional[int] = None) -> None
```

## Parameters


| Parameter          | Input/Output | Description                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| compile_stage    | Input      | Meaning: Controls which stage of compilation is executed. <br> Description: <br> ALL_COMPLETE: No effect; normal compilation and execution. <br> TENSOR_GRAPH: Stop after generating the final tensor graph during compilation. <br> TILE_GRAPH: Terminate after generating the final tile graph during compilation. <br> EXECUTE_GRAPH: Terminate after generating the final execution graph during compilation. <br> CODEGEN_INSTRUCTION: Terminate after generating instruction code during compilation. <br> CODEGEN_BINARY: Terminate after generating the code binary during compilation; compilation phase ends. <br> Value range: CompStage (ALL_COMPLETE/TENSOR_GRAPH/TILE_GRAPH/EXECUTE_GRAPH/CODEGEN_INSTRUCTION/CODEGEN_BINARY) <br> Default: ALL_COMPLETE |
| compile_monitor_enable    | Input      | Meaning: Controls whether to enable compilation progress monitoring during the compilation phase. <br> Description: <br> True: Enable monitoring. <br> False: Disable monitoring. <br> Value range: bool (True/False) <br> Default: False |
| compile_timeout    | Input      | Meaning: When compilation progress monitoring is enabled, a timeout warning is printed if the total compilation time exceeds this value. <br> Description: Only takes effect when compile_monitor_enable is True. Unit: seconds. A value of 0 disables the warning print. <br> Data type: int. <br> Value range: int [0, 2147483647] <br> Default: 600 |
| compile_timeout_stage    | Input      | Meaning: When compilation progress monitoring is enabled, a timeout warning is printed if the time spent in a single compilation stage exceeds this value. <br> Description: Only takes effect when compile_monitor_enable is True. Unit: seconds. A value of 0 disables the warning print. <br> Data type: int. <br> Value range: int [0, 2147483647] <br> Default: 0 (disabled) |
| compile_monitor_print_interval    | Input      | Meaning: When compilation progress monitoring is enabled, progress is printed at this interval after a single compilation stage has exceeded 60 seconds. <br> Description: Only takes effect when compile_monitor_enable is True. Unit: seconds. <br> Data type: int. <br> Value range: int [0, 2147483647] <br> Default: 60 |

## Return Value

void: Set methods have no return value. The setting takes effect immediately upon success.

## Constraints

-   Type safety: Ensure that the type of the value passed matches the type defined for the parameter exactly; otherwise, undefined behavior or runtime errors may occur.
-   Scope: Parameter settings are global and will affect all subsequent compilation processes.

## Example

```python
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH,
                       compile_monitor_enable=True,
                       compile_timeout=120,
                       compile_timeout_stage=30,
                       compile_monitor_print_interval=20
                       )
```

