# pypto.set\_debug\_options

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Sets debug options.

## Function Prototype

```python
set_debug_options(*,
                  compile_debug_mode: Optional[int] = None,
                  runtime_debug_mode: Optional[int] = None,
                  ) -> None
```

## Parameters


| Parameter               | Input/Output | Description                                                                 |
|----------------------|-----------|----------------------------------------------------------------------|
| compile_debug_mode   | Input      | Meaning: Sets the debug mode for the compilation phase. <br> Description: 0: Default — disables compilation-phase debug mode. <br> 1: Graph mode — enables compilation-phase debug mode, activating all graph compilation-related configurations with a single switch; currently includes only the computation graph. <br> Type: int <br> Value range: 0 or 1 <br> Default: 0 <br> Affected pass scope: NA |
| runtime_debug_mode   | Input      | Meaning: Sets the debug mode for the execution phase. <br> Description: 0: Default — disables execution-phase debug mode. <br> 1: Enables execution-phase debug mode, activating all graph execution-related configurations with a single switch; currently includes only the swimlane graph. <br> Type: int <br> Value range: 0 or 1 <br> Default: 0 <br> Affected pass scope: NA |

## Return Value

void: Set methods have no return value. The setting takes effect immediately upon success.

## Constraints

None.

## Example

```python
pypto.set_debug_options(compile_debug_mode=1)
pypto.set_debug_options(runtime_debug_mode=1)
```

