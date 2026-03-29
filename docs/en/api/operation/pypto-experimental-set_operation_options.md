# pypto.experimental.set\_operation\_options

## Supported Products

| Product | Supported |
|:--------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series | √ |
| Atlas A2 Training Series / Atlas A2 Inference Series | √ |

## Description

This interface is a core part of the runtime dynamic configuration management feature provided by the compilation framework. It converts parameters that were previously statically written in the configuration file `tile_fwk_config.json` into dynamic, programmable instructions.

## Function Prototype

```python
set_operation_options(*, force_combine_axis: Optional[bool] = None,
                      combine_axis: Optional[bool] = None)
```

## Parameters

| Parameter | Input/Output | Description |
|-----------|--------------|-------------|
| combine_axis | Input | **Meaning**: Enables tail-axis broadcast inline during the code generation stage. <br> **Description**: For a binary operation (32, 1) + (32, 128), instead of first broadcasting (32, 1) to (32, 128), the (32, 1) tensor is expanded to (32, 8) using a `brcb` instruction, and then (32, 8) + (32, 128) is performed. A prerequisite is that (32, 1) must be contiguous in memory. <br> **Type**: bool <br> **Value range**: {True, False} <br> **Default**: False |
| force_combine_axis | Input | **Meaning**: Same as `combine_axis`. <br> **Description**: An earlier-version option with many functional constraints; it will be gradually deprecated. Keep the default value. <br> **Type**: bool <br> **Value range**: {True, False} <br> **Default**: False |

## Return Value

`void`: Set methods have no return value. The setting takes effect upon successful execution.

## Constraints

-   **Usage scenario**: For tail-axis broadcast, the tail axis of the input must be contiguous in memory, otherwise the feature will not work. If the preceding node is a tail-axis reduce operation, the reduce interface guarantees this; if the preceding node is a `COPY_IN`, continuity on GM must be ensured at the frontend.
-   **Type safety**: The type of the passed-in value must exactly match the type defined for the parameter; otherwise, undefined behavior or runtime errors may occur.
-   **Scope**: Parameter settings are local and only affect the compilation process within the current `jit`/`loop`. If not set, the settings are inherited from the enclosing `jit`/`loop` scope.

## Example

```python
pypto.experimental.set_operation_options(combine_axis=True)
pypto.experimental.set_operation_options(force_combine_axis=False)
```
