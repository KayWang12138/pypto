# pypto.set\_pass\_config

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Modifies the configuration for a specified Pass.

## Function Prototype

```python
set_pass_config(strategy: str, identifier: str, key: PassConfigKey, value: bool)
```

## Parameters


| Parameter       | Input/Output | Description                                                                 |
|--------------|-----------|----------------------------------------------------------------------|
| strategy     | Input      | Pass strategy name, e.g., "PVC2_OOO" |
| identifier   | Input      | Pass name, e.g., "ExpandFunction" |
| key          | Input      | PassConfigKey enum <br> KEY_DUMP_GRAPH: Dump the Pass computation graph |
| value        | Input      | Configuration value |

## Return Value

None

## Constraints

-   Timing: Must be called before graph compilation begins.
-   Scope: Configuration is global and will affect all subsequent compilation processes.

## Example

```python
pypto.set_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```

