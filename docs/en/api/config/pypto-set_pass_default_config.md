# pypto.set\_pass\_default\_config

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Modifies the default configuration for a Pass. Its primary function is to dynamically modify the runtime behavior configuration of a Pass. Currently, the dump computation graph switch can be configured to facilitate analysis and debugging.

## Function Prototype

```python
set_pass_default_config(key: PassConfigKey, value: bool)
```

## Parameters


| Parameter  | Input/Output | Description                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| key     | Input      | PassConfigKey enum <br> KEY_DUMP_GRAPH: Dump the Pass computation graph |
| value   | Input      | Configuration value |

## Return Value

None

## Constraints

-   Timing: Must be called before graph compilation begins.
-   Scope: Configuration is global and will affect all subsequent compilation processes.

## Example

```python
pypto.set_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```

