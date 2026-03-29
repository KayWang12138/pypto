# pypto.get\_pass\_default\_config

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the default configuration for a Pass.

## Function Prototype

```python
get_pass_default_config(key: PassConfigKey, default_value: bool) -> bool
```

## Parameters


| Parameter          | Input/Output | Description                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| key             | Input      | PassConfigKey enum. <br> KEY_DUMP_GRAPH: Dump the Pass computation graph. |
| default_value   | Input      | If the configuration value for the specified key is not found, this default value is returned. |

## Return Value

The default configuration value for the Pass named key. If it does not exist, returns default\_value.

## Constraints

Only allowed enum values may be passed for key.

## Example

```python
pypto.get_pass_default_config(pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
```

