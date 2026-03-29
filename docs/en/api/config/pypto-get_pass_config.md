# pypto.get\_pass\_config

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the configuration for a specified Pass.

## Function Prototype

```python
get_pass_config(strategy: str, identifier: str, key: PassConfigKey, default_value: bool) -> bool
```

## Parameters


| Parameter          | Input/Output | Description                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| strategy        | Input      | Pass strategy name, e.g., "PVC2_OOO". |
| identifier      | Input      | Pass name, e.g., "ExpandFunction". |
| key             | Input      | PassConfigKey enum. <br> KEY_DUMP_GRAPH: Dump the Pass computation graph. |
| default_value   | Input      | If the configuration value for the specified key is not found, this default value is returned. |

## Return Value

The configuration value for the key in the Pass named identifier under the strategy.

## Constraints

Only allowed enum values may be passed for key.

## Example

```python
pypto.get_pass_config("PVC2_OOO", "ExpandFunction", pypto.PassConfigKey.KEY_DUMP_GRAPH, False)
```

