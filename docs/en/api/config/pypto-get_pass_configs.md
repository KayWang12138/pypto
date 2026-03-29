# pypto.get\_pass\_configs

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves all configuration information for a specified Pass.

## Function Prototype

```python
get_pass_configs(strategy: str, identifier: str) -> PassConfigs
```

## Parameters


| Parameter     | Input/Output | Description                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| strategy   | Input      | Pass strategy name, e.g., "PVC2_OOO". |
| identifier | Input      | Pass name, e.g., "ExpandFunction". |

## Return Value

A PassConfigs object with the following read-only attributes:


| Attribute               | Description                                                                 |
|--------------------|----------------------------------------------------------------------|
| printGraph         | Dump the computation graph IR. |
| dumpGraph          | Dump the computation graph. |
| dumpPassTimeCost   | Dump Pass time cost. |
| preCheck           | Perform validation before Pass execution. |
| postCheck          | Perform validation after Pass execution. |
| disablePass        | Skip execution of the current Pass. |
| healthCheck        | Execute a health check and generate a report. |

## Constraints

None

## Example

```python
pypto.get_pass_configs("PVC2_OOO", "ExpandFunction")
```

