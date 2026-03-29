# pypto.get\_pass\_options

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves Pass optimization parameter information.

## Function Prototype

```python
get_pass_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns a dict containing all Pass parameter information.

## Constraints

None.

## Example

```python
pypto.get_pass_options()
```

