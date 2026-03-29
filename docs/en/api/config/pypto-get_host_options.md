# pypto.get\_host\_options

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the host configuration.

## Function Prototype

```python
get_host_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns a dict containing all host configuration options.

## Constraints

None.

## Example

```python
pypto.get_host_options()
```

