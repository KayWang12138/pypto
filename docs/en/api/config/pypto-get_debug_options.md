# pypto.get\_debug\_options

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the debug configuration.

## Function Prototype

```python
get_debug_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns a dict containing all debug configuration options.

## Constraints

None.

## Example

```python
pypto.get_debug_options()
```

