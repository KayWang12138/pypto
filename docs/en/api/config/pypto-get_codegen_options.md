# pypto.get\_codegen\_options

## Supported Products

| Product             | Supported |
|:-----------------|:--------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Retrieves the codegen configuration.

## Function Prototype

```python
get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns a dict containing all codegen configuration options.

## Constraints

None.

## Example

```python
pypto.get_codegen_options()
```

