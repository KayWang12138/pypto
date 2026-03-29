# pypto.get\_verify\_options

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Retrieves the currently configured precision debugging Verify feature options.

## Function Prototype

```python
get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns the current configured values for the precision debugging Verify feature.

## Constraints

## Example

```python
pypto.get_verify_options()
```

