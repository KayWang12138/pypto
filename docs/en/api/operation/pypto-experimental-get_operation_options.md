# pypto.experimental.get\_operation\_options

## Supported Products

| Product | Supported |
|:--------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series | √ |
| Atlas A2 Training Series / Atlas A2 Inference Series | √ |

## Description

Retrieves the operation configuration.

## Function Prototype

```python
get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]
```

## Parameters

None.

## Return Value

Returns a dict containing all configuration items for the operation.

## Constraints

None.

## Example

```python
pypto.get_operation_options()
```
