# pypto.reset\_options

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Resets all configuration options to their default values, including codegen\_options, host\_options, pass\_options, runtime\_options, and verify\_options.

## Function Prototype

```python
reset_options() -> None
```

## Parameters

None.

## Return Value

None: no return value. The reset takes effect immediately upon success.

## Constraints

None.

## Example

```python
pypto.reset_options()
```

