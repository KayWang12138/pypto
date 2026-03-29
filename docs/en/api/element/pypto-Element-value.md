# pypto.Element.value

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

Gets the data value.

## Function Prototype

```python
def value(self) -> int | float
```

## Parameters

NA

## Return Value

Returns the data stored in the Element.

## Constraints

Read-only property.

## Example

```python
t = pypto.element(pypto.DT_FP32, 3)
t.value
```

