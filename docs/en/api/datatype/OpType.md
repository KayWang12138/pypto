# OpType

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series/Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series/Atlas A2 Inference Series |    √     |

## Description

OpType defines the types of comparison operations, used to specify the operation type when performing comparison operations between symbolic scalars. Primarily used in conditional evaluation and logical operations.

## Prototype Definition

```python
class OpType(enum.Enum):
     EQ = ...  # Equal operation: determines whether two values are equal
     NE = ...  # Not-equal operation: determines whether two values are not equal
     LT = ...  # Less-than operation: determines whether the left value is less than the right value
     LE = ...  # Less-than-or-equal operation: determines whether the left value is less than or equal to the right value
     GT = ...  # Greater-than operation: determines whether the left value is greater than the right value
     GE = ...  # Greater-than-or-equal operation: determines whether the left value is greater than or equal to the right value
```

