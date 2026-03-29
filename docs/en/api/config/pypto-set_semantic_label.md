# pypto.set\_semantic\_label

## Supported Products

| Product          | Supported |
|:-----------------|:---------:|
| Atlas A3 Training Series / Atlas A3 Inference Series |    √     |
| Atlas A2 Training Series / Atlas A2 Inference Series |    √     |

## Description

Sets the semantic label for a code segment. The PyPTO Toolkit will recognize the lines of code following this label until the next semantic label is encountered, which helps users more easily locate issues.

## Function Prototype

```python
set_semantic_label(label: str) -> None
```

## Parameters


| Parameter | Input/Output | Description                                     |
|-----------|--------------|-------------------------------------------------|
| label     | Input        | The name of the semantic label to set. Accepts any string. |

## Return Value

No return value. The setting takes effect immediately upon success.

## Constraints

None.

## Example

```python
pypto.set_semantic_label("kv")
compressed_kv = pypto.view(kv_tmp, [tile_b, s, kv_lora_rank], [0, 0, 0])
...
```

