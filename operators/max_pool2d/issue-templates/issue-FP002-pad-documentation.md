# [DOC] pypto.pad 文档应明确说明不支持左/上填充的限制

> **Issue 类型**: Documentation
> **优先级**: P1 (高)
> **来源**: max_pool2d 算子开发断裂点检测
> **断裂点 ID**: FP-002
> **生成时间**: 2026-03-29

---

## 文档问题描述

`pypto.pad` 文档说明了"当前仅支持右侧和底部填充"，但这个限制的描述不够醒目，且没有明确说明：

1. **为什么有这个限制** - 技术原因或设计决策
2. **如何处理需要左/上填充的场景** - 是否有替代方案
3. **是否有替代方案** - 没有提供指导

## 影响范围

- [x] API 文档
- [ ] 使用教程
- [ ] 示例代码
- [ ] 其他

**受影响用户**: 所有使用 `pypto.pad` 的开发者，特别是实现池化算子（需要 symmetric padding）的开发者

## 当前文档内容

从 `docs/api/operation/pypto- pad.md`:

```markdown
## 约束说明

1. `pad` 参数的长度必须为 2 或 4。
2. 当前**仅支持多维情况下在右侧（Right）和底部（Bottom）进行填充，或者1维情况下在右侧（Right）填充**。
   即 `pad` 序列中的向左和向上的填充量必须为 0（例如 `(0, pad_right)` 或 `(0, pad_right, 0, pad_bottom)`）。
```

虽然文档有说明，但：

- ❌ 不够醒目（隐藏在"约束说明"中）
- ❌ 没有解释原因
- ❌ 没有提供替代方案

## 建议的修改

### 1. 在"功能说明"章节开头添加醒目提示框

```markdown
> **Warning: 填充方向限制**
>
> `pypto.pad` 当前**仅支持右侧和底部填充**，不支持左侧和顶部填充。
>
> 如果你的场景需要左侧/顶部填充（如 symmetric padding），请参考 [替代方案](#替代方案) 章节。
```

### 2. 添加"常见问题"章节

```markdown
## 常见问题

### 如何实现左侧/顶部填充？

当前 `pypto.pad` 不支持左侧/顶部填充。如果需要 symmetric padding，可以：

**方案 1: 在 kernel 内部处理边界**

```python
# 在 kernel 内部通过条件判断处理边界
@pypto.frontend.jit
def pool_with_padding(x, output, padding):
    # 在 kernel 内部通过条件判断处理左/上边界
    if position < padding:
        # 左/上边界区域，填充 0 或特定值
        ...
    else:
        # 正常计算区域
        ...
```

**方案 2: 使用 torch.nn.functional.pad 预处理**

```python
# 在 wrapper 中预处理 padding
def pool_wrapper(x, kernel_size, padding):
    if padding > 0:
        # 使用 PyTorch 进行 symmetric padding
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding))
    # 然后调用 PyPTO kernel（无需 padding）
    return pool_kernel(x)
```

### 3. 提供完整示例

在文档中添加一个展示如何在 kernel 内部处理 padding 边界的标准模式：

```python
# 参考: models/experimental/vector/AvgPool2d/avg_pool2d.py
# 该示例展示了在 kernel 内部通过边界检查处理 padding 的标准模式
```
```

## 相关链接

- 断裂点报告: `operators/max_pool2d/fracture-point-2026-03-29-050312.md`
- API 文档: `docs/api/operation/pypto-pad.md`
- 参考示例: `models/experimental/vector/AvgPool2d/avg_pool2d.py`

## 补充信息

在开发 max_pool2d 算子时，由于 pypto.pad 不支持左侧/顶部填充，开发者需要通过试错才能发现这个限制，然后被迫在 kernel 内部实现边界检查。增加了开发时间。

如果文档能提供更醒目的提示和替代方案指导，可以减少开发者的试错成本。

---

**建议标签**: `documentation`, `enhancement`, `needs-triage`
