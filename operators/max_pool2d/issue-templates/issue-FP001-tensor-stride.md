# [Feature] pypto tensor 切片不支持 stride 参数，导致 dilation 操作无法实现

> **Issue 类型**: Feature Request
> **优先级**: P0 (致命)
> **来源**: max_pool2d 算子开发断裂点检测
> **断裂点 ID**: FP-001
> **生成时间**: 2026-03-29

---

## 功能描述

PyPTO tensor 切片操作不支持 `stride` 参数，导致所有需要 dilation（空洞池化）的算子无法直接实现。

## 使用场景

在实现 max_pool2d、avg_pool2d 等池化算子时，需要支持 dilation 功能。按照标准实现方式，需要使用带 stride 的切片来提取窗口：

```python
# 标准 Python/PyTorch 切片语法
input_h_window = input_cur[:, h_start:h_end_clamped:stride_h, :]
```

但 PyPTO 编译时报错：

```
pypto.frontend.parser.error.RenderedParserError: ValueError: step must be 1 or None
```

## 期望的 API

### 方案 1: 支持 stride 参数（推荐）

```python
# 期望支持的标准切片语法
tensor[start:end:stride]

# 示例： dilation=2 的池化
input_window = input[:, h::stride, w::stride]
```

### 方案 2: 提供专门的 dilation window 提取 API

```python
# 新增 dilation API
pypto.extract_window(tensor, start, end, stride)
```

## 动机

1. **标准兼容性**: NumPy、PyTorch、TensorFlow 都支持 stride 切片
2. **功能完整性**: dilation 是池化算子的核心功能之一
3. **开发者体验**: 减少变通实现的工作量

## 生态对比

| 框架 | stride 支持 | 示例 |
|------|-------------|------|
| **PyTorch** | ✅ 支持 | `tensor[start:end:stride]` |
| **NumPy** | ✅ 支持 | `array[start:end:step]` |
| **Triton** | ✅ 支持 | `tl.load(ptr + offsets, mask=..., other=..., eviction_policy=..., cache_modifier=...)` |
| **PyPTO** | ❌ 不支持 | 仅支持 `step=1` 或 `step=None` |

## 当前替代方案

在 stride 支持实现之前，可以使用以下变通方法：

### 方法 1: 使用 gather 实现带 stride 的切片

```python
# 原始: tensor[start:end:stride]
# 替代:
indices = torch.arange(start, end, stride)
result = tensor.index_select(dim, indices)
```

### 方法 2: 循环逐元素提取

```python
# 仅用于 stride > 1 的小规模场景
result = []
for i in range(start, end, stride):
    result.append(tensor[..., i, :])
result = pypto.stack(result, dim=dim)
```

## 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: 2cc92d8a308e9c672f75a8a3f97762083892c71f
- **服务器类型**: A3 (Atlas A3)
- **Python 版本**: 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 相关链接

- 断裂点报告: `operators/max_pool2d/fracture-point-2026-03-29-050312.md`
- 相关算子: `operators/max_pool2d/max_pool2d_impl.py`
- PyTorch max_pool2d dilation: https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html

---

**建议标签**: `enhancement`, `api`, `needs-discussion`
