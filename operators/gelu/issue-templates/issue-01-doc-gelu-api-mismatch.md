# [Doc] pypto.gelu 文档与实际行为不一致

**类型**: Documentation
**标签**: `documentation`
**优先级**: 致命
**置信度**: 高
**关联断裂点**: FP-1

---

## 文档问题描述

`docs/tutorials/development/tensor_operation.md` 第 79 行列出 `pypto.gelu(x)` API，声称可以直接调用。但实际运行时报错 `AttributeError: module 'pypto' has no attribute 'gelu'. Did you mean: 'relu'?`，该 API 实际不存在。

## 影响范围

- [x] API 文档
- [x] 使用教程
- [x] 示例代码
- [ ] 其他

## 当前文档内容

```python
# docs/tutorials/development/tensor_operation.md:79-81
    result = pypto.gelu(x)     # GELU激活
    result = pypto.relu(x)     # ReLU激活
    result = x.gelu()
```

## 实际行为

```python
>>> import pypto
>>> x = pypto.Tensor([1.0, 2.0, 3.0])
>>> pypto.gelu(x)
AttributeError: module 'pypto' has no attribute 'gelu'. Did you mean: 'relu'?

>>> 'gelu' in dir(pypto)
False
```

## 建议的修改

**方案 A**（推荐）: 实现 `pypto.gelu()` API
- 添加 `pypto.gelu()` 作为内置激活函数
- 与文档描述保持一致

**方案 B**: 更新文档
- 移除不存在的 API 引用
- 添加 "使用基础运算组合实现 GELU" 的说明
- 示例代码：
```python
# GELU tanh 近似实现
sqrt_2_over_pi = pypto.Tensor(0.7978845608028654)
coeff = pypto.Tensor(0.044715)
x_cubed = x * x * x
inner = x + coeff * x_cubed
inner = inner * sqrt_2_over_pi
tanh_inner = pypto.tanh(inner)
result = x * 0.5 * (1.0 + tanh_inner)
```

## 相关链接

- `docs/tutorials/development/tensor_operation.md:79`
- `docs/tutorials/network_integration/pypto_torch_api_diff.md:432` (已承认 "公开 pypto.gelu/pypto.silu 文档接口未见")

## 补充信息

- 发现于: gelu 算子开发 (2026-03-28)
- 影响: 用户按文档调用会导致运行时错误
- 临时方案: 使用基础运算组合实现 GELU
