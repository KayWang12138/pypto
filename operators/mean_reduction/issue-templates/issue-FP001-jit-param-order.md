# [Doc] pypto.frontend.jit 参数顺序约束文档与实际行为不一致

> **Issue 类型**: Documentation
> **优先级**: P0 (致命)
> **来源**: mean_reduction 算子开发断裂点检测
> **断裂点 ID**: FP-001
> **生成时间**: 2026-03-28

---

## 文档问题描述

`pypto.frontend.jit` 装饰器对 kernel 函数的参数顺序有严格约束：**张量参数必须放在所有非张量参数之前**。

然而，这个重要约束：
1. 未在公开的 API 文档中说明
2. 仅在 skill 内部参考文档 `references/execution-constraints.md` 中提及
3. 开发者只能在运行时报错后才能发现此约束

## 影响范围

- [x] API 文档
- [x] 使用教程
- [ ] 示例代码
- [ ] 其他

**受影响用户**: 所有使用 `@pypto.frontend.jit` 装饰器开发算子的用户

## 当前文档内容

公开文档中未提及此约束，导致开发者按照常规函数参数顺序编写代码时遇到运行时错误。

## 建议的修改

### 方案 1: 文档层面（推荐优先实施）

在 `docs/api/pypto-frontend-jit.md` 中增加醒目的 "Note" 或 "Warning" 部分：

```markdown
> **Warning: 参数顺序约束**
>
> JIT 装饰的 kernel 函数中，**所有张量参数必须放在非张量参数之前**。
>
> ```python
> # 正确示例
> @pypto.frontend.jit
> def kernel(input_tensor, output_tensor, dim, keepdim):  # 张量在前
>     ...
>
> # 错误示例
> @pypto.frontend.jit
> def kernel(input_tensor, dim, output_tensor, keepdim):  # 张量在非张量之后
>     ...
> ```
```

### 方案 2: 错误信息改进

改进运行时错误信息，使其更加友好和具体：

**当前错误信息**:
```
[ERROR] Runtime error: Non-tensor parameters must come after all tensor parameters. Found tensor parameter 'output_tensor' after non-tensor parameter(s).
```

**建议改进为**:
```
[ERROR] Parameter order constraint violated: 'output_tensor' (position 3, tensor) comes after 'dim' (position 2, non-tensor).
All tensor parameters must come before non-tensor parameters.
See: docs/api/pypto-frontend-jit.md#parameter-order
```

### 方案 3: 编译时检查（长期）

如果可能，在编译阶段就检查参数顺序，而不是等到运行时才报错。

## 相关链接

- 断裂点报告: `operators/mean_reduction/fracture-point-2026-03-28-223200.md`
- 内部参考文档: `.claude/skills/pypto-op-develop/references/execution-constraints.md`
- 受影响算子: `operators/mean_reduction/mean_reduction_impl.py`

## 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: e0a43bba (2026-03-28)
- **服务器类型**: A3
- **Python 版本**: 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 复现步骤

1. 创建一个使用 `@pypto.frontend.jit` 装饰的 kernel 函数
2. 将非 tensor 参数 (如 int, bool) 放在 tensor 参数之前或中间
3. 调用该 kernel 函数

## 最小复现代码

```python
import pypto
import torch

@pypto.frontend.jit
def buggy_kernel(input_tensor, dim, output_tensor):  # dim 在 output_tensor 之前
    # 这会导致运行时错误
    pypto.sum(input_tensor, dim=dim, out=output_tensor)

# 调用
x = torch.randn(2, 4)
y = torch.empty(2)
buggy_kernel(x, 1, y)  # 运行时错误
```

## 补充信息

此问题是在 mean_reduction 算子开发过程中发现的。开发者在编写 kernel 函数时，自然地将相关参数放在一起（如 `input, dim, output`），但违反了 JIT 的参数顺序约束，导致需要额外的调试时间。

---

**建议标签**: `documentation`, `needs-triage`
