# [API] pypto.assemble 对多维 tensor 的 offset 维度限制未在文档中明确说明

> **Issue 类型**: Bug Report
> **优先级**: P0 (致命)
> **来源**: rms_norm 算子开发断裂点检测
> **断裂点 ID**: FP-001
> **生成时间**: 2026-03-29

---

## Bug 描述

`pypto.assemble` API 要求 offset 参数的维度必须与 output tensor 维度完全匹配，但文档未说明此限制，导致开发者在处理多维 tensor 时遇到编译失败。

## 复现步骤

1. 创建 3D tensor 输出 `[2, 128, 4096]`
2. 使用 `pypto.assemble(result, [0, 0], output)` （2D offset）
3. 编译失败

## 最小复现代码

```python
import pypto
import torch

@pypto.frontend.jit
def rms_norm_kernel(x, gamma, output, eps=1e-6):
    pypto.set_vec_tile_shapes(64, 128)
    result = pypto.rms_norm(x, gamma, eps)
    # 3D tensor 但 offset 只有 2 个元素 - 失败
    pypto.assemble(result, [0, 0], output)  # ❌ 编译失败

# 正确用法
@pypto.frontend.jit
def rms_norm_kernel_fixed(x, gamma, output, eps=1e-6):
    pypto.set_vec_tile_shapes(64, 128)
    result = pypto.rms_norm(x, gamma, eps)
    pypto.assemble(result, [0, 0, 0], output)  # ✅ 3D offset
```

## 期望行为

1. 文档应明确说明 offset 维度要求
2. 错误信息应提示正确用法

## 实际行为

```
ERROR: RuntimeError: CHECK FAILED: dest.GetShape().size() == dynOffset.size()
location: ??:0:0
Assemble: dynOffset and dest requires same shape, func Assemble, file operation_impl.cpp, line 1022
```

错误信息虽然明确了问题，但没有给出修复建议。

## 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: 2cc92d8a308e9c672f75a8a3f97762083892c71f
- **服务器类型**: A3 (Atlas A3)
- **Python 版本**: 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 建议修复

### 1. 文档改进

在 `docs/api/operation/pypto-assemble.md` 中添加：

```markdown
> **Warning: offset 维度要求**
>
> `offset` 参数的维度必须与 output tensor 的维度完全匹配。
>
> | Output Shape | offset | 状态 |
> |--------------|--------|------|
> | [32, 128] | [0, 0] | ✅ 正确 |
> | [2, 128, 4096] | [0, 0] | ❌ 错误 |
> | [2, 128, 4096] | [0, 0, 0] | ✅ 正确 |
```

### 2. 错误信息改进

```cpp
// 当前: "dest.GetShape().size() == dynOffset.size()"
// 建议: "offset dimension (2) must match output tensor dimension (3). Expected: [0, 0, 0]"
```

## 影响范围

- **直接影响**: 无法使用 `pypto.assemble` 处理 3D+ tensor
- **间接影响**: 所有需要多维 tensor 输出的算子开发受阻

## 相关链接

- 断裂点报告: `operators/rms_norm/fracture-point-2026-03-29-021900.md`
- 相关 API: `docs/api/operation/pypto-assemble.md`

---

**建议标签**: `bug`, `api`, `documentation`, `needs-triage`
