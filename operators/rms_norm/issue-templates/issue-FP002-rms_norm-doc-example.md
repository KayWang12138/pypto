# [DOC] pypto.rms_norm 缺少 JIT kernel 完整使用示例

> **Issue 类型**: Documentation
> **优先级**: P1 (高)
> **来源**: rms_norm 算子开发断裂点检测
> **断裂点 ID**: FP-002
> **生成时间**: 2026-03-29

---

## 文档问题描述

`pypto.rms_norm` API 文档只提供了简单的 tensor 示例，缺少 JIT kernel 的完整使用示例，包括：
- 如何在 `@pypto.frontend.jit` 函数中使用
- tiling 配置 (`pypto.set_vec_tile_shapes`)
- assemble 操作 (`pypto.assemble`)
- 与 `torch.Tensor` 的集成方式

## 影响范围

- [x] API 文档
- [x] 使用教程
- [ ] 示例代码
- [ ] 其他

**受影响用户**: 所有使用 `pypto.rms_norm` 开发算子的用户

## 当前文档内容

文档中只有简单的示例：

```python
x = pypto.tensor([2, 4], pypto.DT_FP32)
gamma = pypto.tensor([4], pypto.DT_FP32)
y = pypto.rms_norm(x, gamma)
```

这个示例：
- ❌ 不涉及 JIT kernel
- ❌ 不展示 tiling 配置
- ❌ 不展示 assemble 操作
- ❌ 不展示与 torch.Tensor 的集成

## 建议的修改

在 `docs/api/operation/pypto-rms_norm.md` 中添加完整的使用示例：

```python
## 完整使用示例

以下示例展示如何在 JIT kernel 中使用 `pypto.rms_norm`：

### 基础用法

```python
import pypto
import torch

@pypto.frontend.jit
def rms_norm_kernel(
    x: pypto.Tensor(),           # 输入张量 [*, hidden_size]
    gamma: pypto.Tensor(),       # 缩放参数 [hidden_size]
    output: pypto.Tensor(),      # 输出张量 [*, hidden_size]
    eps: float = 1e-6,           # 防止除零的小常数
):
    # 1. 设置 tiling 配置
    pypto.set_vec_tile_shapes(64, 128)

    # 2. 计算 RMS Norm
    result = pypto.rms_norm(x, gamma, eps)

    # 3. 组装输出 (注意: offset 维度必须与 output 维度匹配)
    pypto.assemble(result, [0, 0], output)


def rms_norm_wrapper(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMS Norm 包装函数

    Args:
        x: 输入张量 [batch, seq_len, hidden_size]
        gamma: 缩放参数 [hidden_size]
        eps: 防止除零的小常数

    Returns:
        输出张量 [batch, seq_len, hidden_size]
    """
    output = torch.empty_like(x)
    rms_norm_kernel(x, gamma, output, eps)
    return output
```

### 3D Tensor 用法 (LLM 场景)

```python
@pypto.frontend.jit
def rms_norm_3d_kernel(
    x: pypto.Tensor(),           # [batch, seq_len, hidden_size]
    gamma: pypto.Tensor(),       # [hidden_size]
    output: pypto.Tensor(),      # [batch, seq_len, hidden_size]
    eps: float = 1e-6,
):
    pypto.set_vec_tile_shapes(64, 128)
    result = pypto.rms_norm(x, gamma, eps)
    # 重要: 3D tensor 需要使用 3D offset
    pypto.assemble(result, [0, 0, 0], output)
```

### 参考实现

完整的参考实现请参见：
- `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py`
- `pypto/examples/language/intermediate/rms_norm.py`
```

## 相关链接

- 断裂点报告: `operators/rms_norm/fracture-point-2026-03-29-021900.md`
- API 文档: `docs/api/operation/pypto-rms_norm.md`
- 参考示例: `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py`

## 补充信息

在开发 rms_norm 算子时，需要查看多个示例文件才能理解完整用法，增加了开发时间。

---

**建议标签**: `documentation`, `enhancement`, `needs-triage`
