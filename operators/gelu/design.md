# GELU 算子设计方案

## 概述

GELU 激活函数使用 PyPTO 内置 `pypto.gelu()` API 实现，无需手动组合基础运算。

## API 映射

| PyTorch API | PyPTO API | 说明 |
|-------------|-----------|------|
| `torch.nn.functional.gelu(x, approximate='tanh')` | `pypto.gelu(x)` | PyPTO 内置 GELU |

## 实现策略

### 方案选择

**方案 A（选中）：使用内置 `pypto.gelu()`**
- 优点：简洁、高效、无需手动组合
- 缺点：无法自定义近似公式

### Tiling 策略

GELU 是 element-wise 算子，无需特殊 tiling。直接处理整个输入张量。

### Loop 结构

由于 GELU 是 element-wise 操作，可以整体处理，无需循环分块。

```python
def gelu_kernel(x: pypto.Tensor, y: pypto.Tensor) -> None:
    """GELU 算子 kernel

    Args:
        x: 输入张量 [任意shape]
        y: 输出张量 [与输入同shape]
    """
    # 直接使用内置 GELU
    y_res = pypto.gelu(x)

    # 写回输出
    pypto.assemble(y_res, [0] * len(x.shape), y)
```

## 数据类型支持

| 输入 dtype | 输出 dtype | 说明 |
|------------|------------|------|
| float16 | float16 | 支持 |
| float32 | float32 | 支持 |

## 精度保证

使用 PyPTO 内置实现，精度由框架保证。预期：
- float32: atol=0.001, rtol=0.001
- float16: atol=0.01, rtol=0.01

## 性能考虑

1. 内置 API 经过优化，性能应优于手动组合
2. 无额外内存开销
3. 无需 workspace

## 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 内置 gelu 行为与 PyTorch 不一致 | 低 | 验证精度对比 |
| 特殊值处理不当 | 低 | 测试边界条件 |

---

*设计时间: 2026-03-27*
