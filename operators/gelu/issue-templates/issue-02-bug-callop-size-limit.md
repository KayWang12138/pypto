# [Bug] gelu_kernel 编译限制：大 shape 导致 CallOpSize 超限

**类型**: Bug Report
**标签**: `bug`, `needs-triage`
**优先级**: 致命
**置信度**: 高
**关联断裂点**: FP-2
**关联 Issue**: FP-3 (错误信息模糊)

---

## Bug 描述

当输入 shape 较大（如 [8, 1024, 4096]，共 33,554,432 个元素）时，PyPTO 编译器报错 `CallOpSize: 32770 CallOpmaxSize: 20000`，超出编译器限制。这导致 LLaMA 等大模型场景的 GELU 算子无法编译。

## 复现步骤

1. 创建 GELU kernel，使用 tanh 近似公式（需要约 10 个基础运算）
2. 输入 shape 为 [8, 1024, 4096]
3. 运行编译
4. 观察编译失败

## 期望行为

PyPTO 应支持大 shape 的编译，或至少在文档中明确说明 shape 限制。

## 实际行为

编译失败，错误信息仅显示内部错误码，无修复建议。

## 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: 2cc92d8a308e9c672f75a8a3f97762083892c71f (2026-03-02 17:19:27 +0800)
- **服务器类型**: Ascend 910 A3
- **Python 版本**: Python 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 日志/报错信息

```
[1m[95mINTERNAL BUG[0m [1m/workspace/code/pypto/operators/gelu/gelu_impl.py:1:0:[0m Errcode: FFFFFF!
 loopFunction: TENSOR_LOOP_L0_lIdx_Unroll8_PATH0_hiddenfunc0_root_29088 CallOpSize: 32770 CallOpmaxSize: 20000, func OverCallOpMaxNum, file backend.cpp, line 793
libtile_fwk_compiler.so(+0x55ee8) [0xfffdb6ce5ee8]
libtile_fwk_compiler.so(+0x63778) [0xfffdb6cf3778]
...
```

## 最小复现代码

```python
import pypto
import torch

# GELU tanh 近似实现
def gelu_kernel(x):
    sqrt_2_over_pi = pypto.Tensor(0.7978845608028654)
    coeff = pypto.Tensor(0.044715)
    x_cubed = x * x * x
    inner = x + coeff * x_cubed
    inner = inner * sqrt_2_over_pi
    tanh_inner = pypto.tanh(inner)
    return x * 0.5 * (1.0 + tanh_inner)

# 大 shape 触发编译限制
x = pypto.Tensor.randn([8, 1024, 4096], dtype=pypto.Dtype.Float32)
result = gelu_kernel(x)  # 编译失败
```

## 补充信息

- **通过测试**: [4, 128], [1, 128, 768], [4, 512, 1024]
- **失败测试**: [8, 1024, 4096] (LLaMA 场景)
- **编译限制常量**: `CallOpmaxSize: 20000`
- **临时方案**: 减小 tile 大小或拆分计算
