# [Bug] gelu_kernel 编译错误信息模糊

**类型**: Bug Report
**标签**: `bug`, `needs-triage`
**优先级**: 高
**置信度**: 高
**关联断裂点**: FP-3
**关联 Issue**: FP-2 (CallOpSize 限制)

---

## Bug 描述

编译失败时错误信息过于技术化，仅显示内部错误码 `Errcode: FFFFFF` 和底层堆栈，缺少用户友好的问题描述和修复建议。

## 复现步骤

1. 触发任何编译超限错误（如大 shape 输入）
2. 观察错误输出

## 期望行为

错误信息应包含：
- 问题描述（如 "计算图复杂度超限"）
- 可能的原因（如 "输入 shape 过大"、"循环展开过多"）
- 修复建议（如 "减小 tile 大小"、"减少 unroll_list 选项"）

## 实际行为

仅显示内部错误码和底层 C++ 堆栈，普通用户难以理解。

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
libtile_fwk_compiler.so(+0x5ed8c) [0xfffdb6ceed8c]
libtile_fwk_compiler.so(+0x2efefc) [0xfffdb6f1fefc]
...
```

## 最小复现代码

```python
# 任何触发编译限制的代码都会产生类似错误
import pypto

x = pypto.Tensor.randn([8, 1024, 4096], dtype=pypto.Dtype.Float32)
# 使用复杂计算图触发编译限制
```

## 补充信息

### 当前错误信息问题

1. **内部错误码**: `Errcode: FFFFFF` 对用户无意义
2. **技术化堆栈**: C++ 符号地址对 Python 用户无帮助
3. **缺少建议**: 用户不知道如何修复

### 建议的改进

**改进后的错误信息示例**:

```
CompileError: 计算图复杂度超限 (CallOpSize: 32770 > 最大值 20000)

可能原因:
  - 输入 shape 过大 (当前: [8, 1024, 4096])
  - 循环展开过多 (当前 unroll_list 配置)

修复建议:
  - 减小 tile 大小 (尝试 tile=[1024, 1024])
  - 减少 unroll_list 选项 (尝试 unroll_list=['none'])
  - 拆分计算为多个 kernel

相关文档: docs/troubleshooting/compile-errors.md#callop-size
```

### 参考实现

- **PyTorch**: 提供清晰的 CUDA 编译错误信息
- **Triton**: 错误信息包含行号和修复建议
