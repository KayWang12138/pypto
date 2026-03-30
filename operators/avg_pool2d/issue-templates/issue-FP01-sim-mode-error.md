# [Feature] pypto.RunMode.SIM 模式需要 NPU 设备，错误信息未提供替代方案

> **Issue 类型**: Feature Request
> **优先级**: P1 (高)
> **来源**: avg_pool2d 算子开发断裂点检测
> **断裂点 ID**: FP-01
> **生成时间**: 2026-03-29

---

## 功能描述

在 `--run_mode sim` 模式下运行 PyPTO kernel 时，仍然报错 "Not npu device"。这表明 SIM 模式实际上仍需要 NPU 设备支持，但错误信息没有说明这一点，也没有提供 CPU 纯模拟的替代方案。

## 使用场景

开发者在没有 NPU 设备的环境中尝试使用 SIM 模式进行算子开发和调试：

```bash
python test_avg_pool2d.py --run_mode sim
```

期望在 CPU 上进行模拟运行，但实际报错：

```
Running in simulation mode (CPU)

[SAME_P0] Running test...
  Input shape: (2, 3, 6, 6)
  Kernel: (2, 2), Stride: (2, 2), Padding: SAME
  ERROR: Not npu device
Traceback (most recent call last):
  ...
RuntimeError: Not npu device
```

## 期望行为

SIM 模式应该：
1. 在 CPU 上运行模拟，不需要实际 NPU 设备；或
2. 如果必须需要 NPU 设备，错误信息应该明确说明 "SIM mode still requires NPU device" 并提供替代方案

## 实际行为

错误信息 "Not npu device" 没有解释为什么 SIM 模式仍然需要 NPU，也没有提供任何修复建议或替代方案。

## 建议改进

### 1. 改进错误信息

```python
# 当前错误
RuntimeError: Not npu device

# 建议错误
RuntimeError: SIM mode still requires NPU device for execution.
If you don't have NPU access, please refer to docs/guides/development-without-npu.md for alternative workflows.
```

### 2. 更新文档

在 `docs/api/config/pypto-jit.md` 中明确说明 SIM 模式的实际要求：

```markdown
## RunMode.SIM

SIM (Simulation) 模式用于在开发阶段进行算子逻辑验证。

> **注意**: SIM 模式仍然需要 NPU 设备支持。如果你没有 NPU 环境，请考虑：
> 1. 使用远程 NPU 开发环境
> 2. 在本地进行 Python 原型开发，然后在 NPU 环境验证
```

### 3. 提供替代方案

为无 NPU 环境的开发者提供指导文档。

## 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: 2cc92d8a308e9c672d75a8a3f97762083892c71f
- **服务器类型**: Ascend 910B (A3)
- **Python 版本**: 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 相关链接

- 断裂点报告: `operators/avg_pool2d/fracture-point-2026-03-29.md`
- API 文档: `docs/api/config/pypto-jit.md`

---

**建议标签**: `enhancement`, `error-message`, `documentation`, `needs-discussion`
