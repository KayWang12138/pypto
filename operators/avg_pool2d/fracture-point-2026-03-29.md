# 断裂点识别报告

## 1. 摘要

- **会话时间**: 2026-03-29
- **检测时间**: 2026-03-29
- **断裂点总数**: 2 个
- **按优先级分布**:
  - 致命: 0 个
  - 高: 2 个
  - 中: 0 个
- **按根因归属分布**:
  - 文档: 0 个
  - 框架: 1 个
  - 两者: 1 个
- **按 Issue 类型分布**:
  - Bug Report: 1 个
  - Documentation: 0 个
  - Feature Request: 1 个

## 2. 环境信息

- **CANN 版本**: 8.5.0
- **PyPTO Commit**: 2cc92d8a308e9c672f75a8a3f97762083892c71f (2026-03-02 17:19:27)
- **服务器类型**: Ascend 910B (A3)
- **Python 版本**: 3.10.12
- **操作系统**: Ubuntu 22.04.5 LTS

## 3. 优先修复列表

| 优先级 | 编号 | 类型 | 实体 | Issue 类型 | 置信度 |
|--------|------|------|------|------------|--------|
| 高 | FP-01 | E4-错误信息无建议 | pypto.RunMode.SIM | Feature Request | 高 |
| 高 | FP-02 | A2-API行为异常 | avg_pool2d_golden.py | Bug Report | 中 |

## 4. Session 级断裂点

<!-- 无 Session 级断裂点 -->

## 5. 实体级断裂点详情

### 5.1 [FP-01] E4-错误信息无建议: pypto.RunMode.SIM

- **类型**: E4-错误信息无建议
- **优先级**: 高
- **根因归属**: 框架
- **Issue 类型**: Feature Request
- **建议 Issue 标题**: `pypto.RunMode.SIM 模式需要 NPU 设备，错误信息未提供替代方案`
- **置信度**: 高
- **可能关联**: FP-02

#### 问题描述

在 `--run_mode sim` 模式下运行 PyPTO kernel 时，仍然报错 "Not npu device"。这表明 SIM 模式实际上仍需要 NPU 设备支持，但错误信息没有说明这一点，也没有提供 CPU 纯模拟的替代方案。

#### 证据片段

```
Running in simulation mode (CPU)

[SAME_P0] Running test...
  Input shape: (2, 3, 6, 6)
  Kernel: (2, 2), Stride: (2, 2), Padding: SAME
  ERROR: Not npu device
Traceback (most recent call last):
  File "/workspace/code/pypto/operators/avg_pool2d/test_avg_pool2d.py", line 208, in run_tests
    if run_single_test(case, device):
  File "/workspace/code/pypto/operators/avg_pool2d/test_avg_pool2d.py", line 134, in run_single_test
    impl_output = avg_pool2d_wrapper(
  File "/workspace/code/pypto/operators/avg_pool2d/avg_pool2d_impl.py", line 221, in avg_pool2d_wrapper
    kernel(x, output)
  ...
RuntimeError: Not npu device
```

#### 复现步骤

1. 设置 `--run_mode sim` 参数
2. 调用任何 PyPTO JIT kernel
3. 观察到 "Not npu device" 错误

#### 期望行为

SIM 模式应该：
1. 在 CPU 上运行模拟，不需要实际 NPU 设备；或
2. 如果必须需要 NPU 设备，错误信息应该明确说明 "SIM mode still requires NPU device" 并提供替代方案

#### 实际行为

错误信息 "Not npu device" 没有解释为什么 SIM 模式仍然需要 NPU，也没有提供任何修复建议或替代方案。

#### 相关链接

- docs/api/config/pypto-jit.md

#### 优化建议

1. **改进错误信息**: 当 SIM 模式检测到没有 NPU 设备时，提供更明确的错误信息，说明 SIM 模式的实际要求
2. **更新文档**: 在 `pypto-jit.md` 中明确说明 SIM 模式是否需要 NPU 设备
3. **提供替代方案**: 如果确实需要 NPU，建议在文档中提供无 NPU 环境下的开发流程指导

---

### 5.2 [FP-02] A2-API行为异常: avg_pool2d_golden.py

- **类型**: A2-API行为异常
- **优先级**: 高
- **根因归属**: 两者
- **Issue 类型**: Bug Report
- **建议 Issue 标题**: `avg_pool2d golden SAME padding 输出 shape 计算与 PyTorch 不一致`
- **置信度**: 中
- **可能关联**: FP-01

#### 问题描述

在 golden 实现验证过程中，SAME_large 测试用例的输出 shape 不匹配。预期输出 shape 为 [8, 64, 28, 28]，但实际得到 [8, 64, 27, 27]。这表明 SAME padding 的 ceil 模式实现与 PyTorch 的行为存在差异。

#### 证据片段

```
[SAME_large]: FAILED - shape mismatch. Expected (8, 64, 28, 28), got torch.Size([8, 64, 27, 27])
```

修复后的验证通过：
```
[SAME_large]: PASSED - shape torch.Size([8, 64, 28, 28])
```

#### 复现步骤

1. 使用 PyTorch avg_pool2d 的 `ceil_mode=True` 实现 SAME padding
2. 输入 shape: [8, 64, 56, 56], kernel=(3,3), stride=(2,2)
3. 预期输出 shape: [8, 64, 28, 28]
4. 观察到 shape 不匹配

#### 期望行为

SAME padding 模式下，输出 shape 应该为 ceil(in_h / stride)，即 28x28。

#### 实际行为

初始实现使用 floor 模式，导致输出 shape 为 27x27。

#### 相关链接

- docs/api/operation/pypto-sum.md
- models/experimental/vector/AvgPool2d/avg_pool2d.py

#### 优化建议

1. **Skill 改进**: `pypto-golden-generator` skill 应该自动处理 SAME padding 的 ceil_mode 设置
2. **增加验证**: 在 golden 生成后，自动验证 SAME/VALID 两种模式的输出 shape 是否符合预期
3. **文档补充**: 在 skill 文档中添加 SAME padding 与 PyTorch API 的对照说明

---

## 6. 总结

本次 avg_pool2d 算子开发 session 检测到 2 个高优先级断裂点：

1. **FP-01**: PyPTO SIM 模式错误信息缺乏指导性建议，影响开发者在无 NPU 环境下的调试效率
2. **FP-02**: Golden 生成 skill 对 SAME padding 的处理不够完善，需要手动修复

**fps_total**: 2
**fps_confirmed**: 1 (FP-01 为高置信度)
