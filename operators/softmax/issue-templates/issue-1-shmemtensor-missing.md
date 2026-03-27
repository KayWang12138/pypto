# [Bug] PyPTO 框架编译/安装后缺少 ShmemTensor 属性，阻塞算子运行时验证

## 问题现象

在运行 PyPTO 算子的运行时验证时，尝试导入 PyPTO 模块失败。错误信息显示 `pypto.pypto_impl` 模块中不存在 `ShmemTensor` 属性。

```python
AttributeError: module 'pypto.pypto_impl' has no attribute 'ShmemTensor'
```

该属性是 PyPTO 框架的核心组件，用于在 NPU 上创建和操作张量。此问题导致无法进行任何运行时验证。

## 复现步骤

1. 安装 PyPTO 框架（commit: 2cc92d8a308e9c672f75a8a3f97762083892c71f）
2. 运行任意 PyPTO 算子测试，例如：
   ```python
   import pypto
   # 尝试运行算子
   ```
3. 观察错误信息

## 期望行为

应该能够成功导入 `pypto.pypto_impl.ShmemTensor` 并运行算子测试。

## 环境信息

| 项目 | 值 |
|------|------|
| CANN 版本 | 8.5.0 |
| PyPTO Commit | 2cc92d8a308e9c672f75a8a3f97762083892c71f (2026-03-02 17:19:27 +0800) |
| 服务器类型 | A3 (910B) |
| Python 版本 | 3.10.12 |
| 操作系统 | Ubuntu 22.04.5 LTS |

## 影响范围

- **严重程度**: P0（致命）
- **影响范围**: 所有需要运行时验证的 PyPTO 算子
- **重现性**: 100%（官方示例也受影响）

## 日志/错误信息

```
AttributeError: module 'pypto.pypto_impl' has no attribute 'ShmemTensor'
```

## 可能原因

1. PyPTO 框架编译流程不完整，`ShmemTensor` 未正确导出
2. 安装包不完整，必要组件被遗漏
3. 环境变量配置问题

## 建议修复方向

1. 检查 PyPTO 框架编译流程，确认 `ShmemTensor` 是否正确导出
2. 验证安装包完整性，确保必要组件未被遗漏
3. 在 CI/CD 中添加导入验证步骤，及早发现类似问题
4. 提供环境诊断脚本，帮助开发者检测安装完整性

## 附加信息

- 在开发 softmax 算子时发现此问题
- 所有工件（spec、design、impl、test、README）已成功生成
- Golden 验证（纯 PyTorch 实现）通过
- 仅运行时验证被此问题阻塞

## 标签

`bug`, `needs-triage`, `P0`
