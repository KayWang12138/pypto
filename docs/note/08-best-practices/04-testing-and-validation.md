# 测试与验证最佳实践

> **适用对象：** 需要建立测试验证流程的开发者  
> **学习时间：** 20-30分钟  
> **前置知识：** 已阅读[测试方法](../06-testing/00-methodology.md)和[开发指南](01-development-guide.md)  
> **学习目标：** 掌握如何把验证做成默认流程，避免仅在示例里验证、上线后才发现问题

## 概述

本文档补充"如何把验证做成默认流程"，避免仅在示例里验证、上线后才发现问题。

**相关文档：**
- [测试方法](../06-testing/00-methodology.md) - 测试体系与方法
- [测试结果与定位](../06-testing/01-results.md) - 测试结果分析
- [常见问题与已知问题库](../05-debugging/03-troubleshooting-and-known-issues.md) - 问题排查

---

## 1. 最小验证闭环

### 1.1 固定输入与随机性

- 使用固定随机种子确保可复现性
- 避免测试结果因随机性而波动

### 1.2 固定输出目录（便于对比）

- 使用 `TILE_FWK_OUTPUT_DIR` 环境变量固定输出目录
- 便于多次运行的结果对比

### 1.3 运行用例并查看 `run.log`

- 运行测试用例后，优先查看 `run.log` 确认执行状态
- 参考：[run.log 文件详细说明](../03-mechanisms/output-files/run-log.md)

---

## 2. 测试验证流程

### 2.1 单元测试

- 参考：[测试方法](../06-testing/00-methodology.md) 中的单元测试部分
- 确保每个功能模块都有对应的测试用例

### 2.2 集成测试

- 验证模块间的集成是否正确
- 检查数据流和接口兼容性

### 2.3 回归测试

- 建立回归测试套件
- 确保新功能不影响已有功能

---

## 3. 常见问题排查

- **测试失败**：参考 [测试结果与定位](../06-testing/01-results.md)
- **精度问题**：参考 [精度调试](../05-debugging/02-precision-debugging.md)
- **性能问题**：参考 [性能优化指南](../07-features/01-performance-optimization.md)

---

## 相关文档

- [测试方法](../06-testing/00-methodology.md)
- [测试结果与定位](../06-testing/01-results.md)
- [常见问题与已知问题库](../05-debugging/03-troubleshooting-and-known-issues.md)
- [开发指南](01-development-guide.md)


