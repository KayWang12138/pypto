# 兼容性说明

> **适用对象：** 需要了解 PyPTO 兼容性约束的开发者  
> **学习时间：** 15-25分钟  
> **前置知识：** 已了解 PyPTO 基本使用  
> **学习目标：** 了解版本兼容性、环境组合和平台差异

## 概述

本页用于记录"文档层面"的兼容性约束与容易踩坑的组合（例如版本矩阵、入口变更）。

**相关文档：**
- [环境准备与安装](../00-getting-started/01-environment-setup.md) - 环境配置
- [迁移指南](01-migration-guide.md) - 版本迁移
- [升级检查清单](04-upgrade-checklist.md) - 升级检查

---

## 1. 版本兼容性矩阵

### 1.1 Python / PyTorch / torch_npu / CANN 的推荐组合

**推荐组合（以实际项目验证为准）：**

| Python | PyTorch | torch_npu | CANN | 说明 |
|--------|---------|-----------|------|------|
| 3.9+ | 2.0+ | 对应版本 | 7.0+ | 推荐组合 |
| 3.10+ | 2.1+ | 对应版本 | 8.0+ | 最新组合 |

**注意：** 具体版本要求请参考 [环境准备与安装](../00-getting-started/01-environment-setup.md)

### 1.2 版本约束

- Python：3.9+（推荐 3.10+）
- PyTorch：2.0+（需与 torch_npu 版本匹配）
- torch_npu：需与 PyTorch 版本匹配
- CANN：7.0+（NPU 模式必需）

---

## 2. 运行模式差异

### 2.1 SIM 与 NPU 场景能力差异

**SIM 模式（仿真）：**
- 无需 NPU 硬件
- 使用 cost_model 进行性能估算
- 适用于开发和调试

**NPU 模式（真实硬件）：**
- 需要 NPU 硬件和 CANN 环境
- 真实执行，性能数据准确
- 适用于生产环境

**详细说明：** 参考 [上手指南](../00-getting-started/00-quick-start.md#第5步常用配置速查-⚙️)

---

## 3. 构建系统兼容性

### 3.1 `build_ci.py` 参数与构建链路的变更点

- 参考：[Build 系统文档](../02-core/11-build.md) 了解构建参数
- 参考：[从源码构建与跑测试](../00-getting-started/02-build-and-test.md) 了解构建流程

---

## 4. 文档兼容性

### 4.1 文档链接变更

- 文件重命名后的链接已统一更新
- 参考：[迁移指南](01-migration-guide.md) 了解文档变更

---

## 相关文档

- [环境准备与安装](../00-getting-started/01-environment-setup.md)
- [迁移指南](01-migration-guide.md)
- [升级检查清单](04-upgrade-checklist.md)


