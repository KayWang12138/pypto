# 迁移指南

> **适用对象：** 需要从旧版本迁移到新版本的开发者  
> **学习时间：** 15-25分钟  
> **前置知识：** 已了解 PyPTO 基本使用  
> **学习目标：** 了解版本迁移时的注意事项和兼容性问题

## 概述

本页用于记录文档/目录结构迁移时的注意事项与常见兼容问题。

**相关文档：**
- [兼容性说明](02-compatibility-notes.md) - 兼容性约束
- [升级检查清单](04-upgrade-checklist.md) - 升级检查项
- [贡献指南](03-contributing.md) - 贡献流程

---

## 1. 文档迁移注意事项

### 1.1 文件重命名后的内部链接更新

**重要变更（2025-01）：**
- `output-files/` 目录下的文件已统一为英文命名：
  - `run-log.md.md` → `run-log.md`
  - `topo-json.md.md` → `topo-json.md`
  - `program-json.md.md` → `program-json.md`
  - `kernel-aicore.md.md` → `kernel-aicore.md`
  - `kernel-aicpu.md.md` → `kernel-aicpu.md`
  - `built-in.md.md` → `built-in.md`

- `08-best-practices/` 目录文件已统一命名：
  - `03-memory-management.md` → `03-memory-management.md`
  - `04-testing-and-validation.md` → `04-testing-and-validation.md`

- `09-contributing/` 目录文件已统一命名：
  - `migration-guide.md` → `01-migration-guide.md`
  - `compatibility-notes.md` → `02-compatibility-notes.md`
  - `upgrade-checklist.md` → `04-upgrade-checklist.md`

### 1.2 新旧入口文档的替换关系

- 上手指南：统一使用 `00-getting-started/00-quick-start.md`
- 索引页：统一使用 `README.md`

---

## 2. API 迁移注意事项

### 2.1 版本兼容性

- 参考：[兼容性说明](02-compatibility-notes.md) 了解版本兼容性
- 检查 API 变更：参考 [API 参考](../02-core/02-api-reference.md)

### 2.2 配置参数变更

- 检查配置参数是否有变更
- 参考各模块文档中的配置说明

---

## 3. 升级检查清单

升级前请参考：[升级检查清单](04-upgrade-checklist.md)

---

## 相关文档

- [兼容性说明](02-compatibility-notes.md)
- [升级检查清单](04-upgrade-checklist.md)
- [贡献指南](03-contributing.md)


