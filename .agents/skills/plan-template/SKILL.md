---
name: pypto-kernel-plan-template
description: 算子计划文件模板（custom/plan/<op>.md）。定义必填章节、机器可读字段和更新节奏。
---

# PyPTO Complex Kernel — 计划模板

本 Skill 包含计划模板，Agent 在每个 kernel 实现开始时将其复制到 `custom/plan/<operator_name>.md`。

## 目录

| 文件 | 用途 |
|------|------|
| **`plan.template.md`** | 实际模板 — 复制到 `custom/plan/<op>.md` 并填写 |

---

## 何时使用

- **Phase 0：** 将 `plan.template.md` 复制到 `custom/plan/<operator_name>.md` 作为第一个动作
- **每一轮：** 更新 `active_module`、`modules_pypto_verified`、`current_staged_file`、`next_mandatory_step`
- **Phase 1：** 填写 Golden 函数清单
- **Phase 2：** 填写模块分解、模块契约、阶段模块文件表
- **Phase 3：** 每次边界检查后追加到逐模块验证日志
- **Phase 4：** 最终 Golden 函数清单交叉检查
- **调试：** 粘贴 `extract_pypto_calls.py` 输出，追加到开发与调试日志

## 模板中的关键章节

| 章节 | 何时填写 | 必填？ |
|------|----------|--------|
| Agent 状态（YAML） | 每一轮 | 是 |
| 任务摘要 | Phase 0 | 是 |
| 验证 | Phase 0 | 是 |
| 模块分解及理由 | Phase 2 | 是 |
| 阶段模块文件表 | Phase 2（创建），Phase 3（更新） | 是 |
| 逐模块验证日志 | 每次 GATE 3 通过 | 是 |
| API 映射 | Phase 0 | 是 |
| Golden 函数清单 | Phase 1（创建），Phase 3-4（交叉检查） | 是 |
| 模块契约 | Phase 2 | 是 |
| 设计格式合规性 | Phase 2 | 是 |
| DEBUG.md §9 预写检查清单 | Phase 3 之前 | 是 |
| 开发与调试日志 | 每次错误/修复 | 是 |
| 人工审查里程碑 | 可选 | 否 |
