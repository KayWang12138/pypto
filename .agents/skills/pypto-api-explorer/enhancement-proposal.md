# pypto-api-explorer Skill 增强方案（最终版）

> 状态：**已实施**

## 背景

### 问题描述

原 `.agents/skills/pypto-api-explorer/SKILL.md` 核心工作流只有 5 个阶段（输入解析 → 公式分解 → API 探索 → 约束探索 → 生成报告），搜索范围仅覆盖 `docs/api/`，**未搜索 `models/` 和 `examples/` 下的参考实现**。

### 影响

- 与 AGENTS.md「理解官方示例原理后实现」原则不一致
- 开发者无法在 API 报告阶段获取可复用的实现模式（API 用法、Tiling 配置、Loop 结构）
- 下游 Skill（design、develop）缺少参考实现线索
- `models/` 下已有大量生产级算子实现，是最有价值的参考来源

### 设计原则

> **Skill 不绑定具体输入文件**。Skill 只描述需要的输入信息类型，输入约定由 workflow 和 orchestrator 在流程控制时负责。

---

## 优化后的方案

### 1. 新增 Stage 3.5：参考实现搜索

在 Stage 3（API 探索）之后、Stage 4（约束探索）之前插入。

搜索策略（三轮，按优先级递减）：
1. **模型实现搜索**：在 `models/` 下按算子名称、操作类型搜索已有的生产级实现
2. **精确匹配**：按算子名称搜索 `examples/` 下的目录名和文件名
3. **操作类型匹配**：按公式分解出的操作类型搜索相关示例

> **⚠️ 注意：不要找到一个就停止**
> - 必须遍历所有候选目录，收集**所有匹配的参考实现**
> - 对多个候选进行对比评估，选择**最佳匹配**（相似度最高、置信度最高、可复用点最多）
> - 若存在多个高质量参考，在报告中列出 Top 3，并说明推荐首选及理由

搜索目录与置信度：

| 目录 | 内容特征 | 搜索优先级 | 置信度 |
|------|----------|------------|--------|
| `models/`（排除 experimental） | 生产级模型算子实现（attention、matmul、moe 等） | **首选** | 高 |
| `models/experimental/` | 实验性算子实现，未充分验证 | **首选** | **中** |
| `examples/02_intermediate/operators/` | 完整算子实现（如 softmax、activation） | 次选 | 高 |
| `examples/01_beginner/compute/` | 基础计算模式（elementwise、reduce、matmul） | 次选 | 高 |
| `examples/01_beginner/tiling/` | Tiling 配置示例 | 条件 | 高 |
| `examples/03_advanced/patterns/` | 高级组合模式 | 参考 | 高 |

> `models/experimental/` 下的实现可能未经过充分的精度和性能验证，引用时须在报告中标注置信度为「中」，并提示下游 Skill 需额外验证。

### 2. 模板新增「参考实现」章节

在原 `## 5. Tiling 需求` 之后插入 `## 6. 参考实现`，后续章节重新编号：

- `## 6. 风险评估` → `## 7. 风险评估`
- `## 7. 证据索引` → `## 8. 证据索引`
- `## 8. 结论` → `## 9. 结论`

新章节包含：匹配示例表（含来源、置信度列）、可复用模式、差异分析。

### 3. 更新搜索目录总表

增加 `models/`（排除 experimental）和 `models/experimental/` 两行，分别标注优先级和置信度。

### 4. 更新 Checklist

新增门禁项：`## 6. 参考实现` 章节必须存在（可标注「无匹配」但不可缺失）。

### 5. 更新错误处理

新增场景：无匹配参考实现 → 标注「无匹配」，不阻断流程。

### 6. pypto-op-design 联动修改

**设计原则**：pypto-op-design 不直接绑定 api_report.md 文件，而是：
- 描述需要的输入信息类型（参考实现信息）
- 由 orchestrator 在调用时从 api_report.md 提取并传入

修改点：
- 阶段 2 信息收集：从「读取 api_report.md」改为「复用调用者传入的参考实现信息」
- §7 来源优先级：从「api_report.md」改为「调用者传入的参考实现信息」

---

## 已修改文件

| 文件 | 修改内容 |
|------|----------|
| `pypto-api-explorer/SKILL.md` | Stage 3.5 搜索策略、搜索目录表、Checklist、错误处理 |
| `pypto-api-explorer/templates/api_report.md` | 新增 `## 6. 参考实现` 章节，重编号后续章节 |
| `pypto-op-design/SKILL.md` | 阶段 2 改为「复用调用者传入的参考实现信息」，§7 来源优先级调整 |
