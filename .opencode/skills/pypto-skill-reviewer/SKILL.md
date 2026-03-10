---
name: pypto-skill-reviewer
description: >-
  对 PyPTO 项目 skill 进行深度定性审查并生成结构化改进报告。基于官方最佳实践的
  10 维度评分体系，覆盖 frontmatter、简洁性、渐进披露、工作流等维度。
  适用于审查单个 pypto-* skill 的质量、审核 skill PR、或在发布前改进现有 skill。
  触发词："审查skill"、"review skill"、"skill质量"、"skill改进"、"pypto skill review"、
  "skill best practices check"、"检查skill"。
---

# PyPTO Skill 审查工具

基于官方最佳实践，对单个 PyPTO skill 进行深度定性审查。
输出带有严重度排序的结构化报告和可操作的修复建议。
## 审查流程

```
读取 skill → 加载检查清单 → 逐维度评分 → 生成报告 → 建议修复
```

### 第 1 步：确定目标 Skill

接受以下输入形式：
- Skill 名称：`pypto-performance-analyzer`
- Skill 路径：`.opencode/skills/pypto-performance-analyzer/`
- "所有 pypto skill" → 批量模式，逐个审查

定位 skill 根目录后读取内容。

### 第 2 步：收集 Skill 文件

读取目标 skill 目录下的所有文件：
1. `$SKILL_ROOT/SKILL.md` — 主文档（必需）
2. `$SKILL_ROOT/references/*.md` — 参考文档
3. `$SKILL_ROOT/scripts/*` — 自动化脚本
4. `$SKILL_ROOT/assets/*` — 打包资源

统计行数：`wc -l $SKILL_ROOT/SKILL.md`

**异常处理**：
- 若 `SKILL.md` 不存在 → 报告为 ❌ 关键错误，终止该 skill 审查
- 若目录为空 → 报告为未初始化的脚手架，跳过

### 第 3 步：加载审查标准

加载审查检查清单：@references/best-practices-checklist.md

10 个审查维度，每个维度评分为：
- ✅ **通过** — 符合最佳实践
- ⚠️ **警告** — 轻微偏离，建议改进
- ❌ **不通过** — 违反最佳实践，必须修复

### 第 4 步：执行审查

对每个维度逐项检查。每条发现使用以下格式：

```markdown
### [维度名称] — [✅/⚠️/❌]

**发现**: 观察到的问题
**期望**: 最佳实践要求
**影响**: 问题的后果（token 浪费 / 触发失败 / 可维护性下降等）
**修复**: 具体可操作的建议，附带代码/文本示例
```

#### 维度优先级顺序（按此顺序审查）：

1. **Frontmatter 质量** — 门控可发现性；description 失败则 skill 永远不会触发
2. **简洁性与 Token 效率** — 直接影响上下文窗口成本
3. **渐进披露** — 结构可扩展性
4. **内容规范** — 语言、术语、反模式
5. **自由度匹配** — 任务脆性与指令精确度对齐
6. **工作流与反馈环** — 执行指引的完整性
7. **常用模式** — 模式使用的恰当性
8. **反模式检测** — 已知坏实践
9. **脚本/代码质量** — 自动化可靠性
10. **文件组织** — 可维护性

### 第 5 步：计算评分

评分规则：
- 每个维度：通过=2 分，警告=1 分，不通过=0 分
- 总分：求和 / 20 × 100 = 百分比
- 等级：A (≥90%) / B (≥75%) / C (≥60%) / D (≥40%) / F (<40%)

**评分争议处理**：当某维度介于两个等级之间时，倾向保守评分（选较低等级），并在备注中说明原因。

### 第 6 步：生成报告

输出格式：

```markdown
# Skill 审查报告: <skill-name>

**审查工具**: pypto-skill-reviewer v1.0
**日期**: <date>
**评分**: <score>/20 (<percentage>%) — 等级 <grade>

## 摘要

<2-3 句概述 skill 质量和最优先修复项>

## 审查发现

### 1. Frontmatter 质量 — [✅/⚠️/❌]
...

### 2. 简洁性与 Token 效率 — [✅/⚠️/❌]
...

（全部 10 个维度）

## 优先修复清单

| # | 严重度 | 维度 | 修复描述 | 工作量 |
|---|--------|------|---------|--------|
| 1 | ❌ 关键 | ... | ... | S/M/L |
| 2 | ⚠️ 警告 | ... | ... | S/M/L |

## 重写示例（针对 ❌ 发现）

对每个关键发现，提供具体的重写示例展示修复方案。
```

## 批量模式

审查所有 pypto skill 时：

1. 列出 `.opencode/skills/pypto-*` 下的所有 skill
2. 对每个 skill 执行第 1-6 步
3. 生成逐个 skill 的报告
4. 生成汇总对比表：

```markdown
# PyPTO Skill 批量审查汇总

| Skill | 评分 | 等级 | 关键问题 | 警告 | 首要问题 |
|-------|------|------|---------|------|---------|
| pypto-code-tracer | 14/20 | B | 1 | 3 | 长度超限 |
| ... | ... | ... | ... | ... | ... |
```

## 已知 PyPTO Skill 常见问题

> 基于对全部 7 个现有 pypto skill 的分析总结，供审查时快速比对。

- **长度超限**: 部分 SKILL.md 超过 500 行限制 → 提取到 references/
- **硬编码绝对路径** → 使用相对路径或环境变量（如 `$SKILL_DIR`）
- **body 中出现"触发场景"** → "何时使用"仅属于 frontmatter description
- **结构性元素语言混杂** → 统一选择一种语言用于标题和结构
- **缺少渐进披露** → 大型 skill 应将详细内容提取到 references/
- **非标准 frontmatter 字段** → `license` 等不是标准字段

## 参考文件

| 文件 | 用途 |
|------|------|
| @references/best-practices-checklist.md | 10 维度详细检查清单（含通过/不通过判定标准） |
| @references/official-best-practices-summary.md | 官方最佳实践精华摘要 |
