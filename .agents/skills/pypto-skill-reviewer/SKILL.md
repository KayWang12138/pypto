---
{
  "name": "pypto-skill-reviewer",
  "description": "对指定 skill 目录进行质量与最佳实践合规性评审并打分。用于审计 skill、检查是否符合约定规范，或在发布前做质量评估。先运行 Python 静态检查脚本，再按语义检查清单逐条评估，最终输出带评分与可执行修复建议的报告。",
  "user-invocable": true
}
---

# PyPTO Skill 评审器

基于 9 个维度的 51 条规则评审一个 skill 目录，并产出可执行的评分报告与具体修复建议。

## 输入

用户提供一个 `<skill-path>`：待评审 skill 目录路径。该目录必须包含 `SKILL.md`。

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [references/rules.json](references/rules.json) | 51 条规则的权威来源：维度、权重、严重级别等 | Phase 1 和 Phase 2 开始时读取 |
| [references/scoring-spec.md](references/scoring-spec.md) | 评分算法：维度权重、扣分公式、S0 否决、等级映射 | Phase 3 开始时读取 |
| [references/semantic-checklist.md](references/semantic-checklist.md) | 22 条语义规则的检查步骤、证据标准与判断口径 | Phase 2 开始时读取 |
| [scripts/validate_skill.py](scripts/validate_skill.py) | 对 29 条规则做确定性静态检查并输出 JSON findings | Phase 1 执行 |
| [templates/report-template.md](templates/report-template.md) | 最终评审报告的 Markdown 模板 | Phase 3 开始时读取 |

## 工作流

执行三个阶段。Phase 1 与 Phase 2 无数据依赖，可并行运行。

### Phase 1：静态检查

1. 读取 [references/rules.json](references/rules.json)。
2. 对目标 skill 运行静态检查脚本：
   ```bash
   python3 <reviewer-dir>/scripts/validate_skill.py <skill-path>
   ```
   `<reviewer-dir>` 为本评审器自身目录（`.opencode/skills/pypto-skill-reviewer`）。
3. 捕获 JSON 输出（finding 对象数组）。
4. 确认脚本成功退出；如失败，报告错误并继续 Phase 2。

### Phase 2：语义评审

1. 读取 [references/rules.json](references/rules.json)，筛选 `type: "semantic"` 的规则。
2. 读取 [references/semantic-checklist.md](references/semantic-checklist.md)，严格按清单执行。
3. 读取目标 skill 目录下所有文件（`SKILL.md` 及所有子目录文件）。
4. 对每条语义规则输出一个 finding：
   ```json
   {
     "rule_id": "R07",
     "status": "FAIL|PASS|SKIP",
     "severity": "S1",
     "dimension": "D1",
     "message": "(必须引用目标 skill 中的具体内容)",
     "evidence": {
       "file": "SKILL.md",
       "line": 3,
       "snippet": "(逐字摘录，长度≥10字符)"
     },
     "suggested_fix": "(针对该 skill 的具体修改建议)"
   }
   ```
5. 对语义 findings 做自校验：
   - **Snippet 匹配**：每个 `evidence.snippet` 必须能在目标文件中逐字匹配。
   - **唯一性**：不允许重复的 `message` 文本。
   - **特指性**：`message` 与 `suggested_fix` 必须指向该 skill 的具体内容。

### Phase 3：评分与报告

1. 读取 [references/scoring-spec.md](references/scoring-spec.md)。
2. 读取 [templates/report-template.md](templates/report-template.md)。
3. 合并 Phase 1 与 Phase 2 findings。
4. 按位置聚合为 issue：`file + line_range`（±5 行合并），每个 issue 给一条统一修复建议（含 before/after）。
5. 计算维度分数与总分，应用 S0 否决与等级映射，并统计规则覆盖率。
6. 应用质量闸门（过滤内部误绑与证据不足条目），并在报告中记录。
7. 按模板输出最终 Markdown 报告。

## 输出

直接输出完整 Markdown 评审报告，包含：评审摘要、维度得分、规则覆盖率、质量闸门、问题清单、通过项汇总。

## 错误处理

- **SKILL.md 不存在**：作为单条 S0 finding（R01）报告，跳过其他检查，输出最小报告（总分 0，等级 F）。
- **脚本执行失败**：记录错误，继续语义评审，并在报告中注明静态检查不完整。
- **空目录**：等同于 SKILL.md 不存在。
- **frontmatter 非法**：R01 FAIL 触发 S0 否决；尽可能继续不依赖 frontmatter 的检查。

## 约束

- 不得修改目标 skill 目录文件（只读评审）。
- 不得伪造证据；所有 snippet 必须来自目标文件。
- 严格按 rules.json 定义执行与计分，不得自创或跳过。
