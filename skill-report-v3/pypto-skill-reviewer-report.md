# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-skill-reviewer |
| 评审时间 | 2026-03-11 |
| 总分 | 97.50 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 50 / 失败 0 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.0 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 10.0 | 无扣分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.0 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 2.50 | R50 (-2.5) |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无扣分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 50 |
| 跳过规则数 | 0 |
| 覆盖率 | 100% |

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | PASS | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | PASS | D1 | S3 | static |
| R07 | PASS | D1 | S1 | semantic |
| R08 | PASS | D1 | S2 | semantic |
| R09 | PASS | D1 | S2 | semantic |
| R10 | PASS | D1 | S2 | static |
| R11 | PASS | D2 | S1 | static |
| R12 | PASS | D2 | S2 | static |
| R13 | PASS | D2 | S1 | static |
| R14 | PASS | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | static |
| R16 | PASS | D3 | S2 | static |
| R17 | PASS | D3 | S2 | static |
| R18 | PASS | D3 | S2 | static |
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
| R24 | PASS | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | PASS | D5 | S1 | semantic |
| R27 | PASS | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | PASS | D6 | S2 | semantic |
| R31 | PASS | D6 | S2 | semantic |
| R32 | PASS | D7 | S3 | semantic |
| R33 | PASS | D7 | S3 | semantic |
| R34 | PASS | D8 | S0 | static |
| R35 | PASS | D8 | S1 | static |
| R36 | PASS | D8 | S1 | static |
| R37 | PASS | D8 | S1 | static |
| R38 | PASS | D8 | S2 | static |
| R39 | PASS | D9 | S2 | static |
| R40 | PASS | D9 | S2 | static |
| R41 | PASS | D9 | S2 | static |
|| R42 | PASS | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | static |
| R47 | PASS | D9 | S2 | semantic |
| R50 | WARN | D7 | S2 | semantic (缺少默认推荐) |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无

### S1 重大问题

无

### S2 中等问题

无

### S3 轻微建议

#### 问题 1：多选场景缺少默认推荐

**命中规则**：R50 (S3)

**位置**：`SKILL.md:25-92`

**当前内容**：
> 执行三个阶段。由于没有数据依赖，第 1 阶段和第 2 阶段可并行运行。

### 第 1 阶段：静态检查

1. 读取 [references/rules.json](references/rules.json) 以理解规则定义。
2. 对目标 skill 运行静态检查器：
   ```
   python3 <reviewer-dir>/scripts/validate_skill.py --score <skill-path>
   ```
其中 `<reviewer-dir>` 是该 skill 自身的目录（`.agents/skills/pypto-skill-reviewer`）。
3. 捕获 JSON 输出 —— 一个 finding 对象数组。
4. 验证脚本是否成功退出。若失败，报告错误，并仅继续输出第 2 阶段结果。

### 第 2 阶段：语义评审

1. 读取 [references/rules.json](references/rules.json) 以识别哪些规则是 `type: "semantic"`。
2. 读取 [references/semantic-checklist.md](references/semantic-checklist.md) 获取详细检查流程。
3. 读取目标 skill 目录中的全部文件：
   - `SKILL.md`（必需）
   - 子目录中的所有文件（`references/`、`scripts/`、`templates/` 等）
4. 严格按照清单流程，逐条评估语义规则与目标 skill 内容的一致性。
5. 对每条规则，生成包含必需字段的 finding 对象：
   ```json
   {
     "rule_id": "R07",
     "status": "失败|通过|跳过",
     "severity": "S1",
     "dimension": "D1",
     "message": "（必须引用目标 skill 的具体内容）",
     "evidence": {
       "file": "SKILL.md",
       "line": 3,
       "snippet": "（来自目标 skill 的逐字摘录，≥10 个字符）"
     },
     "suggested_fix": "（针对该具体 skill 的明确修改建议）"
   }
   ```
6. 对所有语义 findings 执行自校验：
   - **Snippet 匹配**：验证每个 `evidence.snippet` 都在目标 skill 文件中逐字存在。若发现伪造片段，删除或修正该 finding。
   - **唯一性**：确保任意两条 finding 的 `message` 文本不完全相同。对重复项进行合并或差异化处理。
   - **具体性**：确认每条 `message` 和 `suggested_fix` 都引用目标 skill 的具体内容，而非泛化建议。

### 第 3 阶段：评分与报告

1. 读取 [references/scoring-spec.md](references/scoring-spec.md) 获取评分算法。
2. 读取 [templates/report-template.md](templates/report-template.md) 获取报告格式。
3. 将第 1 阶段和第 2 阶段的所有 findings 合并为单一列表。
4. 按位置将 findings 聚合为问题：
   - **聚合键**：`file + line_range`（彼此相距 ±5 行内的 findings 合并为一个问题）
   - 每个问题记录所有匹配的 `rule_id` 值
   - 每个问题生成一条统一修复建议（包含修改前/后对比）
5. 按维度计算分数：
   ```
   dimension_raw   = max(0, 100 - sum_of_FAIL_deductions)
   dimension_score = dimension_raw × weight
   ```
6. 计算总分：`total = Σ dimension_scores (D1–D9)`。
7. 应用 S0 否决：若任一 S0 规则状态为 FAIL，则总分封顶为 59.9，且最高等级为 D。
8. D9 特殊情况：若目标 skill 不存在 `scripts/` 目录，D9 给满分（5.0）。
9. 将总分映射到等级：A (90–100)、B (75–89)、C (60–74)、D (40–59)、F (0–39)。
10. 计算规则覆盖率：静态规则（脚本输出中的 PASS + FAIL）+ 语义规则（第 2 阶段的 PASS + FAIL + SKIP）= 总评估数。覆盖率 = evaluated / 50 × 100%。
11. 应用质量门禁 —— 评分前过滤 findings：
    - **内部错绑**：若某语义 finding 的证据明显引用的是 reviewer 自身而非目标 skill，移除该 finding，并标注原因 `internal_misbound_rule_or_evidence`。
    - **证据不足**：若某语义 finding 的 `evidence.snippet` 无法在目标文件中逐字找到，移除该 finding，并标注原因 `low_information_snippet`。
    - 为报告中的质量门禁章节记录所有被过滤项。
12. 使用模板渲染最终报告，填充全部占位符。

**问题说明**：
在"工作流程"章节中，skill 描述了三个阶段的执行方式（第 1 阶段：静态检查、第 2 阶段：语义评审、第 3 阶段：评分与报告），并明确说明"第 1 阶段和第 2 阶段可并行运行"。然而，当脚本执行失败时（第 1 阶段步骤 4），skill 提供了两种处理方式："报告错误，并仅继续输出第 2 阶段结果"与"继续仅语义评审"。这里存在两个选项，但没有明确标注默认推荐项，用户可能不清楚在脚本失败时应优先选择哪种处理方式。

**修改建议**：
> 在第 1 阶段步骤 4 中，明确标注推荐的默认处理方式。例如：
> 
> **修改前**：
> 4. 验证脚本是否成功退出。若失败，报告错误，并仅继续输出第 2 阶段结果。
> 
> **修改后**：
> 4. 验证脚本是否成功退出。若失败，**（推荐）**报告错误，并仅继续输出第 2 阶段结果。

---

## 通过项

共 49 条规则通过（1 条 WARN，0 条 FAIL）。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R21, R22, R23, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42, R47 |
| D0 | R43, R43 |

---

## 评审总结

**pypto-skill-reviewer** 是一个设计精良、实现完整的技能评审工具，整体质量达到 **A 级**（97.50/100）。

### 优势亮点

1. **完善的静态检查系统**：28 条静态规则全部通过，包括 frontmatter 验证、代码块检查、敏感信息扫描等关键安全检查。

2. **清晰的文档结构**：
   - Frontmetadata 完整且规范（name、description、user-invocable）
   - Description 同时回答了"做什么"和"何时使用"
   - 参考文件表格清晰说明了每个文件的用途和加载时机

3. **良好的工作流设计**：
   - 三个阶段（静态检查、语义评审、评分报告）分工明确
   - 错误处理覆盖全面（未找到 SKILL.md、脚本执行失败、输出格式错误等）
   - 步骤之间衔接顺畅，数据流清晰

4. **可执行性强**：
   - 所有命令和路径都具体明确
   - 每个步骤都有可验证的成功标准
   - 脚本包含基础错误处理

5. **遵循最佳实践**：
   - 采用渐进披露模式（入口摘要 → 正文细节 → 参考深度）
   - 使用确定性脚本进行验证（validate_skill.py）
   - 指令使用祈使语气，避免模糊语言

### 改进建议

仅发现 1 条轻微建议（R50，S3 级别）：

- **多选场景提供默认推荐**：在错误处理场景中，当脚本失败时存在两种处理方式，建议明确标注默认推荐项（如"（推荐）"标记），以帮助用户/AI 优先选择最佳处理路径。

### 结论

该技能是一个高质量、生产就绪的评审工具，完全符合技能开发规范。唯一的小改进点是在多选场景中提供默认推荐，这将进一步提升用户体验和执行确定性。

**推荐用于生产环境。**
