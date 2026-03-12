# 技能评审报告

## 1. 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | skill-creator |
| 目标路径 | `/root/.config/opencode/skills/skill-creator` |
| 评审时间 | 2026-03-11 |
| 总分 | **81.15 / 100** |
| 等级 | **B** |
| S0 否决 | 否 |
| 规则统计（按规则去重） | PASS 38 / FAIL 12 / WARN 0 / SKIP 1 |
| 静态检查状态 | 完成（`validate_skill.py --score` 正常返回 JSON） |

## 2. 维度评分表

| 维度 | 原始分(0-100) | 权重 | 加权分 | 扣分明细（FAIL） |
|------|---------------|------|--------|------------------|
| D1 | 98 | 25% | 24.50 | R06(-2) |
| D2 | 65 | 15% | 9.75 | R11(-10), R13×2(-20), R14(-5) |
| D3 | 100 | 10% | 10.00 | 无 |
| D4 | 94 | 10% | 9.40 | R46×3(-6) |
| D5 | 80 | 10% | 8.00 | R24(-5), R25(-5), R26(-10) |
| D6 | 100 | 10% | 10.00 | 无 |
| D7 | 95 | 5% | 4.75 | R51(-5) |
| D8 | 0 | 10% | 0.00 | R35×4(-40), R47×18(-90) |
| D9 | 95 | 5% | 4.75 | R50(-5) |

总分校验：24.50 + 9.75 + 10.00 + 9.40 + 8.00 + 10.00 + 4.75 + 0.00 + 4.75 = **81.15**

## 3. 规则覆盖率

- 静态规则：PASS 23 + FAIL 6 = 29（`validate_skill.py` 输出按 rule_id 去重）
- 语义规则：PASS 15 + FAIL 6 + SKIP 1 = 22
- 总评估数：29 + 22 = **51**
- 覆盖率：51 / 51 = **100.00%**

状态拆分（全量 51 规则）：

| 状态 | 数量 |
|------|------|
| PASS | 38 |
| FAIL | 12 |
| WARN | 0 |
| SKIP | 1 |

SKIP 规则：`R44`（frontmatter 未设置 `context: fork`，不适用）

## 4. 质量门禁

| 过滤类型 | 数量 | 说明 |
|----------|------|------|
| internal_misbound_rule_or_evidence | 0 | 未发现 reviewer 自身内容错绑到目标 skill 的语义证据 |
| low_information_snippet | 0 | 语义 finding 的 `evidence.snippet` 均可在目标文件中逐字匹配 |

质量门禁结论：本次无条目被过滤，评分直接基于静态+语义 findings 合并结果。

## 5. 问题列表

### S0 致命缺陷

- 无

### S1 重大问题

#### 问题 1：文档体量超出上限（R11）
- 规则：`R11(S1)`
- 位置：`SKILL.md:1`
- 证据：`After testing the skill, users may request improvements.`（`SKILL.md` 实际总行数 556）
- 问题说明：`SKILL.md` 总行数为 556，超过 500 行限制。
- 修复建议（before/after）：
```diff
- 将大量示例、路径说明、配置片段、URL 更新说明都保留在 SKILL.md 主体
+ 将高细节内容迁移到 references/（如 path-portability.md、url-update.md）
+ SKILL.md 仅保留触发、主流程、分流导航，目标控制在 <=500 行
```

#### 问题 2：出现 TODO/HACK 占位语义（R13）
- 规则：`R13(S1)`
- 位置：`SKILL.md:408`
- 证据：`Example: When designing a \`frontend-webapp-builder\` skill for queries like "Build me a todo app"`
- 问题说明：代码块外出现 `todo` 触发词，且 `SKILL.md:444` 还有 `TODO placeholders`，被静态规则判定为占位标记。
- 修复建议（before/after）：
```diff
- "Build me a todo app"
- TODO placeholders
+ "Build me a task-tracking app"
+ implementation placeholders
```

#### 问题 3：硬编码用户绝对路径（R35）
- 规则：`R35(S1)`
- 位置：`SKILL.md:336`
- 证据：`| \`${OPENCODE_HOME}/skills/\` | \`/root/.config/opencode/skills/\` |`
- 问题说明：在可执行指导中直接给出 `/root/...`、`/home/user/...` 等用户路径，破坏可移植性。
- 修复建议（before/after）：
```diff
- /root/.config/opencode/skills/
- /home/user/.config/opencode/skills/
+ ${OPENCODE_HOME}/skills/
+ ~/.config/opencode/skills/   # 仅作用户态示例，不绑定具体用户名
```

#### 问题 4：部分操作缺乏可执行方法细节（R26）
- 规则：`R26(S1)`
- 位置：`SKILL.md:463`
- 证据：`Fetch new content from the source URL, compare with current version, and apply changes`
- 问题说明：提到“抓取、对比、应用变更”但未给出具体命令或脚本路径，执行者需要自行猜测实现。
- 修复建议（before/after）：
```diff
- Fetch new content from the source URL, compare with current version, and apply changes
+ Run: python scripts/update_from_url.py --skill <skill-dir> --source <metadata.source_url>
+ Verify: command exits 0 and prints changed file list
```

### S2 中等问题

#### 问题 5：章节间存在重复指令（R14）
- 规则：`R14(S2)`
- 位置：`SKILL.md:465`
- 证据：`When creating a skill from a URL, store the source URL in frontmatter's \`metadata.source_url\``
- 问题说明：URL 技能更新要求在 `URL-Based Skills`、`Important: URL-Based Skill Updates`、末尾迭代段重复出现，内容高度重叠。
- 修复建议（before/after）：
```diff
- 在多个章节重复描述 source_url 存储与更新流程
+ 保留一处“URL 更新规范”主段
+ 其他位置改为链接跳转：See "URL Update Workflow"
```

#### 问题 6：步骤成功标准不充分（R24）
- 规则：`R24(S2)`
- 位置：`SKILL.md:394`
- 证据：`Conclude this step when there is a clear sense of the functionality`
- 问题说明：以“clear sense”作为结束条件，缺少可验证、可复现的完成判定。
- 修复建议（before/after）：
```diff
- clear sense of the functionality
+ produce >=3 validated trigger examples and a confirmed feature scope list
+ record them in SKILL.md "Overview" section
```

#### 问题 7：命令包含未解析占位符（R25）
- 规则：`R25(S2)`
- 位置：`SKILL.md:438`
- 证据：`python scripts/init_skill.py <skill-name> --path <output-directory>`
- 问题说明：关键命令使用 `<...>` 占位符，缺少可直接复制运行的示例值。
- 修复建议（before/after）：
```diff
- python scripts/init_skill.py <skill-name> --path <output-directory>
+ python scripts/init_skill.py api-doc-helper --path /workspace/code/skills/library/shared
```

#### 问题 8：多选场景缺少默认推荐（R51）
- 规则：`R51(S2)`
- 位置：`SKILL.md:31`
- 证据：`[TODO: Choose the structure that best fits this skill's purpose. Common patterns:`
- 问题说明：给出 4 种结构模式，但未给出默认优先方案，决策成本高。
- 修复建议（before/after）：
```diff
- Workflow-Based / Task-Based / Reference / Capabilities (并列)
+ 默认推荐：Workflow-Based（若任务存在明确时序）
+ 例外条件：仅在“工具集合型”场景改用 Task-Based
```

#### 问题 9：超长行过多，影响可读性与审阅性（R47）
- 规则：`R47(S2)`
- 位置：`SKILL.md:17`
- 证据：`Skills are modular, self-contained packages that extend AI agent capabilities by providing specialized knowledge, workflows, and tools.`
- 问题说明：存在 18 处非代码/非表格超长行（>200 字符），降低 diff 审阅效率。
- 修复建议（before/after）：
```diff
- 单行写入完整长段落
+ 每行控制在 <=120 字符（最多 <=200）
+ 复杂长句拆分为短句或列表项
```

#### 问题 10：脚本依赖缺失时缺少友好兜底（R50）
- 规则：`R50(S2)`
- 位置：`scripts/quick_validate.py:9`
- 证据：`import yaml`
- 问题说明：`quick_validate.py` 直接导入 `yaml`，若环境缺少 PyYAML 将抛 traceback，未提供安装提示。
- 修复建议（before/after）：
```diff
- import yaml
+ try:
+     import yaml
+ except ImportError:
+     print("Missing dependency: pyyaml. Install with: pip install pyyaml")
+     sys.exit(2)
```

### S3 轻微建议

#### 问题 11：frontmatter 包含未知字段（R06）
- 规则：`R06(S3)`
- 位置：`SKILL.md:4`
- 证据：`license: Apache-2.0` / `metadata:`
- 问题说明：根据当前 reviewer 规则白名单，这两个字段被判为未知字段告警。
- 修复建议（before/after）：
```diff
- license: Apache-2.0
- metadata:
-   audience: developers
-   workflow: skill-development
+ （若需严格通过本规则）仅保留 name / description / 允许字段
+ 或在 reviewer 规则中显式纳入该组织扩展字段
```

#### 问题 12：存在未标注语言的围栏代码块（R46）
- 规则：`R46(S3)`
- 位置：`SKILL.md:52`
- 证据：`skill-name/`（其前一行为未标注语言的围栏起始符号 ` ``` `；另见 `SKILL.md:152`, `SKILL.md:166`）
- 问题说明：3 处代码块未声明语言，降低渲染与静态检查可用性。
- 修复建议（before/after）：
```diff
- ```
+ ```text
```

## 6. 通过规则汇总

按维度汇总 PASS 规则（D0 不计分但列出）：

| 维度 | 通过规则 |
|------|----------|
| D0 | R43, R43 |
| D1 | R01, R02, R03, R04, R05, R07, R08, R09, R10, R44 |
| D2 | R12, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R21, R22, R23 |
| D5 | R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33 |
| D8 | R34, R36, R37, R38 |
| D9 | R39, R40, R41, R42 |

---

### 附：本次语义规则判定（22条）

- PASS: R07, R08, R09, R19, R20, R21, R23, R27, R28, R29, R30, R31, R32, R33, R42
- FAIL: R14, R24, R25, R26, R50, R51
- SKIP: R44
