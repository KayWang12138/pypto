# 技能评审报告

## 1) 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | `pypto-pr-creator` |
| 评审时间 | 2026-03-11 |
| 总分 | **94.20 / 100** |
| 等级 | **A** |
| S0 否决 | 否 |
| 规则统计 | PASS 39 / FAIL 9 / WARN 0 / SKIP 3 |

说明：本次评审按三阶段执行（静态检查 + 语义评审 + 评分汇总），覆盖全部 51 条规则；未触发 S0 否决。

## 2) 维度评分表

| 维度 | 原始分 (0-100) | 权重 | 加权分 | 扣分明细（FAIL） |
|------|---------------:|-----:|-------:|------------------|
| D1 Frontmatter 元数据 | 100 | 0.25 | 25.00 | 无 |
| D2 简洁性与效率 | 85 | 0.15 | 12.75 | R11(-10), R14(-5) |
| D3 文件结构与导航 | 95 | 0.10 | 9.50 | R19(-5) |
| D4 语言与表达 | 92 | 0.10 | 9.20 | R46(-2)×4 |
| D5 精确性与可执行性 | 85 | 0.10 | 8.50 | R24(-5), R25(-5), R27(-5) |
| D6 工作流完整性 | 100 | 0.10 | 10.00 | 无 |
| D7 模式与最佳实践 | 95 | 0.05 | 4.75 | R51(-5) |
| D8 反模式检测 | 95 | 0.10 | 9.50 | R47(-5) |
| D9 脚本与代码质量 | 100 | 0.05 | 5.00 | 无（无 `scripts/`，按规范满分） |

总分校验：25.00 + 12.75 + 9.50 + 9.20 + 8.50 + 10.00 + 4.75 + 9.50 + 5.00 = **94.20**。

## 3) 规则覆盖率

- 静态规则：29 条（PASS 26 / FAIL 3）
- 语义规则：22 条（PASS 13 / FAIL 6 / SKIP 3）
- 总评估数：29 + 22 = **51**
- 状态拆分：PASS 39 / FAIL 9 / SKIP 3
- 覆盖率：51 / 51 = **100.00%**

## 4) 质量门禁

| 过滤类型 | 数量 | 说明 |
|---------|-----:|------|
| internal_misbound_rule_or_evidence | 0 | 未发现引用 reviewer 自身内容的错绑证据 |
| low_information_snippet | 0 | 未发现无法在目标 skill 逐字匹配的语义证据 |

结论：质量门禁未过滤任何条目，全部语义 findings 保留并参与评分。

## 5) 问题列表

### S0（致命）

- 无。

### S1（重大）

#### 问题 1：`SKILL.md` 超过长度上限
- 命中规则：`[S1] R11`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:1`
- 证据：`SKILL.md 共 575 行（最大 500 行）`
- 问题描述：主文档体量超过规则上限，降低可维护性与检索效率。
- 修复建议（before/after）：
```diff
- 在 SKILL.md 中完整内联认证方案、CLA 处理模板、陷阱详表等大段内容
+ 将大块操作说明下沉到 references/（如 auth-guide.md、cla-guide.md、troubleshooting.md）
+ 在 SKILL.md 仅保留流程骨架与跳转链接，控制总行数 <= 500
```

### S2（中等）

#### 问题 2：文件引用缺少“加载时机”说明（部分条目）
- 命中规则：`[S2] R19`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:565`
- 证据：`## 相关文件`
- 问题描述：`相关文件`表给出了路径（如 `$PYPTO_REPO/CONTRIBUTION.md`、`$PYPTO_REPO/docs/contribute/pull-request.md`），但未逐项说明“在第几阶段读取、用于什么判断”，仅有路径罗列。
- 修复建议（before/after）：
```diff
- | 贡献指南 | $PYPTO_REPO/CONTRIBUTION.md |
+ | 贡献指南 | $PYPTO_REPO/CONTRIBUTION.md | 阶段3用户确认前读取，用于核对贡献流程约束 |
```

#### 问题 3：部分步骤缺少可验证成功标准
- 命中规则：`[S2] R24`, `[S2] R27`, `[S2] R51`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:490`
- 证据：`请选择操作:`
- 问题描述：在 CLA 失败分支中给出 1/2/3 选项，但未定义默认推荐路径，也未给出“本阶段完成”的可验证条件（如标签变化、PR 状态变化）。整体完成标准也未集中收敛到单一“DONE 条件”。
- 修复建议（before/after）：
```diff
- 请选择操作:
- 1. 修改 Git 配置
- 2. 在 GitCode 添加邮箱
- 3. 跳过，稍后处理
+ 默认推荐：1（立即修复本地邮箱并重推）
+ 完成判定：PR 标签出现 `cann-cla/yes` 或 `cla/pass`，且最新 commit 作者邮箱与 GitCode 主邮箱一致
+ 若选择 2/3：明确记录阻塞状态，并输出“未完成”结论与后续动作
```

#### 问题 4：命令中存在未解析占位符，直接可执行性不足
- 命中规则：`[S2] R25`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:240`
- 证据：`git -C "$PYPTO_REPO" checkout -b <branch_name>`
- 问题描述：多处命令使用 `<branch_name>`、`<username>`、`<files>` 等占位符，但未在同一节给出“如何替换”的强制映射步骤，降低直接执行性。
- 修复建议（before/after）：
```diff
- git -C "$PYPTO_REPO" add <files>
+ BRANCH_NAME="feat/add-pr-creator"
+ FILES=".agents/skills/pypto-pr-creator/SKILL.md"
+ git -C "$PYPTO_REPO" add $FILES
```

#### 问题 5：内容重复，存在同主题多处复述
- 命中规则：`[S2] R14`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:509`
- 证据：`| 格式 | `tag(scope): Summary` |`
- 问题描述：commit message 规范在阶段 5（含正则）与“PR 标题与 Body 规范/速查”重复出现，且“常见陷阱”再次复述同类约束，信息密度高但冗余明显。
- 修复建议（before/after）：
```diff
- 阶段 5 + 速查 + 常见陷阱中分别重复解释同一 commit 规范
+ 仅在 references/pr-spec.md 保留唯一规范源
+ SKILL.md 各处改为“引用 + 一句话约束 + 链接”
```

#### 问题 6：存在超长行（可读性差）
- 命中规则：`[S2] R47`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:248`
- 证据：`> **⚠️ Commit Message 格式（Pre-receive Hook 强制验证）**：`^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}``
- 问题描述：单行包含规则解释 + 正则 + 验证命令，超过 200 字符，阅读负担高。
- 修复建议（before/after）：
```diff
- 一行内写完正则、说明、验证命令
+ 拆为 3 行：规则说明 / 正则模式 / 验证命令（分别独立）
```

### S3（轻微）

#### 问题 7：围栏代码块缺少语言标注（位置 1）
- 命中规则：`[S3] R46`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:14`
- 证据：`┌─────────────────────┐      PR       ┌─────────────────────┐`
- 修复建议（before/after）：
```diff
- ```
+ ```text
```

#### 问题 8：围栏代码块缺少语言标注（位置 2）
- 命中规则：`[S3] R46`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:146`
- 证据：`检测到当前认证状态：`
- 修复建议（before/after）：
```diff
- ```
+ ```text
```

#### 问题 9：围栏代码块缺少语言标注（位置 3）
- 命中规则：`[S3] R46`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:459`
- 证据：`📋 CLA 检查结果: ❌ 未通过`
- 修复建议（before/after）：
```diff
- ```
+ ```text
```

#### 问题 10：围栏代码块缺少语言标注（位置 4）
- 命中规则：`[S3] R46`
- 位置：`.agents/skills/pypto-pr-creator/SKILL.md:479`
- 证据：`检测到 CLA 检查未通过（标签: cann-cla/no）。`
- 修复建议（before/after）：
```diff
- ```
+ ```text
```

## 6) 通过规则汇总

- D1：`R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44`
- D2：`R12, R13, R45`
- D3：`R15, R16, R17, R18`
- D4：`R20, R21, R22, R23`
- D5：`R26`
- D6：`R28, R29, R30, R31`
- D7：`R32, R33`
- D8：`R34, R35, R36, R37, R38`
- D9：`R39, R40, R41`
- D0：`R43, R43`

（SKIP 规则：`R42, R44, R50`）
