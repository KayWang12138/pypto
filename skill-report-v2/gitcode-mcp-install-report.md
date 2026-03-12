# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | gitcode-mcp-install |
| 评审时间 | 2026-03-11 13:58:39 |
| 总分 | 97.55 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 44 / 失败 4 / 警告 0 / 跳过 3 |

## 维度评分表

| 维度 | 名称 | 原始分(0-100) | 权重 | 加权分 | 扣分明细 |
|------|------|--------------|------|--------|----------|
| D1 | Frontmatter 元数据 | 100.00 | 25% | 25.00 | 无扣分 |
| D2 | 简洁性与效率 | 95.00 | 15% | 14.25 | R14(-5) |
| D3 | 文件结构与导航 | 100.00 | 10% | 10.00 | 无扣分 |
| D4 | 语言与表达 | 98.00 | 10% | 9.80 | R21(-2) |
| D5 | 精确性与可执行性 | 85.00 | 10% | 8.50 | R25(-5)；R26(-10) |
| D6 | 工作流完整性 | 100.00 | 10% | 10.00 | 无扣分 |
| D7 | 模式与最佳实践 | 100.00 | 5% | 5.00 | 无扣分 |
| D8 | 反模式检测 | 100.00 | 10% | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 100.00 | 5% | 5.00 | 无扣分（无 scripts/ 目录，按规则满分） |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 静态评估数（PASS+FAIL） | 29 |
| 语义评估数（PASS+FAIL+SKIP） | 22 |
| 总评估数 | 51 |
| 期望规则数 | 51 |
| 状态拆分 | PASS=44, FAIL=4, SKIP=3, WARN=0 |
| 覆盖率 | 100.00% |

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
| R14 | FAIL | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | static |
| R16 | PASS | D3 | S2 | static |
| R17 | PASS | D3 | S2 | static |
| R18 | PASS | D3 | S2 | static |
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
| R24 | PASS | D5 | S2 | semantic |
| R25 | FAIL | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
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
| R42 | SKIP | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | static |
| R47 | PASS | D8 | S2 | static |
| R50 | SKIP | D9 | S2 | semantic |
| R51 | PASS | D7 | S2 | semantic |

## 质量门禁

| 类型 | 数量 | 说明 |
|------|------|------|
| internal_misbound | 0 | 发现与目标 skill 无关的规则/证据并移除 |
| low_information_snippet | 0 | snippet 无法在目标文件中逐字匹配并移除 |

## 问题列表

### S0 致命缺陷

- 无

### S1 重大问题

#### 问题 1：关键操作缺少具体执行方法

**命中规则**：R26(S1)

**位置**：`SKILL.md:117`

**证据**：
> > **注意**：修改 `~/.config/opencode/opencode.json` 后需重启 OpenCode 才能生效。

**问题说明**：
要求“修改后需重启 OpenCode 才能生效”，但未提供重启 OpenCode 的具体操作方法。

**修复建议（Before/After）**：
> Before: 仅写“修改后需重启 OpenCode 才能生效”。\nAfter: 增加可执行步骤，如“关闭当前 OpenCode 进程并重新启动会话，再执行验证命令”。

---

### S2 中等问题

#### 问题 1：跨章节重复说明影响简洁性

**命中规则**：R14(S2)

**位置**：`SKILL.md:66`

**证据**：
> 你只需要在安装完成后把占位符替换成真实 token，**修改后需重启 OpenCode 才能生效**。

**问题说明**：
“修改后需重启 OpenCode 才能生效”在配置章节与验证章节重复表达，存在跨章节冗余。

**修复建议（Before/After）**：
> Before: 在配置与验证两处重复写“修改后需重启 OpenCode 才能生效”。\nAfter: 仅在“OpenCode 配置”保留完整说明；在“验证”改为“重启要求见上文 OpenCode 配置章节”。

---

#### 问题 2：验证命令包含未解析占位符

**命中规则**：R25(S2)

**位置**：`SKILL.md:109`

**证据**：
> curl -s "https://api.gitcode.com/api/v5/user/repos?access_token=<YOUR_TOKEN>&per_page=5" | jq '.[].full_name'

**问题说明**：
可执行命令中保留 `<YOUR_TOKEN>` 占位符，未提供可直接执行的最终命令形式。

**修复建议（Before/After）**：
> Before: `curl ...access_token=<YOUR_TOKEN>...` 直接使用占位符。\nAfter: 先执行 `export GITCODE_TOKEN=...`，再使用 `curl -s "https://api.gitcode.com/api/v5/user/repos?access_token=$GITCODE_TOKEN&per_page=5" | jq '.[].full_name'`。

---

### S3 轻微建议

#### 问题 1：指令语气存在弱化措辞

**命中规则**：R21(S3)

**位置**：`SKILL.md:94`

**证据**：
> 2. 创建 Personal Access Token（建议包含 `repo`、`read:user` 权限）

**问题说明**：
可执行指令中使用了弱化措辞“建议包含”，降低了执行确定性。

**修复建议（Before/After）**：
> Before: “建议包含 `repo`、`read:user` 权限”。\nAfter: “必须包含 `repo`、`read:user` 权限，否则后续仓库查询会失败”。

---

## 通过规则汇总

共 44 条规则通过。

| 维度 | 通过规则 |
|------|----------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R22, R23, R46 |
| D5 | R24, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R51 |
| D8 | R34, R35, R36, R37, R38, R47 |
| D9 | R39, R40, R41 |
| D0 | R43, R43 |
