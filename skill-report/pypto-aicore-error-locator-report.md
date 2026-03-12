# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-aicore-error-locator |
| 评审时间 | 2026-03-12 |
| 总分 | 99.0 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 42 / 失败 2 / 警告 0 / 跳过 3 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.0 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 10.0 | 无扣分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 9.5 | R25 (-5分) |
| D6 | 工作流完整性 | 10% | 10.0 | 9.5 | R30 (-5分) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.0 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无扣分（不存在 scripts/ 目录，自动满分） |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 47 |
| 已评估规则数 | 47 |
| 跳过规则数 | 3 |
| 覆盖率 | 100.0% |

**跳过的规则**：R19、R42、R47（原因：不适用于此技能）

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 | 说明 |
|------|------|------|--------|------|------|
| R01 | PASS | D1 | S0 | static | frontmatter 格式正确 |
| R02 | PASS | D1 | S0 | static | name 字段有效 |
| R03 | PASS | D1 | S0 | static | description 字段有效 |
| R04 | PASS | D1 | S1 | static | name 与目录名一致 |
| R05 | PASS | D1 | S2 | static | description 长度合理 |
| R06 | PASS | D1 | S3 | static | frontmatter 字段规范 |
| R07 | PASS | D1 | S1 | semantic | description 回答了"做什么"和"何时使用" |
| R08 | PASS | D1 | S2 | semantic | description 包含自然触发短语 |
| R09 | PASS | D1 | S2 | semantic | description 以结果为导向 |
| R10 | PASS | D1 | S2 | static | 布尔字段格式正确 |
| R11 | PASS | D2 | S1 | static | 文件行数未超限 |
| R12 | PASS | D2 | S2 | static | 词汇量未超限 |
| R13 | PASS | D2 | S1 | static | 无 TODO/FIXME 占位符 |
| R14 | PASS | D2 | S2 | semantic | 无冗余重复内容 |
| R15 | PASS | D3 | S2 | static | 目录名格式正确 |
| R16 | PASS | D3 | S2 | static | 无大型内联内容块 |
| R17 | PASS | D3 | S2 | static | 引用路径使用相对路径 |
| R18 | PASS | D3 | S2 | static | 所有引用路径都存在 |
| R19 | SKIP | D3 | S2 | semantic | 未引用外部文件 |
| R20 | PASS | D4 | S2 | semantic | 指令使用祈使语气 |
| R21 | PASS | D4 | S3 | semantic | 无模糊语言 |
| R22 | PASS | D4 | S2 | static | 代码块正确闭合 |
| R23 | PASS | D4 | S2 | semantic | 指令解释原因 |
| R24 | PASS | D5 | S2 | semantic | 步骤有可验证的成功标准 |
| R25 | FAIL | D5 | S2 | semantic | 命令使用占位符变量 |
| R26 | PASS | D5 | S1 | semantic | 所有操作提供具体方法 |
| R27 | PASS | D5 | S2 | semantic | 定义了完成标准 |
| R28 | PASS | D6 | S1 | semantic | 定义了清晰的分步工作流 |
| R29 | PASS | D6 | S2 | semantic | 工作流步骤衔接顺畅 |
| R30 | FAIL | D6 | S2 | semantic | 缺少错误处理说明 |
| R31 | PASS | D6 | S2 | semantic | 条件分支描述完整 |
| R32 | PASS | D7 | S3 | semantic | 采用渐进式披露模式 |
| R33 | PASS | D7 | S3 | semantic | 使用确定性脚本验证 |
| R34 | PASS | D8 | S0 | static | 无敏感数据 |
| R35 | PASS | D8 | S1 | static | 无硬编码路径 |
| R36 | PASS | D8 | S1 | static | 无大型内联数据块 |
| R37 | PASS | D8 | S1 | static | frontmatter 无 XML 标签 |
| R38 | PASS | D8 | S2 | static | 无 Windows 风格路径 |
| R39 | PASS | D9 | S2 | static | Python 脚本语法有效 |
| R40 | PASS | D9 | S2 | static | 脚本包含 shebang |
| R41 | PASS | D9 | S2 | static | 脚本路径可移植 |
| R42 | SKIP | D9 | S2 | semantic | 不存在 scripts/ 目录 |
| R43 | PASS | D3 | S3 | static | 子目录使用标准命名 |
| R44 | PASS | D1 | S2 | static | description 无尖括号占位符 |
| R45 | PASS | D2 | S2 | static | 无重复章节标题 |
| R46 | PASS | D4 | S3 | static | 代码块有语言标注 |
| R47 | SKIP | D7 | S2 | semantic | 无多选场景 |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 未发现与目标技能无关的条目 |
| 证据不足条目 | 0 | 所有 snippet 都能在源文件中逐字匹配 |

## 问题清单

### S0 致命缺陷

无

### S1 重大问题

无

### S2 中等问题

#### 问题 1：命令使用占位符变量未提供具体示例

**命中规则**：R25 [S2]

> 规则内容：命令和路径必须具体且可执行

**位置**：`SKILL.md:82`

**当前内容**：
> **重要**: <log-file>在 device log 落盘路径中的debug文件夹下，且命名包含device关键字，同时后缀为log

**问题说明**：
步骤6中的命令使用了占位符变量 `<log-file>`，虽然文本中说明了该变量的定义（在 device log 落盘路径中的debug文件夹下，且命名包含device关键字，同时后缀为log），但未提供具体示例路径，可能导致执行困难。用户在执行 `grep -rn "trace" <log-file> | grep "LActStart"` 时，需要自行理解并替换 `<log-file>`，增加了认知负担。

**修改建议**：
> 建议在步骤6中补充一个具体示例，例如：
> 
> **示例**：假设 device log 落盘路径为 `/tmp/device_logs`，则 `<log-file>` 可能是 `/tmp/device_logs/debug/device_0.log`
> 
> 或提供完整的命令示例：
> ```bash
> grep -rn "trace" /tmp/device_logs/debug/device_*.log | grep "LActStart"
> ```

---

#### 问题 2：缺少错误处理或失败恢复说明

**命中规则**：R30 [S2]

> 规则内容：必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:77`

**当前内容**：
> **重要**: 运行测试的打屏日志中必须出现aicore error，如果未出现，则不适用于该SKILL，请停止运行

**问题说明**：
虽然步骤5提到"如果未出现 aicore error，则不适用于该SKILL，请停止运行"，但未说明其他可能出现的错误情况及处理方法。工作流中涉及多个步骤（配置文件修改、编译安装、日志清理、测试运行、日志分析等），每个步骤都可能失败，但文档未提供相应的错误处理指导。

**修改建议**：
> 建议在"工作流程"章节后增加"错误处理"小节，说明常见错误场景及处理方法，例如：
> 
> **错误处理**：
> 1. **配置文件不存在**：检查 pypto 目录路径是否正确，确认文件名为 `tile_fwk_config.json`、`aicore_entry.h`、`device_switch.h`
> 2. **编译失败**：检查依赖是否完整，查看编译错误日志，确认 CANN 环境已正确配置
> 3. **日志文件不存在**：确认 `ASCEND_PROCESS_LOG_PATH` 环境变量是否正确设置，检查目录权限
> 4. **grep 未找到匹配项**：检查日志级别是否为 0，确认测试是否正常运行并产生了日志
> 5. **多个缺失 Uid 对应不同 CCE_ID**：需分别查找并输出所有相关文件，不要遗漏
> 6. **kernel_aicore 目录不存在**：确认测试已成功运行，检查编译输出目录

---

### S3 轻微建议

无

## 通过项

共 42 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R43 |
| D4 | R20, R21, R22, R23, R46 |
| D5 | R24, R26, R27 |
| D6 | R28, R29, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |

---

## 总体评价

该技能总体质量**优秀**（等级 A，得分 99.0/100），体现了良好的技能设计实践：

**优势**：
- ✅ Frontmatter 元数据完整且规范，description 清晰描述了功能和使用场景
- ✅ 工作流结构清晰，8 个步骤有序衔接，每步都有明确指令
- ✅ 指令语言简洁、使用祈使语气，无模糊表达
- ✅ 采用了渐进式披露模式，结构层次分明
- ✅ 验证任务使用确定性脚本（grep 命令），而非依赖 LLM 判断
- ✅ 无安全风险（无敏感数据、无硬编码路径）

**待改进**：
- ⚠️ 命令中的占位符变量 `<log-file>` 应提供具体示例，降低执行难度
- ⚠️ 应增加错误处理章节，覆盖常见失败场景

**建议优先级**：
1. **高**：补充错误处理说明（影响工作流完整性）
2. **中**：提供命令示例（提升可执行性）

修复以上 2 个问题后，该技能有望达到满分（100.0/100）。
