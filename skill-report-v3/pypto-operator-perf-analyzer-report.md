# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-analyzer |
| 评审时间 | 2026-03-11 15:28:55 |
| 总分 | 97.30 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 44 / 失败 6 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 0分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 0分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.00 | 10分 |
| D4 | 语言与表达 | 10% | 10.0 | 9.30 | 7分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 9.50 | 5分 |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | 5分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 0分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 0分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 0分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 50 |
| 跳过规则数 | 1 |
| 覆盖率 | 100.0% |

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
| R18 | FAIL | D3 | S2 | static |
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | PASS | D5 | S1 | semantic |
| R27 | PASS | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | FAIL | D6 | S2 | semantic |
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
| R42 | PASS | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
| R47 | PASS | D9 | S2 | semantic |
| R50 | PASS | D7 | S2 | semantic |
| RR09 | PASS | D1 | S2 | semantic |

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

#### 问题 1：引用路径 `../../docs/tutorials/debug/performance.md` 不...

**命中规则**：R18 (S2)

**位置**：`SKILL.md:397`

**当前内容**：
> - [性能调优文档](../../docs/tutorials/debug/performance.md)

**问题说明**：
引用路径 `../../docs/tutorials/debug/performance.md` 不存在

**修改建议**：
> 见问题说明

---
#### 问题 2：引用路径 `../../docs/tutorials/debug/matmul_performanc...

**命中规则**：R18 (S2)

**位置**：`SKILL.md:398`

**当前内容**：
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)

**问题说明**：
引用路径 `../../docs/tutorials/debug/matmul_performance_guide.md` 不存在

**修改建议**：
> 见问题说明

---
#### 问题 3：引用路径 `../../docs/tutorials/debug/performance_case_...

**命中规则**：R18 (S2)

**位置**：`SKILL.md:399`

**当前内容**：
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)

**问题说明**：
引用路径 `../../docs/tutorials/debug/performance_case_quantindexerprolog.md` 不存在

**修改建议**：
> 见问题说明

---
#### 问题 4：Referenced files in the References section (lines ...

**命中规则**：R19 (S2)

**位置**：`SKILL.md:395`

**当前内容**：
> ## 参考资料

- [性能调优文档](../../docs/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../docs/tutorials

**问题说明**：
Referenced files in the References section (lines 397-399) do not have explicit usage and timing explanations. The skill mentions three documentation files but does not explain when they should be loaded or what specific purpose each serves in the workflow

**修改建议**：
> Add usage context for each reference. For example: '## 参考资料

这些文档在以下场景中查阅：
- [性能调优文档](../../docs/tutorials/debug/performance.md): 在步骤5性能瓶颈分析时查阅，了解性能调优理论基础
- [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md): 在优化矩阵乘法类算子时参考
- [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md): 在步骤6生成优化建议时参考实际优化案例'

---
#### 问题 5：Several key instructions lack rationale. For examp...

**命中规则**：R23 (S2)

**位置**：`SKILL.md:37`

**当前内容**：
> 从 `bubble_analysis.log` 中提取以下指标：

**问题说明**：
Several key instructions lack rationale. For example, step 1 says to find the latest output directory but does not explain why (to ensure analyzing the most recent performance data). Step 2 lists metrics to extract but does not explain why these specific metrics are important for performance analysis

**修改建议**：
> Add rationale for key steps. For example: '从 `bubble_analysis.log` 中提取以下指标（这些指标是评估NPU核心利用率和调度效率的关键）：'

---
#### 问题 6：Most workflow steps lack explicit success criteria...

**命中规则**：R24 (S2)

**位置**：`SKILL.md:22`

**当前内容**：
> ### 步骤 1：定位性能数据文件

**问题说明**：
Most workflow steps lack explicit success criteria. For example, step 1 '定位性能数据文件' does not specify how to confirm the files were found correctly. Step 2 '提取核心性能指标' does not specify how to verify extraction was successful

**修改建议**：
> Add success criteria to each step. For example: '### 步骤 1：定位性能数据文件

成功标准：确认 bubble_analysis.log、merged_swimlane.json 和 machine_runtime_operator_trace.json 三个文件都存在于输出目录中'

---
#### 问题 7：No error handling or failure recovery instructions...

**命中规则**：R30 (S2)

**位置**：`SKILL.md:20`

**当前内容**：
> ## 工作流程

**问题说明**：
No error handling or failure recovery instructions are provided. The skill does not explain what to do if bubble_analysis.log is missing, if the script fails, or if performance data is malformed

**修改建议**：
> Add error handling section. For example: '## 错误处理

- 如果 bubble_analysis.log 文件不存在：检查算子是否正确编译并运行，确认性能数据生成开关已开启
- 如果脚本执行失败：检查Python环境，确认依赖包已安装，查看错误日志
- 如果性能数据格式异常：检查PyPTO版本，确认数据格式兼容性'

---
### S3 轻微建议

#### 问题 1：代码块缺少语言标注...

**命中规则**：R46 (S3)

**位置**：`SKILL.md:59`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 见问题说明

---
#### 问题 2：代码块缺少语言标注...

**命中规则**：R46 (S3)

**位置**：`SKILL.md:65`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 见问题说明

---
#### 问题 3：代码块缺少语言标注...

**命中规则**：R46 (S3)

**位置**：`SKILL.md:71`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 见问题说明

---
#### 问题 4：代码块缺少语言标注...

**命中规则**：R46 (S3)

**位置**：`SKILL.md:77`

**当前内容**：
> ```

**问题说明**：
代码块缺少语言标注

**修改建议**：
> 见问题说明

---
## 通过项

共 44 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, RR09, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17 |
| D4 | R20, R21, R22 |
| D5 | R25, R26, R27 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R50 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42, R47 |
