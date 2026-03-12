# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-autotuner |
| 评审时间 | 2026-03-11 |
| 总分 | 96.75 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 44 / 失败 8 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.0 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 7.0 | R18(S2x5): -25, R19(S2): -5 |
| D4 | 语言与表达 | 10% | 10.0 | 10.0 | 无扣分 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.0 | 无扣分 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.75 | R50(S2): -5 |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无 scripts/ 目录，满分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 50 |
| 跳过规则数 | 3 |
| 覆盖率 | 100% |

**跳过的规则**：R42, R47, R44（原因：不适用于此技能）

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
| R33 | FAIL | D9 | S2 | semantic |
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
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | static |
| R47 | SKIP | D9 | S2 | semantic |
| R50 | FAIL | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无 S0 级别问题。

### S1 重大问题

无 S1 级别问题。

### S2 中等问题

#### 问题 1：参考资料文件路径不存在

**命中规则**：R18(S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:102`

**当前内容**：
> - [性能调优文档](../../docs/tutorials/debug/performance.md)

**问题说明**：
参考资料部分引用了 5 个文档文件，但这些文件在相对路径 `../../docs/tutorials/debug/` 下不存在。这会导致用户点击链接时无法访问文档。

**修改建议**：
> 将参考资料链接改为指向实际存在的文档路径，或者移除不存在的链接。建议：
`> ```markdown
> ## 参考资料
> 
> - [性能调优文档](docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](docs/tutorials/debug/performance_case_quantindexerprolog.md)
> - [功能调试](docs/tutorials/debug/debug.md)
> - [精度调试](docs/tutorials/debug/precision.md)
> ```
> 或者如果这些文档确实不存在，请提供正确的文档路径或说明文档位置。

---

#### 问题 2：参考资料文件路径不存在

**命中规则**：R18(S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:103`

**当前内容**：
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)

**问题说明**：
同问题 1，此文件路径不存在。

**修改建议**：
> 同问题 1 的建议。

---

#### 问题 3：参考资料文件路径不存在

**命中规则**：R18(S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:104`

**当前内容**：
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)

**问题说明**：
同问题 1，此文件路径不存在。

**修改建议**：
> 同问题 1 的建议。

---

#### 问题 4：参考资料文件路径不存在

**命中规则**：R18(S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:105`

**当前内容**：
> - [功能调试](../../docs/tutorials/debug/debug.md)

**问题说明**：
同问题 1，此文件路径不存在。

**修改建议**：
> 同问题 1 的建议。

---

#### 问题 5：参考资料文件路径不存在

**命中规则**：R18(S2)

> 规则内容：被引用的文件路径必须真实存在

**位置**：`SKILL.md:106`

**当前内容**：
> - [精度调试](../../docs/tutorials/debug/precision.md)

**问题说明**：
同问题 1，此文件路径不存在。

**修改建议**：
> 同问题 1 的建议。

---

#### 问题 6：被引用文件缺少用途和时机说明

**命中规则**：R19(S2)

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:100-106`

**当前内容**：
> ## 参考资料
> 
> - [性能调优文档](../../docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
> - [功能调试](../../docs/tutorials/debug/debug.md)
> - [精度调试](../../docs/tutorials/debug/precision.md)

**问题说明**：
参考资料章节列出了 5 个文档链接，但没有说明这些文档的用途（何时需要查阅）和加载时机（在哪个步骤使用）。这会让用户不清楚何时需要参考这些文档。

**修改建议**：
> 为每个参考资料添加用途说明：
> ```markdown
> ## 参考资料
> 
> - [性能调优文档](../../docs/tutorials/debug/performance.md) - 了解性能调优的基本概念和方法
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md) - 学习矩阵乘法的高性能编程技巧
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md) - 参考实际性能优化案例
> - [功能调试](../../docs/tutorials/debug/debug.md) - 遇到功能问题时查阅
> - [精度调试](../../docs/tutorials/debug/precision.md) - 遇到精度问题时查阅
> ```

---

#### 问题 7：缺少精度检查的具体方法

**命中规则**：R33(S2)

> 规则内容：验证逻辑应优先使用确定性脚本

**位置**：`SKILL.md:54-58`

**当前内容**：
> ### 步骤 2：检查精度
> 
> 进行精度检查，根据检查结果选择执行下面步骤：
> 1. 如果检查通过，继续执行步骤3
> 2. 如果检查失败，是首轮失败，提示用户精度失败，是否选择修复或者继续进行优化
> 3. 如果检查失败，失败是由于上一轮修改导致，回退修改，尝试其它优化方案

**问题说明**：
步骤 2 提到"检查精度"，但没有说明如何进行精度检查。是使用脚本、工具还是手动对比？缺少具体的验证方法，可能导致执行者不知道如何操作。

**修改建议**：
> 提供具体的精度检查方法：
> ```markdown
> ### 步骤 2：检查精度
> 
> 运行算子的测试用例，对比 golden 值和实际输出：
> ```bash
> python3 custom/operator_name/operator.py --run-mode cpu --test
> ```
> 
> 根据检查结果选择执行下面步骤：
> 1. 如果检查通过（相对误差 < 1e-3，绝对误差 < 1e-5），继续执行步骤3
> 2. 如果检查失败，是首轮失败，提示用户精度失败，是否选择修复或者继续进行优化
> 3. 如果检查失败，失败是由于上一轮修改导致，回退修改，尝试其它优化方案
> ```

---

#### 问题 8：多选场景未提供默认推荐

**命中规则**：R50(S2)

> 规则内容：提供多个选项时，应给出默认推荐

**位置**：`SKILL.md:93-98`

**当前内容**：
> ### Q5: 如何选择合适的 Tilesize？
> 
> A: 
> - 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
> - 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
> - 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑

**问题说明**：
常见问题 Q5 提供了多个 Tilesize 选项，但没有标注默认推荐值。用户在面对多个选项时可能不知道优先选择哪一个。

**修改建议**：
> 为每个计算类型标注默认推荐：
> ```markdown
> ### Q5: 如何选择合适的 Tilesize？
> 
> A: 
> - 对于 Cube 计算：推荐使用 [128, 128]（默认）, [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
> - 对于 Vector 计算：推荐使用 [32, 512]（默认）或 [64, 512]
> - 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑
> ```

---

### S3 轻微建议

无 S3 级别问题。

## 通过项

共 44 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17 |
| D4 | R20, R21, R22, R23, R46 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
| D0 | R43, R43 |

---

**报告生成时间**：2026-03-11  
**评审工具**：PyPTO Skill Reviewer v1.0.1
