# 技能评审报告

## 1. 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-analyzer |
| 评审时间 | 2026-03-11 13:49:59 +0800 |
| 总分 | 94.10 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | PASS 38 / FAIL 12 / WARN 0 / SKIP 1 |

## 2. 维度评分表

| 维度 | 原始分(0-100) | 权重 | 加权分 | 扣分明细（FAIL） |
|------|---------------|------|--------|------------------|
| D1 | 100 | 25% | 25.00 | 无 |
| D2 | 95 | 15% | 14.25 | R14(-5) |
| D3 | 90 | 10% | 9.00 | R18(-5), R19(-5) |
| D4 | 86 | 10% | 8.60 | R20(-5), R21(-2), R23(-5), R46(-2) |
| D5 | 80 | 10% | 8.00 | R24(-5), R26(-10), R27(-5) |
| D6 | 95 | 10% | 9.50 | R30(-5) |
| D7 | 100 | 5% | 5.00 | 无 |
| D8 | 100 | 10% | 10.00 | 无 |
| D9 | 95 | 5% | 4.75 | R42(-5) |

- 总分校验：25.00 + 14.25 + 9.00 + 8.60 + 8.00 + 9.50 + 5.00 + 10.00 + 4.75 = **94.10**

## 3. 规则覆盖率

- 静态规则：29（PASS 27 / FAIL 2 / SKIP 0）
- 语义规则：22（PASS 11 / FAIL 10 / SKIP 1）
- 总评估数：29 + 22 = **51**
- 状态汇总：PASS 38 / FAIL 12 / SKIP 1
- 覆盖率：51 / 51 × 100% = **100.00%**

## 4. 质量门禁

| 过滤类型 | 数量 | 说明 |
|----------|------|------|
| internal_misbound_rule_or_evidence | 0 | 未发现引用 reviewer 自身而非目标 skill 的错绑证据 |
| low_information_snippet | 0 | 所有 FAIL/SKIP 证据片段均可在目标文件逐字匹配 |

- 被过滤条目：无

## 5. 问题列表

### S0 致命缺陷

- 无。

### S1 重大问题

#### 问题 1：部分操作仅给出“要做什么”，未给出执行方法

- 命中规则：`[S1] R26`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:103`
- 证据：`- 识别气泡率最高的核心`
- 问题说明：该条与同组“分析要点”只描述检查目标，没有给出具体命令、字段或脚本调用方式，执行者无法直接复现。
- 修改建议（before/after）：

```diff
- - 识别气泡率最高的核心
+ - 运行 `python3 .opencode/skills/pypto-operator-perf-analyzer/scripts/analyze_perf.py <output_dir>` 生成报告
+ - 在报告“3. 气泡分析”表格中按“气泡率”降序排序，记录 Top1 核心及数值
```

### S2 中等问题

#### 问题 2：章节存在实质重复，降低可读性

- 命中规则：`[S2] R14`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:3`
- 证据：`description: 分析 PyPTO 算子的性能指标。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。`
- 问题说明：frontmatter 的 `description` 与“概述”段落内容几乎逐字重复（`SKILL.md:10`），信息增量不足。
- 修改建议（before/after）：

```diff
- 此技能专注于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。
+ 此技能用于将 `bubble_analysis.log` 与 `merged_swimlane.json` 转化为可执行的性能结论，输出“瓶颈定位 + 优先级优化建议 + 报告文件路径”。
```

#### 问题 3：引用文件缺少可达性与用途/时机说明

- 命中规则：`[S2] R18`, `R19`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:397`
- 证据：`- [性能调优文档](../../docs/tutorials/debug/performance.md)`
- 问题说明：三条参考链接均解析失败（`R18`），且仅给出路径，未说明“何时读”“读它解决什么问题”（`R19`）。
- 修改建议（before/after）：

```diff
- - [性能调优文档](../../docs/tutorials/debug/performance.md)
+ - [性能调优文档](../../docs/tutorials/debug/performance.md)（用途：解释核心利用率/气泡率阈值来源；时机：步骤4评级前阅读）
```

> 同位置还需同步修复：
> - `../../docs/tutorials/debug/matmul_performance_guide.md`
> - `../../docs/tutorials/debug/performance_case_quantindexerprolog.md`

#### 问题 4：关键指令使用建议式表达且缺少“为什么”

- 命中规则：`[S2] R20`, `R23`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:136`
- 证据：`性能分析报告应包含以下内容：`
- 问题说明：“应包含”是建议式表达，不是可执行指令；同时未说明这些段落是为了保证可复核性与跨任务可比性。
- 修改建议（before/after）：

```diff
- 性能分析报告应包含以下内容：
+ 按以下固定结构生成报告，以保证不同算子结果可横向对比并可被自动审阅：
```

#### 问题 5：步骤缺少可验证成功标准与全局完成定义

- 命中规则：`[S2] R24`, `R27`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:22`
- 证据：`### 步骤 1：定位性能数据文件`
- 问题说明：步骤描述未定义“如何判定完成”；文档也未给出最终完成条件（例如报告产出路径、最小字段集合、错误为0）。
- 修改建议（before/after）：

```diff
- ### 步骤 1：定位性能数据文件
+ ### 步骤 1：定位性能数据文件（成功标准：目标目录存在且包含3个文件）
```

```diff
- ## 性能分析报告模板
+ ## 完成标准
+ - 已生成 `<output_dir>/performance_analysis_report.md`
+ - 报告包含“指标统计/评级/瓶颈/优化建议”四个必需章节
+ - 关键命令执行返回码为 0
```

#### 问题 6：工作流未定义失败恢复路径

- 命中规则：`[S2] R30`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:20`
- 证据：`## 工作流程`
- 问题说明：流程仅定义正常路径，未覆盖输入目录不存在、日志缺失、解析失败等常见异常场景。
- 修改建议（before/after）：

```diff
- ## 工作流程
+ ## 工作流程
+ ### 错误处理
+ - 若 `<output_dir>/bubble_analysis.log` 不存在：输出明确错误并终止
+ - 若日志解析结果核心数为 0：提示日志格式不匹配并给出样例正则
+ - 若报告写入失败：返回非零退出码并打印目标路径权限信息
```

#### 问题 7：脚本缺少对文件读取异常的基础捕获

- 命中规则：`[S2] R42`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/scripts/analyze_perf.py:60`
- 证据：`with open(log_path, 'r') as f:`
- 问题说明：当前读取路径仅依赖上游存在性检查，未处理权限、编码、I/O 中断等异常，失败时可能直接抛 traceback，用户提示不稳定。
- 修改建议（before/after）：

```diff
- with open(log_path, 'r') as f:
-     content = f.read()
+ try:
+     with open(log_path, 'r', encoding='utf-8') as f:
+         content = f.read()
+ except OSError as exc:
+     logging.error(f"读取日志失败: {log_path} ({exc})")
+     sys.exit(2)
```

### S3 轻微建议

#### 问题 8：存在模糊措辞，降低执行确定性

- 命中规则：`[S3] R21`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:97`
- 证据：`**高气泡率（>10%）可能原因：**`
- 问题说明：多处使用“可能原因”但未给出判断条件与排查顺序，执行者难以按统一标准复盘。
- 修改建议（before/after）：

```diff
- **高气泡率（>10%）可能原因：**
+ **高气泡率（>10%）判定与排查顺序：**
```

#### 问题 9：多个围栏代码块缺少语言标注

- 命中规则：`[S3] R46`
- 位置：`.agents/skills/pypto-operator-perf-analyzer/SKILL.md:59`
- 证据：````` ``` `````
- 问题说明：3.1~3.4 的公式代码块未声明语言，不利于渲染器高亮与结构化抽取。
- 修改建议（before/after）：

```diff
- ```
+ ```text
  核心利用率 = AicoreTime / (AicoreTime + 等待总时间) × 100%
  ```
```

## 6. 通过规则汇总

| 维度 | 通过规则 |
|------|----------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R45 |
| D3 | R15, R16, R17 |
| D4 | R22 |
| D5 | R25 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R51 |
| D8 | R34, R35, R36, R37, R38, R47 |
| D9 | R39, R40, R41, R50 |
| D0 | R43, R43 |
