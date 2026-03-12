# 技能评审报告

## 1. 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-accuracy-verify |
| 评审时间 | 2026-03-11 13:56:50 |
| 总分 | 95.85 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | PASS 40 / FAIL 8 / WARN 0 / SKIP 3 |

## 2. 维度评分表

| 维度 | 名称 | 原始分(0-100) | 权重 | 加权分 | 扣分明细 |
|------|------|---------------|------|--------|----------|
| D1 | Frontmatter 元数据 | 98.00 | 25% | 24.50 | R06(-2) |
| D2 | 简洁性与效率 | 95.00 | 15% | 14.25 | R14(-5) |
| D3 | 文件结构与导航 | 95.00 | 10% | 9.50 | R19(-5) |
| D4 | 语言与表达 | 96.00 | 10% | 9.60 | R46(-2)；R21(-2) |
| D5 | 精确性与可执行性 | 80.00 | 10% | 8.00 | R24(-5)；R25(-5)；R26(-10) |
| D6 | 工作流完整性 | 100.00 | 10% | 10.00 | 无 |
| D7 | 模式与最佳实践 | 100.00 | 5% | 5.00 | 无 |
| D8 | 反模式检测 | 100.00 | 10% | 10.00 | 无 |
| D9 | 脚本与代码质量 | 100.00 | 5% | 5.00 | 无（`scripts/` 不存在，D9 满分） |
| 合计 | - | - | 100% | 95.85 | - |

## 3. 规则覆盖率

- 静态规则评估: 29 / 29
- 语义规则评估: PASS 13 / FAIL 6 / SKIP 3（共 22 条）
- 总评估数: 51 / 51
- 覆盖率: 100.00%

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | PASS | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | FAIL | D1 | S3 | static |
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
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
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
| R46 | FAIL | D4 | S3 | static |
| R47 | PASS | D8 | S2 | static |
| R50 | SKIP | D9 | S2 | semantic |
| R51 | PASS | D7 | S2 | semantic |

## 4. 质量门禁

- 过滤条目总数: 0
- internal_misbound_rule_or_evidence: 0
- low_information_snippet: 0

## 5. 问题列表

### S0
- 无

### S1
### 问题 1: tile size 检查缺少可执行方法
- 命中规则: [R26/S1]
- 位置: `SKILL.md:359`
- 证据: "3. **检查 tile size**：确认 tile 配置是否合理"
- 问题说明: “检查 tile size”只有目标描述，没有给出具体执行方法与判定阈值。
- 修复建议:
```text
修改前：3. **检查 tile size**：确认 tile 配置是否合理
修改后：增加可执行步骤：检查命令、关键参数、通过阈值与失败后的调整动作。
```

### S2
### 问题 1: 流程步骤缺少逐步成功标准
- 命中规则: [R24/S2]
- 位置: `SKILL.md:149`
- 证据: "### 步骤 1：准备测试数据"
- 问题说明: “步骤1~3”仅描述动作，缺少每步可验证成功标准。
- 修复建议:
```text
修改前：### 步骤 1：准备测试数据
修改后：为每个步骤增加完成判定，例如“步骤1通过条件：shape/dtype/seed 均满足预期”。
```
### 问题 2: 示例调用使用占位符不可直接执行
- 命中规则: [R25/S2]
- 位置: `SKILL.md:168`
- 证据: "your_npu_operator(input_data, actual_result)"
- 问题说明: 流程示例存在占位函数调用，命令与路径不够可直接执行。
- 修复建议:
```text
修改前：your_npu_operator(input_data, actual_result)
修改后：补充最小可运行示例（真实导入路径+函数名），并单独标注可替换项与替换规则。
```
### 问题 3: 容差策略在文档中重复出现
- 命中规则: [R14/S2]
- 位置: `SKILL.md:366`
- 证据: "# 策略1: 涉及指数/除法/Softmax 的算子"
- 问题说明: “容差策略”内容在“容差选择原则”和“容差调整策略”两处重复，正文存在冗余。
- 修复建议:
```text
修改前：# 策略1: 涉及指数/除法/Softmax 的算子
修改后：将“容差调整策略”改为引用前文“容差选择原则”，仅保留新增内容（如失败后的排查优先级），删除重复代码块。
```
### 问题 4: 参考路径缺少用途与加载时机
- 命中规则: [R19/S2]
- 位置: `SKILL.md:451`
- 证据: "- 精度验证示例: `models/glm_v4_5/`"
- 问题说明: 参考路径仅列出文件名，未逐项说明用途与加载时机（如 `models/glm_v4_5/`、`docs/api/`）。
- 修复建议:
```text
修改前：- 精度验证示例: `models/glm_v4_5/`
修改后：在“参考资料”中为每个路径补充“用途+何时读取”，例如“`docs/api/`：当需要确认 API 参数与 dtype 限制时读取”。
```

### S3
### 问题 1: frontmatter 包含未知字段 `license`
- 命中规则: [R06/S3]
- 位置: `SKILL.md:1`
- 证据: "license"
- 问题说明: 未知的 frontmatter 字段: license
- 修复建议:
```text
修改前：`license: 完整条款见 LICENSE.txt`
修改后：删除该 frontmatter 字段，并在正文新增“## 许可证”段落说明 `LICENSE.txt` 位置。
```
### 问题 2: 存在未标注语言的代码块
- 命中规则: [R46/S3]
- 位置: `SKILL.md:100`
- 证据: "```"
- 问题说明: 代码块缺少语言标注
- 修复建议:
```text
修改前：```
修改后：为该代码块补充语言标注（示例：```text）。
```
### 问题 3: 指令含弱化词影响可执行性
- 命中规则: [R21/S3]
- 位置: `SKILL.md:302`
- 证据: "对于复杂算子，可以分段验证中间结果："
- 问题说明: 可执行步骤使用弱化词“可以”，降低指令确定性。
- 修复建议:
```text
修改前：对于复杂算子，可以分段验证中间结果：
修改后：改为确定式语句：“对于复杂算子，按以下步骤分段验证中间结果”。
```

## 6. 通过规则汇总

- D1: R01、R02、R03、R04、R05、R07、R08、R09、R10、R44
- D2: R11、R12、R13、R45
- D3: R15、R16、R17、R18
- D4: R20、R22、R23
- D5: R27
- D6: R28、R29、R30、R31
- D7: R32、R33、R51
- D8: R34、R35、R36、R37、R38、R47
- D9: R39、R40、R41
- D0: R43、R43
