# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-binary-search-verify |
| 评审时间 | 2026-03-11  |
| 总分 | 94.45 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 40 / 失败 8 / 警告 0 / 跳过 3 |

## 维度评分表

| 维度 | 原始分(0-100) | 权重 | 加权分 | 扣分明细（FAIL） |
|------|---------------|------|--------|------------------|
| D1 | 88 | 25% | 22.00 | R04(-10), R06(-2) |
| D2 | 95 | 15% | 14.25 | R14(-5) |
| D3 | 95 | 10% | 9.50 | R19(-5) |
| D4 | 92 | 10% | 9.20 | R46×3(-6), R21(-2) |
| D5 | 95 | 10% | 9.50 | R25(-5) |
| D6 | 100 | 10% | 10.00 | 无 |
| D7 | 100 | 5% | 5.00 | 无 |
| D8 | 100 | 10% | 10.00 | 无 |
| D9 | 100 | 5% | 5.00 | 无（未检测到 `scripts/` 目录，按规范满分） |

- 总分计算：22.00 + 14.25 + 9.50 + 9.20 + 9.50 + 10.00 + 5.00 + 10.00 + 5.00 = **94.45**

## 规则覆盖率

- 静态规则：29 = PASS 25 + FAIL 4 + SKIP 0
- 语义规则：22 = PASS 15 + FAIL 4 + SKIP 3
- 总评估：51 = PASS 40 + FAIL 8 + SKIP 3
- 覆盖率：51 / 51 = **100.00%**
- SKIP 规则：R42, R44, R50
  - R42/R50：目标 skill 无 `scripts/` 目录（脚本位于根目录 `verify_binary_search.py`，不属于 `scripts/` 检查域）
  - R44：frontmatter 未设置 `context: fork`

## 质量门禁

- 被过滤条目总数：0
- internal_misbound_rule_or_evidence：0
- low_information_snippet：0
- 说明：本次语义 findings 均绑定目标 skill，且证据片段均可在源文件逐字匹配。

## 问题列表

### S0 致命缺陷

- 无。

### S1 重大问题

#### 问题 1：frontmatter 标识与目录命名不一致，且包含未知字段
- 命中规则：R04(S1), R06(S3)
- 位置：`SKILL.md:2`, `SKILL.md:4`
- 证据：
  - `name: pypto-verify-binary-search`
  - `license: 完整条款见 LICENSE.txt`
- 问题说明：`name` 与目录名 `pypto-binary-search-verify` 不一致；同时 `license` 不在允许字段白名单中。
- 修改建议（before/after）：
```diff
- name: pypto-verify-binary-search
- license: 完整条款见 LICENSE.txt
+ name: pypto-binary-search-verify
```

### S2 中等问题

#### 问题 2：存在非标准子目录
- 命中规则：R43(S2)
- 位置：`__pycache__:0`
- 证据：`__pycache__`
- 问题说明：skill 根目录出现非标准子目录 `__pycache__`，不在标准集合（references/scripts/templates/assets/examples）中。
- 修改建议（before/after）：
```diff
- .agents/skills/pypto-binary-search-verify/__pycache__/
+ （删除该目录，避免将运行时缓存纳入 skill 资产）
```

#### 问题 3：工作流与检查清单存在重复指令
- 命中规则：R14(S2)
- 位置：`SKILL.md:160`, `SKILL.md:356`
- 证据：
  - `在 jit 和 golden 函数中插入对应的检查点（参考原则 2 和 3）。`
  - `- [ ] **步骤 0**：先验证整体结果`
- 问题说明：`完整工作流程` 已给出步骤 0~5，`检查清单` 再次逐条重复，信息增量低。
- 修改建议（before/after）：
```diff
- ## 检查清单
- （重复列出步骤0-5及其子项）
+ ## 检查清单
+ - [ ] 已完成“完整工作流程”的步骤 0~5
+ - [ ] 已清理调试文件并回归验证
```

#### 问题 4：被引用文件的用途/加载时机描述与实际路径不一致
- 命中规则：R19(S2), R25(S2)
- 位置：`SKILL.md:23`, `SKILL.md:29`
- 证据：
  - `本技能提供了通用对比脚本 `scripts/verify_binary_search.py`，自动完成检查点扫描和对比。`
  - `python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py`
- 问题说明：文档引用 `scripts/verify_binary_search.py`，但实际文件位于 skill 根目录 `verify_binary_search.py`；命令路径也使用了旧 skill 名 `pypto-verify-binary-search`。
- 修改建议（before/after）：
```diff
- 本技能提供了通用对比脚本 `scripts/verify_binary_search.py`
- python3 .opencode/skills/pypto-verify-binary-search/scripts/verify_binary_search.py
+ 本技能提供了通用对比脚本 `verify_binary_search.py`（在本 skill 根目录）
+ python3 .agents/skills/pypto-binary-search-verify/verify_binary_search.py
```

### S3 轻微建议

#### 问题 5：代码块缺少语言标注（二分策略示意）
- 命中规则：R46(S3)
- 位置：`SKILL.md:123`
- 证据：`输入 [op1] [op2] [op3] ... [opN] 输出`
- 问题说明：围栏代码块使用 ``` 而未标注语言。
- 修改建议（before/after）：
```diff
- ```
+ ```text
```

#### 问题 6：代码块缺少语言标注（工具建议输出示例）
- 命中规则：R46(S3)
- 位置：`SKILL.md:180`
- 证据：`✗ 检查点 checkpoint1 匹配，但 checkpoint2 不匹配`
- 问题说明：围栏代码块使用 ``` 而未标注语言。
- 修改建议（before/after）：
```diff
- ```
+ ```text
```

#### 问题 7：代码块缺少语言标注（渐进式二分示意）
- 命中规则：R46(S3)
- 位置：`SKILL.md:249`
- 证据：`第1轮：输入 → 中间 → 输出（3个检查点）`
- 问题说明：围栏代码块使用 ``` 而未标注语言。
- 修改建议（before/after）：
```diff
- ```
+ ```text
```

#### 问题 8：存在弱化措辞，降低指令确定性
- 命中规则：R21(S3)
- 位置：`SKILL.md:261`
- 证据：`对于大数据量的 tensor，可以：`
- 问题说明：`可以` 属于模糊/弱化措辞，执行优先级不够明确。
- 修改建议（before/after）：
```diff
- 对于大数据量的 tensor，可以：
+ 对于大数据量的 tensor，按以下方式处理：
```

## 通过规则汇总

- D1：R01, R02, R03, R05, R07, R08, R09, R10, R44
- D2：R11, R12, R13, R45
- D3：R15, R16, R17, R18
- D4：R20, R22, R23
- D5：R24, R26, R27
- D6：R28, R29, R30, R31
- D7：R32, R33, R51
- D8：R34, R35, R36, R37, R38, R47
- D9：R39, R40, R41
- D0：R43
