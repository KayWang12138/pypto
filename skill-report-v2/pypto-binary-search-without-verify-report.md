# 技能评审报告

## 评审摘要
| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-binary-search-without-verify |
| 评审时间 | 2026-03-11 14:01:30 |
| 总分 | 94.45 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | PASS 40 / FAIL 8 / WARN 0 / SKIP 3 |

## 维度评分表
| 维度 | 原始分(0-100) | 权重 | 加权分 | 扣分明细 |
|------|---------------|------|--------|----------|
| D1 | 88.00 | 25% | 22.00 | R04(-10)；R06(-2) |
| D2 | 100.00 | 15% | 15.00 | 无 |
| D3 | 95.00 | 10% | 9.50 | R19(-5) |
| D4 | 98.00 | 10% | 9.80 | R21(-2) |
| D5 | 85.00 | 10% | 8.50 | R24(-5)；R26(-10) |
| D6 | 100.00 | 10% | 10.00 | 无 |
| D7 | 93.00 | 5% | 4.65 | R33(-2)；R51(-5) |
| D8 | 100.00 | 10% | 10.00 | 无 |
| D9 | 100.00 | 5% | 5.00 | 无（无 `scripts/` 目录，按规则满分） |
| 合计 | - | - | 94.45 | - |

## 规则覆盖率
- 静态规则：29（PASS 27 / FAIL 2 / SKIP 0）
- 语义规则：22（PASS 13 / FAIL 6 / SKIP 3）
- 总评估数：51 / 51
- 覆盖率：100.00%
- 跳过规则：R42、R44、R50

### 规则状态明细
| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | FAIL | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | FAIL | D1 | S3 | static |
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
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
| R27 | PASS | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | PASS | D6 | S2 | semantic |
| R31 | PASS | D6 | S2 | semantic |
| R32 | PASS | D7 | S3 | semantic |
| R33 | FAIL | D7 | S3 | semantic |
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
| R51 | FAIL | D7 | S2 | semantic |

## 质量门禁
| 类型 | 数量 | 说明 |
|------|------|------|
| internal_misbound_rule_or_evidence | 0 | 未发现引用 reviewer 自身内容的错绑证据 |
| low_information_snippet | 0 | 未发现无法在目标文件逐字匹配的证据片段 |

## 问题列表
### S0 致命缺陷
- 无

### S1 重大问题
#### 问题 1：frontmatter 名称与目录不一致且包含未知字段

**命中规则**：[S1] R04、[S3] R06  

**位置**：`SKILL.md:2`  

**证据**：> name: pypto-precision-multiple-checkpoints  

**问题说明**：`name` 字段与目录名 `pypto-binary-search-without-verify` 不一致，同时 frontmatter 出现白名单外字段 `license`，降低 skill 元数据可发现性与一致性。  

**修改建议（before/after）**：
```diff
- name: pypto-precision-multiple-checkpoints
- license: 完整条款见 LICENSE.txt
+ name: pypto-binary-search-without-verify
+ # 将许可证信息移至 README 或仓库 LICENSE，frontmatter 仅保留受支持字段
```

#### 问题 2：关键执行与验证缺少确定性命令且包含模糊表述

**命中规则**：[S3] R21、[S1] R26、[S3] R33  

**位置**：`SKILL.md:66`  

**证据**：> 创建检查点tensor，执行kernel（kernel内部会原地修改checkpoint），执行golden，对比最终结果和所有检查点。  

**问题说明**：文档要求“运行测试并分析结果”，但未给出可直接执行的命令/脚本（R26、R33）；同时同段落出现“可能是卡冲突了”这类弱化措辞（R21），影响故障定位确定性。  

**修改建议（before/after）**：
```diff
- 创建检查点tensor，执行kernel（kernel内部会原地修改checkpoint），执行golden，对比最终结果和所有检查点。
- 注意：...device输出为0，可能是卡冲突了或者检查点添加有误。
+ 创建检查点tensor后执行以下命令：`python3 test_xxx.py --case level0`。
+ 成功标准：日志中 `final_result: PASS` 且 `checkpoint_mismatch_count=0`。
+ 若 `device_output=0`，先执行 `npu-smi info` 检查设备占用；再逐项核对 checkpoint 的 shape/dtype 与 golden 是否一致。
```

### S2 中等问题
#### 问题 1：步骤缺少可验证成功标准

**命中规则**：[S2] R24  

**位置**：`SKILL.md:20`  

**证据**：> ### 步骤 1：分析代码结构，确定检查点  

**问题说明**：流程步骤提供了动作但未给出“完成判据”，执行者难以判断步骤是否通过（如检查点是否覆盖关键路径、是否满足可比对性）。  

**修改建议（before/after）**：
```diff
- ### 步骤 1：分析代码结构，确定检查点
+ ### 步骤 1：分析代码结构，确定检查点
+ 成功标准：输出一份检查点清单（变量名、shape、dtype、所在计算节点），并确认每个检查点在 golden 中有一一对应变量。
```

#### 问题 2：多选方案未给出默认推荐路径

**命中规则**：[S2] R51  

**位置**：`SKILL.md:131`  

**证据**：> 使用切片赋值或者view, assemble赋值，根据代码逻辑选择合适的方法。  

**问题说明**：在“切片赋值”和“view+assemble”两种方案并列时，未标注默认优先选项，执行者难以快速决策。  

**修改建议（before/after）**：
```diff
- 使用切片赋值或者view, assemble赋值，根据代码逻辑选择合适的方法。
+ 默认优先使用切片赋值（推荐），仅当存在跨tile重排或布局不连续时再使用 view + assemble。
```

#### 问题 3：引用文件缺少用途与加载时机说明

**命中规则**：[S2] R19  

**位置**：`SKILL.md:147`  

**证据**：> - PyPTO API: `docs/api/`  

**问题说明**：参考项仅给出路径，未说明何时读取 `docs/api/` 以及该目录在流程中的具体作用。  

**修改建议（before/after）**：
```diff
- - PyPTO API: `docs/api/`
+ - PyPTO API: `docs/api/`（用途：核对 API 参数与约束；时机：步骤2修改 kernel 前、步骤3对齐 golden 前必读）
```

### S3 轻微建议
- 无


## 通过规则汇总
- D1：R01, R02, R03, R05, R07, R08, R09, R10, R44
- D2：R11, R12, R13, R14, R45
- D3：R15, R16, R17, R18
- D4：R20, R22, R23, R46
- D5：R25, R27
- D6：R28, R29, R30, R31
- D7：R32
- D8：R34, R35, R36, R37, R38, R47
- D9：R39, R40, R41
- D0：R43, R43
