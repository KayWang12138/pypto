# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-develop-workflow |
| 评审时间 | 2026-03-11 13:58:35 +0800 |
| 总分 | 94.35 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | PASS 39 / FAIL 11 / WARN 0 / SKIP 1 |

## 维度评分表

| 维度 | 原始分(0-100) | 权重 | 加权分 | 扣分明细 |
|------|---:|---:|---:|------|
| D1 | 98 | 25% | 24.50 | R06(-2) |
| D2 | 95 | 15% | 14.25 | R14(-5) |
| D3 | 90 | 10% | 9.00 | R18(-5), R19(-5) |
| D4 | 91 | 10% | 9.10 | R46(-2), R21(-2), R23(-5) |
| D5 | 80 | 10% | 8.00 | R24(-5), R26(-10), R27(-5) |
| D6 | 95 | 10% | 9.50 | R31(-5) |
| D7 | 100 | 5% | 5.00 | 无 |
| D8 | 100 | 10% | 10.00 | 无 |
| D9 | 100 | 5% | 5.00 | 无 |

总分校验：24.50 + 14.25 + 9.00 + 9.10 + 8.00 + 9.50 + 5.00 + 10.00 + 5.00 = **94.35**

## 规则覆盖率

- 静态规则：29（PASS 26 / FAIL 3 / SKIP 0）
- 语义规则：22（PASS 13 / FAIL 8 / SKIP 1）
- 总评估数：51 / 51
- 覆盖率：100.00%

### 规则状态明细（51条）

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
| R18 | FAIL | D3 | S2 | static |
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
| R27 | FAIL | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | PASS | D6 | S2 | semantic |
| R31 | FAIL | D6 | S2 | semantic |
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
| R43 | PASS | D0 | S2 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
| R47 | PASS | D8 | S2 | static |
| R50 | PASS | D9 | S2 | semantic |
| R51 | PASS | D7 | S2 | semantic |

## 质量门禁

| 类型 | 数量 | 说明 |
|------|---:|------|
| internal_misbound_rule_or_evidence | 0 | 未发现引用 reviewer 自身内容的语义 finding |
| low_information_snippet | 0 | 所有纳入评分的证据片段均可在目标 skill 中逐字匹配 |

- 被过滤条目：无

## 问题列表

### S0

- 无

### S1

#### 问题 1：阶段动作缺少可执行实现方法（R26）
- 命中规则：[R26|S1]
- 位置：`SKILL.md:142`
- 证据：`当阶段五保证算子基础版本正确后，请做以下高阶参数的使能`
- 问题说明：该段要求“使能高阶参数”，但未给出具体执行方法（命令、API、配置文件位置、最小示例）。
- 修复建议（修改前/后）：
  - before：`1. 涉及loop的话，请使能loop_unroll`
  - after：`1. 在算子入口显式设置 pass 选项，例如 set_pass_options({"loop_unroll": true})，并在日志中确认 loop_unroll 已生效。`

### S2

#### 问题 2：内容存在跨章节重复（R14）
- 命中规则：[R14|S2]
- 位置：`SKILL.md:123`
- 证据：`优先使用npu模式进行验证` / `2. 优先使用npu模式进行精度验证`
- 问题说明：同一条关键指令在“阶段五”和“注意事项”重复出现，造成冗余。
- 修复建议（修改前/后）：
  - before：两处均保留相同指令
  - after：在“阶段五”保留主指令，在“注意事项”改为引用式补充：`详见“阶段五：测试验证”的执行要求。`

#### 问题 3：引用文件不存在且缺少可用替代（R18, R19）
- 命中规则：[R18|S2], [R19|S2]
- 位置：`SKILL.md:46`
- 证据：`详细的错误示例、正确做法和经验教训请查看：**[common_issues.md](./common_issues.md)**`
- 问题说明：`./common_issues.md` 文件不存在；同时该引用未提供“不可用时如何处理”的时机说明。
- 修复建议（修改前/后）：
  - before：`[common_issues.md](./common_issues.md)`
  - after：`[references/common_issues.md](./references/common_issues.md)`（并补充：若文件缺失，改查 `docs/install/` 与 `examples/` 的对应章节）

#### 问题 4：关键指令缺少“为什么”（R23）
- 命中规则：[R23|S2]
- 位置：`SKILL.md:142`
- 证据：`当阶段五保证算子基础版本正确后，请做以下高阶参数的使能`
- 问题说明：要求进入“高阶参数使能”，但未解释目的（例如性能收益、避免回归、覆盖特定算子模式）。
- 修复建议（修改前/后）：
  - before：仅要求“做以下高阶参数的使能”
  - after：补充原因：`为降低循环开销并提升内存访问效率，在基础正确性通过后再开启高阶参数，避免将功能错误与调优错误混淆。`

#### 问题 5：多个步骤缺少可验证成功标准（R24）
- 命中规则：[R24|S2]
- 位置：`SKILL.md:50`
- 证据：`### 阶段一：需求检查`
- 问题说明：阶段一/三/四/六仅列动作，没有“通过判据”（例如必须产出何文件、满足何阈值、出现何日志）。
- 修复建议（修改前/后）：
  - before：`必需信息清单：...`
  - after：新增“成功标准”小节：`信息项完整率=100%，缺失项均填默认值并记录；否则禁止进入下一阶段。`

#### 问题 6：缺少全局完成标准（R27）
- 命中规则：[R27|S2]
- 位置：`SKILL.md:48`
- 证据：`## 开发阶段`
- 问题说明：文档未定义“整体任务完成”的判定条件，用户无法确认何时交付结束。
- 修复建议（修改前/后）：
  - before：无“完成定义”章节
  - after：新增 `## 完成标准`：至少包含功能通过、精度阈值达标、性能数据已采集、README已更新四项硬门槛。

#### 问题 7：条件分支覆盖不完整（R31）
- 命中规则：[R31|S2]
- 位置：`SKILL.md:128`
- 证据：`检查到存在npu卡的时候，直接使用run_mode=npu执行`
- 问题说明：仅定义“有 NPU 卡”的路径，未定义“无 NPU 卡”或“检测失败”的兜底路径。
- 修复建议（修改前/后）：
  - before：仅给出 `run_mode=npu`
  - after：补充分支：`若无NPU卡或检测失败，则使用 run_mode=sim 执行基础功能校验，并标注“仅功能验证，不做性能结论”。`

### S3

#### 问题 8：frontmatter 出现未知字段（R06）
- 命中规则：[R06|S3]
- 位置：`SKILL.md:4`
- 证据：`tag: [PyPTO，算子开发]`
- 问题说明：`tag` 不在允许字段白名单（name/description/context/agent/allowed-tools/user-invocable/intercept/model）中。
- 修复建议（修改前/后）：
  - before：`tag: [PyPTO，算子开发]`
  - after：删除该字段，或将触发语义并入 `description`。

#### 问题 9：语句包含模糊措辞（R21）
- 命中规则：[R21|S3]
- 位置：`SKILL.md:74`
- 证据：`如果不存在，初始化也不成功，可以参考`docs/install/prepare_environment.md`获取pto-isa源码章节，并进行环境变量的设置。`
- 问题说明：`可以参考` 属于弱化表达，执行优先级与动作边界不清晰。
- 修复建议（修改前/后）：
  - before：`可以参考 ... 并进行环境变量的设置`
  - after：`必须阅读 docs/install/prepare_environment.md 中 pto-isa 章节，并按文档完成环境变量设置后再继续。`

#### 问题 10：代码块缺少语言标注（R46）
- 命中规则：[R46|S3]
- 位置：`SKILL.md:13`
- 证据：`需求检查 → 环境准备 → Plan 模式 → 开发实现 → 测试验证 → 高阶参数使能`
- 问题说明：围栏代码块使用 ``` 但未标注语言，影响渲染与可读性。
- 修复建议（修改前/后）：
  - before：`\n```\n需求检查 → ...\n```\n`
  - after：`\n```text\n需求检查 → ...\n```\n`

## 通过规则汇总

- D0：R43, R43
- D1：R01, R02, R03, R04, R05, R07, R08, R09, R10, R44
- D2：R11, R12, R13, R45
- D3：R15, R16, R17
- D4：R20, R22
- D5：R25
- D6：R28, R29, R30
- D7：R32, R33, R51
- D8：R34, R35, R36, R37, R38, R47
- D9：R39, R40, R41, R42, R50
