# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-pass-ut-generate |
| 评审时间 | 2026-03-11 |
| 总分 | 88.00 / 100 |
| 等级 | B |
| S0 否决 | 否 |
| 规则统计 | 通过 40 / 失败 6 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 19.50 | R04(-10), R06(-2) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.00 | R19(-5) |
| D4 | 语言与表达 | 10% | 10.0 | 8.00 | R46(-2) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.0 | 无 |
| D6 | 工作流完整性工作流完整性 | 10% | 10.0 | 10.0 | 无 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.50 | R50(-2) |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无（无 scripts/ 目录） |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 46 |
| 跳过规则数 | 4 |
| 覆盖率 | 92% |

**跳过的规则**：R42, R47（原因：无 scripts/ 目录），R44（原因：context 不是 fork）

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
| R15 | | PASS | D3 | S2 | static |
| R16 | PASS | D3 | S2 | static |
| R17 | PASS | D3 | S2 | static |
| R18 | PASS | D3 | S2 | static |
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
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
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

#### 问题 1：name 字段与目录名不匹配

**命中规则**：R04 (S1)

**位置**：`SKILL.md:2`

**当前内容**：
> name: pass-ut-generate

**问题说明**：
frontmatter 中的 `name` 字段值为 `pass-ut-generate`，但实际目录名为 `pypto-pass-ut-generate`。根据规范，name 值应与 skill 目录名一致，以便于技能识别和调用。

**修改建议**：
> 将 name 字段修改为与目录名一致：
> ```yaml
> name: pypto-pass-ut-generate
> ```

---

### S2 中等问题

#### 问题 2：未知的 frontmatter 字段

**命中规则**：R06 (S3)

**位置**：`SKILL.md:4`

**当前内容**：
> license: 完整条款见 LICENSE.txt

**问题说明**：
frontmatter 中包含了 `license` 字段，该字段不在已知的 frontmatter 字段白名单中（name, description, context, agent, allowed-tools, user-invocable, intercept, model）。

**修改建议**：
> 删除未知的 license 字段，或将其移至文档正文中：
> ```yaml
> ---
> name: pypto-pass-ut-generate
> description: 根据Pass业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应Pass的Ut用例。
> ---
> 
> **许可证**：完整条款见 LICENSE.txt
> ```

---

#### 问题 3：被引用文件缺少用途与加载时机说明

**命中规则**：R19 (S2)

**位置**：`SKILL.md:210-224`

**当前内容**：
> | 类别 | 文件路径 |
> |------|----------|
> | 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |
> | 生成流程二示例 | `pypto/framework/tests/ut/passes/src/test_cube_process.cpp` |
> | ComputationalGraphBuilder | `pypto/framework/tests/ut/pares/src/computational_graph_builder.h` |
> | function信息 | `pypto/framework/src/interface/program/program.cpp和pypto/framework/src/interface/function/function.h`|
> | Opcode 定义 | `pypto/framework/src/interface/operation/opcode.h` |
> | op属性定义 | `framework/src/interface/operation/attribute.h` |
> | DataType 定义 | `pypto/framework/include/tilefwk/data_type.h` |
> | LogicalTensor 定义 | `pypto/framework/src/interface/tensor/logical_tensor.h` |
> | Operation信息 | `pypto/framework/src/interface/operation/operation.cpp`  |
> | COMPILE_STAGE策略 | `pypto/framework/src/interface/configs/config_manager_ng.h` |

**问题说明**：
参考资料章节列出了多个被引用的文件路径，但未说明每个文件的用途以及应在何时读取/加载这些文件。例如，未说明 ComputationalGraphBuilder.h 应在何时引用，或 test_removeredundantop.cpp 示例应在哪个步骤查看。

**修改建议**：
> 为每个被引用文件添加用途说明和加载时机：
> ```markdown
> ## 参考资料
> 
> | 类别 | 文件路径 | 用途 | 加载时机 |
> |------|----------|------|----------|
> | 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` | 手动构建计算图的完整示例 | 步骤4-6参考 |
> | 生成流程二示例 | `pypto/framework/tests/ut/passes/src/test_cube_process.cpp` | 使用 ComputationalGraphBuilder 的示例 | 步骤4参考 |
> | ComputationalGraphBuilder | `pypto/framework/tests/ut/passes/src/computational_graph_builder.h` | 辅助构建计算图的工具类 | 步骤4（流程二） |
> | function信息 | `pypto/framework/src/interface/program/program.cpp` 等 | Function 和 Program类API参考 | 步骤4（流程一） |
> | Opcode 定义 | `pypto/framework/src/interface/operation/opcode.h` | 操作码枚举定义 | 步骤6 |
> | COMPILE_STAGE策略 | `pypto/framework/src/interface/configs/config_manager_ng.h` | 编译阶段配置常量 | 步骤2 |
> ```

---

#### 问题 4：多选场景未提供默认推荐

**命中规则**：R50 (S2)

**位置**：`SKILL.md:31-203`

**当前内容**：
> ## UT生成流程一
> ...
> ## UT生成流程二
> ...

**问题说明**：
技能提供了两种 UT 生成流程（"UT生成流程一"和"UT生成流程二"），但未说明默认应使用哪种流程，或在什么情况下优先选择哪种流程。这可能导致用户/AI 在选择时产生困惑。

**修改建议**：
> 在两个流程前添加选择指南，标注默认推荐：
> ```markdown
> ## UT生成流程选择
> 
> 本技能提供两种 UT 生成方式：
> - **流程一（推荐）**：手动构建 Function、Tensor 和 Operation，适用于需要精细控制计算图结构的场景
> - **流程二**：使用 ComputationalGraphBuilder 辅助类，适用于快速构建标准计算图的场景
> 
> 若无特殊需求，默认使用流程一。
> 
> ---
> 
> ## UT生成流程一
> ...
> ```

---

### S3 轻微建议

#### 问题 5：代码块缺少语言标注

**命中规则**：R46 (S3)

**位置**：`SKILL.md:81, 92, 99, 129, 148`

**当前内容**：
> ```

**问题说明**：
文档中存在多个代码块使用 ``` 标记但未指定语言类型（如 ```cpp），这会影响代码高亮显示和阅读体验。

**修改建议**：
> 为所有代码块添加语言标注：
> ```markdown
> ```cpp
> void SetUp() override {
>     Program::GetInstance().Reset();
>     ...
> }
> ```
> ```

---

## 通过项

共 40 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R05, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18 |
| D4 | R20, R21, R22, R23 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
| D0 | R43, R43 |
