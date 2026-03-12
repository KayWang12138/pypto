# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-accuracy-verify |
| 评审时间 | 2026-03-11 |
| 总分 | 97.30 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 46 / 失败 3 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 24.50 | R06(S3,-2) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | - |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.50 | R19(S2,-5) |
| D4 | 语言与表达 | 10% | 10.0 | 9.80 | R46(S3,-2) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 9.50 | R50(S2,-5) |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | - |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | - |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | - |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | - |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 50 |
| 跳过规则数 | 0 |
| 覆盖率 | 100% |

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
| R14 | PASS | D2 | S2 | semantic |
| R15 | PASS | D3 | S2 | static |
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
| R42 | PASS | D9 | S2 | semantic |
| R43 | PASS | D0 | S2 | static |
| R44 | SKIP | D0 | S2 | semantic |
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
| R47 | PASS | D9 | S2 | semantic |
| R50 | FAIL | D5 | S2 | semantic |

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

#### 问题 1：被引用文件缺少用途和加载时机说明

**命中规则**：R19 (S2)

**位置**：`SKILL.md:71`

**当前内容**：
> 使用 `utils/np_compare.py` 中的 `detailed_allclose_manual` 进行详细分析：

**问题说明**：
技能中引用了 `utils/np_compare.py` 文件，但没有说明该文件的用途（是什么工具）和加载时机（何时应该读取或使用该文件）。虽然在第 452 行的参考资料中提到了该文件，但未明确说明其用途。

**修改建议**：
> 使用 `utils/np_compare.py` 中的 `detailed_allclose_manual` 进行详细分析（该文件提供详细的数组比较功能，包含异常元素定位和统计信息；在进行精度问题调试时读取使用）：

---

#### 问题 2：多选场景未提供默认推荐

**命中规则**：R50 (S2)

**位置**：`SKILL.md:44-49`

**当前内容**：
> | 算子类型 | rtol | atol | 说明 |
> |---------|------|------|------|
> | **复杂算子** (Attention, FFN) | `0.0078125` | `0.0001` | 涉及 Softmax、除法等不稳定运算 |
> | **简单算子** (Gate, MatMul) | `5e-3` | `5e-3` | 简单矩阵运算，容差稍宽 |
> | **通用算子** | `1e-3` | `1e-3` | 默认容差 |
> | **高精度要求** | `1e-5` | `1e-5` | 对精度要求极高的场景 |

**问题说明**：
容差配置策略表格提供了多种选项，但没有标注哪一个是默认推荐选项。用户可能不清楚在不确定算子类型时应选择哪一组容差参数。

**修改建议**：
> | 算子类型 | rtol | atol | 说明 |
> |---------|------|------|------|
> | **通用算子** (推荐) | `1e-3` | `1e-3` | 默认容差，适用于大多数场景 |
> | **复杂算子** (Attention, FFN) | `0.0078125` | `0.0001` | 涉及 Softmax、除法等不稳定运算 |
> | **简单算子** (Gate, MatMul) | `5e-3` | `5e-3` | 简单矩阵运算，容差稍宽 |
> | **高精度要求** | `1e-5` | `1e-5` | 对精度要求极高的场景 |

---

### S3 轻微建议

#### 问题 3：未知的 frontmatter 字段

**命中规则**：R06 (S3)

**位置**：`SKILL.md:4`

**当前内容**：
> license: 完整条款见 LICENSE.txt

**问题说明**：
frontmatter 中包含了 `license` 字段，该字段不在已知的 frontmatter 字段白名单中（name, description, context, agent, allowed-tools, user-invocable, intercept, model）。

**修改建议**：
> 删除 `license` 字段，或将许可证信息移至文档正文部分。

---

#### 问题 4：代码块缺少语言标注

**命中规则**：R46 (S3)

**位置**：`SKILL.md:100`

**当前内容**：
> ```
> 开始比较数组，形状: (32, 5120), 总元素数: 163840

**问题说明**：
第 100 行的代码块使用 ``` 开始，但没有指定语言标识符（如 python、bash 等）。这会影响代码高亮显示。

**修改建议**：
> ```text
> 开始比较数组，形状: (32, 5120), 总元素数: 163840

---

## 通过项

共 46 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18 |
| D4 | R20, R21, R22, R23 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42, R47 |
| D0 | R43, R43 |
