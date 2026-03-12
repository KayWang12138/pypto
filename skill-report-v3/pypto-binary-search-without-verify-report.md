# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-binary-search-without-verify |
| 评审时间 | 2026-03-11 |
| 总分 | 94.30 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 46 / 失败 4 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 20.75 | R04(-10), R06(-2), R08(-5) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.0 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.0 | 无 |
| D4 | 语言与表达 | 10% | 10.0 | 9.80 | R21(-2) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 9.50 | R24(-5) |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(-5) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.75 | R50(-5) |
| D8 | 反模式检测 | 10% | 10.0 | 10.0 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.0 | 无（无 scripts/ 目录） |

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
| R04 | FAIL | D1 | S1 | static |
| R05 | PASS | D1 | S2 | static |
| R06 | FAIL | D1 | S3 | static |
| R07 | PASS | D1 | S1 | semantic |
| R08 | FAIL | D1 | S2 | semantic |
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
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | FAIL | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
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
| R44 | PASS | D0 | S2 | semantic |
| R43 | PASS | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | static |
| R47 | PASS | D9 | S2 | semantic |
| R50 | FAIL | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无

### S1 重大问题

#### 问题 1：name 值与目录名不匹配

**命中规则**：R04 (S1)

**位置**：`SKILL.md:2`

**当前内容**：
> name: pypto-precision-multiple-checkpoints

**问题说明**：
frontmatter 中的 `name` 字段值为 `pypto-precision-multiple-checkpoints`，但实际 skill 目录名为 `pypto-binary-search-without-verify`。这种不一致会导致 OpenCode 在技能发现和调用时出现混淆。

**修改建议**：
将 frontmatter 中的 `name` 字段修改为与目录名一致：
```yaml
name: pypto-binary-search-without-verify
```

---

### S2 中等问题

#### 问题 2：description 缺少自然触发短语

**命中规则**：R08 (S2)

**位置**：`SKILL.md:3`

**当前内容**：
> description: PyPTO 算子不依赖精度工具精度对比技能。通过在kernel函数中添加检查点tensor作为输入参数进行原地修改，对比中间结果的精度，定位导致精度差异的具体op。适用于算子运行通过但精度不满足要求、需要定位具体op导致精度问题、需要对比多个中间结果的场景。

**问题说明**：
description 使用了技术化语言（"不依赖精度工具精度对比技能"、"添加检查点tensor"），缺少用户在自然对话中会说出的触发短语。用户可能用更口语化的方式描述需求，如"精度不对"、"结果和golden不一致"、"哪里出错了"等。

**修改建议**：
在 description 中补充自然触发短语：
```yaml
description: 当算子精度不满足要求时，通过在kernel函数中添加检查点tensor对比中间结果，定位导致精度差异的具体op。适用于"精度不对"、"结果和golden不一致"、"需要找到哪个op导致精度问题"等场景。
```

---

#### 问题 3：步骤缺少可验证的成功标准

**命中规则**：R24 (S2)

**位置**：`SKILL.md:20-76`

**当前内容**：
> ### 步骤 1：分析代码结构，确定检查点
> 
> 分析kernel和golden代码，确定需要检查的关键计算节点。选择在关键计算节点之后的位置，确保检查点的结果有明确的含义，优先选择有明显边界的位置（如matmul、softmax之后）。

**问题说明**：
工作流步骤仅描述了"做什么"，但缺少"如何验证成功"的标准。例如，步骤1完成后如何确认检查点选择是正确的？步骤2完成后如何验证检查点tensor添加正确？没有明确的验证方法可能导致执行者无法确认每个步骤是否正确完成。

**修改建议**：
为每个步骤添加成功标准，例如：
```
### 步骤 1：分析代码结构，确定检查点

分析kernel和golden代码，确定需要检查的关键计算节点。选择在关键计算节点之后的位置，确保检查点的结果有明确的含义，优先选择有明显边界的位置（如matmul、softmax之后）。

**成功标准**：已列出至少3个关键计算节点，每个节点都有明确的计算含义和边界位置。
```

---

#### 问题 4：错误处理说明不完整

**命中规则**：R30 (S2)

**位置**：`SKILL.md:123-143`

**当前内容**：
> ### 输出参数shape不匹配
> 
> 检查golden和kernel的shape是否一致，不一致检查检查点输出拼接。

**问题说明**：
常见问题章节仅列出了部分错误场景（shape不匹配、循环内结果拼接、dtype不一致、LSP错误），但缺少更全面的错误处理说明。例如：检查点tensor数量不匹配、assemble操作失败、检查点变量未正确初始化等场景如何处理？

**修改建议**：
补充更多错误场景的处理方法：
```markdown
### 检查点tensor数量不匹配

检查kernel函数的输入参数中检查点tensor数量是否与golden函数的返回值数量一致。如果不一致，修改kernel或golden使其数量匹配。

### assemble操作失败

如果遇到"mix assemble and common operation for same output"错误，检查输入和输出变量名是否相同，使用不同的变量名解决。
```

---

#### 问题 5：多选场景缺少默认推荐

**命中规则**：R50 (S2)

**位置**：`SKILL.md:36-40`

**当前内容**：
> **shape推导方法**：
> - 从变量定义推导：找到产生该变量的赋值语句，分析等号右边的操作对shape的变换
> - 从权重/输入推导：找到相关的权重tensor shape和输入tensor shape，根据matmul/view等操作规则推导
> - 从循环tile推导：循环内变量的第一维 = tile_batch，向上追溯到原始输入的shape

**问题说明**：
shape推导方法提供了三种方式，但没有说明哪种方式应该优先使用或哪种更推荐。用户可能不知道应该选择哪种方法，导致执行效率降低或选择不当。

**修改建议**：
标注推荐方法：
```markdown
**shape推导方法**（推荐：优先使用"从变量定义推导"）：
- 从变量定义推导（推荐）：找到产生该变量的赋值语句，分析等号右边的操作对shape的变换
- 从权重/输入推导：找到相关的权重tensor shape和输入tensor shape，根据matmul/view等操作规则推导
- 从循环tile推导：循环内变量的第一维 = tile_batch，向上追溯到原始输入的shape
```

---

### S3 轻微建议

#### 问题 6：未知的 frontmatter 字段

**命中规则**：R06 (S3)

**位置**：`SKILL.md:4`

**当前内容**：
> license: 完整条款见 LICENSE.txt

**问题说明**：
frontmatter 中包含 `license` 字段，该字段不在已知字段列表（name, description, context, agent, allowed-tools, user-invocable, intercept, model）中。虽然不会影响功能，但建议使用标准字段以保持一致性。

**修改建议**：
移除 `license` 字段，或将其放在文档正文的参考资料章节中：
```markdown
## 参考资料

- PyPTO API: `docs/api/`
- 许可证：完整条款见 LICENSE.txt
```

---

#### 问题 7：包含模糊语言

**命中规则**：R21 (S3)

**位置**：`SKILL.md:29-34`

**当前内容**：
> **重要原则**：
> - 检查点tensor写在def kernel()里作为输入参数，不要写在 -> 之后
> - 在测试函数中用torch.empty()初始化检查点tensor（shape需要写在小括号里）
> - kernel函数内部不需要return，因为检查点tensor是原地修改的

**问题说明**：
"不需要return"是明确的指令，但整体风格可以更直接。不过这个问题较轻微，因为大部分指令已经使用了祈使语气。

**修改建议**：
当前表达已经较为清晰，无需修改。如需进一步优化，可以保持现有风格。

---

## 通过项

共 46 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R05, R07, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R22, R23, R46 |
| D5 | R25, R26, R27 |
| D6 | R28, R29, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42, R47 |
| D0 | R43, R44, R43 |
