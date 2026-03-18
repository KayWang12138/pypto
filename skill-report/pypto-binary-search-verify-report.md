# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-binary-search-verify |
| 评审时间 | 2026-03-16 |
| 总分 | 96.30 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 43 / 失败 4 / 警告 1 / 跳过 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 22.50 | R04(S1): -10 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无扣分 |
| D4 | 语言与表达 | 10% | 10.0 | 9.30 | R23(S2): -5, R46(S3): -2 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无扣分 |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(S2): -5 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无扣分 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无扣分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 48 |
| 已评估规则数 | 48 |
| 跳过规则数 | 0 |
| 覆盖率 | 100.0% |

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | FAIL | D1 | S1 | static |
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
| R18 | PASS | D3 | S2 | static |
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | PASS | D5 | S2 | semantic |
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
| R43 | PASS | D3 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | semantic |
| R47 | PASS | D7 | S2 | semantic |
| R48 | WARN | D1 | S3 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 无 |
| 证据不足条目 | 0 | 无 |

## 问题清单

### S0 致命缺陷

无

### S1 重大问题

#### 问题 1：name 字段与目录名不匹配

**命中规则**：R04 (S1)

> 规则内容：name 值应与 skill 目录名一致

**位置**：`SKILL.md:1`

**当前内容**：
> name: pypto-verify-binary-search

**问题说明**：
frontmatter 中的 name 字段值为 `pypto-verify-binary-search`，但实际目录名为 `pypto-binary-search-verify`，两者不一致。这可能导致技能调用时的路径匹配问题。

**修改建议**：
将 frontmatter 中的 name 字段从 `pypto-verify-binary-search` 改为 `pypto-binary-search-verify`，使其与目录名保持一致。

**修改前/后对比**：
```yaml
# 修改前
name: pypto-verify-binary-search

# 修改后
name: pypto-binary-search-verify
```

---

### S2 中等问题

#### 问题 2：部分指令缺少原因解释

**命中规则**：R23 (S2)

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:96`

**当前内容**：
> 必须设置 `verify_options={"enable_pass_verify": True}`

**问题说明**：
该指令仅说明要做什么（设置 verify_options），未解释为何必须设置此选项。缺少原因解释会降低指令的可理解性和执行意愿。

**修改建议**：
补充原因说明，例如："必须设置 `verify_options={"enable_pass_verify": True}`，因为 pass_verify_save 功能默认是关闭的，需要显式启用才能输出中间结果"

**修改前/后对比**：
```markdown
# 修改前
必须设置 `verify_options={"enable_pass_verify": True}`

# 修改后
必须设置 `verify_options={"enable_pass_verify": True}`，因为 pass_verify_save 功能默认是关闭的，需要显式启用才能输出中间结果
```

---

#### 问题 3：缺少错误处理说明

**命中规则**：R30 (S2)

> 规则内容：必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:339`

**当前内容**：
> ## 常见问题

**问题说明**：
工作流中缺少错误处理说明。例如：如果找不到 output 目录怎么办？如果 golden 文件不存在怎么处理？工具返回错误时应该如何排查？虽然有"常见问题"章节，但未在主工作流中明确说明错误处理步骤。

**修改建议**：
在"完整工作流程"章节中增加"错误处理"子章节，说明常见错误场景及处理方法，如：output 目录不存在时检查是否正确运行了测试；golden 文件不存在时检查是否正确插入了保存代码。

**建议新增内容**：
```markdown
### 错误处理

在执行二分查找过程中可能遇到以下错误：

1. **未找到 output 目录**
   - 检查是否已运行测试生成数据
   - 确认 verify_options 是否正确设置

2. **未找到 golden 文件**
   - 检查 golden 函数中是否正确调用了 tofile()
   - 确认文件命名是否符合约定（golden_{checkpoint_name}.bin）

3. **检查点名称不匹配**
   - 确保 jit 和 golden 的检查点名称一致
   - 注意去掉 golden_ 前缀和数字后缀后名称应对应
```

---

### S3 轻微建议

#### 问题 4：代码块缺少语言标注

**命中规则**：R46 (S3)

> 规则内容：代码块应带有语言标注，纯文本块除外

**位置**：`SKILL.md:123`（同时影响第180行、第249行）

**当前内容**：
```
输入 [op1] [op2] [op3] ... [opN] 输出
```

**问题说明**：
第123行、第180行、第249行的代码块缺少语言标注。这些代码块虽然是纯文本图表，但添加语言标注可以提高可读性和一致性。

**修改建议**：
为这三个代码块添加 `text` 语言标注。

**修改前/后对比**：
````markdown
# 修改前
```
输入 [op1] [op2] [op3] ... [opN] 输出
```

# 修改后
```text
输入 [op1] [op2] [op3] ... [opN] 输出
```
````

---

#### 问题 5：description 缺少扩展触发短语

**命中规则**：R48 (S3) - WARN

> 规则内容：description 应该 pushy 一些，避免在技能本应发挥作用的场景下却不使用

**位置**：`SKILL.md:3`

**当前内容**：
> 当需要调试 PyPTO 算子精度、定位精度差异来源或进行中间结果对比时使用此技能

**问题说明**：
description 仅列出显式触发关键词（调试 PyPTO 算子精度、定位精度差异、中间结果对比），缺少扩展触发短语如 "whenever"、"including" 等，可能导致在相关场景下未能主动使用该技能。

**修改建议**：
在 description 中添加扩展触发短语，使其能够覆盖更多相关场景。

**建议修改**：
```markdown
PyPTO 算子二分查找调试技能。利用精度工具通过二分查找方法快速定位算子精度问题。Whenever encountering precision issues in PyPTO operators, including but not limited to accuracy mismatches, numerical differences, or any intermediate result comparison needs, use this skill for systematic binary search debugging.
```

---

## 通过项

共 43 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19, R43 |
| D4 | R20, R21, R22 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R31 |
| D7 | R32, R33, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42 |
