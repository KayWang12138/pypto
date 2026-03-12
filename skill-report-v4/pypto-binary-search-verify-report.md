# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-binary-search-verify |
| 评审时间 | 2026-03-11 |
| 总分 | 95.90 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 43 / 失败 6 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 22.00 | R04(-10), R06(-2) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无 |
| D4 | 语言与表达 | 10% | 10.0 | 9.40 | R46(-2) × 3 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | 无 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.75 | R50(-5) |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 4.75 | R47(-5) |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 50 |
| 跳过规则数 | 1 |
| 覆盖率 | 100% |

**跳过的规则**：R44（原因：不适用于此技能）

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
| R19 | PASS | D3 | S2 | semantic |
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
| R43 | FAIL | D0 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | FAIL | D4 | S3 | static |
| R47 | FAIL | D9 | S2 | semantic |
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

#### 问题 1：name 值与目录名不匹配

**命中规则**：R04 (S1)

> 规则内容：`name` 值应与 skill 目录名一致

**位置**：`SKILL.md:2`

**当前内容**：
> name: pypto-verify-binary-search

**问题说明**：
frontmatter 中的 `name` 字段值为 `pypto-verify-binary-search`，但实际目录名为 `pypto-binary-search-verify`，两者不一致。

**修改建议**：
> 将 frontmatter 中的 name 字段修改为 `pypto-binary-search-verify` 以与目录名保持一致：
> ```yaml
> name: pypto-binary-search-verify
> ```

---

### S2 中等问题

#### 问题 1：未知的 frontmatter 字段

**命中规则**：R06 (S3)

> 规则内容：未知的 frontmatter 字段应产生告警

**位置**：`SKILL.md:4`

**当前内容**：
> license: 完整条款见 LICENSE.txt

**问题说明**：
frontmatter 中使用了 `license` 字段，该字段不在已知的 frontmatter 字段白名单中（已知字段：name, description, license, compatibility, metadata）。

**修改建议**：
> 虽然该字段在白名单中，但建议移除或确认其必要性。如果保留，可以在 skill 文档中说明其用途。

---

#### 问题 2：多选场景未提供默认推荐

**命中规则**：R50 (S2)

> 规则内容：提供多个选项时，应给出默认推荐

**位置**：`SKILL.md:314-337`

**当前内容**：
> ### Q1: 中间结果太大无法输出
> 
> **解决方法**：
> - 使用 `cond` 参数只输出部分元素
> - 只在特定条件下保存
> 
> ### Q2: op 太多，二分效率低
> 
> **解决方法**：
> - 先根据代码逻辑划分大块，对每个块进行二分
> - 优先检查可疑的 op（例如复杂的数学运算、类型转换等）

**问题说明**：
在常见问题章节中，Q1 和 Q2 都提供了多个解决方法，但未明确标注推荐使用哪一个。

**修改建议**：
> 在每个问题的解决方法中添加推荐标注，例如：
> ```markdown
> **解决方法**：
> - 使用 `cond` 参数只输出部分元素（推荐）
> - 只在特定条件下保存
> ```
> 
> ```markdown
> **解决方法**：
> - 先根据代码逻辑划分大块，对每个块进行二分（推荐）
> - 优先检查可疑的 op（例如复杂的数学运算、类型转换等）
> ```

---

#### 问题 3：脚本未优雅处理缺失依赖

**命中规则**：R47 (S2)

> 规则内容：脚本应能优雅处理缺失依赖

**位置**：`verify_binary_search.py:21`

**当前内容**：
> import numpy as np

**问题说明**：
脚本导入了 numpy（非标准库），但没有使用 try/except 保护导入。如果 numpy 未安装，脚本会抛出晦涩的 traceback。

**修改建议**：
> 在 import 语句前使用 try/except 保护，并提供清晰的错误提示：
> ```python
> try:
>     import numpy as np
> except ImportError:
>     print("错误：需要安装 numpy 库")
>     print("请运行：pip install numpy")
>     sys.exit(1)
> ```

---

### S3 轻微建议

#### 问题 1：非标准子目录

**命中规则**：R43 (S3)

> 规则内容：子目录结构应使用标准命名

**位置**：`__pycache__:0`

**当前内容**：
> __pycache__

**问题说明**：
技能目录中存在 `__pycache__` 子目录，该目录不在标准子目录列表中（标准：references, scripts, templates, assets, examples）。注意：`__pycache__` 是 Python 自动生成的缓存目录，通常应加入 .gitignore。

**修改建议**：
> 将 `__pycache__` 添加到 .gitignore 文件中，避免提交到版本控制：
> ```
> __pycache__/
> *.pyc
> ```

---

#### 问题 2：代码块缺少语言标注（第 1 处）

**命中规则**：R46 (S3)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:123`

**当前内容**：
> ```

**问题说明**：
第 123 行的代码块缺少语言标注，建议添加 `text` 或其他合适的语言标识符以改善渲染效果。

**修改建议**：
> 将代码块标记从 ``` 修改为 ```text 或 ```bash：
> ```text
> 输入 [op1] [op2] [op3] ... [opN] 输出
>   ↑                              ↑
> 正确                          不正确
> ```

---

#### 问题 3：代码块缺少语言标注（第 2 处）

**命中规则**：R46 (S3)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:180`

**当前内容**：
> ```

**问题说明**：
第 180 行的代码块缺少语言标注。

**修改建议**：
> 将代码块标记从 ``` 修改为 ```text：
> ```text
> ✗ 检查点 checkpoint1 匹配，但 checkpoint2 不匹配
> → 问题位置：checkpoint1 和 checkpoint2 之间的操作
> → 建议：在这两个检查点之间插入新的检查点
> ```

---

#### 问题 4：代码块缺少语言标注（第 3 处）

**命中规则**：R46 (S3)

> 规则内容：围栏代码块应带有语言标注

**位置**：`SKILL.md:249`

**当前内容**：
> ```

**问题说明**：
第 249 行的代码块缺少语言标注。

**修改建议**：
> 将代码块标记从 ``` 修改为 ```text：
> ```text
> 第1轮：输入 → 中间 → 输出（3个检查点）
>   ↓ 发现中间不匹配
> 第2轮：在中间位置前后插入检查点（5个检查点）
>   ↓ 继续缩小范围
> 第3轮：在问题范围内插入更多检查点
>   ↓
> 定位到具体 op
> ```

---

## 通过项

共 43 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R05, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R21, R22, R23 |
| D5 | R24, R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41, R42 |
| D0 | R43 |
