# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-binary-search-without-verify |
| 评审时间 | 2026-03-12 |
| 总分 | 85.90 / 100 |
| 等级 | B |
| S0 否决 | 否 |
| 规则统计 | 通过 38 / 失败 8 / 警告 0 / 跳过 1 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 无扣分 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无扣分 |
| D3 | 文件结构与导航 | 10% | 10.0 | 9.50 | R19(S2): -5 |
| D4 | 语言与表达 | 10% | 10.0 | 9.50 | R23(S2): -5 |
| D5 | 精确性与可执行性 | 10% | 10.0 | 7.50 | R24(S2): -5, R25(S2): -5, R26(S1): -10, R27(S2): -5 |
| D6 | 工作流完整性 | 10% | 10.0 | 9.50 | R30(S2): -5 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.90 | R33(S3): -2 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无扣分 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 不存在 scripts/ 目录，自动满分 |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 47 |
| 已评估规则数 | 46 |
| 跳过规则数 | 1 |
| 覆盖率 | 97.87% |

**跳过的规则**：R42（原因：不存在 scripts/ 目录，D9 自动获得满分）

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
| R01 | PASS | D1 | S0 | static |
| R02 | PASS | D1 | S0 | static |
| R03 | PASS | D1 | S0 | static |
| R04 | PASS | D1 | S1 | static |
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
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S1 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | FAIL | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
| R27 | FAIL | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | FAIL | D6 | S2 | semantic |
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
| R43 | PASS | D3 | S3 | static |
| R44 | PASS | D1 | S2 | static |
| R45 | PASS | D2 | S2 | static |
| R46 | PASS | D4 | S3 | static |
| R47 | PASS | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 无 |
| 证据不足条目 | 0 | 无 |

## 问题清单

### S1 重大问题

#### 问题 1：步骤1中"分析kernel和golden代码"缺少具体的分析方法

**命中规则**：R26 [S1]

> 规则内容：所有提及的操作都必须提供具体实现方法

**位置**：`SKILL.md:22`

**当前内容**：
> 分析kernel和golden代码，确定需要检查的关键计算节点

**问题说明**：
步骤1中"分析kernel和golden代码"缺少具体的分析方法。用户不知道应该用什么方法、按什么顺序、关注哪些方面来分析代码。

**修改建议**：
> 补充具体分析方法：
> ```markdown
> 分析kernel和golden代码（方法：
> 1. 从输出向输入追溯，识别每个计算节点
> 2. 标记涉及数值变换的op（如matmul、softmax、cast）
> 3. 对比kernel和golden的计算流程差异
> ），确定需要检查的关键计算节点
> ```

---

### S2 中等问题

#### 问题 2：参考资料部分引用了 docs/api/ 但未说明具体用途和加载时机

**命中规则**：R19 [S2]

> 规则内容：SKILL.md 应说明每个被引用文件的用途和加载时机

**位置**：`SKILL.md:147`

**当前内容**：
> - PyPTO API: `docs/api/`

**问题说明**：
参考资料部分引用了 `docs/api/` 但未说明具体用途和加载时机。用户不知道应该在什么情况下查阅这个路径，也不知道应该查找哪些具体的API。

**修改建议**：
> 将参考资料改为具体说明：
> ```markdown
> ## 参考资料
> - PyPTO API 文档：`docs/api/`（查阅 pypto.assemble、pypto.cast 等API的详细用法，在需要了解具体API参数时加载）
> ```

---

#### 问题 3：步骤1中"选择在关键计算节点之后的位置"没有解释原因

**命中规则**：R23 [S2]

> 规则内容：指令应解释"为什么"，而不仅是"做什么"

**位置**：`SKILL.md:22`

**当前内容**：
> 选择在关键计算节点之后的位置，确保检查点的结果有明确的含义，优先选择有明显边界的位置（如matmul、softmax之后）

**问题说明**：
步骤1中"选择在关键计算节点之后的位置"没有解释为什么应该选择这些位置。缺少原因说明，用户无法理解背后的逻辑。

**修改建议**：
> 补充原因说明：
> ```markdown
> 选择在关键计算节点之后的位置（因为关键计算如matmul、softmax是精度问题的高发区域，在其后添加检查点可以快速定位问题来源），确保检查点的结果有明确的含义
> ```

---

#### 问题 4：步骤1"分析代码结构，确定检查点"缺少可验证的成功标准

**命中规则**：R24 [S2]

> 规则内容：每个步骤都应具备可验证的成功标准

**位置**：`SKILL.md:20`

**当前内容**：
> ### 步骤 1：分析代码结构，确定检查点

**问题说明**：
步骤1"分析代码结构，确定检查点"缺少可验证的成功标准。用户不知道如何判断"分析完成"或"检查点确定正确"。

**修改建议**：
> 补充成功标准：
> ```markdown
> ### 步骤 1：分析代码结构，确定检查点
> 
> **成功标准**：已识别出所有关键计算节点，并为每个节点确定了检查点的shape和dtype
> ```

---

#### 问题 5：参考资料路径 docs/api/ 过于模糊

**命中规则**：R25 [S2]

> 规则内容：命令和路径必须具体且可执行

**位置**：`SKILL.md:147`

**当前内容**：
> - PyPTO API: `docs/api/`

**问题说明**：
参考资料路径 `docs/api/` 过于模糊，没有指定具体的API文档路径或文件名，用户无法直接定位到需要的信息。

**修改建议**：
> 将路径具体化：
> ```markdown
> - PyPTO assemble API: `docs/api/pypto_assemble.md`（查看assemble函数的详细用法和参数说明）
> - PyPTO cast API: `docs/api/pypto_cast.md`（查看dtype转换函数的详细用法）
> ```

---

#### 问题 6：skill缺少整体完成标准的明确定义

**命中规则**：R27 [S2]

> 规则内容：完成标准必须明确定义

**位置**：`SKILL.md:74`

**当前内容**：
> ### 步骤 6：二分定位精度问题

**问题说明**：
skill缺少整体完成标准的明确定义。用户不知道何时可以认为"精度问题定位完成"。

**修改建议**：
> 在步骤6后添加完成标准：
> ```markdown
> ## 完成标准
> 
> 精度问题定位完成需满足以下条件：
> 1. 已定位到导致精度不匹配的具体op
> 2. 该op的kernel实现与golden实现存在可识别的差异
> 3. 差异原因已明确（如dtype转换、计算顺序等）
> ```

---

#### 问题 7：缺少系统性的错误处理说明

**命中规则**：R30 [S2]

> 规则内容：必须包含错误处理或失败恢复说明

**位置**：`SKILL.md:123`

**当前内容**：
> ## 常见问题

**问题说明**：
虽然有"常见问题"部分，但缺少系统性的错误处理说明。例如：如果检查点shape不匹配怎么办？如果所有检查点都通过但最终结果仍不对怎么办？

**修改建议**：
> 在工作流程中添加错误处理说明：
> ```markdown
> ## 错误处理
> 
> 在执行过程中可能遇到以下问题及解决方案：
> 1. **检查点shape不匹配**：检查kernel和golden的计算路径是否一致，确认检查点添加位置相同
> 2. **所有检查点都通过但最终结果仍不对**：检查输出层的计算，可能问题在最后一步
> 3. **device输出为0**：检查NPU卡状态或检查点添加是否正确
> 4. **assemble报错"mix assemble and common operation"**：使用不同的变量名作为输入和输出
> 5. **assemble报错"Source dtype must be same with dst dtype"**：使用cast转换dtype后再assemble
> ```

---

### S3 轻微建议

#### 问题 8：验证任务完全依赖人工判断，缺少确定性脚本辅助验证

**命中规则**：R33 [S3]

> 规则内容：验证逻辑应优先使用确定性脚本

**位置**：`SKILL.md:66`

**当前内容**：
> 创建检查点tensor，执行kernel（kernel内部会原地修改checkpoint），执行golden，对比最终结果和所有检查点

**问题说明**：
验证任务（检查点对比、精度验证）完全依赖人工判断，缺少确定性脚本辅助验证。

**修改建议**：
> 建议提供验证脚本模板或具体的torch.allclose验证命令：
> ```markdown
> ### 步骤 4：修改测试函数，对比所有结果
> 
> 使用以下代码模板进行精度对比：
> ```python
> for name, (cp_kernel, cp_golden) in checkpoints.items():
>     is_close = torch.allclose(cp_kernel.float(), cp_golden.float(), atol=1e-3, rtol=1e-3)
>     print(f"{name}: {'PASS' if is_close else 'FAIL'}")
>     if not is_close:
>         print(f"  max diff: {(cp_kernel.float() - cp_golden.float()).abs().max()}")
> ```
> ```

---

## 通过项

共 38 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R43 |
| D4 | R20, R21, R22, R46 |
| D5 | 无（4条规则失败） |
| D6 | R28, R29, R31 |
| D7 | R32, R47 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
