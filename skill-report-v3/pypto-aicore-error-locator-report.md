# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-aicore-error-locator |
| 评审时间 | 2026-03-11 |
| 总分 | 97.30 / 100 |
| 等级 | A |
| S0 否决 | 否 |
| 规则统计 | 通过 47 / 失败 3 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 24.50 | R06(-2) |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 10.00 | 无 |
| D4 | 语言与表达 | 10% | 10.0 | 9.80 | R46(-2) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 10.00 | 无 |
| D6 | 工作流完整性 | 10% | 10.0 | 10.00 | 无 |
| D7 | 模式与最佳实践 | 5% | 5.0 | 5.00 | 无 |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无 |

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
| R19 | PASS | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | PASS | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
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
| R50 | PASS | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无 S0 致命缺陷。

### S1 重大问题

无 S1 重大问题。

### S2 中等问题

#### 问题 1：步骤缺少可验证的成功标准

**命中规则**：R24 (S2)

**位置**：`SKILL.md:39-53`

**当前内容**：
> ### 2. 启用追踪日志
> 
> 根据用户提供的 pypto 目录路径，修改以下配置以启用详细的追踪日志：
> 
> - **配置文件**: 搜索修改 `tile_fwk_config.json`
>   - 设置 `"fixed_output_path"` 为 `true`
>   - 设置 `"force_overwrite"` 为 `false`
> 
> - **头文件**: 搜索修改 `aicore_entry.h`
>   - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`
> 
> - **工具头文件**: 搜索修改 `device_switch.h`
>   - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
>   - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

**问题说明**：
步骤 2 描述了需要修改的配置文件和宏定义，但没有说明如何验证这些修改是否成功应用。例如，没有提及检查文件内容、验证配置生效或确认修改是否保存成功的方法。

**修改建议**：
> ### 2. 启用追踪日志
> 
> 根据用户提供的 pypto 目录路径，修改以下配置以启用详细的追踪日志：
> 
> - **配置文件**: 搜索修改 `tile_fwk_config.json`
>   - 设置 `"fixed_output_path"` 为 `true`
>   - 设置 `"force_overwrite"` 为 `false`
>   - **验证**: 使用 `grep` 命令确认配置已正确设置
> 
> - **头文件**: 搜索修改 `aicore_entry.h`
>   - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`
>   - **验证**: 检查头文件中该宏定义的值
> 
> - **工具头文件**: 搜索修改 `device_switch.h`
>   - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`
>   - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`
>   - **验证**: 确认两个宏定义都已正确设置

---

### S3 轻微建议

#### 问题 1：未知的 frontmatter 字段

**命中规则**：R06 (S3)

**位置**：`SKILL.md:4`

**当前内容**：
> license: 完整条款见 LICENSE.txt

**问题说明**：
frontmatter 中使用了 `license` 字段，该字段不在已知的标准字段列表中（name, description, context, agent, allowed-tools, user-invocable, intercept, model）。

**修改建议**：
> 删除 `license` 字段，或将许可证信息放在文档正文的合适位置。

---

#### 问题 2：代码块缺少语言标注

**命中规则**：R46 (S3)

**位置**：`SKILL.md:23`

**当前内容**：
> ```

**问题说明**：
第 23 行的代码块使用了 ``` 开头，但没有指定语言标识符（如 ```yaml 或 ```json），这会影响代码高亮和可读性。

**修改建议**：
> ```yaml
> question: 
>   - header: "PyPTO配置"
>     question: "请提供 pypto 目录的完整路径"
>     options: []
>   - header: "日志路径"
>     question: "请提供 device log 落盘路径（不存在将自动创建）"
>     options: []
>   - header: "运行命令"
>     question: "请提供触发 aicore error 的测试命令"
>     options: []
>   - header: "运行目录"
>     question: "请提供运行测试命令的目录路径"
>     options: []
> ```

---

## 通过项

共 47 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17, R18, R19 |
| D4 | R20, R21, R22, R23 |
| D5 | R25, R26, R27 |
| D6 | R28, R29, R30, R31 |
| D7 | R32, R33, R50 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
| D0 | R43, R43 |
