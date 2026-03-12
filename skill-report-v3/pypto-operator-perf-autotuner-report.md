# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | pypto-operator-perf-autotuner |
| 评审时间 | 2026-03-11 |
| 总分 | 77.50 / 100 |
| 等级 | B |
| S0 否决 | 否 |
| 规则统计 | 通过 41 / 失败 9 / 警告 0 |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | 25.00 | 无 |
| D2 | 简洁性与效率 | 15% | 15.0 | 15.00 |.00 | 无 |
| D3 | 文件结构与导航 | 10% | 10.0 | 7.50 | R18(-5×5) |
| D4 | 语言与表达 | 10% | 10.0 | 8.00 | R23(-2) |
| D5 | 精确性与可执行性 | 10% | 10.0 | 5.00 | R24(-2), R26(-10), R27(-3) |
| D6 | 工作流完整性 | 10% | 10.0 | 8.00 | R30(-2) |
| D7 | 模式与最佳实践 | 5% | 5.0 | 4.60 | R33(-1), R50(-0.4) |
| D8 | 反模式检测 | 10% | 10.0 | 10.00 | 无 |
| D9 | 脚本与代码质量 | 5% | 5.0 | 5.00 | 无（无 scripts/ 目录） |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | 50 |
| 已评估规则数 | 50 |
| 跳过规则数 | 2 |
| 覆盖率 | 100% |

**跳过的规则**：R42, R47（原因：不存在 scripts/ 目录）

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
| R18 | FAIL | D3 | S2 | static |
| R19 | FAIL | D3 | S2 | semantic |
| R20 | PASS | D4 | S2 | semantic |
| R21 | PASS | D4 | S3 | semantic |
| R22 | PASS | D4 | S2 | static |
| R23 | FAIL | D4 | S2 | semantic |
| R24 | FAIL | D5 | S2 | semantic |
| R25 | PASS | D5 | S2 | semantic |
| R26 | FAIL | D5 | S1 | semantic |
| R27 | FAIL | D5 | S2 | semantic |
| R28 | PASS | D6 | S1 | semantic |
| R29 | PASS | D6 | S2 | semantic |
| R30 | FAIL | D6 | S2 | semantic |
| R31 | PASS | D6 | S2 | semantic |
| R32 | PASS | D7 | S3 | semantic |
| R33 | WARN | D7 | S3 | semantic |
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
| R46 | PASS | D4 | S3 | static |
| R47 | SKIP | D9 | S2 | semantic |
| R50 | WARN | D7 | S2 | semantic |

## 质量闸门

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | 0 | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | 0 | snippet 无法在源文件中匹配，已从评分中剔除 |

## 问题清单

### S0 致命缺陷

无 S0 致命缺陷。

### S1 重大问题

#### 问题 1：缺少操作的具体实现方法

**命中规则**：R26 (S1)

**位置**：`SKILL.md:61`

**当前内容**：
> 使用pypto-operator-perf-analyzer分析性能，生成性能报告和性能优化建议

**问题说明**：
步骤 3 提到了"使用pypto-operator-perf-analyzer分析性能"，但没有说明如何调用这个技能、需要提供什么参数、如何查看生成的报告。用户无法直接执行这一步。

**修改建议**：
> 调用 pypto-operator-perf-analyzer 技能，提供以下参数：
> - 算子代码路径：custom/operator_name/operator.py
> - 泳道图数据路径：output/output_*/merged_swimlane.json
> - 输出报告路径：custom/operator_name/perf_report.md
> 
> 执行后，检查生成的性能报告是否包含：
> - 性能瓶颈分析
> - 优化建议列表
> - 关键指标统计

---

### S2 中等问题

#### 问题 1：被引用的文件缺少用途和时机说明

**命中规则**：R19 (S2)

**位置**：`SKILL.md:100-106`

**当前内容**：
> ## 参考资料
> 
> - [性能调优文档](../../docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
> - [功能调试](../../docs/tutorials/debug/debug.md)
> - [精度调试](../../docs/tutorials/debug/precision.md)

**问题说明**：
参考资料部分列出了多个文档链接，但没有说明每个文档的用途以及在什么时机应该查阅这些文档。用户不知道何时需要参考这些文档。

**修改建议**：
> ## 参考资料
> 
> 以下文档在性能调优的不同阶段提供参考：
> 
> - [性能调优文档](../../docs/tutorials/debug/performance.md) - 在步骤 4 执行优化前查阅，了解可用的优化策略
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md) - 当算子包含矩阵乘法时参考，提供专项优化指导
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md) - 在步骤 4 寻找优化思路时参考，提供实际案例
> - [功能调试](../../docs/tutorials/debug/debug.md) - 在出现运行时错误时查阅，帮助定位问题
> - [精度调试](../../docs/tutorials/debug/precision.md) - 在步骤 2 精度检查失败时查阅，提供调试方法

---

#### 问题 2：指令缺少原因说明

**命中规则**：R23 (S2)

**位置**：`SKILL.md:35-47`

**当前内容**：
> #### 1.2 重新编译并运行
> 如果非第一次运行，没有修改framework或python下的代码，则不需要重新编译，跳过此节。
> 
> ```bash
> # 设置环境变量
> export TILE_FWK_DEVICE_ID=0
> export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/
> 
> # 编译 whl 包
> python3 build_ci.py -f python3 --disable_auto_execute
> 
> # 运行算子（生成泳道图数据）
> python3 custom/operator_name/operator.py --run-mode npu
> ```

**问题说明**：
该步骤列出了编译和运行的命令，但没有说明为什么要设置这些环境变量、为什么要重新编译、运行后会生成什么文件。用户不理解这些操作的目的。

**修改建议**：
> #### 1.2 重新编译并运行
> 如果非第一次运行，没有修改framework或python下的代码，则不需要重新编译，跳过此节。
> 
> 重新编译的原因：启用性能数据采集需要修改 debug_options，这会影响生成的可执行代码，因此必须重新编译。
> 
> ```bash
> # 设置环境变量
> export TILE_FWK_DEVICE_ID=0  # 指定使用的 NPU 设备 ID
> export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/  # 指定 PTO 虚拟指令库路径
> 
> # 编译 whl 包（重新生成包含性能采集功能的可执行代码）
> python3 build_ci.py -f python3 --disable_auto_execute
> 
> # 运行算子（生成泳道图数据）
> python3 custom/operator_name/operator.py --run-mode npu
> ```
> 
> 运行后会在 output/output_*/ 目录下生成性能数据文件，包括泳道图数据和性能追踪信息。

---

#### 问题 3：步骤缺少可验证的成功标准

**命中规则**：R24 (S2)

**位置**：`SKILL.md:60-73`

**当前内容**：
> ### 步骤 3：分析性能数据
> 使用pypto-operator-perf-analyzer分析性能，生成性能报告和性能优化建议
> 
> ### 步骤 4：执行性能优化
> 根据上一步的性能优化建议, 执行性能优化
> 
> ### 步骤 5：迭代调优
> 1. 采集性能数据
> 2. 分析性能瓶颈
> 3. 应用优化策略
> 4. 重新编译运行
> 5. 检查精度
> 6. 对比性能提升，如果性能出现回退则回退修改
> 7. 重复步骤 1-6 直到达到目标性能

**问题说明**：
步骤 3、4、5 都没有明确的成功标准。用户不知道如何判断性能分析是否完成、优化是否成功、何时达到目标性能。

**修改建议**：
> ### 步骤 3：分析性能数据
> 使用pypto-operator-perf-analyzer分析性能，生成性能报告和性能优化建议。
> 
> **成功标准**：
> - 生成了性能报告文件（如 perf_report.md）
> - 报告中包含性能瓶颈分析（识别出主要耗时算子）
> - 报告中提供了至少一条优化建议
> 
> ### 步骤 4：执行性能优化
> 根据上一步的性能优化建议, 执行性能优化。
> 
> **成功标准**：
> - 代码已根据优化建议进行修改
> - 修改后的代码语法正确，无编译错误
> 
> ### 步骤 5：迭代调优
> 1. 采集性能数据
> 2. 分析性能瓶颈
> 3. 应用优化策略
> 4. 重新编译运行
> 5. 检查精度
> 6. 对比性能提升，如果性能出现回退则回退修改
> 7. 重复步骤 1-6 直到达到目标性能
> 
> **成功标准**：
> - 性能指标（如总耗时、吞吐量）达到或超过目标值
> - 精度检查通过（与 golden 结果的误差在容忍范围内）
> - 性能提升显著（相比初始版本提升 > 10% 或达到平台最优水平）

---

#### 问题 4：缺少整体任务的完成标准

**命中规则**：R27 (S2)

**位置**：`SKILL.md:1-106`

**当前内容**：
> ---
> name: pypto-operator-perf-autotuner
> description: PyPTO算子性能分析和自动调优技能。用于生成泳道图、分析性能数据、查看性能统计和提供优化建议。当用户需要分析 PyPTO 算子的性能、生成性能报告或进行性能调优时使用此技能。
> ---
> 
> # PyPTO 算子性能分析和自动调优
> 
> ## 概述
> 
> 此技能提供 PyPTO 算子性能调优的完整工作流程，包括性能数据采集、性能分析、优化策略和迭代调优。

**问题说明**：
skill 描述了工作流程，但没有明确说明整个任务何时算完成。用户不知道最终应该交付什么、如何确认性能调优任务已完成。

**修改建议**：
在文档开头添加"完成标准"章节：
> 
> ## 完成标准
> 
> 性能调优任务完成后，应满足以下条件：
> 1. **性能达标**：算子性能达到或超过目标指标（如目标耗时、目标吞吐量）
> 2. **精度保持**：优化后的算子输出与 golden 结果的误差在容忍范围内
> 3. **文档完备**：生成了性能分析报告，记录了优化过程和最终性能数据
> 4. **代码规范**：优化后的代码符合 PyPTO 编码规范，无编译警告
> 
> 验证方法：
> - 检查性能报告中的最终性能指标
> - 运行精度测试用例，确认所有测试通过
> - 查看编译日志，确认无警告信息

---

#### 问题 5：缺少错误处理或失败恢复说明

**命中规则**：R30 (S2)

**位置**：`SKILL.md:35-47`

**当前内容**：
> #### 1.2 重新编译并运行
> 如果非第一次运行，没有修改framework或python下的代码，则不需要重新编译，跳过此节。
> 
> ```bash
> # 设置环境变量
> export TILE_FWK_DEVICE_ID=0
> export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/
> 
> # 编译 whl 包
> python3 build_ci.py -f python3 --disable_auto_execute
> 
> # 运行算子（生成泳道图数据）
> python3 custom/operator_name/operator.py --run-mode npu
> ```

**问题说明**：
该步骤没有说明如果编译失败或运行失败应该如何处理。用户遇到错误时不知道如何排查和恢复。

**修改建议**：
> #### 1.2 重新编译并运行
> 如果非第一次运行，没有修改framework或python下的代码，则不需要重新编译，跳过此节。
> 
> ```bash
> # 设置环境变量
> export TILE_FWK_DEVICE_ID=0
> export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/
> 
> # 编译 whl 包
> python3 build_ci.py -f python3 --disable_auto_execute
> 
> # 运行算子（生成泳道图数据）
> python3 custom/operator_name/operator.py --run-mode npu
> ```
> 
> **错误处理**：
> - 如果编译失败：检查代码语法错误，确认 debug_options 参数格式正确
> - 如果运行失败：检查 NPU 设备是否可用（运行 npu-smi info），确认环境变量设置正确
> - 如果未生成泳道图文件：确认 debug_options={"runtime_debug_mode": 1} 已正确设置，重新编译运行
> - 如果 NPU 设备不可用：检查 TILE_FWK_DEVICE_ID 值，使用 npu-smi info 查看可用设备

---

#### 问题 6：引用的文件路径不存在

**命中规则**：R18 (S2)

**位置**：`SKILL.md:102-106`

**当前内容**：
> - [性能调优文档](../../docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](../../docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
> - [功能调试](../../docs/tutorials/debug/debug.md)
> - [精度调试](../../docs/tutorials/debug/precision.md)

**问题说明**：
所有 5 个引用的文档路径都不存在。这些路径是相对于 skill 目录的相对路径，但目标文件在 skill 目录中找不到。

**修改建议**：
需要确认这些文档的正确路径。如果这些文档在项目根目录的 docs/ 下，应该使用正确的相对路径。例如：
> 
> - [性能调优文档](../../../../docs/tutorials/debug/performance.md)
> - [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
> - [性能优化案例](../../../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
> - [功能调试](../../../../docs/tutorials/debug/debug.md)
> - [精度调试](../../../../docs/tutorials/debug/precision.md)
> 
> 或者，如果这些文档在 skill 目录的 references/ 下，应该移动到 references/ 目录并更新引用：
> 
> - [性能调优文档](references/performance.md)
> - [Matmul 高性能编程](references/matmul_performance_guide.md)
> - [性能优化案例](references/performance_case_quantindexerprolog.md)
> - [功能调试](references/debug.md)
> - [精度调试](references/precision.md)

---

### S3 轻微建议

#### 问题 1：未使用确定性脚本进行验证

**命中规则**：R33 (S3)

**位置**：`SKILL.md:54-59`

**当前内容**：
> ### 步骤 2：检查精度
> 进行精度检查，根据检查结果选择执行下面步骤：
> 1. 如果检查通过，继续执行步骤3
> 2. 如果检查失败，是首轮失败，提示用户精度失败，是否选择修复或者继续进行优化
> 3. 如果检查失败，失败是由于上一轮修改导致，回退修改，尝试其它优化方案

**问题说明**：
步骤 2 提到"检查精度"，但没有说明如何进行精度检查。如果依赖人工判断或 LLM 判断，可能不够客观和可重复。建议提供具体的精度检查脚本或命令。

**修改建议**：
> ### 步骤 2：检查精度
> 运行精度测试脚本，检查优化后的算子输出是否与 golden 结果一致：
> 
> ```bash
> python3 custom/operator_name/operator.py --test-mode golden
> ```
> 
> 根据检查结果选择执行下面步骤：
> 1. 如果检查通过（所有测试用例误差在容忍范围内），继续执行步骤3
> 2. 如果检查失败，是首轮失败，提示用户精度失败，是否显示误差详情、修复或继续进行优化
> 3. 如果检查失败，失败是由于上一轮修改导致，回退修改，尝试其它优化方案
> 
> **验证标准**：测试脚本应返回明确的通过/失败状态，并输出误差统计信息

---

#### 问题 2：多选项场景未提供默认推荐

**命中规则**：R50 (S3)

**位置**：`SKILL.md:95-98`

**当前内容**：
> - 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
> - 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
> - 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑

**问题说明**：
Q5 问题提供了多个 Tilesize 选项，但没有明确标注哪个是首选或默认推荐。用户在选择时可能感到困惑。

**修改建议**：
> - 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]（**首选 [128, 128]**）
> - 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]（**首选 [64, 512]**）
> - 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑

---

## 通过项

共 41 条规则通过。

| 维度 | 通过规则 |
|------|---------|
| D1 | R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R44 |
| D2 | R11, R12, R13, R14, R45 |
| D3 | R15, R16, R17 |
| D4 | R20, R21, R22, R46 |
| D5 | R25 |
| D6 | R28, R29, R31 |
| D7 | R32 |
| D8 | R34, R35, R36, R37, R38 |
| D9 | R39, R40, R41 |
| D0 | R43, R43 |
