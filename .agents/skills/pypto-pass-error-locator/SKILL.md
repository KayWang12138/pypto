---
name: pypto-pass-error-locator
description: PyPTO Pass 模块问题定位技能。用于解析 PyPTO 运行日志，提取错误信息、堆栈跟踪等关键信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，为用户自行修复和通过skill自动修复提供参考。当需要分析 PyPTO pass 模块错误时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO Pass Error locator Skill

## 概述

本技能提供 PyPTO Pass 模块错误的完整分析能力，包括日志解析、错误识别、原因分析和修复建议。

## 使用场景

当需要根据执行异常日志，分析定位某pass异常原因以及如何修复该问题时使用此技能。

## 功能概述

- 解析 PyPTO 运行日志文件，提取错误信息
- 识别错误相关的 pass 模块
- 分析错误产生的根本原因
- 提供错误分类和严重程度评估
- 生成详细的错误分析报告
- 提供问题修复策略建议

## 触发机制

当用户输入包含以下关键字时，自动触发此技能：
- **执行xx用例失败，分析pass失败原因**：根据用例执行日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因
- **根据xx日志信息，分析pass失败原因**：根据用户提供的错误日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因

## 工作流程

### 步骤 1：获取日志

1. 如果用户提供了具体的错误信息，直接使用用户提供的日志内容
2. 如果用户提供了具体的错误日志路径，从用户指定的错误文件中获取日志内容
3. 如果用户提供了具体的错误日志目录，从用户指定的日志目录下查找日志文件并获取日志内容
4. 如果用户没提供具体错误信息或日志路径，从如下位置获取日志文件
    - `$HOME/ascend/log/` 目录（默认）
    - `$ASCEND_PROCESS_LOG_PATH/` 目录（如果设置）
    - `$ASCEND_WORK_PATH/` 目录（如果设置且未设置 ASCEND_PROCESS_LOG_PATH）

### 步骤 1：解析日志

**日志格式示例**：
```
[INFO ] PYPTO(592313):2026-03-13 14:26:37.767 [l1_copy_reuse.cpp:642][PASS]:[L1CopyInReuseMerge.Operation]:Op 704 feature: [63, 128, 0, 32, 512, 148].
```

**解析模式**：

1. **日志级别识别**
    - 模式：`\[(DEBUG|INFO|WARNING|ERROR)\s*\]`
    - DEBUG (0 ASCEND_GLOBAL_LOG_LEVEL): 调试信息
    - INFO (1 ASCEND_GLOBAL_LOG_LEVEL): 一般信息
    - WARNING (2 ASCEND_GLOBAL_LOG_LEVEL): 警告信息
    - ERROR (3 ASCEND_GLOBAL_LOG_LEVEL): 致命错误

2. **模块名提取**
   - 模式：`PYPTO\((\d+)\):`
   - 模块名：`PYPTO`
   - 进程ID：`592313`

3. **时间戳提取**
   - 模式：`\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}`
   - 示例：`2026-03-13 14:26:37.767`

4. **文件位置提取**
   - 模式：`\[([a-zA-Z0-9_]+\.(cpp|py|h|cc)):(\d+)\]`
   - 文件名：`l1_copy_reuse.cpp`
   - 行号：`642`

5. **所属模块识别**
   - 模式：`\[(FUNCTION|PASS|CODEGEN|MACHINE|DISTRIBUTED|SIMULATION|VERIFY)\]:`
   - FUNCTION: 表示这是函数调用相关的日志，记录函数入口、出口、参数等信息
   - PASS: 表示这是pass模块的日志，记录各个pass阶段的执行情况
   - CODEGEN: 表示这是代码生成相关的日志，记录中间表示(IR)生成、优化等信息
   - MACHINE: 表示这是机器码生成相关的日志，记录汇编代码生成、指令调度等信息
   - DISTRIBUTED: 表示这是分布式相关的日志，记录多设备通信、数据分发等信息
   - SIMULATION: 表示这是仿真相关的日志，记录NPU仿真执行、内存模拟等信息
   - VERIFY: 表示这是验证相关的日志，记录结果验证、精度检查等信息
   - 示例：`[PASS]:`

6. **Pass 模块信息提取**
   - 模式：`\[([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\]:`
   - Pass模块名：`L1CopyInReuseMerge.Operation`

7. **消息内容提取**
    - 提取最后一个冒号后的所有内容
    - 示例：`Op 704 feature: [63, 128, 0, 32, 512, 148].`

8. **堆栈跟踪提取**（Python异常）
    - 识别 `Traceback (most recent call last):` 开始的堆栈
    - 提取每个堆栈帧的文件、行号、函数名

### 步骤 3：Pass 模块识别

**识别策略**：

1. **直接识别**：从错误日志解析获取的关键信息中提取 pass 名称
   - 模式：`Pass\[([^\]]+)\]`
   - 模式：`in pass ([a-zA-Z0-9_]+)`

2. **堆栈分析**：从堆栈跟踪中推断 pass 模块
   - 检查文件路径：`framework/passes/`
   - 检查函数名：`run_pass`, `apply_pass`, `execute`

3. **错误模式匹配**：根据错误特征推断 pass 模块
   - 冗余消除 → RemoveUndrivenView、RemoveRedundantOp、CommonOperationEliminate
   - 切图合图 → GraphPartition、ReduceCopyMerge、NBufferMerge、L1CopyInReuseMerge
   - 数据通路 → IntraSubgraphAdapter、GenerateMoveOp
   - 动态推导 → InferDynShape、InferParamIndex
   - 内存管理 → InferMemoryConflict、SrcDstBufferMerge、AddAlloc、OoOSchedule、RemoveAlloc、GlobalMemoryReuse
   - 调度优化 → OoOSchedule

### 步骤 4：错误原因分析

**分析维度**：

1. **输入数据问题**
    - Shape 不对齐
    - Dtype 不匹配
    - 数据范围异常
    - 空张量/零维度
 
2. **API 使用问题**
    - 参数错误
    - 不支持的配置
    - 缺少必需参数
    - 参数冲突
 
3. **环境问题**
    - NPU 设备不可用
    - 内存不足
    - CANN 版本不兼容
    - 驱动问题
 
4. **代码逻辑问题**
    - 算子实现错误
    - 边界条件处理不当
    - 类型转换错误
    - 内存访问越界

**Pass 模块详细知识库**：

详细的 Pass 模块知识库请参考：`references/pass-modules-knowledge-base.md`

该文档包含了 Pass 模块常见错误原因知识库，涵盖：
- 每个模块的详细功能描述
- 常见错误类型列表
- 可能的根本原因分析
- 错误定位指南

在分析 Pass 模块错误时，请先查阅该知识库以获取详细的错误模式信息。
 
### 步骤 5：错误严重程度评估

**评估标准**：

- **CRITICAL**: 导致程序崩溃、无法继续执行
- **HIGH**: 影响核心功能、结果错误
- **MEDIUM**: 影响性能、功能受限
- **LOW**: 警告信息、可忽略

### 步骤 6：生成分析报告

**报告格式**（Markdown）：

报告模板参见 `references/pass-error-report-template.md` 文件。

## 错误模式匹配规则

### 1. Pass 模块相关错误

**模式识别**：
```
[INFO ] PYPTO(592313):2026-03-13 14:26:37.767 [l1_copy_reuse.cpp:642][PASS]:[L1CopyInReuseMerge.Operation]:Op 704 feature: [63, 128, 0, 32, 512, 148].
[ERROR] PYPTO(592314):2026-03-13 14:26:38.123 [shape_inference.cpp:123][ERROR]:[ShapeInference]:Shape mismatch at dim 2.
[ERROR] PYPTO(592315):2026-03-13 14:26:38.456 [constant_folding.cpp:78][ERROR]:[ConstantFolding]:Folding failed for op 123.
```

**提取策略**：
- 使用正则表达式 `\[([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\]:` 提取 pass 名称
- 识别 pass 相关的关键词：`folding`, `inference`, `lowering`, `optimization`, `reuse`, `merge`

### 2. Shape 相关错误

**正则模式**：
```python
SHAPE_ERROR_PATTERNS = [
    r"Shape mismatch",
    r"Invalid shape",
    r"Dimension mismatch",
    r"Shape not compatible"
]
```

**分析策略**：
- 提取期望的 shape 和实际的 shape
- 检查是否是动态 shape 问题
- 分析 shape 不匹配的位置

### 3. 类型相关错误

**正则模式**：
```python
TYPE_ERROR_PATTERNS = [
    r"Type mismatch",
    r"Invalid dtype",
    r"Unsupported type",
    r"Type conversion failed"
]
```

**分析策略**：
- 提取期望的 dtype 和实际的 dtype
- 检查类型转换是否支持
- 分析类型不兼容的原因

### 4. 内存相关错误

**正则模式**：
```python
MEMORY_ERROR_PATTERNS = [
    r"Out of memory",
    r"Allocation failed",
    r"Buffer overflow",
    r"Memory access violation"
]
```

**分析策略**：
- 估算所需内存大小
- 检查可用内存
- 分析内存分配失败的原因

### 5. 设备相关错误

**模式识别**：
```
Invalid Device
NPU not available
Device not found
```

## 修复策略建议

### 策略 1：输入数据修复

**适用场景**：Shape 不匹配、类型不匹配

**建议**：
1. 检查输入张量的 shape 和 dtype
2. 使用 `reshape()` 或 `view()` 调整 shape
3. 使用 `to()` 转换 dtype
4. 添加 shape 和 dtype 的断言

### 策略 2：API 参数修复

**适用场景**：参数错误、不支持的配置

**建议**：
1. 检查 API 文档，确认参数正确性
2. 使用默认参数
3. 添加参数验证
4. 打印参数值进行调试

### 策略 3：环境配置修复

**适用场景**：设备不可用、内存不足

**建议**：
1. 检查 NPU 设备状态：`npu-smi info`
2. 设置正确的设备 ID：`export TILE_FWK_DEVICE_ID=0`
3. 减少批处理大小
4. 释放不必要的内存

### 策略 4：代码逻辑修复

**适用场景**：算子实现错误、边界条件问题

**建议**：
1. 使用 golden 函数验证结果
2. 添加中间结果打印
3. 检查边界条件处理
4. 使用单元测试覆盖边界情况

## 常见问题

常见问题详见 `references/pass-error-locating-faq.md` 文件。

## 输出验证

**验证检查点**：
- [ ] 错误信息完整提取
- [ ] 堆栈跟踪准确解析
- [ ] Pass 模块名称正确识别
- [ ] 时间戳格式正确
- [ ] 错误原因分析准确
- [ ] 修复建议合理可行

## 参考文件

| 文件 | 内容 |
|------|----------|
| `scripts/parse_log.py` | 日志解析脚本 |
| `scripts/analyze_pass_error.py` | Pass 模块错误分析脚本 |
| `tests/test_log_parser.py` | 日志解析单元测试 |
| `tests/test_pass_analyzer.py` | Pass 分析单元测试 |
| `knowledge_base/pass_errors.json` | Pass 模块错误知识库 |
