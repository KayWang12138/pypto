---
name: pypto-pass-error-analyzer
description: PyPTO Pass 模块错误分析技能。用于解析 PyPTO 运行日志，提取错误信息、堆栈跟踪等关键信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，为用户自行修复和通过skill自动修复提供参考。当需要分析 PyPTO pass 模块错误时使用此技能。
tag: [PyPTO, Pass分析, 错误诊断, 日志解析]
---

# PyPTO Pass 模块错误分析技能

本技能提供 PyPTO Pass 模块错误的完整分析能力，包括日志解析、错误识别、原因分析和修复建议。

## 功能概述

- 解析 PyPTO 运行日志文件，提取错误信息
- 识别错误相关的 pass 模块
- 分析错误产生的根本原因
- 提供错误分类和严重程度评估
- 生成详细的错误分析报告
- 为自动修复提供策略建议

## 工作流程

### 步骤 1：日志文件定位与解析

**日志目录配置**：

PyPTO 日志目录的确定规则（按优先级从高到低）：
1. **ASCEND_PROCESS_LOG_PATH** 环境变量（最高优先级）
2. **ASCEND_WORK_PATH** 环境变量
3. **$HOME/ascend/log**（默认目录）

**日志级别配置**：

通过 **ASCEND_GLOBAL_LOG_LEVEL** 环境变量设置：
- `0`: DEBUG 级别（最详细）
- `1`: INFO 级别
- `2`: WARNING 级别
- `3`: ERROR 级别（仅错误）

**查找日志文件**：
```bash
# 获取日志目录
if [ -n "$ASCEND_PROCESS_LOG_PATH" ]; then
    LOG_DIR="$ASCEND_PROCESS_LOG_PATH"
elif [ -n "$ASCEND_WORK_PATH" ]; then
    LOG_DIR="$ASCEND_WORK_PATH"
else
    LOG_DIR="$HOME/ascend/log"
fi

# 查找日志文件
find $LOG_DIR -name "*.log" -type f
find $LOG_DIR -name "pypto_*.log" -type f
```

**常见日志位置**：
- `$HOME/ascend/log/` 目录（默认）
- `$ASCEND_PROCESS_LOG_PATH/` 目录（如果设置）
- `$ASCEND_WORK_PATH/` 目录（如果设置且未设置 ASCEND_PROCESS_LOG_PATH）

**日志解析规则**：

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

### 步骤 2：Pass 模块识别

**识别策略**：

1. **直接识别**：从错误消息中提取 pass 名称
   - 模式：`Pass\[([^\]]+)\]`
   - 模式：`in pass ([a-zA-Z0-9_]+)`

2. **堆栈分析**：从堆栈跟踪中推断 pass 模块
   - 检查文件路径：`framework/passes/`
   - 检查函数名：`run_pass`, `apply_pass`, `execute`

3. **错误模式匹配**：根据错误特征推断 pass 模块
   - Shape 相关 → ShapeInference
   - 类型相关 → TypeInference
   - 常量折叠 → ConstantFolding
   - 内存分配 → MemoryPlanning

### 步骤 3：错误原因分析

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

### 步骤 4：Pass 模块知识库

**常见 Pass 模块及错误模式**：

| Pass 模块 | 功能 | 常见错误 | 可能原因 |
|----------|------|---------|---------|
| **ShapeInference** | 形状推断 | Shape mismatch, Invalid shape | 输入shape不匹配、动态shape处理不当 |
| **TypeInference** | 类型推断 | Type mismatch, Invalid dtype | dtype不兼容、类型转换失败 |
| **ConstantFolding** | 常量折叠 | Folding failed, Invalid constant | 常量超出范围、不支持的操作 |
| **Lowering** | 算子下沉 | Lowering failed, Unsupported op | 算子不支持、缺少实现 |
| **MemoryPlanning** | 内存规划 | Out of memory, Allocation failed | 内存不足、内存碎片严重 |
| **ScheduleOptimization** | 调度优化 | Schedule invalid, Loop unroll failed | 循环边界错误、依赖关系错误 |
| **FuseOps** | 算子融合 | Fusion failed, Pattern not matched | 融合模式不匹配、算子属性冲突 |
| **LayoutTransform** | 布局转换 | Layout mismatch, Transform failed | 布局不兼容、转换规则缺失 |

### 步骤 5：错误严重程度评估

**评估标准**：

- **CRITICAL**: 导致程序崩溃、无法继续执行
- **HIGH**: 影响核心功能、结果错误
- **MEDIUM**: 影响性能、功能受限
- **LOW**: 警告信息、可忽略

### 步骤 6：生成分析报告

**报告格式**（Markdown）：

报告模板参见 `references/error_report_template.md` 文件。

## 使用场景

本技能适用于以下场景：

### 场景 1：分析单个日志文件中的 Pass 错误

当您有一个日志文件，需要快速定位和分析其中包含的 Pass 模块错误时使用。

**示例**：
```python
from pypto_pass_error_analyzer import parse_and_analyze_log

result = parse_and_analyze_log("path/to/error.log")
print(result)
```

### 场景 2：分析日志内容字符串

当您直接拥有日志内容字符串，需要分析其中的错误时使用。

**示例**：
```python
from pypto_pass_error_analyzer import parse_and_analyze_log_content

log_content = """
[INFO ] PYPTO(592313):2026-03-13 14:26:37.767 [l1_copy_reuse.cpp:642][PASS]:[L1CopyInReuseMerge.Operation]:Op 704 feature: [63, 128, 0, 32, 512, 148].
[ERROR] PYPTO(592314):2026-03-13 14:26:38.123 [shape_inference.cpp:123][ERROR]:[ShapeInference]:Shape mismatch at dim 2.
"""

result = parse_and_analyze_log_content(log_content)
print(result)
```

### 场景 3：批量分析多个日志文件

当您需要同时分析多个日志文件中的错误时使用。

**示例**：
```python
from pypto_pass_error_analyzer import analyze_logs_batch

results = analyze_logs_batch([
    "log1.log",
    "log2.log",
    "log3.log"
])
```

### 场景 4：按 Pass 模块分组分析

当您需要按不同的 Pass 模块对错误进行分类统计时使用。

**示例**：
```python
from pypto_pass_error_analyzer import group_by_pass

result = parse_and_analyze_log("error.log")
grouped = group_by_pass(result["errors"])

for pass_name, pass_errors in grouped.items():
    print(f"Pass: {pass_name}, Errors: {len(pass_errors)}")
```

### 场景 5：流式处理大日志文件

当日志文件非常大，需要逐行处理以避免内存溢出时使用。

**示例**：
```python
from pypto_pass_error_analyzer import parse_and_analyze_stream

with open("large.log", "r") as f:
    for error_analysis in parse_and_analyze_stream(f):
        process_error(error_analysis)
```

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

常见问题详见 `references/faq.md` 文件。

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
