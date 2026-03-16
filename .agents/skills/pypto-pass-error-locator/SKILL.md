---
name: pypto-pass-error-diagnosis-and-fixer
description: PyPTO Pass 模块错误诊断与自动修复技能。整合错误定位、原因分析和自动修复功能，提供从问题发现到修复验证的完整工作流程。当遇到 PyPTO Pass 模块错误时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO Pass 错误诊断与自动修复技能

本技能提供 PyPTO Pass 模块错误的完整诊断和修复能力，从错误日志解析到自动修复验证的端到端解决方案。

## 概述

整合错误定位和自动修复功能，提供连贯的问题解决流程：
- 错误日志解析和信息提取
- Pass 模块识别和错误定位
- 错误原因分析和严重程度评估
- 自动修复策略应用和代码修复
- 修复验证和报告生成

## 使用场景

当需要根据执行异常日志，分析定位某pass异常原因并自动修复该问题时使用此技能。

## 功能概述

- 解析 PyPTO 运行日志，提取错误信息
- 识别错误相关的 Pass 模块
- 分析错误产生的根本原因
- 提供错误分类和严重程度评估
- 生成详细的错误分析报告
- 自动定位问题代码位置
- 应用自动修复策略
- 验证修复结果
- 支持回滚操作

## 触发机制

当用户输入包含以下关键字时，自动触发此技能：
- **执行用例失败，分析并修复pass失败**：根据用例执行日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，并尝试自动修复
- **根据日志信息，分析并修复pass失败**：根据用户提供的错误日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，并尝试自动修复

## 工作流程

### 步骤 1：获取日志

1. 如果用户提供了具体的错误信息，直接使用用户提供的日志内容
2. 如果用户提供了具体的错误日志路径，从用户指定的错误文件中获取日志内容
3. 如果用户提供了具体的错误日志目录，从用户指定的日志目录下查找日志文件并获取日志内容
4. 如果用户没提供具体错误信息或日志路径，从如下位置获取日志文件
    - `$HOME/ascend/log/` 目录（默认）
    - `$ASCEND_PROCESS_LOG_PATH/` 目录（如果设置）
    - `$ASCEND_WORK_PATH/` 目录（如果设置且未设置 ASCEND_PROCESS_LOG_PATH）

### 步骤 2：解析日志

**日志格式示例**：
```
[INFO ] PYPTO(592313):2026-03-13 14:26:37.767 [l1_copy_reuse.cpp:642][PASS]:[L1CopyInReuseMerge.Operation]:Op 704 feature: [63, 128, 0, 32, 512, 148].
```

**解析模式**：

1. **日志级别识别**
    - 模式：`\[(DEBUG|INFO|WARNING|ERROR)\s*\]`
    - DEBUG: 调试信息
    - INFO: 一般信息
    - WARNING: 警告信息
    - ERROR: 致命错误

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
    - Pass模块名：`L1CopyInReuseMerge`
    - Element类型：`Operation`

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

**定位策略**：

1. **基于堆栈跟踪定位**
    - 从堆栈中提取文件路径和行号
    - 定位到具体的错误代码行

2. **基于 pass 模块定位**
    - 在 `framework/passes/` 目录中搜索 pass 模块
    - 查找 pass 模块的实现代码

3. **基于错误模式定位**
    - 使用 grep 搜索错误消息
    - 在相关文件中查找匹配的代码

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

该文档包含了 Pass 模块常见错误原因的知识库，涵盖：
- 每个模块的详细功能描述
- 常见错误类型列表
- 可能的根本原因分析
- 修复建议

在分析 Pass 模块错误时，请先查阅该知识库以获取详细的错误模式信息。

### 步骤 5：代码修复

根据上一步分析的错误原因及相关修改建议，执行对应的修复操作

**修复操作类型**：

1. **参数修复**
```python
# 修复前
result = pypto.matmul(x, y)

# 修复后
result = pypto.matmul(x, y, transpose_b=True)
```

2. **类型转换修复**
```python
# 修复前
result = pypto.add(x, y)

# 修复后
x = x.to(pypto.float32)
y = y.to(pypto.float32)
result = pypto.add(x, y)
```

3. **Shape 调整修复**
```python
# 修复前
result = pypto.reshape(x, [batch, -1])

# 修复后
x = x.reshape([batch, seq_len, hidden_dim])
result = pypto.reshape(x, [batch, -1])
```

4. **内存优化修复**
```python
# 修复前
x = pypto.zeros([10000, 10000])

# 修复后
x = pypto.zeros([batch_size, seq_len, hidden_dim])
```

### 步骤 6：修复验证

**验证步骤**：

1. **语法验证**
```bash
python3 -m py_compile file.py
```

2. **编译验证**
```bash
python3 build_ci.py -f python3 --disable_auto_execute
```

3. **运行验证**
```bash
python3 custom/operator/operator.py --run_mode npu
```

4. **精度验证**
```bash
python3 custom/operator/operator.py --verify
```

## 使用方法

### 方式 1：自动诊断和修复单个错误

```python
from pypto_error_diagnosis_fixer import auto_diagnosis_fix

# 从错误日志得到的结果
log_content = """
[ERROR] PYPTO(592314):2026-03-13 14:26:38.123 [shape_inference.cpp:123][ERROR]:[ShapeInference]:Shape mismatch at dim 2.
"""

fix_result = auto_diagnosis_fix(log_content)
print(fix_result)
```

### 方式 2：交互式诊断和修复

```python
from pypto_error_diagnosis_fixer import interactive_diagnosis_fix

# 显示诊断和修复建议，等待用户确认
fix_result = interactive_diagnosis_fix(log_content)
```

### 方式 3：批量诊断和修复

```python
from pypto_error_diagnosis_fixer import batch_diagnosis_fix

logs = [log1, log2, log3]
fix_results = batch_diagnosis_fix(logs)
```

### 方式 4：回滚修复

```python
from pypto_error_diagnosis_fixer import rollback_fix

# 回滚指定的修复
rollback_fix(fix_id)
```

## 输出验证

**验证检查点**：
- [x] 错误信息完整提取
- [x] 堆栈跟踪准确解析
- [x] Pass 模块名称正确识别
- [x] 时间戳格式正确
- [x] 错误原因分析准确
- [x] 修复建议合理可行
- [x] 自动修复正确应用
- [x] 修复验证通过

## 参考文件

| 文件 | 内容 |
|------|----------|
| `scripts/parse_log.py` | 日志解析脚本 |
| `scripts/analyze_pass_error.py` | Pass 模块错误分析脚本 |
| `scripts/auto_fix.py` | 自动修复脚本 |
| `scripts/verify_fix.py` | 修复验证脚本 |
| `tests/test_log_parser.py` | 日志解析单元测试 |
| `tests/test_pass_analyzer.py` | Pass 分析单元测试 |
| `tests/test_auto_fixer.py` | 自动修复单元测试 |
| `references/pass-modules-knowledge-base.md` | Pass 模块知识库 |
| `references/pass-error-report-template.md` | 错误报告模板 |
| `knowledge_base/fix_strategies.json` | 修复策略知识库 |
