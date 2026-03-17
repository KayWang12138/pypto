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
- **执行xx用例失败，分析并修复pass失败**：根据用例执行日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，并尝试自动修复
- **根据xx用例失败日志信息，分析并修复pass失败**：根据用户提供的错误日志信息，识别错误是哪个 pass 模块导致的，分析错误产生的可能原因，并尝试自动修复

## 工作流程

### 步骤 1：获取日志内容

1. 如果用户提供了具体的错误信息，直接使用用户提供的日志内容
2. 如果用户提供了具体的错误日志路径，从用户指定的错误文件中获取日志内容
3. 如果用户提供了具体的错误日志目录，从用户指定的日志目录下查找日志文件并获取日志内容
4. 如果用户没提供具体错误信息或日志路径，从如下位置获取日志文件
    - `$HOME/ascend/log/` 目录（默认）
    - `$ASCEND_PROCESS_LOG_PATH/` 目录（如果设置）
    - `$ASCEND_WORK_PATH/` 目录（如果设置且未设置 ASCEND_PROCESS_LOG_PATH）

### 步骤 2：解析日志关键信息

以如下pass模块打印的日志格式为例，分段解析：
```
[ERROR] PYPTO(638465):2026-03-16 10:02:24.711 [n_buffer_merge.cpp:530][PASS]:[NBufferMerge.Config]:The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.
```

1. 日志级别：[ERROR]
2. 模块名及进程：PYPTO(638465)
3. 日志打印时间：2026-03-16 10:02:24.711
4. 代码位置及行号：[n_buffer_merge.cpp:530]
5. 所属模块类型：[PASS]
6. 具体模块信息：[NBufferMerge.Config] Pass模块名：`NBufferMerge`，Element类型：`Config`
7. 日志内容：The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.

### 步骤 3：错误原因分析

#### 3.1 分析流程

1. 分析用例代码，识别用例涉及的业务场景，使用 `pypto-pass-workflow-analyzer` 技能，了解涉及的各个 pass 模块在当前场景中的职责、执行顺序依赖关系及数据流转过程
2. 根据错误日志信息中获取的相关数据，反向分析 错误行 -> 所属函数 -> 所属pass模块，
3. 结合错误日志内容，模块的代码、功能和设计，异常方法在该pass中的作用推断可能的失败原因

如果能分析出具体原因，如果不能分析出具体原因详细分析 更高的日志级别、打印分析计算图、IR、打开precheck、postcheck、涉及外部调用的，如果是文档的找文档约束，python的用例找到涉及的算子的文档约束

#### 3.2 分析维度
错误原因的分析可从以下维度入手，但不仅限于这些维度

1. 输入数据问题：Shape 不对齐、Dtype 不匹配、数据范围异常、空张量/零维度等
2. API 使用问题： 参数错误、不支持的配置、缺少必需参数、参数冲突等
3. 环境问题：NPU 设备不可用、内存不足、CANN 版本不兼容、驱动问题等
4. 代码逻辑问题：算子实现错误、边界条件处理不当、类型转换错误、内存访问越界等

各pass模块常见错误及修复建议可参考：[常见错误及修复建议](references/pass-modules-knowledge-base.md)

#### 3.3 分析技巧
1. 错误信息中通常带有处理建议
2. 参数配置类错误可以参数文档中参数的约束和作用判断参数是否合法
3. 用例中涉及的算子，可以通过文档了解各算子的用法和约束
4. 打开precheck/postcheck可以校验图是否符合约束
5. 要了解是用户代码的哪一行异常，可以反向推到，错误日志中op的magic -> 计算图 -> 前段代码行
6. 环境问题可以通过环境setup技能来检查
### 步骤 4：代码修复

问题修复从以下两个维度进行
- 可自动修复（参数配置错误、类型转换错误、数据类型不匹配等）→ 直接修复
- 需人工判断（算子实现错误、边界条件处理不当等）→ 生成修复建议


分析错误原因，简单问题自动修复并验证，复杂问题给出修复建议

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

## 参考文档

- [常见错误及修复建议](references/pass-modules-knowledge-base.md) — 各Pass模块常见错误原因及对应修复建议


