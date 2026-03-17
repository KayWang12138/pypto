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

### 尝试复现
根据用户提供的复现命令，尝试复现相关用户问题

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
2. 根据错误日志信息中获取的相关数据，定位错误 所属代码行 -> 所属函数 -> 所属pass模块，分析相关代码理解上下文
3. 根据错误日志内容，及对应代码上下文逻辑，初步分析错误类型确定排查方向
4. 综合分析日志上下文相关信息，结合用例代码及业务逻辑代码，推断可能的错误原因，列举出可能的错误原因，对应的代码位置及相关问题的修复建议

#### 3.2 分析维度

错误原因的分析可从以下维度入手，但不仅限于这些维度

1. 输入数据问题：Shape 不对齐、Dtype 不匹配、数据范围异常、空张量/零维度等
2. API 使用问题： 参数错误、不支持的配置、缺少必需参数、参数冲突等
3. 环境问题：NPU 设备不可用、内存不足、CANN 版本不兼容、驱动问题等
4. 代码逻辑问题：算子实现错误、边界条件处理不当、类型转换错误、内存访问越界等

各pass模块常见错误及修复建议可参考：[常见错误及修复建议](references/pass-modules-knowledge-base.md)

#### 3.3 分析技巧

1. 错误信息中通常带有处理建议，可根据错误内容推断可能的错误类型和原因
2. 参数配置类错误可以参数文档中参数的约束和作用判断参数是否合法，配置参数文档通常位于 `docs/api/config` 目录下
3. python用例中涉及的算子，可以通过文档了解各算子的用法和约束，算子结束文档通常位于 `docs/api/operation` 目录下
4. 打开precheck/postcheck可以校验图是否符合约束，可以帮助定位错误位置
5. 要了解是用户代码的哪一行异常，可以反向推到，错误日志中op的magic -> 计算图 -> 前端代码行
6. 环境问题可以通过 `pypto-environment-setup` 技能来检查环境状态
7. 日志上下文信息不足，可通过调整日志级别打印更多辅助信息，日志相关配置可参考 `references/pass-error-locating-faq.md` 中的Q7、Q8

### 步骤 4：问题修复

根据上一步分析的错误原因及相关修改建议，执行对应的修复操作
- 可自动修复（参数配置错误、类型转换错误、数据类型不匹配等）→ 直接修复
- 需人工判断（算子实现错误、边界条件处理不当等）→ 生成修复建议


### 步骤 6：修复验证

执行用户提供的用例执行命令进行验证，检查运行结果是否符合预期

UT执行命令示例
```
python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestNBufferMerge
```

ST执行命令示例
```
python3 build_ci.py -s=python/tests/st/test_adds_onboard.py::test_adds_onboard
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

- [Pass常见错误及修复建议](references/pass-modules-knowledge-base.md) — 各Pass模块常见错误原因及对应修复建议
- [错误分析常见问题](references/pass-error-locating-faq.md) - 问题定位过程中可能遇到的场景问题

