---
name: pypto-pass-error-fixer
description: PyPTO Pass 模块错误诊断与修复技能。包含错误定位、原因分析和问题修复功能，提供从问题定位到修复验证的完整工作流程。当遇到 PyPTO Pass 模块抛出错误时使用此技能。
license: 完整条款见 LICENSE.txt
---

# PyPTO Pass 错误诊断与修复技能

本技能提供 PyPTO Pass 模块错误的完整诊断和修复能力，从错误原因定位到问题修复验证的端到端解决方案。

## 使用场景

当某业务场景执行失败，需要分析定位某pass抛出异常原因并修复该问题时使用此技能。

## 功能概述

- 解析 PyPTO 运行日志，提取错误信息
- 识别错误相关的 Pass 模块
- 分析错误产生的根本原因
- 定位问题代码位置
- 提供修复方案并等待用户确认
- 执行用户确认的修复操作
- 验证修复结果

## 前置条件

使用此技能前需要满足以下条件：

1. **环境要求**
   - PyPTO 开发环境已正确配置
   - 可访问 PyPTO 源代码目录
   - 具有日志文件读取权限

2. **输入要求**
   - 用户必须提供复现问题的执行命令
   - 可选：提供相关的测试用例或脚本路径

3. **依赖技能**
   - `pypto-environment-setup`：用于检查环境状态

## 触发机制

当用户输入包含以下错误日志或关键字时，自动触发此技能：

- **执行xx异常，修复pass问题**：定位Pass抛出异常的具体原因，并提供修复方案
- **执行xx时，pass报错**：定位Pass抛出异常的具体原因，并提供修复方案
- **定位pass错误**：定位Pass抛出异常的具体原因，并提供修复方案
- **分析pass异常**：定位Pass抛出异常的具体原因，并提供修复方案
- **分析pass失败原因**：定位Pass抛出异常的具体原因，并提供修复方案

**触发示例**
- 执行 python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4 异常，pass报错
- 执行 python3 test.py异常，分析pass失败原因

## 工作流程

### 步骤 1：问题复现

1. 如果用户执行的是python相关的脚本，开启图编译阶段调试模式开关：

   ```python
    @pypto.frontend.jit(
        debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1}
    )
    ```

2. 设置日志相关环境变量：
   - 设置日志输出目录环境变量：`export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs/$(date +%Y%m%d%H%M%S) `
   - 设置日志输出级别环境变量：`export ASCEND_GLOBAL_LOG_LEVEL=0 `

3. 根据用户提供的执行命令，复现用户问题

**重要**: 上述步骤中的 `设置日志相关环境变量` 与 `根据用户提供的执行命令，复现用户问题`，必须同一个会话中完成，否则环境变量的设置无法生效。

**验证检查点**：
- [ ] 问题成功复现
- [ ] 获得可复现的执行命令
- [ ] 记录复现时的环境信息

### 步骤 2：获取日志内容

**日志查找策略：**

1. 从 `$ASCEND_PROCESS_LOG_PATH/debug/plog` 目录下获取 `pypto-*.log`
2. 优先查找pass模块相关的[ERROR]、[WARN]级别的日志

**日志文件验证：**
- 确认日志文件包含 [PASS] 标记
- 确认日志文件包含文件路径和行号信息

**验证检查点**：
- [ ] 成功定位日志文件
- [ ] 日志内容可读取
- [ ] 日志内容包含错误或警告信息
- [ ] 日志内容包含 Pass 模块信息

### 步骤 3：解析日志关键信息

**日志格式示例**：

以如下pass模块打印的日志格式为例，分段解析：
```
[ERROR] PYPTO(638465):2026-03-16 10:02:24.711 [n_buffer_merge.cpp:530][PASS]:[NBufferMerge.Config]:The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.
```

**解析字段**：

1. 日志级别：[ERROR]
2. 模块名及进程：PYPTO(638465)
3. 日志打印时间：2026-03-16 10:02:24.711
4. 代码文件及行号：[n_buffer_merge.cpp:530] 此处 `n_buffer_merge.cpp` 为文件名，`530` 为打印该日志代码所在行号
5. 所属模块类型：[PASS]
6. 具体模块信息：[NBufferMerge.Config]，此处 `NBufferMerge` 可能是pass模块名，也可能是其他工具模块名，如果不是pass模块名称要根据日志上下文分析是哪个pass模块调用的该方法
7. 日志内容：The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.

**多行错误处理**：
- 检查错误信息是否跨越多行
- 合并相关联的错误上下文
- 提取完整的错误堆栈信息

**日志级别处理**：
- ERROR：必须处理的关键错误
- WARNING：可能导致问题的警告信息
- INFO：辅助调试的信息日志

**验证检查点**：
- [ ] 日志级别正确识别
- [ ] 代码位置准确提取
- [ ] Pass模块名称正确解析
- [ ] 错误内容完整提取
- [ ] 时间戳格式正确解析

### 步骤 4：错误原因分析

#### 4.1 分析流程

**分析步骤**：

1. **理解用户业务代码**
   - 根据用户提供的复现命令，找到用户对应代码位置
   - 分析用户业务代码，了解业务场景 

2. **获取关键信息**
   - 获取日志关键信息：错误内容、文件与行号、堆栈轨迹、上下文相关日志
   - 获取计算图和IR目录信息（如果有的话）：计算图和IR默认输出在 `output/output_xxx` 目录下，例如：`output/output_20260324_162232_912961_1605290_C0A8451A`

3. **理解Pass模块代码**
   - 根据日志中获取的相关信息定位异常pass模块及相关的源文件代码位置
   - 从 `framework/src/passes` 目录下获取对应Pass模块代码
   - 重点分析异常代码上下文，不仅仅是出错的那一行，通常需要向上取 20 行，向下取 10 行
   - 使用 `pypto-pass-module-analyzer` 技能分析对应pass模块整体业务逻辑

4. **理解计算图（如果有的话）**
   - 根据时间戳找到最新的计算图输出目录，在最新目录下查找异常pass模块的 `.json` 计算图文件
   - 根据 `references/computation-graph-parse.md` 指导分析计算图，推断可能的错误原因
   - 根据日志内容及计算图信息，定位异常节点对应的用户代码行，了解对应算子的用法与约束

5. **理解IR（如果有的话）**
   - 根据时间戳找到最新的IR输出目录，在最新目录下查找异常pass模块的 `.tifwkgr` IR文件
   - 根据 `references/ir-analysis-guide.md` 指导分析IR，推断可能的错误原因

6. **分析错误原因**
   - 综合上述步骤信息，分析推导错误原因
   - 列举可能的错误原因、代码位置、修复建议

#### 4.2 分析技巧

1. **错误日志内容分析**
   - 错误信息中通常包含明确的错误原因，可以根据内容推断错误类型和原因
   - 错误信息中通常包含针对该错误的处理建议，可参考对应的建议进一步分析或者生成修复建议

2. **了解参数配置约束**
   - pass参数配置参考文档：`docs/api/config/pypto-set_pass_options.md`
   - 分析错误类型涉及参数配置，可根据参数配置文档进行判断

3. **了解算子用法和约束**
   - 算子文档通常位于 `docs/api/operation` 目录下
   - 分析用户提供的复现脚本，可以找到对应代码涉及的算子
   - 通过文档了解各算子的用法和约束，判断用户提供的代码是否正确

4. **定位异常代码行**
   - 根据日志信息中代码文件及行号可以定位到打印对应的日志的代码行
   - 根据日志内容中Operation的magic（如果有的话）可以再计算图中找到节点信息，根据计算图中Operation中源代码文件及源代码行号信息，定位对应的用户代码位置
   - 根据日志内容中Tensor的magic（如果有的话），可以在计算图中找到其生产者、消费者 Operation，再根据Operation定位用户代码位置

**验证检查点**：
- [ ] 错误类型正确分类
- [ ] 错误原因分析准确
- [ ] 代码位置精确定位
- [ ] 修复建议合理可行

### 步骤 5：问题修复

按如下**修复步骤**定义的顺序执行修复，不预设固定修复方案，由 LLM 根据步骤4分析得到的错误原因及修复建议，制定合理的修复方案

**修复步骤**：

1. **用户确认修复方案**
   - 使用 `question` 工具列举可行的修复方案，让用户选择修复方案
   - 说明每个方案的优缺点
   - 标注推荐方案（如有）
   - 用户可选择确认修复、跳过修复或提供新的方案

2. **备份原始代码**
   - 创建修复前的代码备份
   - 记录修改的文件列表

3. **执行修复操作**
   - 根据用户确认的方案执行修复
   - 记录修改的详细内容

4. **代码质量检查**
   - 检查改动的代码是否与原代码风格一致
   - 确保没有引入新的问题

**失败处理机制**：

- 如果修复失败，自动回退到原始代码
- 记录失败原因和错误信息
- 提供其他修复方案供用户选择

**验证检查点**：
- [ ] 修复方案已展示给用户
- [ ] 用户已确认修复方案
- [ ] 备份文件成功创建
- [ ] 代码修改准确执行
- [ ] 代码质量检查通过

### 步骤 6：修复验证

#### 6.1 执行测试

1. **执行相关用例**
   - 重新编译 pypto 包 `python3 build_ci.py -f python3 --disable_auto_execute`，并pip安装 `pip install build_out/pypto*.whl --force --no-deps`
   - 根据用户提供的用例，执行命令验证相关用例
   - 检查运行结果是否符合预期
   - 检查日志中是否还有ERROR级别的日志

#### 6.2 回归检查（可选）

对pass业务逻辑做了改变需要通过回归检查确保修复未影响其他功能，无pass业务逻辑改变可以不做回归检查

1. **运行相关测试用例集**
   - 执行相关的回归测试
   - 确保修复未破坏其他功能

2. **回归测试范围**
   - 同一Pass模块的其他测试用例
   - 依赖该Pass模块的测试用例
   - 相关业务场景的测试用例

#### 6.3 失败回滚

1. **测试不通过情况**
   - 如果测试不通过，将修改内容还原
   - 记录回滚原因
   - 分析失败原因，提供新的修复建议

2. **回滚操作**
   - 恢复备份的原始代码
   - 清理临时文件
   - 记录回滚日志

#### 6.4 性能验证（可选）

如果修复涉及性能优化，需要进行性能验证：
- 对比修复前后的性能数据
- 确保性能没有下降
- 记录性能改进情况

**验证检查点**：
- [ ] 测试用例执行成功
- [ ] 运行结果符合预期
- [ ] 日志中无ERROR级别日志
- [ ] 回归测试通过
- [ ] 未引入新的问题
- [ ] 性能验证通过（如适用）

## 输出格式

### 错误分析报告格式

```markdown
## PyPTO Pass 错误分析报告

### 一、基本信息
- **错误级别**: ERROR
- **发生时间**: 2026-03-16 10:02:24.711
- **进程ID**: 638465
- **复现命令**: python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4

### 二、错误位置
- **文件**: n_buffer_merge.cpp
- **行号**: 530
- **Pass模块**: NBufferMerge
- **Element类型**: Config

### 三、错误信息
```
The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.
```

### 四、错误原因分析
1. **主要原因**: 参数配置错误
2. **详细分析**: VEC_NBUFFER_SETTING 参数值 -3 超出有效范围
3. **影响范围**: 仅影响当前Pass模块
4. **相关代码片段**:
   ```cpp
   // n_buffer_merge.cpp:530
   if (key < -1 || key > max_hash_order) {
       GELOGE(INTERNAL_ERROR, "The VEC_NBUFFER_SETTING key %d is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder %d.", key, max_hash_order);
       return INTERNAL_ERROR;
   }
   ```

### 五、修复建议
1. **修复方案**: 调整 VEC_NBUFFER_SETTING 参数值
2. **具体步骤**:
   - 检查参数配置文件
   - 将参数值调整为有效范围（-1 到 0）
   - 重新执行测试用例
3. **风险提示**:
   - 修改参数可能影响性能
   - 建议先在测试环境验证

### 六、修复结果
- **修复状态**: 成功/失败/跳过
- **测试结果**: 通过/失败
- **回归测试**: 通过/失败/未执行
- **性能影响**: 无/轻微/显著

### 七、附录
- **完整日志**: [日志文件路径]
- **相关文档**: [文档链接]
- **参考案例**: [案例链接]

## 版本兼容性

本技能支持以下 PyPTO 版本：
- PyPTO 8.5.0+
- 建议使用最新版本以获得最佳支持

不同版本可能存在以下差异：
- 日志格式可能略有不同
- Pass模块名称可能变化
- 错误信息内容可能更新

## 性能优化建议

处理大规模日志文件时：
1. 使用流式读取，避免一次性加载整个文件
2. 优先搜索ERROR级别日志，减少处理范围
3. 使用多线程并行处理多个日志文件
4. 缓存已解析的日志信息，避免重复解析

## 使用示例

### 示例 1：参数配置错误

**用户输入：**
```
执行 python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4 异常，pass报错
```

**执行流程：**
1. 设置调试选项和环境变量
2. 执行命令复现问题
3. 从日志中提取提取错误信息
4. 定位到 NBufferMerge.Config Pass
5. 分析参数配置错误
6. 提供修复建议
7. 用户确认修复
8. 验证修复结果

**输出：**
```markdown
## 错误分析报告

### 错误位置
- 文件: n_buffer_merge.cpp
- 行号: 530
- Pass模块: NBufferMerge
- Element类型: Config

### 错误信息
The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.

### 修复建议
将 VEC_NBUFFER_SETTING 参数值从 -3 改为 -1
```

### 示例 2：复杂错误场景

**用户输入：**
```
执行 python3 test.py 异常，分析pass失败原因
```

**执行流程：**
1. 设置调试选项和环境变量
2. 执行命令复现问题
3. 从日志中提取提取错误信息
4. 识别多个Pass相关错误
5. 逐个分析错误
6. 提供综合修复建议
7. 用户确认修复
8. 验证修复结果

## 参考文档

### 核心文档
- [查看计算图](docs/tools/computation_graph/查看计算图.md)
- [PyPTO IR分析指导](references/ir-analysis-guide.md)
- [计算图JSON解析指导](references/computation-graph-parse.md)

### 相关技能
- [pypto-environment-setup](../pypto-environment-setup/SKILL.md)
- [pypto-pass-module-analyzer](../pypto-pass-module-analyzer/SKILL.md)
- [pypto-pass-workflow-analyzer](../pypto-pass-workflow-analyzer/SKILL.md)

### API文档
- [Pass配置API](docs/api/config/pypto-set_pass_options.md)
