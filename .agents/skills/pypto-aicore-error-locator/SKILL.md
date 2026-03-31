---
name: pypto-aicore-error-locator
description: “定位测试案例中出现 aicore error 时的问题 CCE 文件、问题代码行及对应的前端源代码。Triggers: aicore error、定位aicore error的原因、帮我定位aicore error报错”。
---

# AICore Error 定位器

此技能用于定位 PyPTO 测试中出现的 aicore error，通过系统化的排查流程，找出导致错误的 CCE 文件和具体代码行。

## 工作流程概述

1. 收集必要信息（必须执行）
2. 排除 machine 框架调度问题
3. 启用追踪日志
4. 重新编译和安装
5. 清理日志并运行测试
6. 分析追踪日志并定位 CCE 文件
7. 二分查找定位问题代码行
8. 映射到前端源代码
9. 输出结果

---

## 步骤 1：收集必要信息（必须执行）

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **pypto_path**: pypto 项目的根目录路径（绝对路径）
- **device_log_path**: device log 的落盘路径（若不存在则需创建，绝对路径）
- **test_cmd**: 触发 aicore error 的完整测试命令
- **run_path**: 运行测试命令的目录路径（绝对路径）

将收集的路径全部转换成绝对路径，收集到所有信息后才能继续后续步骤。

---
**⚠️ 重要提示**: 将bash运行命令超时时间设置为1800000ms
## 步骤 2：排除 machine 框架调度问题

### 2.1 注释 CallSubFuncTask

使用脚本注释 `aicore_entry.h` 中的 CallSubFuncTask 部分：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/modify_callsubfunctask.py comment pypto_path
```

**脚本参数说明**:
- `comment`: 注释 CallSubFuncTask
- `pypto_path`: pypto 项目的根目录路径（绝对路径或相对路径）

**脚本功能**:
- 在指定 pypto 目录中搜索 `aicore_entry.h` 文件
- 找到 `CallSubFuncTask` 函数
- 注释 `CallSubFuncTask` 及相关代码行

### 2.2 编译安装

进入 `pypto_path`，重新编译 pypto 包并 pip 安装。

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

### 2.3 运行验证

进入 `run_path`，运行测试。

```bash
cd run_path && test_cmd
cd -
```

**⚠️ 重要提示**：
- **若有aicore error**: 说明是 machine 调度框架的问题，已找到问题原因，**停止执行后续步骤！**
- **若没有aicore error**: 说明问题在 kernel 代码中，**请继续执行后续步骤！**

### 2.4 取消注释 CallSubFuncTask

使用脚本取消注释 `aicore_entry.h` 中的 CallSubFuncTask 部分：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/modify_callsubfunctask.py uncomment pypto_path
```

**脚本参数说明**:
- `uncomment`: 取消注释 CallSubFuncTask
- `pypto_path`: pypto 项目的根目录路径（绝对路径或相对路径）

**脚本功能**:
- 在指定 pypto 目录中搜索 `aicore_entry.h` 文件
- 找到 `CallSubFuncTask` 函数
- 取消注释 `CallSubFuncTask` 及相关代码行

---

## 步骤 3：启用追踪日志

进入 `pypto_path`，修改以下配置：

- **配置文件**: 修改 `tile_fwk_config.json`
  - 设置 `"fixed_output_path"` 为 `true`
  - 设置 `"force_overwrite"` 为 `false`

- **头文件**: 修改 `aicore_print.h`
  - 设置 `#define ENABLE_AICORE_PRINT` 为 `1`

- **工具头文件**: 修改 `device_switch.h`
  - 设置 `#define ENABLE_COMPILE_VERBOSE_LOG` 为 `1`

---

## 步骤 4：重新编译和安装

进入 `pypto_path`，重新编译 pypto 包并 pip 安装。

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

---

## 步骤 5：清理日志并运行测试

### 5.1 运行测试

进入 `run_path`，配置环境变量并运行测试。

```bash
rm -rf device_log_path/* && rm -rf run_path/kernel_aic* && cd run_path && export ASCEND_PROCESS_LOG_PATH=device_log_path && export ASCEND_GLOBAL_LOG_LEVEL=0 && test_cmd
cd -
```

**⚠️ 重要提示**: 运行测试的打屏日志中必须出现 aicore error，如果未出现，则不适用于该 SKILL，**停止执行后续步骤**


### 5.2 获取 program.json 路径

运行脚本获取最新的 program.json 路径：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/get_latest_program_json.py run_path/output
```

记录输出的 `program_json_path`，该路径将在步骤 8 中使用。

---

## 步骤 6：分析追踪日志并定位 CCE 文件

### 6.1 查找 trace 日志、分析缺失 leaf index 并定位问题 CCE 文件

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/analyze_trace.py device_log_path run_path/kernel_aicore
```

**⚠️ 重要提示**: 若未定位到问题 CCE 文件，请说明原因，**停止执行后续步骤**

### 6.2 测试验证 CCE 文件

如果有多个问题 CCE 文件，需要分别测试每个文件，以确定哪个是问题文件。若只有一个问题 CCE 文件，测试验证该文件是否为问题文件：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/test_cce_file.py <cce_file> test_cmd run_path
```

**⚠️ 重要提示**:
- 若未定位到问题 CCE 文件，请说明原因，**停止执行后续步骤**
- 若打印的 error 中包含 `ld.lld: error: undefined` 关键字，则修改 `tile_fwk_config.json` 中的 `parallel_compile` 为 `1`，再从步骤 1 开始重新执行一遍

---

## 步骤 7：二分查找定位问题代码行

### 7.1 获取 ERROR_IN_T 的值（错误是否在 T 操作中）

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/determine_error_scope.py <cce_file> test_cmd run_path
```

**⚠️ 重要提示**:
- `cce_file` 为步骤 6.2 的输出
- 若打印的 error 中包含 `ld.lld: error: undefined` 关键字，则修改 `tile_fwk_config.json` 中的 `parallel_compile` 为 `1`，再从步骤 1 开始重新执行一遍

### 7.2 获取二分查找初始范围

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/get_commentable_range.py <cce_file> ERROR_IN_T
```

记录输出的 `LEFT` 和 `RIGHT` 值。

### 7.3 执行二分查找迭代

根据上一步的 `LEFT` 和 `RIGHT` 值，执行第一次迭代：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/binary_search_iteration.py <cce_file> test_cmd run_path <left> <right> ERROR_IN_T
```

记录输出的 `NEXT_LEFT` 和 `NEXT_RIGHT` 值。

**判断逻辑**:
- 如果 `NEXT_LEFT` 等于 `NEXT_RIGHT`，则已找到问题行（输出 `FOUND <problem_line>`）
- 否则，使用新的 `NEXT_LEFT` 和 `NEXT_RIGHT` 作为下一轮的 `left` 和 `right`，重复执行此步骤

**⚠️ 重要提示**:
- `cce_file` 为步骤 6.2 的输出
- 若未定位到问题代码行，请说明原因，**停止执行后续步骤**

---

## 步骤 8：映射到前端源代码

使用以下命令将 CCE 问题代码行映射到前端源代码：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py <cce_file> <program_json_path> <problem_line>
```

**参数说明**:
- `<cce_file>`: 步骤 6.2 输出的问题 CCE 文件路径
- `<program_json_path>`: 步骤 5.2 输出的 program.json 文件路径
- `<problem_line>`: 步骤 7 输出的问题代码行号

**输出说明**:
- 若匹配成功，将输出前端源代码文件路径和行号
- 若匹配失败，将说明原因（例如：框架自动生成代码、操作数不匹配等）

---

## 步骤 9：输出结果

输出以下信息：
- 找到的 CCE 文件路径
- 问题代码行号
- 问题代码内容
- 前端源代码文件路径（如果步骤 8 映射成功）
- 前端源代码行号（如果步骤 8 映射成功）
- 前端源代码内容（如果步骤 8 映射成功）

---

## 错误处理

### 步骤 1：收集必要信息

**可能错误**：
- 用户取消输入或提供无效路径
- 路径不存在或不可访问

**错误处理**：
- 输出：收集到的信息和缺失的信息
- 建议：重新执行步骤 1，提供有效的绝对路径
- 降级方案：如果无法获取完整信息，说明技能无法继续执行

### 步骤 2：排除 machine 框架调度问题

**可能错误**：
- `aicore_entry.h` 文件未找到
- `CallSubFuncTask` 函数未找到
- 编译失败
- 安装失败
- 测试执行失败

**错误处理**：
- **文件未找到**：输出搜索路径和可能的文件位置，建议检查 PyPTO 版本
- **编译失败**：输出编译日志，定位编译错误，建议检查环境配置
- **安装失败**：输出安装日志，检查依赖冲突，建议使用 pip 修复
- **测试执行失败**：输出测试错误，区分是否为 aicore error
  - 如果是 aicore error：输出 "machine 框架调度问题"，停止执行
  - 如果不是 aicore error：继续执行步骤 3
- **降级方案**：如果无法注释/取消注释，尝试手动修改文件

### 步骤 3：启用追踪日志

**可能错误**：
- 配置文件未找到
- 头文件未找到
- 文件修改失败

**错误处理**：
- **文件未找到**：输出搜索路径和可能的文件位置，建议检查 PyPTO 版本
- **修改失败**：输出文件权限和当前内容，建议检查文件权限
- **降级方案**：如果无法自动修改，提供手动修改指南

### 步骤 4：重新编译和安装

**可能错误**：
- 编译失败
- 安装失败

**错误处理**：
- **编译失败**：输出编译日志，定位编译错误，建议检查配置修改是否正确
- **安装失败**：输出安装日志，检查依赖冲突，建议使用 pip 修复
- **降级方案**：如果编译和失败，尝试回滚配置修改，使用之前的工作版本

### 步骤 5：清理日志并运行测试

**可能错误**：
- 日志清理失败
- 测试执行失败
- 未出现 aicore error
- 出现链接错误（ld.lld: error: undefined）
- 出现段错误（segment fault）
- 出现其他错误
- program.json 未找到

**错误处理**：
- **日志清理失败**：输出路径权限，建议手动清理
- **测试执行失败**：输出测试错误，检查是否为环境问题
- **未出现 aicore error**：输出 "未检测到 aicore error，技能不适用"，停止执行
- **链接错误**：
  - 输出：链接错误信息摘要（符号名称、引用位置）
  - 建议：
    - 检查 `tile_fwk_config.json` 中的 `parallel_compile` 设置
    - 尝试将 `parallel_compile` 设置为 `1` 并从步骤 1 重新执行
    - 检查分布式算子的特殊配置需求
  - 降级方案：
    - 手动分析链接错误信息
    - 检查相关头文件和源代码
    - 使用其他编译调试方法
  - 停止执行
- **段错误**：
  - 输出：段错误发生的位置和上下文（如果有）
  - 建议：
    - 使用调试工具（如 gdb、core dump）分析段错误
    - 检查空指针访问、数组越界等常见原因
    - 简化测试场景，逐步缩小问题范围
  - 降级方案：
    - 使用 pypto-pass-error-fixer 技能
    - 手动添加调试输出，定位问题代码
    - 检查内存访问相关的代码逻辑
  - 停止执行
- **其他错误**：
  - 输出：错误类型和错误信息摘要
  - 建议：根据错误类型使用对应的调试技能
  - 降级方案：
    - 编译错误：使用 pypto-pass-error-fixer 技能
    - 精度错误：使用 pypto-precision-debugger 技能
    - Python 异常：使用常规 Python 调试方法
  - 停止执行
- **program.json 未找到**：输出搜索路径和可能的文件位置，建议检查输出配置
- **降级方案**：如果无法获取 program.json，尝试使用其他调试方法

### 步骤 6：分析追踪日志并定位 CCE 文件

**可能错误**：
- trace 日志未找到
- 脚本执行失败
- 未定位到 CCE 文件
- CCE 文件测试失败
- 出现 `ld.lld: error: undefined` 错误

**错误处理**：
- **trace 日志未找到**：输出日志路径和可能的原因，建议检查日志配置
- **脚本执行失败**：输出脚本错误和堆栈，建议检查脚本依赖
- **未定位到 CCE 文件**：输出分析结果和可能的原因，建议检查日志完整性
- **CCE 文件测试失败**：输出测试错误，检查是否为编译问题
- **ld.lld 错误**：输出 "检测到并行编译问题，需要修改配置"，修改 `parallel_compile` 为 `1`，从步骤 1 重新执行
- **降级方案**：如果无法定位 CCE 文件，尝试手动分析日志

### 步骤 7：二分查找定位问题代码行

**可能错误**：
- 脚本执行失败
- ERROR 无法确定
- 初始范围获取失败
- 二分查找迭代失败
- 未定位到问题代码行
- 出现 `ld.lld: error: undefined` 错误

**错误处理**：
- **脚本执行失败**：输出脚本错误和堆栈，建议检查脚本依赖
- **ERROR 无法确定**：输出分析结果和可能的原因，建议检查 CCE 文件内容
- **初始范围获取失败**：输出文件内容，建议检查 CCE 文件格式
- **二分查找迭代失败**：输出迭代状态和错误，建议检查测试命令稳定性
- **未定位到问题代码行**：输出搜索结果，建议手动检查 CCE 文件
- **ld.lld 错误**：输出 "检测到并行编译问题，需要修改配置"，修改 `parallel_compile` 为 `1`，从步骤 1 重新执行
- **降级方案**：如果二分查找失败，尝试手动注释代码行进行测试

### 步骤 8：映射到前端源代码

**可能错误**：
- program.json 文件未找到
- 映射脚本执行失败
- 映射失败

**错误处理**：
- **program.json 未找到**：输出文件路径，建议检查步骤 5.2 的结果
- **脚本执行失败**：输出脚本错误和堆栈，建议检查脚本依赖
- **映射失败**：输出失败原因（如：框架自动生成代码、操作数不匹配），说明无法映射到前端源代码
- **降级方案**：如果映射失败，提供 CCE 文件和问题行号，建议手动分析

### 步骤 9：输出结果

**可能错误**：
- 信息不完整
- 文件读取失败

**错误处理**：
- **信息不完整**：输出已获取的信息和缺失的信息，说明部分定位成功
- **文件读取失败**：输出文件路径和错误，建议检查文件权限
- **降级方案**：如果无法读取文件，输出文件路径和行号，建议手动查看

### 通用错误处理原则

1. **错误信息清晰**
   - 输出错误发生的步骤和具体原因
   - 输出相关的路径、命令和配置
   - 输出错误堆栈（如果有）

2. **提供恢复建议**
   - 根据错误类型提供具体的修复建议
   - 提供降级方案，说明替代的调试方法
   - 提供相关技能的名称和触发词

3. **避免静默失败**
   - 不得在遇到错误时继续执行后续步骤
   - 不得忽略错误或假设错误已解决
   - 不得伪造成功结果

4. **保持状态一致性**
   - 如果步骤失败，建议回滚相关修改（如配置文件、源码注释）
   - 如果需要重新执行，说明需要从哪个步骤开始
   - 如果技能无法继续，说明已完成的步骤和当前状态

---

## 关键注意事项

1. **fixed 模式**: 确保 `fixed` 模式启用以保持输出路径不变
2. **路径规范**: 所有路径必须使用绝对路径
3. **信息收集**: 第一步必须通过 `question` 工具收集信息，严禁猜测
4. **停止条件**: 遇到不适用的情况或定位失败时，立即停止执行并说明原因
5. **并行编译问题**: 遇到 `ld.lld: error: undefined` 错误时，需要修改 `parallel_compile` 为 `1` 并从头重新执行
6. **段错误处理**: 段错误不是 aicore error，技能不适用。需要使用调试工具（如 gdb、core dump）分析段错误，或检查内存访问相关的代码逻辑
7. **错误类型识别**: 在步骤 5.1 运行测试后，必须识别错误类型，只有确认是 aicore error 才能继续执行后续步骤
