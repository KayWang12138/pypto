---
name: pypto-host-stacktrace-analyzer
description: 分析 host 侧捕获异常后的堆栈信息，通过地址到源码行映射和符号解析，定位问题代码位置。支持 Python traceback、C++ stack trace 和混合堆栈的自动识别与分析。支持编译Debug版本PyPTO包并定位具体代码行。Triggers："堆栈分析"、"堆栈反汇编"、"分析堆栈信息"、"地址到源码行"、"stack trace"、"backtrace"
---

# 堆栈信息分析器

此技能用于分析 host 侧捕获异常后的堆栈信息，通过反汇编和符号解析，定位问题代码位置。

## 工作流程

1. 收集堆栈信息
2. 检查是否有调试符号
3. 如果没有调试符号，编译Debug版本PyPTO
4. 使用综合分析器自动分析
5. 输出完整报告（包含源码行号）

---

## 步骤 1：收集堆栈信息

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **stack_text**: 堆栈信息文本（直接粘贴或文件路径）
- **stack_file**: 堆栈信息文件路径（可选，如果提供了文本则不需要）
- **stack_source**: 堆栈来源的测试用例文件路径（可选，如果编译Debug版本后需要重新运行）

**⚠️ 重要**：如果 `stack_source` 是测试用例文件，编译Debug版本后需要重新运行该用例以获取新的堆栈信息。

将收集的路径全部转换成绝对路径，收集到所有信息后才能继续后续步骤。

---

## 步骤 2：检查调试符号

运行调试符号检查脚本：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/locate_source_code.py --check-debug
```

**输出**：
- ✓ 二进制文件包含调试符号
- ✗ 二进制文件不包含调试符号

---

## 步骤 3：编译Debug版本PyPTO（如果需要）

如果步骤2显示没有调试符号，则需要编译Debug版本：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/build_debug_pypto.py
```

**编译选项**：
- `-p, --pypto-root`: PyPTO项目根目录（默认：当前目录）
- `-t, --timeout`: 编译超时时间（秒，默认：600）
- `--skip-check`: 跳过前提条件检查

**编译完成后会自动**：
- 找到编译生成的wheel文件
- 显示编译信息

---

## 步骤 4：安装Debug版本

编译完成后，安装Debug版本：

```bash
pip install build_out/pypto*.whl --force-reinstall --no-deps
```

---

## 步骤 5：重新运行测试用例并收集新堆栈信息

**⚠️ 重要：如果用户在步骤1中提供了 `stack_source`（测试用例文件路径），则需要执行此步骤**

如果步骤1中用户提供了测试用例文件路径，则需要重新运行该用例以获取新的堆栈信息：

```bash
# 假设用户提供的是测试用例文件 /path/to/test_case.py
python3 /path/to/test_case.py --run_mode sim 2>&1 | tee /tmp/new_stack_trace.log
```

**然后从新输出的日志中提取堆栈信息**：

```bash
# 提取 C++ stack trace
grep -A 50 "libtile_fwk_interface.so" /tmp/new_stack_trace.log > /tmp/new_cpp_stack.txt

# 或提取 Python traceback
grep -A 100 "Traceback" /tmp/new_stack_trace.log > /tmp/new_python_stack.txt

# 或提取完整堆栈（包含错误信息）
grep -A 100 "Run pass failed" /tmp/new_stack_trace.log > /tmp/new_stack_trace.txt
```

**⚠️ 注意事项**：
- 使用与原始堆栈信息相同的运行参数（如 `--run_mode sim`）
- 将新堆栈信息保存到文件中，供后续分析使用
- 确保新堆栈信息包含完整的错误信息

---

## 步骤 6：使用综合分析器自动分析

运行综合分析脚本，自动完成所有分析步骤：

```bash
python3 .agents/skills/pypto-stack-trace-analyzer/scripts/comprehensive_analyzer.py <stack_file> -f
```

**综合分析器会自动完成以下步骤**：

1. **提取错误信息**
   - 错误码 (Errcode)
   - 错误位置 (file, line, func)
   - 错误消息

2. **解析 Python traceback**
   - 自动识别 Python traceback 格式
   - 提取文件、行号、函数名
   - 标记错误触发点

3. **解析 C++ stack trace**
   - 自动识别 C++ stack trace 格式
   - 支持 `libtile.so(function+offset) [address]` 格式
   - 支持 `#0 0xaddress in function at file.c:line` 格式
   - 标记错误发生点

4. **自动查找二进制文件**
   - 从堆栈信息中提取二进制文件名
   - 在多个路径中搜索：
     - 当前目录
     - PATH 环境变量
     - PyPTO 安装路径
     - 常见库路径 (/usr/lib, /usr/local/lib)
   - 验证二进制文件有效性

5. **符号反混淆**
   - 使用 `c++filt` 工具反混淆 C++ 符号
   - 显示原始符号和反混淆后的符号

6. **源码行定位** 
   - 使用 `addr2line` 工具定位地址到源码行
   - 显示函数名、文件名、行号
   - 仅在Debug版本中可用

---

## 步骤 6：输出完整报告

综合分析器会自动生成包含以下内容的完整报告：

### 错误信息
- 错误码
- 错误位置
- 错误消息

### Python Traceback
- 总帧数
- 每帧的详细信息（文件、行号、函数）
- 错误触发点标记

### C++ Stack Trace
- 总帧数
- 每帧的详细信息（二进制、符号、偏移、地址）
- 符号反混淆结果
- **源码行号** 
- 错误发生点标记

### 二进制文件
- 找到的二进制文件路径

---