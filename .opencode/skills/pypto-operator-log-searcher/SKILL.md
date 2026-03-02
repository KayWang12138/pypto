---
name: pypto-operator-log-searcher
description: PyPTO 算子日志关键字检索和分析工具。当用户需要在 PyPTO 算子开发过程中搜索日志文件中的关键字、分析日志模式、提取错误信息或统计关键字出现次数时使用。支持：(1) 单关键字搜索，(2) 多关键字组合搜索，(3) 正则表达式匹配，(4) 上下文信息提取，(5) 统计分析，(6) 错误追踪，(7) 性能分析
---

# PyPTO 算子日志检索器

此技能提供高效的 PyPTO 算子日志文件搜索和分析能力。

## 核心原则

### 高效搜索优先

使用 `rg` (ripgrep) 而非 `grep`，性能更优：
- 递归搜索更快
- 正则表达式支持更完善
- 输出格式更灵活

### 渐进式搜索策略

1. **精确匹配优先**：先尝试精确匹配关键字
2. **模糊匹配**：无结果时使用正则表达式
3. **上下文扩展**：提取匹配行的前后行
4. **统计分析**：统计出现次数和分布

## 使用流程

### 步骤 1：确认搜索目标

明确用户需求：
- **关键字**：要搜索的具体字符串
- **日志位置**：日志文件路径或目录
- **搜索模式**：精确匹配、模糊匹配或正则表达式
- **输出需求**：仅匹配行、包含上下文、统计信息

### 步骤 2：选择搜索方式

#### 方式一：使用脚本（推荐）

使用技能中的日志搜索脚本：

```bash
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py <keyword> <log_path>
```

**常用参数：**
- `--keyword` 或 `-k`：搜索关键字（必需）
- `--path` 或 `-p`：日志文件或目录路径（必需）
- `--regex` 或 `-r`：使用正则表达式匹配
- `--context` 或 `-C`：显示前后行数（默认 0）
- `--count` 或 `-c`：仅统计出现次数
- `--ignore-case` 或 `-i`：忽略大小写
- `--output` 或 `-o`：输出结果到文件

**示例：**

```bash
# 基础搜索
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "ERROR" -p output/

# 带上下文
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "ERROR" -p output/ -C 2

# 正则表达式
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "ERROR.*timeout" -p output/ -r

# 统计次数
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "ERROR" -p output/ -c

# 多关键字（使用脚本多次调用）
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "ERROR" -p output/
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "WARN" -p output/
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "FAIL" -p output/
```

#### 方式二：直接使用 ripgrep

```bash
# 基础搜索
rg "ERROR" logs/

# 带行号和文件名
rg -n "ERROR" logs/

# 包含上下文（前后各 2 行）
rg -C 2 "ERROR" logs/

# 正则表达式匹配
rg -n "ERROR.*timeout" logs/

# 多关键字搜索（OR 逻辑）
rg -n "(ERROR|WARN|FAIL)" logs/

# 统计出现次数
rg -c "ERROR" logs/

# 按文件统计
rg -c "ERROR" logs/ | sort -t: -k2 -rn
```

### 步骤 3：高级分析

#### 时间范围过滤

```bash
rg "2024-01-15.*ERROR" logs/
```

#### 提取特定字段

```bash
rg -o "ERROR.*?\[" logs/
```

#### 按严重级别分类

```bash
rg -n "ERROR" logs/ > errors.log
rg -n "WARN" logs/ > warnings.log
rg -n "INFO" logs/ > info.log
```

## PyPTO 算子日志常见搜索模式

### 编译错误追踪

```bash
# 搜索编译错误
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "error|Error|ERROR" -p output/ -C 3

# 搜索特定编译错误
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "CompilationError|SyntaxError" -p output/ -C 3
```

### 运行时错误追踪

```bash
# 搜索运行时错误
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "RuntimeError|Exception|Traceback" -p output/ -C 5

# 搜索 NPU 相关错误
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "NPU|AICPU|AICORE" -p output/ -C 3
```

### 性能分析

```bash
# 搜索性能指标
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "latency|duration|response_time|execution_time" -p output/

# 搜索慢查询或超时
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "timeout|slow|latency" -p output/ -C 2
```

### 调试信息

```bash
# 搜索调试日志
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "DEBUG|TRACE|verbose" -p output/

# 搜索特定模块或函数
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "jit|compile|assemble" -p output/
```

### 精度验证

```bash
# 搜索精度相关日志
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "precision|accuracy|diff|tolerance" -p output/ -C 2

# 搜索测试结果
python3 .opencode/skills/pypto-operator-log-searcher/scripts/log_searcher.py -k "test|assert|pass|fail" -p output/
```

## 输出建议

根据搜索结果，提供：
- **匹配行数**：总匹配数量
- **文件分布**：哪些文件包含匹配
- **时间分布**：日志时间范围（如可识别）
- **上下文摘要**：关键匹配行的前后信息
- **建议操作**：基于日志内容的下一步建议

## 注意事项

- 大文件搜索时，考虑使用 `--max-count` 限制结果数量
- 使用 `-i` 标志进行不区分大小写的搜索
- 对于压缩日志（.gz），先解压或使用 `zgrep`
- 搜索结果过多时，建议分批处理或添加更多过滤条件
- PyPTO 算子日志通常在 `output/output_*/` 目录下
