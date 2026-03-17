# 常见问题

### Q1: 日志文件很大，如何处理？

**A**: 使用流式解析，逐行读取，避免一次性加载整个文件：
```python
from pypto_pass_error_analyzer import parse_and_analyze_stream

with open("large.log", "r") as f:
    for error_analysis in parse_and_analyze_stream(f):
        process_error(error_analysis)
```

### Q2: 如何过滤特定 pass 模块的错误？

**A**: 解析后过滤：
```python
result = parse_and_analyze_log("error.log")
pass_errors = [e for e in result["errors"] if e["pass_module"] == "ConstantFolding"]
```

### Q3: 堆栈跟踪不完整怎么办？

**A**: 检查日志级别设置，确保 DEBUG 或 INFO 级别日志被记录。

### Q4: 如何识别动态 shape 相关的错误？

**A**: 检查错误消息中是否包含动态 shape 的关键词：
- `dyn_shape`
- `symbolic`
- `unknown dimension`
- `variable shape`

### Q5: 如何区分是用户代码错误还是框架错误？

**A**: 检查堆栈跟踪：
- 用户代码：堆栈中包含 `custom/` 或用户定义的文件
- 框架代码：堆栈中包含 `framework/` 或 `pypto/`

### Q6: 如何获取更详细的错误信息？

**A**:
1. 设置日志级别为 DEBUG：`export ASCEND_GLOBAL_LOG_LEVEL=0`
2. 启用详细堆栈跟踪
3. 使用 `--verbose` 标志运行
4. 检查完整的日志文件

### Q7: 如何修改日志输出目录？

**A**:
```bash
# 方式 1：使用 ASCEND_PROCESS_LOG_PATH（优先级最高）
export ASCEND_PROCESS_LOG_PATH=/custom/log/path

# 方式 2：使用 ASCEND_WORK_PATH（优先级次之）
export ASCEND_WORK_PATH=/custom/work/path

# 方式 3：使用默认目录 $HOME/ascend/log
```

### Q8: 如何设置日志级别？

**A**:
```bash
# DEBUG 级别（最详细）
export ASCEND_GLOBAL_LOG_LEVEL=0

# INFO 级别
export ASCEND_GLOBAL_LOG_LEVEL=1

# WARNING 级别
export ASCEND_GLOBAL_LOG_LEVEL=2

# ERROR 级别（仅错误）
export ASCEND_GLOBAL_LOG_LEVEL=3
```

### 可自动修复的判断标准

满足**全部**条件时可自动修复：

1. **修改意图明确** — 清楚描述了需要什么改动
2. **范围可界定** — 能确定影响哪些文件和代码位置
3. **操作确定性强** — 修改方案唯一或选项有限
4. **无业务判断** — 不涉及设计决策、架构选择、业务逻辑权衡

### 错误严重程度评估标准

- **CRITICAL**: 导致程序崩溃、无法继续执行
- **HIGH**: 影响核心功能、结果错误
- **MEDIUM**: 影响性能、功能受限
- **LOW**: 警告信息、可忽略