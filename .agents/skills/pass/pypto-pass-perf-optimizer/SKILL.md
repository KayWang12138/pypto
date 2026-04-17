---
name: pypto-pass-perf-optimizer
description: PyPTO Pass 编译性能优化技能。用于分析和优化 Pass 模块的编译性能，当 Pass 编译耗时过长需要优化时使用此技能。
---

# PyPTO Pass 编译性能优化技能

## 概述

本技能用于分析和优化 PyPTO Pass 模块的编译性能，帮助识别性能瓶颈并提供优化建议。

## ⚠️ 执行纪律警告（CRITICAL）

**🛑 严禁跳过任何关键步骤！以下步骤必须严格执行：**

### 必须完成的步骤序列

```
步骤1-5（环境准备+性能测量）
    ↓
步骤6（编译Debug版本） ⚠️ 强制，不可跳过
    ↓
步骤7（perf分析） ⚠️ 强制，不可跳过
    ↓
步骤8（内存分析，可选）
    ↓
步骤9（理解Pass功能）
    ↓
步骤10-13（性能分析） ⚠️ 强制，必须基于perf数据
    ↓
步骤14-18（优化实施）
    ↓
步骤19（UT验证） ⚠️ 强制，不可跳过
    ↓
步骤20-22（验证与迭代）
```

### 禁止行为

❌ **严禁跳过步骤6-7**：Debug编译和perf分析是优化的数据依据
❌ **严禁跳过步骤10-13**：必须基于perf数据进行优化决策，禁止凭猜测优化
❌ **严禁在未完成perf分析前修改代码**：这是盲目优化，可能导致错误的方向
❌ **严禁在UT未通过前进行性能验证**：功能正确性优先于性能优化

### 正确的执行方式

**优化前的准备工作**：
1. ✅ 必须完成步骤6：编译Debug版本（perf需要调试符号）
2. ✅ 必须完成步骤7：使用perf record采集数据
3. ✅ 必须查看perf report，了解热点函数
4. ✅ 必须记录 top 5 热点函数及其 cycles 占比
5. ✅ 必须理解热点函数的来源（查看调用栈）

**只有完成以上准备工作后**，才能进入步骤14进行代码优化。

**优化后的验证工作**：
1. ✅ 必须完成步骤19：运行UT验证功能正确性
2. ✅ UT全部通过后，才能进行性能验证
3. ✅ 如果UT失败，必须回退修改或重新优化

### 常见错误示例

**错误 1：跳过perf分析直接优化**
```
步骤1-5 → ❌跳过步骤6-7 → 步骤14修改代码
后果：缺乏数据支撑，优化方向可能错误
```

**错误 2：仅凭代码审查进行优化**
```
步骤1-5 → ❌跳过步骤6-13 → 步骤14修改代码
后果：无法识别真正的性能瓶颈，浪费时间
```

**错误 3：UT未通过就进行性能验证**
```
步骤14修改代码 → ❌跳过步骤19 → 步骤20性能验证
后果：可能引入功能bug，性能优化无效
```

### 关键提醒

**为什么要使用perf分析？**
1. **精确定位热点**：只凭代码审查无法确定实际的热点函数
2. **量化分析**：需要cycles、cache-misses等量化指标
3. **调用栈分析**：perf -g 可以显示完整调用栈，找出真正的性能瓶颈
4. **避免盲目优化**：基于实际数据而非猜测进行优化

**为什么要编译Debug版本？**
1. Debug版本包含完整的调试符号
2. perf需要调试符号才能正确显示函数名
3. Release版本优化后可能丢失符号信息

**为什么必须运行UT？**
1. 性能优化可能引入功能bug
2. 必须确保功能正确性优先
3. 性能提升不能以牺牲功能为代价

## 核心目标

### 性能目标定义

**基准目标**：
- 200,000 Op 场景下，单个 Pass 平均耗时不超过 20s

**关键术语定义**：
- **Op数量**：单个Function中的Operation总数（从expand_function Pass日志中获取）
- **平均耗时**：Pass在所有Function上执行的平均时间（总耗时 / 执行次数）
- **线性关系**：Op数量增加或减少时，目标时间线性增加或减少

**目标时间计算公式**：
```
目标平均耗时 = (实际Op数量 / 200,000) × 20s
```

**示例**：
- 100,000 Op → 目标平均耗时 = (100,000 / 200,000) × 20s = 10s
- 200,000 Op → 目标平均耗时 = (200,000 / 200,000) × 20s = 20s
- 300,000 Op → 目标平均耗时 = (300,000 / 200,000) × 20s = 30s

**性能判断标准**：
- ✅ **达标**：实际平均耗时 ≤ 目标平均耗时
- ⚠️ **未达标**：实际平均耗时 > 目标平均耗时
- 💡 **优化方向**：降低平均耗时，使其不超过目标值

## 触发机制

当用户提到以下关键词时触发:
- "Pass 编译性能优化"
- "Pass 耗时优化"
- "Pass 性能分析"
- "优化 XXX Pass 模块"

## 使用场景

- 分析 Pass 编译耗时
- 识别性能瓶颈
- 优化 Pass 编译性能
- 验证优化效果

## 固定步骤

### 阶段一：环境准备

#### 步骤 1：用户指定算子脚本

用户需要指定要执行的算子脚本（用于触发 Pass 编译流程）：
- 脚本路径
- 脚本参数（如有）

#### 步骤 2：设置日志环境

```bash
# 开启 info 级别日志
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置日志落盘路径（默认为算子脚本同目录下的 logs 文件夹）
export ASCEND_PROCESS_LOG_PATH=$(dirname {user_specified_script})/logs

# 创建日志目录
mkdir -p $ASCEND_PROCESS_LOG_PATH

# 日志文件将落盘到：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-{pid}-{timestamp}.log
# 注意：单个日志文件超过 20M 会自动拆分为多个文件
# 拆分文件命名：每个文件都有独立的时间戳（pypto-log-{pid}-{timestamp}.log）
# 同一次执行的所有拆分文件具有相同的 pid，可通过 pid 识别
```

### 阶段二：性能测量

#### 步骤 3：编译 Release 版本并安装

**⚠️ 重要：确保使用最新编译的 Release 版本进行性能测试**

```bash
# 编译 Python 包（Release 版本，用于性能测试）
python3 build_ci.py -f=python3 --build_type Release --disable_auto_execute

# 安装到 Python 环境
pip install ./build_out/pypto-*.whl --force-reinstall

```

#### 步骤 4：运行算子脚本采集 Pass 耗时

```bash
# 执行用户指定的算子脚本（日志自动落盘）
# 使用超时控制，默认 5 分钟，超时后自动中断并继续后续步骤
bash scripts/run_with_timeout.sh 300 python3 {user_specified_script}.py

# 自定义超时时间（例如 10 分钟）
# bash scripts/run_with_timeout.sh 600 python3 {user_specified_script}.py

# 不使用超时控制直接执行（不推荐，可能长时间等待）
# python3 {user_specified_script}.py

# 日志文件位置：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log
# 注意：单个日志文件超过 20M 会自动拆分为多个文件
# 同一次执行的所有文件具有相同的 pid
```

#### 步骤 5：运行 Python 脚本分析性能

```bash
# 解析日志文件，生成 Pass 耗时排序报告
# 日志文件位于：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log
# 注意：脚本会自动处理拆分的日志文件（同 pid 的所有文件）

# 查找最新的日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "使用日志文件: $latest_log"

python3 scripts/parse_pass_perf.py -l $latest_log

```

#### 步骤 6：编译 Debug 版本

**⚠️ 重要：perf 采样需要 Debug 版本才能正确显示函数名和调用栈**

```bash
# 编译 Python 包（Debug 版本带完整调试符号，用于 perf 采样）
python3 build_ci.py -f=python3 --build_type Debug

# 安装到 Python 环境
pip install ./build_out/pypto-*.whl --force-reinstall
```

#### 步骤 7：使用 perf 分析热点

**⚠️ 重要：perf 能够正确分析当前代码的性能优化点**

##### 选项 A：火焰图方式（推荐）

火焰图提供了直观的可视化视图，更容易识别性能热点。

**生成火焰图：**

```bash
# 生成火焰图（默认 5 分钟超时）
# 参数：超时时间(秒) 输出目录 命令...
bash scripts/generate_flamegraph.sh \
    300 ./flamegraphs python3 {user_specified_script}.py

# 火焰图将保存到 ./flamegraphs/flamegraph_{timestamp}.svg
# 同时生成折叠数据文件 folded_{timestamp}.txt（用于后续对比）

# 使用浏览器打开查看
firefox ./flamegraphs/flamegraph_*.svg
# 或
google-chrome ./flamegraphs/flamegraph_*.svg
```

**🔥 火焰图解读指南：**

**1. 结构说明**
- **X轴**：表示函数调用栈的样本占比（宽度越大，占用 CPU 越多）
- **Y轴**：表示调用栈深度（从下往上，底部是入口，顶部是叶子函数）
- **颜色**：随机分配，仅用于区分不同函数

**2. 识别热点函数**

| 图形特征 | 含义 | 行动建议 |
|---------|------|---------|
| **宽平台** | 顶部宽大的函数块 | 这是性能热点，优先优化 |
| **高塔尖** | 调用栈很深 | 检查是否存在过度封装或递归 |
| **反复出现** | 同一函数多处出现 | 函数被频繁调用，考虑缓存或批处理 |
| **细长条** | 调用栈很窄但很深 | 可能是单线程瓶颈 |

**3. 交互操作**（浏览器中）
- **鼠标悬停**：显示函数名和样本占比百分比
- **点击函数**：放大查看该函数的调用栈细节
- **搜索功能**：按 `Ctrl+F` 搜索特定函数名
- **重置视图**：点击空白处或刷新页面

**4. 常见热点模式及优化方向**

| 热点函数 | 问题类型 | 优化方向 |
|---------|---------|---------|
| `_malloc` / `_free` | 频繁内存分配 | 预分配内存、使用对象池 |
| `std::unordered_map::find` | 哈希表查找 | 优化哈希函数、预分配bucket |
| `std::vector` 扩容 | 动态扩容 | 使用 `reserve()` 预分配 |
| `memcpy` / `memmove` | 大量数据拷贝 | 减少拷贝、使用引用 |
| 字符串操作 | 字符串拼接/转换 | 使用 `std::string_view`、预分配 |
| 循环内函数调用 | 过度循环 | 循环展开、提前计算 |

**5. 火焰图示例解读**

```
                    ┌─────────────────────┐
                    │  hot_function() 15% │  ← 宽平台：性能热点
                    └─────────────────────┘
               ┌────────────────────────────────┐
               │     process_data() 35%         │  ← 宽块：主要耗时函数
               └────────────────────────────────┘
          ┌──────────────────────────────────────────┐
          │          main_loop() 50%                 │  ← 入口函数
          └──────────────────────────────────────────┘
```

- `hot_function()` 占 15% CPU，宽度较宽，是优化重点
- `process_data()` 占 35%，是主要耗时函数
- 点击 `hot_function()` 可以查看其内部调用栈

##### 选项 B：传统 perf report

如果火焰图工具不可用，可以使用传统方式：

```bash
# 使用 perf 采样
bash scripts/run_with_timeout.sh 300 \
    perf record -g -e cycles,instructions,cache-misses -- python3 {user_specified_script}.py

# 查看报告
perf report
```

**perf report 快捷键：**
- `+`：展开调用栈
- `-`：折叠调用栈
- `Enter`：进入函数详情
- `/`：搜索函数名

##### 分析输出要求

完成火焰图分析后，记录以下信息：

```markdown

---

## Detailed Procedures and Script Reference

Read `references/optimization-procedures.md` for:
- Performance analysis recording templates
- Python script usage (parse_pass_perf.py, flamegraph scripts)
- Timeout configuration
- Functional correctness safeguards
- Common commands reference
