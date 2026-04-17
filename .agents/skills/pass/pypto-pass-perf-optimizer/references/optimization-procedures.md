## 性能分析记录

### 热点函数 Top 5（从火焰图识别）
| 排名 | 函数名 | CPU占比 | 所属模块 | 可能原因 |
|-----|--------|---------|---------|---------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### 性能瓶颈类型
- [ ] 计算密集型
- [ ] 内存密集型
- [ ] IO密集型

### 优化方向
1. ...
2. ...
```

#### 步骤 8：内存分析（可选）

```bash
# 使用超时控制执行内存分析，默认 5 分钟
bash scripts/run_with_timeout.sh 300 \
    valgrind --tool=massif -- python3 {user_specified_script}.py

massif-visualizer massif.out.*
```

### 阶段三：理解 Pass 功能

**⚠️ 重要：优化前必须充分理解 Pass 的功能和业务逻辑**

#### 步骤 9：使用 pypto-pass-module-analyzer 分析 Pass

调用 `pypto-pass-module-analyzer` skill 来深入理解目标 Pass：

```
分析 Pass 模块 {PassName} 的功能和业务逻辑
```

**需要了解的内容：**
- Pass 的整体功能是什么
- Pass 的主要处理流程
- Pass 的核心数据结构
- Pass 的关键算法
- Pass 的输入输出

**理解 Pass 功能的好处：**
- 知道哪些数据结构可以优化
- 知道哪些算法可以改进
- 知道哪些步骤是性能瓶颈
- 避免优化破坏功能正确性

### 阶段四：性能分析

#### 步骤 10：分析热点函数

从 perf report 中识别：
- CPU 密集函数（cycles 占比高）
- Cache miss 严重的函数
- 内存分配热点

#### 步骤 11：分析数据结构

检查以下问题：
- 容器选择是否合理（vector/set/unordered_map/map）
- 是否有频繁的内存分配/释放
- 是否有不必要的拷贝操作
- 是否有可以预分配的容器

#### 步骤 12：分析算法复杂度

- 评估时间复杂度（O(n)/O(n²)/O(n³)/O(n log n)）
- 检查是否有重复计算
- 分析循环嵌套层数
- 检查是否有可以提前终止的循环

#### 步骤 13：识别性能瓶颈类型

| 瓶颈类型 | 症状描述 | 典型原因 |
|---------|---------|---------|
| **计算密集型** | CPU 利用率高，cycles 占比高 | 算法复杂度过高、重复计算 |
| **内存密集型** | cache miss 多，内存访问频繁 | 数据结构不合理、缓存不友好 |
| **IO 密集型** | 文件读写、日志输出多 | 过多的日志、文件操作 |

### 阶段五：优化实施

#### 步骤 14：修改代码

根据性能分析结果修改 Pass 代码，实施优化。

#### 步骤 15：添加关键步骤耗时统计

在优化过程中，为 Pass 内部关键步骤添加耗时统计：

```cpp
#include <chrono>

// 在 Pass 的 RunOnFunction 或关键函数中添加
auto stepStart = std::chrono::high_resolution_clock::now();

// ... 关键步骤代码 ...

auto stepEnd = std::chrono::high_resolution_clock::now();
auto stepDuration = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
PASS_LOGI("[{PassName}] Step {step_name} cost %ld us", stepDuration.count());
```

**建议添加耗时统计的场景：**
- 主要循环遍历
- 图遍历/搜索
- 数据结构构建
- 排序/查找操作
- 内存密集操作

#### 步骤 16：算法优化

**常见优化方法：**
- 降低时间复杂度
- 减少不必要的遍历
- 利用索引/哈希加速查找
- 提前终止减少循环次数

**优化示例：**
```cpp
// 优化前：O(n²) 查找
for (auto& op1 : operations) {
    for (auto& op2 : operations) {
        if (op1.id == op2.parentId) { ... }
    }
}

// 优化后：O(n) 使用哈希表
std::unordered_map<int, Operation*> idToOp;
for (auto& op : operations) {
    idToOp[op.id] = &op;
}
for (auto& op : operations) {
    auto it = idToOp.find(op.parentId);
    if (it != idToOp.end()) { ... }
}
```

#### 步骤 17：数据结构优化

**容器选择指南：**
| 场景 | 推荐容器 | 原因 |
|-----|---------|-----|
| 顺序访问 | `std::vector` | 连续内存，缓存友好 |
| 频繁查找 | `std::unordered_map` | O(1) 查找 |
| 需要排序 | `std::set` / `std::map` | 自动排序 |
| 频繁插入删除 | `std::list` | O(1) 插入删除 |

**内存优化：**
```cpp
// 预分配内存
std::vector<Operation*> ops;
ops.reserve(expected_size);  // 避免多次重新分配

// 使用引用避免拷贝
for (const auto& op : operations) { ... }  // 而不是 for (auto op : ...)

// 使用 emplace_back 避免临时对象
ops.emplace_back(newOp);  // 而不是 ops.push_back(newOp)

// 使用 std::move 转移所有权
auto result = std::move(tempVector);
```

#### 步骤 18：缓存优化

```cpp
// 改善数据局部性
struct OpInfo {
    int id;
    int parentId;
    Operation* op;
};
std::vector<OpInfo> opInfos;  // 连续内存，缓存友好

// 避免"指针追逐"
// 优化前：多层指针
for (auto& op : ops) {
    for (auto& consumer : op->consumers) {
        for (auto& child : consumer->children) { ... }
    }
}

// 优化后：预计算索引
std::vector<std::vector<int>> opToChildren;  // 直接索引访问
```

### 阶段六：验证与迭代

#### 步骤 19：运行 UT 确保功能正确（优化后）

**⚠️ 重要：优化后必须验证目标 Pass 的 UT 通过**

```bash
# 查找 Pass 相关 UT 测试文件
# 路径：framework/tests/ut/passes/src/test_{pass_name}.cpp

# 运行指定 Pass 的所有 UT
python3 build_ci.py -f=cpp -u="{PassName}Test.*" -j=24

# 示例：运行 AssignMemoryType 的 UT
python3 build_ci.py -f=cpp -u="AssignMemoryTypeTest.*" -j=24

# 运行单个测试用例
python3 build_ci.py -f=cpp -u="AssignMemoryTypeTest.AddReshape" -j=24

# 如果 UT 失败：
# 1. 检查优化代码是否引入 bug
# 2. 必要时回退修改
# 3. 重新优化
```

**注意：UT 测试是 C++ 测试，必须使用 `-f=cpp` 参数**

#### 步骤 20：编译和安装（Release 版本）

**⚠️ 重要：最终性能验证必须使用 Release 版本，确保性能数据准确**

```bash
# 编译 Python 包（Release 版本，用于最终性能验证）
python3 build_ci.py -f=python3 --build_type Release

# 安装到 Python 环境
pip install ./build_out/pypto-*.whl --force-reinstall
```

#### 步骤 21：对比优化效果

**⚠️ 重要：优化后需要对比验证优化效果**

##### 21.1 重新采集日志数据

```bash
# 重新采集性能数据（日志自动落盘）
# 使用超时控制，默认 5 分钟
bash scripts/run_with_timeout.sh 300 python3 {user_specified_script}.py

# 日志文件位置：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log
# 注意：每次运行会生成新的日志文件，可通过时间戳区分

# 查找最新的日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "最新日志文件: $latest_log"

# 运行 Python 脚本对比
python3 scripts/parse_pass_perf.py \
    -l $latest_log
```

##### 21.2 生成优化后火焰图（可选，推荐）

```bash
# 生成优化后的火焰图和折叠数据
bash scripts/generate_flamegraph.sh \
    300 ./flamegraphs python3 {user_specified_script}.py

# 这会生成新的文件，例如：
#   - folded_20260313_143022.txt (优化前，步骤7生成)
#   - folded_20260313_150335.txt (优化后，刚刚生成)
```

##### 21.3 对比火焰图验证优化效果

```bash
# 对比优化前后的火焰图
bash scripts/compare_flamegraphs.sh \
    ./flamegraphs/folded_20260313_143022.txt \
    ./flamegraphs/folded_20260313_150335.txt

# 差异火焰图将保存到 ./flamegraphs/diff_flamegraph_{timestamp}.svg
```

**差异火焰图颜色说明：**

| 颜色 | 含义 | 解读 |
|------|------|------|
| 🔴 **红色/橙色** | 优化后增加的热点 | ⚠️ 需要关注，可能是新引入的性能问题 |
| 🔵 **蓝色/青色** | 优化后减少的热点 | ✅ 优化有效，这些函数耗时减少 |
| ⚪ **灰色** | 基本不变 | 优化对该函数影响不大 |

**对比分析要点：**
1. **关注红色区域**：优化后新出现或增加的热点
2. **确认蓝色区域**：验证优化是否对目标函数有效
3. **量化对比**：记录主要函数优化前后的占比变化

#### 步骤 22：迭代优化

如果性能未达标：

**⚠️ 重要：重新生成火焰图进行性能瓶颈分析，不要盲目优化**
重复步骤 6-21 直到达标

## Python 脚本使用说明

### 脚本位置

```
scripts/parse_pass_perf.py
```

### 使用方法

```bash
# 基本用法：分析单个日志文件
# 日志文件位置：$ASCEND_PROCESS_LOG_PATH/debug/plog/pypto-log-*.log

# 查找最新日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python3 parse_pass_perf.py -l $latest_log

# 或者直接指定日志文件路径
python3 parse_pass_perf.py -l /path/to/pypto-log-*.log

# 对比两个日志文件
python3 parse_pass_perf.py -l pypto-log-after.log --compare pypto-log-before.log

# 指定 Op 数量阈值（默认 200000）
python3 parse_pass_perf.py -l pypto-log-*.log --ops-threshold 200000

# 指定时间阈值（默认 20s）
python3 parse_pass_perf.py -l pypto-log-*.log --time-threshold 20
```

## 超时控制配置

### 为什么需要超时控制？

某些算子在运行阶段可能执行很长时间（超过 1 小时），但对于 Pass 编译性能分析而言：
- ✅ Pass 编译信息在编译完成后就已保存在日志中
- ✅ 运行阶段的长时间执行对 Pass 性能分析没有帮助
- ✅ 超时中断不会影响 Pass 性能分析结果

### 默认配置

- **默认超时时间**: 300 秒（5 分钟）
- **中断信号**: SIGINT (Ctrl+C)
- **中断后行为**: 继续执行后续分析步骤

### 如何自定义超时时间

```bash
# 设置超时时间为 10 分钟（600 秒）
bash scripts/run_with_timeout.sh 600 python3 test.py

# 设置超时时间为 15 分钟（900 秒）
bash scripts/run_with_timeout.sh 900 python3 test.py

# 设置超时时间为 2 分钟（120 秒）
bash scripts/run_with_timeout.sh 120 python3 test.py
```

### 适用场景

超时控制适用于所有需要执行算子的步骤：
1. ✅ 步骤 4：运行算子脚本采集 Pass 耗时
2. ✅ 步骤 7：使用 perf 分析热点
3. ✅ 步骤 8：内存分析
4. ✅ 步骤 21：对比优化效果

### 中断后继续执行的原因

当算子执行超时被中断后，脚本会返回退出码 0，允许继续执行后续步骤：
- ✅ Pass 编译日志已保存，可以进行分析
- ✅ 可以继续 perf 分析、内存分析等
- ✅ 可以对比优化前后的性能数据

如果算子执行因其他原因失败（非超时中断），脚本会返回非零退出码，停止后续执行。

### 超时控制脚本说明

**脚本位置**: `scripts/run_with_timeout.sh`

**参数说明**:
```bash
bash run_with_timeout.sh <超时秒数> <命令> [命令参数...]
```

**退出码**:
- `0` - 成功完成或超时中断（允许继续后续步骤）
- `非 0` - 执行失败（停止后续步骤）

**技术细节**:
- 使用 SIGINT (Ctrl+C) 信号，允许进程优雅退出
- 可以正确关闭文件、释放资源
- Python 程序可以捕获并处理该信号

## 功能正确性保障

### 优化后验证

**⚠️ 重要：优化后必须验证目标 Pass 的 UT 通过**

```bash
# 运行目标 Pass 的 UT
python3 build_ci.py -f=cpp -u=Test{PassName}.* -j=24

# 如果 UT 失败：
#    - 检查优化代码是否引入 bug
#    - 使用 git diff 查看修改
#    - 必要时回退修改
#    - 重新优化

# 只有 UT 全部通过才能接受优化
```

## 参考资料

| 类别 | 文件路径 |
|------|----------|
| Pass 日志机制 | framework/src/passes/pass_mgr/pass_manager.cpp |
| 耗时统计函数 | LogPassRuntime() |
| Op 数量日志 | expand_function.cpp 中的 Operations().size() |
| 日志配置 | ASCEND_MODULE_LOG_LEVEL=PASS=1 |
| perf 使用 | perf record -g, perf report |
| valgrind 使用 | valgrind --tool=massif |

## 常用命令

### 查找最新的日志文件

```bash
# 方法 1：使用 find 命令查找最新日志文件
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "最新日志文件: $latest_log"

# 方法 2：使用 ls 按时间排序查找
latest_log=$(find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | xargs ls -t | head -1)
echo "最新日志文件: $latest_log"

# 直接使用最新日志文件进行分析
python3 scripts/parse_pass_perf.py -l $latest_log
```

### 处理拆分的日志文件

```bash
# 日志文件拆分规则：单个文件超过 20M 自动拆分
# 拆分文件命名：
#   - pypto-log-{pid}-{timestamp1}.log
#   - pypto-log-{pid}-{timestamp2}.log
#   - pypto-log-{pid}-{timestamp3}.log
#   - ...
# 同一次执行的所有文件具有相同的 pid

# 查看同一次执行产生的所有拆分日志文件
# 方法 1：手动提取 pid 查找
log_file="pypto-log-1051473-20260312202107387.log"
pid=$(echo $log_file | sed 's/pypto-log-\([0-9]*\)-.*/\1/')
ls -lh pypto-log-${pid}-*.log

# 方法 2：自动分析（推荐）
# parse_pass_perf.py 会自动检测并处理所有拆分文件
python3 parse_pass_perf.py -l pypto-log-1051473-20260312202107387.log
```

### 查看所有日志文件

```bash
# 查看所有日志文件（按时间排序）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | xargs ls -lht

# 查看所有日志文件（按大小排序）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f -exec ls -lh {} \; | sort -k5 -h

# 列出所有拆分的日志文件组（按 pid 分组）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | xargs ls -t | head -20
```

### 查找特定进程或时间的日志

```bash
# 查找特定进程ID的所有日志文件
pid="1051473"
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-${pid}-*.log" -type f

# 查找特定时间段的所有日志文件（2026年3月12日 20:21）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*-202603122021*.log" -type f

# 查找特定日期的日志文件（2026年3月12日）
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*-20260312*.log" -type f

# 统计每次执行产生的日志文件数量
find $ASCEND_PROCESS_LOG_PATH/debug/plog -name "pypto-log-*.log" -type f | \
    sed 's/pypto-log-\([0-9]*\)-.*/\1/' | sort | uniq -c
```
