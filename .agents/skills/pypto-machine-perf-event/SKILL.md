---
name: pypto-machine-perf-event
description: PyPTO Machine模块性能打点日志添加技能。分析代码文件中耗时较长的函数，自动添加性能打点日志（PerfMtBegin/PerfMtEnd），配置事件定义和日志开关，并编译验证。适用于需要分析PyPTO Machine模块性能瓶颈的场景。
---
# Machine 模块性能打点日志添加

此技能用于为 PyPTO Machine 模块代码添加性能打点日志，帮助分析代码执行耗时。

## 使用场景

当需要分析 PyPTO Machine 模块性能瓶颈时使用此技能：

- 分析特定代码文件的函数执行耗时
- 为耗时函数添加性能打点日志
- 配置性能事件定义和日志开关
- 编译验证打点日志是否生效

## 前置条件

1. **环境要求**
   - PyPTO 开发环境已正确配置
   - 可访问 PyPTO 源代码目录
   - 具备编译和运行能力

2. **输入要求**
   - 用户必须提供目标代码文件路径
   - 可选：指定需要打点的函数名称列表

3. **依赖技能**
   - `pypto-environment-setup`：用于检查编译环境

## 触发机制

当用户输入包含以下关键字或相关内容时，自动触发此技能：

- **添加性能打点**：为代码添加 PerfMtBegin/PerfMtEnd 打点
- **性能事件定义**：添加 PERF_EVT 事件定义
- **分析函数耗时**：分析代码文件中耗时较长的函数
- **开启性能日志**：打开 ENABLE_PERF_EVT 开关

**触发示例**：
- 为 wrap_manager.h 添加性能打点日志
- 分析 device_ctrl.cpp 中耗时较长的函数并添加打点
- 开启性能打点日志开关

---

## 背景知识

> ⚠️ **重要**：此部分为添加性能打点日志必备的背景知识。

### 性能打点日志机制

PyPTO Machine 模块使用 `PerfMtBegin` 和 `PerfMtEnd` 进行多线程性能打点：

```cpp
PerfMtBegin(PERF_EVT_函数名简写, 调度线程id);
// ... 函数代码 ...
PerfMtEnd(PERF_EVT_函数名简写, 调度线程id);
```

### 性能事件定义规范

性能事件定义位于 `device_utils.h` 的 `PERF_EVENTS` 宏中：

```cpp
#define PERF_EVENTS                  \
    X(层级, 事件名)                   \
    ...
```

**层级规范**：
- 层级 1：入口函数（直接调用的顶层函数）
- 层级 2：一级子函数
- 层级 3：二级子函数（孙子函数）
- 依次类推...

**多线程场景**：使用 `X5` 定义，会展开为 5 个事件（evt, evt1, evt2, evt3, evt4），用于区分不同线程：

```cpp
X5(1, DISPATCH_MIX_CORE_TASK)    // 展开为 PERF_EVT_DISPATCH_MIX_CORE_TASK 等 5 个事件
```

**排列顺序**：相关联的函数按层级顺序排列：
```
一级函数1
  二级函数1
    三级函数1
    三级函数2
  二级函数2
    三级函数1
一级函数2
  二级函数1
    三级函数1
...
```

### 相关文件

| 文件 | 说明 |
|------|------|
| `framework/src/machine/device/dynamic/device_utils.h` | PERF_EVT 事件定义 |
| `framework/src/machine/device/dynamic/device_perf.h` | PerfMtBegin/PerfMtEnd 函数定义 |
| `framework/src/machine/utils/device_switch.h` | ENABLE_PERF_EVT 开关 |
| `framework/src/machine/device/dynamic/*.h` | Machine 模块代码（需添加打点） |

### 获取调度线程 ID

在 WrapManager 类中，通过 `schDevTaskCtx->parallelIdx` 获取调度线程 ID：

```cpp
inline int GetSchedTid() { 
    return schDevTaskCtx != nullptr ? static_cast<int>(schDevTaskCtx->parallelIdx) : 0; 
}
```

---

## 工作流程

**⚠️ 重要提示**：将 bash 运行命令超时时间设置为 300000ms

### 步骤 1：收集必要信息（必须执行）

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息。**

使用 `question` 工具收集以下信息：

- **pypto_path**：PyPTO 项目的根目录路径（绝对路径）
- **target_file**：需要添加打点日志的目标代码文件路径（绝对路径）
- **target_functions**：需要打点的函数名称列表（可选，若不指定则自动分析）
- **compile_cmd**：编译命令（如 `python3 build_ci.py -s --frontend=python3 -j=8`）
- **test_cmd**：验证测试命令（可选）

将收集的路径全部转换成绝对路径。

---

### 步骤 2：分析目标代码文件

若用户未指定函数列表，则自动分析目标代码文件，识别耗时较长的函数。

#### 2.1 读取目标代码文件

```bash
# 使用 Read 工具读取目标文件
```

#### 2.2 分析函数耗时特征

根据以下特征识别耗时较长的函数：

| 特征 | 耗时程度 | 说明 |
|------|----------|------|
| 包含自旋锁（`__sync_bool_compare_and_swap`）| 高 | 忙等待，高竞争时消耗 CPU |
| 多层嵌套循环 | 高 | 循环次数多时耗时显著 |
| 大量数组遍历 | 高 | 遍历大量数据 |
| 线性查找（`for` 循环查找）| 中高 | O(n) 复杂度 |
| 调用多个子函数 | 中 | 函数调用开销 |
| 多次条件判断 | 中 | 分支判断开销 |

#### 2.3 分析函数调用关系

确定各函数之间的调用关系，用于确定层级：

1. 入口函数（层级 1）：直接被外部调用的函数
2. 子函数（层级 2）：被入口函数调用的函数
3. 孙子函数（层级 3）：被子函数调用的函数

#### 2.4 输出分析结果

向用户展示分析结果，列出：
- 高耗时函数列表
- 中等耗时函数列表
- 函数调用层级关系图

---

### 步骤 3：添加性能事件定义

在 `device_utils.h` 的 `PERF_EVENTS` 宏中添加事件定义。

#### 3.1 确定事件名称

事件名称命名规范：`函数名简写`，如：
- `GetAvailableWrapCoreCnt` → `GET_AVAIL_WRAP_CORE`
- `UpdateWrapQueueAndRmvCoreIdx` → `UPDATE_WRAP_QUEUE_RMV_IDX`

#### 3.2 确定层级和定义方式

根据函数调用层级确定：
- 多线程场景使用 `X5` 定义
- 层级数字：入口函数为 1，子函数为 2，孙子函数为 3

#### 3.3 添加事件定义

编辑 `device_utils.h`，在 `PERF_EVENTS` 宏末尾（`X(0, MAX)` 之前）添加：

```cpp
X5(层级, 事件名)             \
```

按层级顺序排列：一级函数 → 二级函数 → 三级函数 → 下一个一级函数...

---

### 步骤 4：添加头文件依赖

若目标文件未包含必要的头文件，需添加：

```cpp
#include "device_perf.h"
#include "device_utils.h"
```

---

### 步骤 5：添加性能打点日志

编辑目标代码文件，为每个需要打点的函数添加 `PerfMtBegin` 和 `PerfMtEnd`。

#### 5.1 确定线程 ID 获取方式

根据代码上下文确定获取线程 ID 的方式：

**方式 A：通过 schDevTaskCtx 获取**
```cpp
inline int GetSchedTid() { 
    return schDevTaskCtx != nullptr ? static_cast<int>(schDevTaskCtx->parallelIdx) : 0; 
}
```

**方式 B：直接使用现有成员变量**
```cpp
PerfMtBegin(PERF_EVT_XXX, aicpuIdx_);
```

**方式 C：使用固定值**
```cpp
PerfMtBegin(PERF_EVT_XXX, 0);
```

#### 5.2 添加打点日志

在函数开始处添加 `PerfMtBegin`，在函数所有返回点添加 `PerfMtEnd`：

```cpp
inline void SomeFunction()
{
    PerfMtBegin(PERF_EVT_SOME_FUNCTION, GetSchedTid());
    
    // ... 函数代码 ...
    
    if (condition) {
        PerfMtEnd(PERF_EVT_SOME_FUNCTION, GetSchedTid());
        return;  // 提前返回
    }
    
    // ... 更多代码 ...
    
    PerfMtEnd(PERF_EVT_SOME_FUNCTION, GetSchedTid());
}
```

**⚠️ 重要**：确保所有返回路径（包括提前 return）都有对应的 `PerfMtEnd`。

---

### 步骤 6：开启性能日志开关

编辑 `device_switch.h`，将 `ENABLE_PERF_EVT` 设置为 1：

```cpp
#define ENABLE_PERF_EVT 1
```

---

### 步骤 7：修改日志输出方式

编辑 `device_perf.h`，修改 `RepeatPuts` 函数中的日志输出：

```cpp
// 将 DEV_INFO 改为 DEV_ERROR(0,
static void RepeatPuts(char c, size_t count)
{
    char buf[80];
    for (size_t i = 0; i < count; i++) {
        buf[i] = c;
    }
    buf[count] = '\0';
    DEV_ERROR(0, "%s.", buf);  // 原为 DEV_INFO("%s.", buf);
}
```

---

### 步骤 8：编译验证

使用用户提供的编译命令编译代码：

```bash
source <CANN_PATH>/cann/set_env.sh
cd <pypto_path>
python3 build_ci.py -s --disable_auto_execute --frontend=python3 -j=16
```

**验证检查点**：
- [ ] 编译成功，无编译错误
- [ ] wheel 包正确生成到 `build_out` 目录

---

### 步骤 9：运行验证（可选）

若用户提供了测试命令，运行测试验证打点日志是否生效：

```bash
source <CANN_PATH>/cann/set_env.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<pypto_path>/build_out/pto/lib/
export PYTHONPATH=${PYTHONPATH}:<pypto_path>/build_out/
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/
export TILE_FWK_DEVICE_ID=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_WORK_PATH=<pypto_path>/

<test_cmd>
```

**验证检查点**：
- [ ] 测试命令执行成功
- [ ] 日志目录下生成相关日志文件

---

### 步骤 10：检查日志输出（可选）

检查日志目录，确认性能打点日志是否生成：

```bash
ls -la <pypto_path>/log/debug/device-0/
ls -la <pypto_path>/log/run/device-0/
```

---

## 完整示例

### 示例：为 wrap_manager.h 添加性能打点

**步骤 1：用户输入**
```
为 wrap_manager.h 中耗时较长的函数添加性能打点日志
```

**步骤 2：分析结果**
- 高耗时函数：`UpdateWrapQueueForThread`, `UpdateWrapQueueAndRmvCoreIdx`, `GetAvailableWrapCoreCnt`, `PushTaskToTasklist`
- 自旋锁函数：`WrapInfoQueueLock`（耗时极高）
- 中等耗时：`DispatchMixCoreTask`, `CalculateTaskCountInSync`, `findWrapInfo`

**步骤 3：层级关系**
```
层级 1：DISPATCH_MIX_CORE_TASK
  └ 层级 2：UPDATE_WRAP_QUEUE
      └ 层级 3：UPDATE_WRAP_QUEUE_RMV_IDX
      └ 层级 3：GET_AVAIL_WRAP_CORE
      └ 层级 3：CALC_TASK_COUNT_SYNC

层级 1：RESOLVE_DEP_MIX_CORE
  └ 层级 2：PUSH_TASK_TO_TASKLIST
      └ 层级 3：FIND_WRAP_INFO

层级 1：UPDATE_FINISH_ID_MIX_CORE
```

**步骤 4：事件定义（device_utils.h）**
```cpp
X5(3, WAIT_AICORE_FINISH)        \
X5(3, DISPATCH_TASK)             \
X5(1, DISPATCH_MIX_CORE_TASK)    \
X5(2, UPDATE_WRAP_QUEUE)         \
X5(3, UPDATE_WRAP_QUEUE_RMV_IDX) \
X5(3, GET_AVAIL_WRAP_CORE)       \
X5(3, CALC_TASK_COUNT_SYNC)      \
X5(1, RESOLVE_DEP_MIX_CORE)      \
X5(2, PUSH_TASK_TO_TASKLIST)     \
X5(3, FIND_WRAP_INFO)            \
X5(1, UPDATE_FINISH_ID_MIX_CORE) \
X(0, MAX)
```

**步骤 5：添加打点日志（wrap_manager.h）**

添加线程 ID 获取函数：
```cpp
inline int GetSchedTid() { 
    return schDevTaskCtx != nullptr ? static_cast<int>(schDevTaskCtx->parallelIdx) : 0; 
}
```

添加打点示例：
```cpp
inline uint32_t GetAvailableWrapCoreCnt(uint32_t& core1c1vCnt, uint32_t& core1c2vCnt, uint32_t maxCoreCnt)
{
    PerfMtBegin(PERF_EVT_GET_AVAIL_WRAP_CORE, GetSchedTid());
    // ... 函数代码 ...
    PerfMtEnd(PERF_EVT_GET_AVAIL_WRAP_CORE, GetSchedTid());
    return core1c2vCnt + core1c1vCnt;
}
```

---

## 关键注意事项

1. **层级规范**：严格按照调用层级定义事件，入口函数为 1，子函数为 2，孙子函数为 3
2. **排列顺序**：相关联的函数按层级顺序排列，便于日志分析
3. **多线程场景**：使用 `X5` 定义，展开为 5 个事件用于区分线程
4. **返回点覆盖**：确保所有 return 路径都有对应的 `PerfMtEnd`
5. **路径规范**：所有路径必须使用绝对路径
6. **日志开关**：必须同时修改 `ENABLE_PERF_EVT` 和 `RepeatPuts` 中的 `DEV_INFO`

## 编译与运行方法

### 编译 PyPTO

**标准编译命令**：

```bash
# 设置 CANN 环境
source <CANN_PATH>/cann/set_env.sh

# 进入 PyPTO 目录并编译
cd <pypto_path>
python3 build_ci.py -s --disable_auto_execute --frontend=python3 -j=8
```

**编译参数说明**：

| 参数 | 说明 |
|------|------|
| `-s` | 静默模式 |
| `--disable_auto_execute` | 禁用自动执行测试 |
| `--frontend=python3` | 使用 Python3 前端 |
| `-j=8` | 并行编译线程数 |

**编译输出**：
- wheel 包位于 `<pypto_path>/build_out/` 目录

---

### 运行测试

**标准运行环境配置**：

```bash
# 1. 设置 CANN 环境
source <CANN_PATH>/cann/set_env.sh

# 2. 配置运行时库路径
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<pypto_path>/build_out/pto/lib/
export PYTHONPATH=${PYTHONPATH}:<pypto_path>/build_out/

# 3. 配置 ISA 库路径
export PTO_TILE_LIB_CODE_PATH=/mnt/workspace/pto-isa/

# 4. 配置设备 ID
export TILE_FWK_DEVICE_ID=0

# 5. 配置日志级别（可选，用于调试）
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_WORK_PATH=<pypto_path>/

# 6. 运行测试
<test_cmd>
```

**环境变量说明**：

| 变量 | 说明 |
|------|------|
| `LD_LIBRARY_PATH` | 动态库搜索路径，需包含 `build_out/pto/lib/` |
| `PYTHONPATH` | Python 模块搜索路径，需包含 `build_out/` |
| `PTO_TILE_LIB_CODE_PATH` | Tile ISA 库路径 |
| `TILE_FWK_DEVICE_ID` | NPU 设备 ID（默认 0） |
| `ASCEND_GLOBAL_LOG_LEVEL` | 日志级别（0-3，3 为 INFO） |
| `ASCEND_WORK_PATH` | 工作路径 |

---

### 查看日志

**日志目录结构**：

```
<pypto_path>/log/
├── debug/                 # 调试日志
│   ├── device-0/         # Device 侧日志（NPU 执行时生成）
│   └── plog/             # Host 侧日志
├── run/                   # 运行日志
│   ├── device-0/         # Device 运行日志
│   └── plog/             # Host 运行日志
└── security/              # 安全日志
```

**查看日志命令**：

```bash
# 查看 Device 日志
ls -la <pypto_path>/log/debug/device-0/
cat <pypto_path>/log/debug/device-0/device-*.log

# 查看 Host 日志
ls -la <pypto_path>/log/run/plog/
grep "perf\|PERF_EVT" <pypto_path>/log/run/plog/plog-*.log
```

---

## 参考文档

| 文件 | 说明 |
|------|------|
| `framework/src/machine/device/dynamic/device_utils.h` | PERF_EVT 事件定义（PERF_EVENTS 宏） |
| `framework/src/machine/device/dynamic/device_perf.h` | PerfMtBegin/PerfMtEnd 函数实现 |
| `framework/src/machine/utils/device_switch.h` | ENABLE_PERF_EVT 开关定义 |