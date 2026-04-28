---
name: pypto-schedule-trace-refine
description: PyPTO Machine 调度性能细化打点技能。用于分析和细化 Machine 调度阶段的性能打点，使用 aicpu_perf.h 中的打点事件和 PerfMtTrace 工具收集调度阶段执行时间，根据采集结果自动细化打点，分析耗时较长和波动较大的代码位置，并进一步分析可能原因。触发词：调度性能打点、细化打点、调度耗时分析、性能打点优化、trace refine、schedule trace。
---

# PyPTO Machine 调度性能细化打点技能

## 概述

本技能用于分析和细化 PyPTO Machine 调度阶段的性能打点，通过迭代式细化打点策略，精准定位调度流程中的性能瓶颈和波动点。

## 核心目标

### 目标定义

**主要目标**：
1. **定位耗时瓶颈**：识别调度阶段中耗时最长的代码位置
2. **分析性能波动**：找出执行时间波动大的代码段，分析波动原因
3. **细化打点层级**：在粗粒度打点基础上，自动插入细粒度打点
4. **提供优化建议**：基于打点数据分析，提供调度性能优化建议

### 性能分析目标

**关键性能指标**：
- 单次调度执行时间（绝对耗时）
- 调度时间占比（相对于算子总执行时间）
- 调度时间波动（多次执行的标准差/方差）
- 任务下发效率（任务下发时间 vs. 任务执行时间）

**优化方向判断**：
- ✅ **调度时间占比高 (>20%)**：需要优化调度流程
- ✅ **波动大 (标准差 > 平均值的 30%)**：需要分析波动原因
- ✅ **解依赖耗时占比高**：需要优化依赖解析逻辑
- ✅ **任务下发耗时占比高**：需要优化任务下发流程

## 触发机制

当用户提到以下关键词时触发：
- "调度性能打点"
- "细化打点"
- "调度耗时分析"
- "性能打点优化"
- "trace refine"
- "schedule trace"
- "Machine 调度性能"
- "DispatchAiCoreTask 性能"

## 使用场景

- 分析 Machine 调度流程性能
- 定位调度阶段性能瓶颈
- 优化调度时间占比
- 分析调度性能波动
- 细化调度性能打点
- 评估调度优化效果

## 核心文件

### 打点相关文件

| 文件路径 | 功能描述 |
|---------|---------|
| `framework/src/interface/machine/device/tilefwk/aicpu_perf.h` | AICPU 打点事件定义（PERF_TRACES 宏） |
| `framework/src/interface/machine/device/tilefwk/aicpu_common.h` | AICore 打点事件定义（AicorePerfTrace 枚举） |
| `framework/src/machine/device/dynamic/device_perf.h` | PerfMtTrace 函数实现 |
| `framework/src/machine/device/dynamic/aicore_manager.h` | 调度核心函数（DispatchAiCoreTask 等） |
| `framework/src/machine/runtime/dump_device_perf.cpp` | 性能数据导出函数（DumpAicpuPerfInfo） |

### 关键函数

| 函数名 | 文件位置 | 功能描述 |
|--------|---------|---------|
| `DispatchAiCoreTask` | aicore_manager.h:940 | 调度主循环（解依赖 + 任务下发） |
| `ResolveDepForAllAiCore` | aicore_manager.h:1125 | 解依赖过程 |
| `TryBatchSendTask` | aicore_manager.h:846 | 任务下发流程 |
| `PerfMtTrace` | device_perf.h:393 | 打点记录函数 |
| `DumpAicpuPerfInfo` | dump_device_perf.cpp:251 | 性能数据导出 |

---

## 工作流程

### 流程总览

```
┌──────────────────────────────────────────────────────────┐
│            调度性能细化打点迭代流程                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  步骤 1: 环境准备                                         │
│     ├─ 设置性能采集环境                                   │
│     ├─ 确认打点文件位置                                   │
│     └─ 理解现有打点结构                                   │
│                                                          │
│  步骤 2: 粗粒度性能采集                                   │
│     ├─ 启用现有打点                                       │
│     ├─ 运行算子采集数据                                   │
│     ├─ 分析粗粒度性能数据                                 │
│     └─ 确定需要细化的代码段                               │
│                                                          │
│  步骤 3: 细化打点设计                                     │
│     ├─ 分析目标代码段内部流程                             │
│     ├─ 设计细化打点位置                                   │
│     ├─ 新增打点事件定义                                   │
│     └─ 插入打点代码                                       │
│                                                          │
│  步骤 4: 细粒度性能采集                                   │
│     ├─ 编译并运行                                         │
│     ├─ 采集细化性能数据                                   │
│     ├─ 分析细粒度耗时分布                                 │
│     └─ 判断是否需要继续细化                               │
│                                                          │
│  步骤 5: 波动分析                                         │
│     ├─ 多次运行采集数据                                   │
│     ├─ 计算执行时间统计量                                 │
│     ├─ 分析波动来源                                       │
│     └─ 提供波动优化建议                                   │
│                                                          │
│  步骤 6: 问题诊断                                         │
│     ├─ 定位耗时最长代码段                                 │
│     ├─ 定位波动最大代码段                                 │
│     ├─ 分析可能的性能瓶颈原因                             │
│     └─ 提供优化建议                                       │
│                                                          │
│  步骤 7: 持续迭代                                         │
│     ├─ 判断是否需要继续细化                               │
│     ├─ 重复步骤 3-6                                       │
│     └─ 达到目标后生成报告                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 步骤 1: 环境准备

### 1.0 设置编译环境（关键步骤）

**⚠️ 编译前必须设置 CANN 环境**：

```bash
# 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 确认 CANN 路径
ls -la /usr/local/Ascend/
# 通常路径: /usr/local/Ascend/cann-8.5.0 或类似

# 设置编译环境变量
export PYPTO_BUILD_EXT_ARGS="--cmake-options=-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0"
```

**编译失败的常见原因**：
- ❌ 未设置 `ASCEND_CANN_PACKAGE_PATH` → `BUILD_WITH_CANN=OFF` → 精度问题
- ❌ 使用错误的编译方式 → 编译不生效
- ❌ 编译缓存未清理 → 修改未生效

### 1.1 设置性能采集环境

**启用性能数据采集（必要配置）**：

```bash
# 设置环境变量
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs
mkdir -p $ASCEND_PROCESS_LOG_PATH

# 启用 aicpu perf 数据采集（必要！）
export DUMP_DEVICE_PERF=true

# 确认 NPU 设备
npu-smi info
```

**⚠️ 重要提示**：
- `DUMP_DEVICE_PERF=true` 是启用 aicpu perf 采集的必要环境变量
- 未设置此环境变量时，`machine_trace_perf_data_*.json` 文件不会生成
- 必须在运行算子测试前设置此环境变量

### 1.2 设置算子调试选项（必要配置）

**在算子测试脚本中添加 debug_options**：

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": "npu"},
    debug_options={"runtime_debug_mode": 1}  # 必要：启用完整调试分析
)
def kernel_function(...):
    pass
```

**⚠️ 重要提示**：
- `debug_options={"runtime_debug_mode": 1}` 是必要配置，用于启用完整调试分析
- 启用后会生成：
  - `machine_trace_perf_data_*.json`（性能打点数据）
  - `merged_swimlane.json`（泳道图数据）
  - `bubble_analysis.log`（Bubble 分析报告）
  - `tilefwk_L1_prof_data.json`（L1 级性能数据）
- 必须同时在算子装饰器中设置此选项，否则部分分析功能不可用

### 1.3 完整配置示例

**完整的环境配置和运行流程**：

```bash
# 1. 设置环境变量
export DUMP_DEVICE_PERF=true
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs
mkdir -p $ASCEND_PROCESS_LOG_PATH

# 2. 运行算子（算子脚本中需已设置 debug_options={"runtime_debug_mode": 1})
python3 custom/operator_name/operator.py --run_mode npu
```

**⚠️ 双重必要配置**：
- **环境变量**: `DUMP_DEVICE_PERF=true`（shell 中设置）
- **算子配置**: `debug_options={"runtime_debug_mode": 1}`（Python 装饰器中设置）
- 两者必须同时设置，缺一不可

### 1.4 理解现有打点结构

**现有打点事件** (aicpu_perf.h):

```cpp
#define PERF_TRACES                           \
    X(BEGIN)                                  \
    X(ALLOC_THREAD_ID)                        \
    X(INIT)                                   \
    X(CORE_HAND_SHAKE)                        \
    XDEVTASK(DEV_TASK_BUILD)                  \
    XDEVTASK(DEV_TASK_RCV)                    \
    XDEVTASK(DEV_TASK_SEND_FIRST_LEAF_TASK)   \
    XDEVTASK(DEV_TASK_SCHED_EXEC)             \
    XDEVTASK(DEV_TASK_SYNC_CORE_STOP)         \
    XDEVTASK(DEV_TASK_RSP)                    \
    X(WAIT_ALL_DEV_TASK_FINISH)               \
    X(WAIT_CORE_EXIT)                         \
    X(EXIT)                                   \
    X(MAX)
```

**打点使用示例**:

```cpp
// 在关键位置插入打点
PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, threadIdx);
PerfMtTrace(PERF_TRACE_DEV_TASK_SCHED_EXEC, aicpuIdx_);
```

### 1.3 确认关键函数位置

**主要调度函数**：
- `DispatchAiCoreTask`: aicore_manager.h:940-969（调度主循环）
- `ResolveDepForAllAiCore`: aicore_manager.h:1125-1179（解依赖）
- `TryBatchSendTask`: aicore_manager.h:846-890（任务下发）

---

## 步骤 2: 粗粒度性能采集

### 2.1 确认现有打点是否覆盖关键流程

**检查现有打点覆盖情况**：

| 流程阶段 | 现有打点事件 | 是否覆盖 |
|---------|-------------|---------|
| 任务构建 | DEV_TASK_BUILD | ✅ |
| 任务接收 | DEV_TASK_RCV | ✅ |
| 首次叶子任务发送 | DEV_TASK_SEND_FIRST_LEAF_TASK | ✅ |
| 任务调度执行 | DEV_TASK_SCHED_EXEC | ✅ |
| 核同步停止 | DEV_TASK_SYNC_CORE_STOP | ✅ |
| 任务响应 | DEV_TASK_RSP | ✅ |

**⚠️ 注意**：现有打点为粗粒度打点，仅覆盖主要阶段，需要细化分析。

### 2.2 运行算子采集粗粒度数据

**运行算子**：

在完成步骤 1 的双重必要配置后，运行算子即可采集完整的性能数据：

```bash
python3 custom/operator_name/operator.py --run_mode npu
```

**必要配置检查清单**：

| 配置项 | 设置位置 | 必要性 | 说明 |
|--------|---------|--------|------|
| `DUMP_DEVICE_PERF=true` | Shell 环境变量 | **必要** | 启用 aicpu perf 数据采集 |
| `debug_options={"runtime_debug_mode": 1}` | Python 装饰器 | **必要** | 启用完整调试分析 |

**生成的文件清单**：

| 文件 | 功能 | 依赖配置 |
|------|------|---------|
| `machine_trace_perf_data_*.json` | 性能打点数据 | DUMP_DEVICE_PERF + runtime_debug_mode |
| `merged_swimlane.json` | 泳道图数据 | runtime_debug_mode |
| `bubble_analysis.log` | Bubble 分析报告 | runtime_debug_mode |
| `tilefwk_L1_prof_data.json` | L1 级性能数据 | runtime_debug_mode |
| `machine_runtime_operator_trace_*.json` | Perfetto trace | runtime_debug_mode |

**⚠️ 注意**：
- 两项配置必须同时设置，缺一不可
- 未设置 `runtime_debug_mode` 时，泳道图和 Bubble 分析等关键功能不可用
- 配置正确后，运行算子即可自动生成完整性能分析数据

### 2.3 分析粗粒度性能数据

**性能数据文件位置**：

```
output/output_时间戳/machine_trace_perf_data_*.json
```

**使用分析脚本**：

```bash
python3 .agents/skills/pypto-schedule-trace-refine/scripts/schedule_trace_analysis.py \
    output/output_*/machine_trace_perf_data_*.json
```

**完整分析输出示例**：

```markdown
====================================================================================================
Machine 调度性能细化打点分析报告
====================================================================================================

AICPU 调度核心阶段耗时分析 (Level 0: 粗粒度)
----------------------------------------------------------------------------------------------------
阶段名称                                        平均(us)      最大(us)      最小(us)      次数     
----------------------------------------------------------------------------------------------------
DEV_TASK_BUILD                                  455.10        455.10        455.10        1        
INIT                                            329.75        431.40        260.00        3        
...

性能瓶颈定位 (耗时最长阶段)
----------------------------------------------------------------------------------------------------
排名  核心类型              阶段名称                            平均耗时(us)    占比      
----------------------------------------------------------------------------------------------------
1     AICPU-CTRL           DEV_TASK_BUILD                      455.10          27.4%     

波动分析 (变化系数 CV)
----------------------------------------------------------------------------------------------------
核心类型              阶段名称                            平均(us)      标准差(us)    CV(%)       次数     
----------------------------------------------------------------------------------------------------
AICPU-SCHED          EXIT                                415.64        373.16        89.8%       3        ⭐⭐⭐
AICPU-SCHED          DEV_TASK_RESOLVE_RELEASE_CORE       86.77         57.70         66.5%       3        ⭐⭐⭐

调度与计算耗时对比
----------------------------------------------------------------------------------------------------
指标                              耗时(us)       占比          
----------------------------------------------------------------------------------------------------
AICPU 调度总耗时                  1662.15        45.2%         
AICore/AIV 计算总耗时             2000.00        54.8%         
算子总执行时间                    3662.15        100.0%        
```

---

## 步骤 3: 细化打点设计

### 3.1 分析目标代码段内部流程

**以 DispatchAiCoreTask 为例**：

```cpp
inline int32_t DispatchAiCoreTask(...) {
    // 流程 1: 解依赖
    if (devTaskCtx->waitTaskCnt[...] > 0) {
        ret = ResolveDepForAllAiCore(...);  // ← 需要细化
    }
    
    // 流程 2: DIE 任务下发（混合架构）
    if (wrapManager.GetIsMixarch()) {
        TryBatchSendTask(..., dieReadyQue, ...);  // ← 需要细化
    }
    
    // 流程 3: 正常任务下发
    TryBatchSendTask(..., readyQue, ...);  // ← 需要细化
    
    // 流程 4: 公平调度
    if (enableFairSch_) { ... }
}
```

### 3.2 设计细化打点位置

**细化打点设计原则**：
1. **分层细化**：先细化主要流程，再细化子流程
2. **关键路径优先**：优先细化占比高的代码段
3. **避免过度细化**：每次只增加必要的打点
4. **考虑栈空间限制**：打点数量不超过 PERF_TRACE_COUNT_DEVTASK_MAX_NUM

**细化打点设计示例**：

**Level 1: DispatchAiCoreTask 内部流程**
- `DEV_TASK_DISPATCH_RESOLVE_DEP`: 解依赖阶段开始/结束
- `DEV_TASK_DISPATCH_MIXARCH_SEND`: 混合架构任务下发
- `DEV_TASK_DISPATCH_NORMAL_SEND`: 正常任务下发
- `DEV_TASK_DISPATCH_FAIR_SCH`: 公平调度处理

**Level 2: ResolveDepForAllAiCore 内部流程**
- `DEV_TASK_RESOLVE_RELEASE_CORE`: 释放核心
- `DEV_TASK_RESOLVE_SEND_AVAILABLE`: 发送任务到可用核心
- `DEV_TASK_RESOLVE_PUSH_READYQUE`: 推入就绪队列

**Level 3: TryBatchSendTask 内部流程**
- `DEV_TASK_SEND_CHECK_READY`: 检查就绪状态
- `DEV_TASK_SEND_LOCK_QUEUE`: 队列加锁
- `DEV_TASK_SEND_POP_TASK`: 弹出任务
- `DEV_TASK_SEND_BATCH_SEND`: 批量发送

### 3.3 新增打点事件定义

**修改 aicpu_perf.h 文件**：

```cpp
// 在 PERF_TRACES 宏中新增打点事件
#define PERF_TRACES                           \
    X(BEGIN)                                  \
    X(ALLOC_THREAD_ID)                        \
    X(INIT)                                   \
    X(CORE_HAND_SHAKE)                        \
    XDEVTASK(DEV_TASK_BUILD)                  \
    XDEVTASK(DEV_TASK_RCV)                    \
    XDEVTASK(DEV_TASK_SEND_FIRST_LEAF_TASK)   \
    XDEVTASK(DEV_TASK_SCHED_EXEC)             \
    XDEVTASK(DEV_TASK_SYNC_CORE_STOP)         \
    XDEVTASK(DEV_TASK_RSP)                    \
    XDEVTASK(DEV_TASK_DISPATCH_RESOLVE_DEP)   /* 新增: 解依赖阶段 */ \
    XDEVTASK(DEV_TASK_DISPATCH_MIXARCH_SEND)  /* 新增: 混合架构下发 */ \
    XDEVTASK(DEV_TASK_DISPATCH_NORMAL_SEND)   /* 新增: 正常下发 */ \
    XDEVTASK(DEV_TASK_RESOLVE_RELEASE_CORE)   /* 新增: 释放核心 */ \
    XDEVTASK(DEV_TASK_RESOLVE_PUSH_READYQUE)  /* 新增: 推入队列 */ \
    XDEVTASK(DEV_TASK_SEND_CHECK_READY)       /* 新增: 检查就绪 */ \
    XDEVTASK(DEV_TASK_SEND_BATCH_SEND)        /* 新增: 批量发送 */ \
    X(WAIT_ALL_DEV_TASK_FINISH)               \
    X(WAIT_CORE_EXIT)                         \
    X(EXIT)                                   \
    X(MAX)
```

### 3.4 插入打点代码

**在 DispatchAiCoreTask 中插入打点**：

```cpp
inline int32_t DispatchAiCoreTask(
    SchDeviceTaskContext* devTaskCtx, CoreType type, ReadyCoreFunctionQueue* readyQue,
    int coreIdxStart, int coreIdxEnd)
{
    int32_t ret = DEVICE_MACHINE_OK;
    auto& wrapManager = devTaskCtx->GetWrapManager();
    
    // 流程 1: 解依赖
    if (devTaskCtx->waitTaskCnt[static_cast<int>(type)] > 0) {
        PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_RESOLVE_DEP, aicpuIdx_);  // 新增打点
        ret = ResolveDepForAllAiCore(devTaskCtx, type, coreIdxStart, coreIdxEnd);
        PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_RESOLVE_DEP, aicpuIdx_);  // 新增打点（结束）
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return ret;
        }
        wrapManager.DispatchMixCoreTask();
    }
    
    // 流程 2: DIE 任务下发
    if (wrapManager.GetIsMixarch()) {
        ReadyCoreFunctionQueue* dieReadyQue = ...;
        if (dieReadyQue != readyQue) {
            PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_MIXARCH_SEND, aicpuIdx_);  // 新增打点
            TryBatchSendTask(devTaskCtx, type, dieReadyQue, coreIdxStart, coreIdxEnd);
            PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_MIXARCH_SEND, aicpuIdx_);  // 新增打点（结束）
        }
    }
    
    // 流程 3: 正常任务下发
    PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_NORMAL_SEND, aicpuIdx_);  // 新增打点
    TryBatchSendTask(devTaskCtx, type, readyQue, coreIdxStart, coreIdxEnd);
    PerfMtTrace(PERF_TRACE_DEV_TASK_DISPATCH_NORMAL_SEND, aicpuIdx_);  // 新增打点（结束）
    
    // 流程 4: 公平调度
    if (enableFairSch_) { ... }
    
    return ret;
}
```

**⚠️ 重要提示**：
- 打点使用成对的 `PerfMtTrace` 来标记开始和结束
- 或者使用 `PerfMtBegin` 和 `PerfMtEnd` 配对
- 确保 `aicpuIdx_` 或 `threadIdx` 参数正确传递

---

## 步骤 4: 细粒度性能采集

### 4.1 编译并运行

**⚠️ 重要：编译前必须正确设置 CANN 路径**

**编译失败的常见原因**：
- 未设置 `ASCEND_CANN_PACKAGE_PATH` 导致 `BUILD_WITH_CANN=OFF`
- 编译出的版本缺少 NPU 支持，运行时会产生精度问题

**正确编译方式**：

```bash
# 步骤1: 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 步骤2: 确认 CANN 路径（通常是 /usr/local/Ascend/cann-x.x.x）
ls -la /usr/local/Ascend/

# 步骤3: 编译时指定 CANN 路径（方式一：pip 环境变量）
export PYPTO_BUILD_EXT_ARGS="--cmake-options=-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0"
python3 -m pip install -e . --no-build-isolation

# 或者（方式二：直接 cmake）
cmake -S . -B build -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0
cmake --build build -j 32
cmake --install build --prefix python
```

**验证编译成功**：

```bash
# 检查编译配置（必须显示 BUILD_WITH_CANN=ON）
grep BUILD_WITH_CANN build/CMakeCache.txt

# 验证精度正确（不设置性能采集环境变量）
python examples/01_beginner/basic/basic_ops.py elementwise_ops::test_elementwise_ops --run_mode npu
```

**运行算子**：

```bash
python3 custom/operator_name/operator.py --run-mode npu
```

### 4.2 采集细化性能数据

**性能数据文件位置**：

```
output/output_时间戳/machine_trace_perf_data_*.json
```

### 4.3 分析细粒度耗时分布

**使用分析脚本**：

```bash
python3 .agents/skills/pypto-schedule-trace-refine/scripts/schedule_trace_analysis.py \
    output/output_*/machine_trace_perf_data_*.json
```

脚本自动识别已细化的打点层级，分析细粒度耗时分布。

**分析输出示例**：

```markdown
====================================================================================================
DEV_TASK_SCHED_EXEC 详细分析
====================================================================================================

DEV_TASK_SCHED_EXEC 平均耗时: 86.77 us
占调度总耗时比例: 10.4%
调用次数: 3

⚠️ DEV_TASK_SCHED_EXEC 耗时占比 > 20%, 建议细化打点
建议新增的细化打点事件:
  1. DEV_TASK_DISPATCH_RESOLVE_DEP - 解依赖阶段
  2. DEV_TASK_DISPATCH_NORMAL_SEND - 正常任务下发
  3. DEV_TASK_SEND_BATCH_SEND - 批量发送
  4. DEV_TASK_RESOLVE_RELEASE_CORE - 释放核心
  5. DEV_TASK_RESOLVE_PUSH_READYQUE - 推入队列
```

### 4.4 判断是否需要继续细化

**⚠️ 重要：满足以下任一条件时，必须继续细化，不能直接终止分析**

**继续细化的判断条件**：
1. ✅ 仍有耗时占比 >30% 的代码段未细化 → **必须细化**
2. ✅ 存在波动 > 平均值 30% (CV > 30%) 的代码段 → **必须细化**
3. ✅ 用户要求进一步分析特定流程 → **按用户要求细化**
4. ✅ 打点数量未超过限制 (PERF_TRACE_COUNT_DEVTASK_MAX_NUM) → **可以细化**

**终止细化的判断条件**（需全部满足）：
1. ✅ 所有主要流程都已细化到 Level 1 或更深
2. ✅ 无 CV > 30% 的高波动代码段（或已定位到波动根源）
3. ✅ 打点数量接近限制，无法继续添加
4. ✅ 用户明确确认当前粒度足够

---

## 步骤 5: 波动分析

### 5.1 多次运行采集数据

**运行多次测试**：

```bash
# 运行 10 次测试，采集波动数据
for i in {1..10}; do
    echo "Run $i"
    python3 custom/operator_name/operator.py --run-mode npu > log_$i.txt 2>&1
    sleep 1
done
```

**收集性能数据文件**：

```bash
# 查找所有性能数据文件
find output -name "machine_trace_perf_data_*.json" -type f | sort
```

### 5.2 计算执行时间统计量

**波动分析已集成在主分析脚本中**：

`schedule_trace_analysis.py` 自动计算所有阶段的波动统计量（CV）并标注高波动阶段。

运行完整分析即可获得波动数据：

```bash
python3 .agents/skills/pypto-schedule-trace-refine/scripts/schedule_trace_analysis.py \
    output/output_*/machine_trace_perf_data_*.json
```

**波动分析输出示例**：

```markdown
====================================================================================================
波动分析 (变化系数 CV)
====================================================================================================
说明: CV = 标准差/平均值 × 100%, CV > 30% 表示高波动

核心类型              阶段名称                            平均(us)      标准差(us)    CV(%)       次数     
----------------------------------------------------------------------------------------------------
AICPU-SCHED          EXIT                                415.64        373.16        89.8%       3        ⭐⭐⭐
AICPU-SCHED          DEV_TASK_RESOLVE_RELEASE_CORE       86.77         57.70         66.5%       3        ⭐⭐⭐
AICPU-SCHED          DEV_TASK_RCV                        57.41         36.62         63.8%       3        ⭐⭐⭐
AICPU-SCHED          CORE_HAND_SHAKE                     68.27         29.10         42.6%       3        ⭐⭐

⚠️ 高波动阶段 (CV > 30%):
   AICPU-SCHED/EXIT: CV=89.8%
   可能原因: 任务数量波动、并发竞争、硬件延迟波动
   建议: 稳定任务调度策略、减少锁竞争、优化数据局部性
```

**统计量计算**：
- 平均值 (Mean)
- 标准差 (Standard Deviation)
- 方差 (Variance)
- 最大值/最小值 (Max/Min)
- 变化系数 (Coefficient of Variation: CV = Std/Mean)
- 波动评级: ⭐⭐⭐ (CV>50%), ⭐⭐ (CV>30%), ⭐ (CV>10%)

### 5.3 分析波动来源

**常见波动原因**：

| 波动类型 | 可能原因 | 诊断方法 |
|---------|---------|---------|
| **任务数量波动** | 不同任务的依赖数不同 | 检查任务队列长度变化 |
| **并发竞争波动** | 多核并发访问共享资源 | 分析锁竞争时间 |
| **硬件延迟波动** | L2 cache、内存延迟变化 | 检查 cache miss 统计 |
| **调度策略波动** | 公平调度、优先级调度切换 | 检查调度策略触发条件 |
| **核心状态波动** | 核心空闲/忙碌状态变化 | 检查核心状态变化次数 |

### 5.4 提供波动优化建议

**波动优化策略**：

| 波动原因 | 优化建议 |
|---------|---------|
| 任务数量波动 | 优化任务分组策略，减少任务数量差异 |
| 并发竞争波动 | 减少锁竞争，使用无锁队列或细粒度锁 |
| 硬件延迟波动 | 优化数据局部性，减少 cache miss |
| 调度策略波动 | 稳定调度策略，减少策略切换次数 |
| 核心状态波动 | 优化核心负载均衡，减少状态切换 |

---

## 步骤 6: 问题诊断

### 6.1 定位耗时最长代码段

**耗时定位方法**：

1. **从粗到细**：先分析粗粒度打点，确定主要瓶颈
2. **逐层细化**：在瓶颈代码段内插入细化打点
3. **持续迭代**：直到定位到具体函数或代码块

**耗时诊断输出示例**：

```markdown
## 耗时瓶颈定位

### 耗时最长路径
DispatchAiCoreTask (XX us)
  → ResolveDepForAllAiCore (XX us, 占比 XX%)
    → ResolveDepWithDfx (XX us, 占比 XX%)
      → BatchPushReadyQueue (XX us, 占比 XX%)
        → PushReadyQue (XX us, 占比 XX%) ← 最耗时点

### 性能瓶颈分析
1. PushReadyQue 耗时最长
   - 代码位置: aicore_manager.h:1216-1218
   - 可能原因: 队列锁竞争、内存拷贝开销
   - 建议优化: 使用无锁队列、减少拷贝次数
```

### 6.2 定位波动最大代码段

**波动定位方法**：

1. **计算 CV**：对所有打点事件计算变化系数
2. **排序波动**：按 CV 从大到小排序
3. **分析高波动**：重点分析 CV > 30% 的代码段

**波动诊断输出示例**：

```markdown
## 波动瓶颈定位

### 波动最大路径
DEV_TASK_DISPATCH_RESOLVE_DEP (CV=XX%)
  → ResolveDepForAllAiCore 内部
    → ResolveDepWithDfx (CV=XX%)
      → 硬件状态检测 (CV=XX%) ← 最高波动点

### 波动原因分析
1. 硬件状态检测波动最大
   - 代码位置: aicore_manager.h:1133-1140
   - 波动原因: 核心空闲状态检测时间波动
   - 建议优化: 稳定检测策略，减少检测次数
```

### 6.3 分析可能的性能瓶颈原因

**性能瓶颈原因分类**：

| 瓶颈类型 | 症状描述 | 典型原因 | 诊断方法 |
|---------|---------|---------|---------|
| **计算瓶颈** | 单次执行时间长 | 复杂算法、重复计算 | 分析算法复杂度 |
| **内存瓶颈** | 内存访问频繁 | 数据拷贝、cache miss | 检查内存操作次数 |
| **锁瓶颈** | 等待时间长 | 锁竞争、并发冲突 | 分析锁等待时间 |
| **IO瓶颈** | IO 操作多 | 设备访问、硬件同步 | 检查 IO 操作次数 |
| **调度瓶颈** | 调度开销大 | 任务调度、依赖解析 | 分析调度流程 |

### 6.4 提供优化建议

**优化建议模板**：

```markdown
## 性能优化建议

### 高耗时优化建议
1. [优化点名称]
   - 当前耗时: XX us
   - 优化方法: [具体方法]
   - 预期收益: 减少 XX us
   - 实施难度: [低/中/高]

### 高波动优化建议
1. [优化点名称]
   - 当前波动: CV = XX%
   - 优化方法: [具体方法]
   - 预期收益: 减少 CV 到 XX%
   - 实施难度: [低/中/高]
```

---

## 步骤 7: 持续迭代

### 7.1 判断是否需要继续细化

**⚠️ 重要：发现波动点后必须深入分析，不能仅在粗粒度层级停止**

**继续细化条件**：
- ✅ 仍有耗时占比 >30% 的代码段 → **必须细化**
- ✅ 存在 CV > 30% 的高波动代码段 → **必须细化并定位波动根源**
- ✅ 用户要求进一步分析 → **按用户要求细化**
- ✅ 打点数量未超过限制 → **可以细化**

**终止细化条件**（需全部满足）：
- ✅ 已定位到具体瓶颈点且波动根源已分析
- ✅ 无 CV > 30% 的高波动代码段
- ✅ 打点数量接近限制 (PERF_TRACE_COUNT_DEVTASK_MAX_NUM)
- ✅ 用户明确确认当前粒度足够

**⚠️ 不允许的终止情况**：
- ❌ 仅完成粗粒度分析就停止（除非无高耗时和高波动）
- ❌ 发现波动点但不深入分析波动根源
- ❌ 用户未确认就自行终止

### 7.2 迭代流程

**迭代细化流程**：

```
判断是否继续细化
    ↓
需要继续 → 重复步骤 3-6
    ↓
    ├─ 分析目标代码段内部流程
    ├─ 设计细化打点位置
    ├─ 新增打点事件定义
    ├─ 插入打点代码
    ├─ 编译运行采集数据
    ├─ 分析细粒度耗时分布
    ├─ 波动分析
    └─ 问题诊断
    ↓
不需要继续 → 生成最终报告 (步骤 8)
```

---

## 步骤 8: 生成最终报告

### 8.1 报告模板

**最终报告模板**：

```markdown
# Machine 调度性能细化打点分析报告

## 概述
- 分析目标: [算子名称]
- 分析范围: Machine 调度流程
- 细化层级: Level X
- 分析时间: YYYY-MM-DD HH:MM:SS

## 性能数据摘要
- 总调度时间: XX us
- 占算子总时间: XX%
- 最大单次调度: XX us
- 最小单次调度: XX us
- 平均调度时间: XX us
- 标准差: XX us
- 变化系数 (CV): XX%

## 细化打点层级

### Level 1: 主要调度流程
| 打点事件 | 平均耗时 (us) | 占比 | 调用次数 |
|---------|--------------|------|---------|
| DEV_TASK_DISPATCH_RESOLVE_DEP | XX | XX% | XX |
| DEV_TASK_DISPATCH_NORMAL_SEND | XX | XX% | XX |

### Level 2: 解依赖流程
| 打点事件 | 平均耗时 (us) | 占比 | 调用次数 |
|---------|--------------|------|---------|
| DEV_TASK_RESOLVE_RELEASE_CORE | XX | XX% | XX |
| DEV_TASK_RESOLVE_PUSH_READYQUE | XX | XX% | XX |

### Level 3: 任务下发流程
| 打点事件 | 平均耗时 (us) | 占比 | 调用次数 |
|---------|--------------|------|---------|
| DEV_TASK_SEND_CHECK_READY | XX | XX% | XX |
| DEV_TASK_SEND_BATCH_SEND | XX | XX% | XX |

## 性能瓶颈定位

### 耗时最长代码段
1. [函数名]
   - 代码位置: [文件:行号]
   - 平均耗时: XX us
   - 占比: XX%
   - 可能原因: [原因分析]
   - 优化建议: [具体建议]

### 波动最大代码段
1. [函数名]
   - 代码位置: [文件:行号]
   - 平均耗时: XX us
   - CV: XX%
   - 波动原因: [原因分析]
   - 优化建议: [具体建议]

## 优化建议汇总

### 高耗时优化建议
| 优化点 | 当前耗时 | 优化方法 | 预期收益 | 实施难度 |
|--------|---------|---------|---------|---------|
| [优化点1] | XX us | [方法] | -XX us | [难度] |
| [优化点2] | XX us | [方法] | -XX us | [难度] |

### 高波动优化建议
| 优化点 | 当前CV | 优化方法 | 预期收益 | 实施难度 |
|--------|--------|---------|---------|---------|
| [优化点1] | XX% | [方法] | -XX% | [难度] |
| [优化点2] | XX% | [方法] | -XX% | [难度] |

## 新增打点列表

### 新增打点事件定义
| 打点事件 | 打点类型 | 打点位置 | 功能描述 |
|---------|---------|---------|---------|
| DEV_TASK_DISPATCH_RESOLVE_DEP | XDEVTASK | DispatchAiCoreTask | 解依赖阶段 |
| DEV_TASK_DISPATCH_NORMAL_SEND | XDEVTASK | DispatchAiCoreTask | 正常任务下发 |

### 新增打点代码位置
| 文件 | 函数 | 行号 | 打点代码 |
|------|------|------|---------|
| aicore_manager.h | DispatchAiCoreTask | XX | PerfMtTrace(...) |

## 注意事项

### 栈空间限制
- 当前打点数量: XX
- 最大限制: PERF_TRACE_COUNT_DEVTASK_MAX_NUM = 20
- 建议: [是否需要调整限制]

### 数据采集优化
- 当前采集: DumpAicpuDevTask + DumpAicoreDevTask
- 建议: [是否需要注释 DumpAicoreDecTask]

## 分析工具使用

### 分析脚本

```bash
python3 .agents/skills/pypto-schedule-trace-refine/scripts/schedule_trace_analysis.py \
    output/output_*/machine_trace_perf_data_*.json
```

### 数据文件

| 文件 | 功能 |
|------|------|
| 输入: `machine_trace_perf_data_*.json` | 性能打点数据 |
| 输出: 终端报告 | 完整分析报告（可重定向保存） |
```

---

## 重要注意事项

### ⚠️ 栈空间限制问题

**问题描述**：
- 打点数据存储在栈空间中
- 打点数量过多可能导致栈空间不足
- 表现为运行时栈溢出报错

**解决方案**：

**方案 1: 撤销无效打点**

```cpp
// 在 aicpu_perf.h 中删除不必要的打点事件
#define PERF_TRACES                           \
    X(BEGIN)                                  \
    // XDEVTASK(UNUSED_TRACE_EVENT)  ← 删除无效打点
    X(EXIT)                                   \
    X(MAX)
```

**方案 2: 修改 PERF_TRACE_COUNT_DEVTASK_MAX_NUM**

```cpp
// 在 aicpu_perf.h 中修改最大数量
inline constexpr uint32_t PERF_TRACE_COUNT_DEVTASK_MAX_NUM = 10;  // 改小
```

**⚠️ 注意**：修改 `PERF_TRACE_COUNT_DEVTASK_MAX_NUM` 需要重新编译 framework 代码。

### ⚠️ 数据采集优化

**减少采集数据量**：

在 `dump_device_perf.cpp` 的 `DumpAicpuPerfInfo` 函数中，可以注释掉部分数据采集：

```cpp
void DumpAicpuPerfInfo(DeviceArgs& args, const std::vector<void*>& perfData, uint32_t freq, bool isLast)
{
    // ...
    DumpAicpuDevTask(args, aicpuPrefArray, freq, sumRoundNum);
    // DumpAicoreDecTask(args, aicpuPrefArray, perfData, freq, sumRoundNum);  ← 注释掉减少采集
    DumpAicoreDevTask(args, aicpuPrefArray, perfData, freq, sumRoundNum);
    // ...
}
```

**⚠️ 注意**：注释掉 `DumpAicoreDecTask` 会减少采集的数据量，可能影响某些分析。

### ⚠️ 打点插入原则

**禁止行为**：
- ❌ 禁止在同一位置插入重复打点
- ❌ 禁止在频繁调用的循环内部插入过多打点
- ❌ 禁止超过 `PERF_TRACE_COUNT_DEVTASK_MAX_NUM` 限制
- ❌ 禁止在非关键路径插入过多打点

**推荐行为**：
- ✅ 优先在高耗时代码段插入打点
- ✅ 使用成对的 `PerfMtBegin/PerfMtEnd` 标记时间段
- ✅ 每次只增加必要的打点，避免过度细化
- ✅ 及时清理无效或低价值的打点

### ⚠️ 编译和运行顺序

**编译顺序**：
1. 修改 `aicpu_perf.h` 添加新事件定义
2. 修改代码文件插入打点代码
3. **设置 ASCEND_CANN_PACKAGE_PATH 环境变量**（关键！）
4. 编译 framework 代码（C++ 代码必须重新编译）
5. 设置性能采集环境变量
6. 运行算子采集数据

**⚠️ 重要提示**：

1. **修改 `aicpu_perf.h` 或任何 C++ 代码后，必须重新编译！**

2. **编译前必须设置 CANN 路径**：
   ```bash
   # 错误：未设置 CANN 路径会导致精度问题
   cmake -S . -B build  # ❌ 缺少 ASCEND_CANN_PACKAGE_PATH
   
   # 正确：必须指定 CANN 路径
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   export PYPTO_BUILD_EXT_ARGS="--cmake-options=-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0"
   python3 -m pip install -e . --no-build-isolation  # ✅ 正确
   ```

3. **编译后验证**：
   ```bash
   # 确认 BUILD_WITH_CANN=ON
   grep BUILD_WITH_CANN build/CMakeCache.txt
   # 预期输出: BUILD_WITH_CANN:BOOL=ON
   
   # 验证精度正确
   python examples/01_beginner/basic/basic_ops.py elementwise_ops::test_elementwise_ops --run_mode npu
   ```

---

## 参考资料

### 打点系统文件

| 文件 | 路径 | 功能 |
|------|------|------|
| AICPU 打点定义 | framework/src/interface/machine/device/tilefwk/aicpu_perf.h | PERF_TRACES 宏定义 |
| AICore 打点定义 | framework/src/interface/machine/device/tilefwk/aicpu_common.h | AicorePerfTrace 枚举 |
| PerfMtTrace 实现 | framework/src/machine/device/dynamic/device_perf.h | 打点记录函数 |
| 调度核心函数 | framework/src/machine/device/dynamic/aicore_manager.h | DispatchAiCoreTask 等 |
| 数据导出函数 | framework/src/machine/runtime/dump_device_perf.cpp | DumpAicpuPerfInfo |

### 调度流程详解

参见：[references/schedule-flow-detail.md](references/schedule-flow-detail.md)

### 打点设计指南

参见：[references/trace-design-guide.md](references/trace-design-guide.md)

### 性能分析案例

参见：[references/perf-analysis-cases.md](references/perf-analysis-cases.md)

---

## 常见问题

### 问题 1: 如何判断需要细化到哪一层级？

**判断方法**：
- Level 1: 当主要流程耗时占比 >30% 时，细化到流程内部
- Level 2: 当子流程耗时占比 >20% 时，继续细化
- Level 3: 当具体函数耗时占比 >10% 时，进一步细化
- 终止: 当最细粒度打点无明显瓶颈时，停止细化

### 问题 2: 打点数量超过了限制怎么办？

**解决方案**：
1. 分析现有打点的价值，删除低价值打点
2. 合并相似打点，减少打点数量
3. 修改 `PERF_TRACE_COUNT_DEVTASK_MAX_NUM` 增大限制（需要重新编译）
4. 分批细化，先分析一批打点，清理后再添加新批

### 问题 3: 如何分析波动原因？

**分析方法**：
1. 检查任务数量是否波动（不同任务依赖数不同）
2. 检查并发竞争（锁等待时间）
3. 检查硬件状态（cache miss、内存延迟）
4. 检查调度策略（策略切换次数）
5. 检查核心状态（空闲/忙碌状态变化）

### 问题 4: 性能数据文件在哪里？

**文件位置**：
```
output/output_时间戳/machine_trace_perf_data_*.json
output/output_时间戳/machine_runtime_operator_trace_*.json
output/output_时间戳/merged_swimlane.json
```

### 问题 5: 如何分析性能数据？

**使用分析脚本**：

```bash
python3 .agents/skills/pypto-schedule-trace-refine/scripts/schedule_trace_analysis.py \
    output/output_*/machine_trace_perf_data_*.json

# 保存报告到文件
python3 .agents/skills/pypto-schedule-trace-refine/scripts/schedule_trace_analysis.py \
    output/output_*/machine_trace_perf_data_*.json > schedule_perf_report.md
```

查看终端输出或将报告重定向保存为 Markdown 文件。

---

## 总结

本技能提供完整的调度性能细化打点流程，帮助用户：
1. ✅ 精准定位调度流程中的性能瓶颈
2. ✅ 分析调度性能波动原因
3. ✅ 提供细粒度打点设计指导
4. ✅ 自动化性能数据分析和报告生成
5. ✅ 提供优化建议和实施指导

通过迭代式细化打点策略，逐步深入分析调度流程，最终定位到具体瓶颈点并提供优化建议。