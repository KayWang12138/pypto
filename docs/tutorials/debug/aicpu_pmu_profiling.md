# AICPU PMU 性能采集技术方案

## 概述

本方案基于 Linux `perf_event_open` 系统调用实现 AICPU 侧的 PMU（Performance Monitoring Unit）性能数据采集，用于分析算子在 AICPU 上的执行性能瓶颈。

核心特性：
- **即插即用**：提供宏接口，仅需在目标代码段前后插入打点即可自动采集
- **零侵入设计**：当 PMU 不可用时自动降级为纯时间测量模式
- **多维度分析**：覆盖 CPU、分支、缓存、系统四大类指标
- **事件组采样**：利用 perf event group 机制保证多事件同步采集

---

## 实现原理

### perf_event_open 机制

PMU 采集基于 Linux 内核提供的 `perf_event_open` 系统调用：

```c
int fd = syscall(__NR_perf_event_open, &perf_event_attr, pid, cpu, group_fd, flags);
```

关键参数：
- `perf_event_attr`：配置事件类型、计数器属性
- `pid`：目标线程 TID（使用 `gettid()` 获取当前线程）
- `group_fd`：事件组首领 fd，用于实现多事件同步采样
- `flags`：附加控制选项

### 事件组机制

为确保多个 PMU 事件同时开始、同时结束计数，采用 **事件组（Event Group）** 机制：

1. 第一个事件作为 **组首领**，`group_fd = -1`
2. 后续事件加入该组，`group_fd = 首领fd`
3. 通过 `ioctl` 对首领 fd 操作可同时控制整组事件：
   ```c
   ioctl(groupFd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);  // 同时启用
   ioctl(groupFd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP); // 同时禁用
   ioctl(groupFd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);   // 同时清零
   ```
4. 读取时通过 `PERF_FORMAT_GROUP` 格式一次读取所有计数

### 采集流程

```
┌─────────────────────────────────────────────────────────────┐
│                    PMU 采集完整流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   构造阶段                                                    │
│   ┌─────────────────────────────────────────────┐           │
│   │  AicpuPerfEventSampler()                    │           │
│   │  ├── 获取当前线程 TID                        │           │
│   │  ├── 注册 10 个 PMU 事件（事件组）            │           │
│   │  └── 记录可用事件数量                        │           │
│   └─────────────────────────────────────────────┘           │
│                         ↓                                    │
│   Begin()                                                    │
│   ├── ioctl RESET + ENABLE（清零并启用事件组）   │           │
│   ├── 记录起始 cycles（高精度时间戳）            │           │
│   └───────────────────────────────────────────┘             │
│                         ↓                                    │
│   ┌─────────────────────────────────────────────┐           │
│   │         目标代码段执行                        │           │
│   │         （PMU 硬件计数器持续计数）            │           │
│   └─────────────────────────────────────────────┘           │
│                         ↓                                    │
│   End()                                                      │
│   ├── 计算执行时间（cycles 差值）               │           │
│   ├── ioctl DISABLE（停止事件组）               │           │
│   └───────────────────────────────────────────┘             │
│                         ↓                                    │
│   Dump()                                                     │
│   ├── read() 读取所有事件计数                   │           │
│   ├── 计算衍生指标（IPC、CPI、命中率等）         │           │
│   └── 分类别打印性能报告                        │           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件说明

### GroupEvent（事件组管理器）

负责 PMU 事件的注册、启用、禁用和读取：

| 方法 | 功能 |
|------|------|
| `AddEvent(type, config, name)` | 注册单个 PMU 事件到事件组 |
| `Enable()` | 启用整组事件计数（RESET + ENABLE） |
| `Disable()` | 禁用整组事件计数 |
| `Read(counts)` | 读取所有事件的计数值 |

### AicpuPerfEventSampler（采集器主体）

提供完整的采集生命周期管理：

| 方法 | 功能 |
|------|------|
| `Begin()` | 开始采集：启用计数器 + 记录起始时间 |
| `End()` | 结束采集：停止计数器 + 计算执行时间 |
| `Dump()` | 输出报告：读取计数 + 计算指标 + 打印 |

### AicpuPerfScopedSampler（作用域采集器）

RAII 封装，实现自动采集：

- 构造时自动 `Begin()`
- 析构时自动 `End() + Dump()`
- 支持提前 `return` 场景自动收口

---

## PMU 事件详解

本方案采集 10 个标准 PMU 事件，分为四大类：

### 1. CPU 核心指标

| 事件 | 类型 | 含义 | 分析价值 |
|------|------|------|----------|
| **CPU_CYCLES** | Hardware | CPU 时钟周期数 | 基础时间度量，计算 IPC/CPI |
| **INSTRUCTIONS** | Hardware | 执行指令数 | 代码效率指标，与 cycles 组合判断 CPU 利用率 |
| **STALL_FRONTEND** | Hardware | 前端停顿周期 | CPU 取指/解码单元阻塞时间，指示指令供给瓶颈 |
| **STALL_BACKEND** | Hardware | 后端停顿周期 | CPU 执行单元阻塞时间，指示执行资源瓶颈 |

**衍生指标**：
- **IPC** (Instructions Per Cycle) = `INSTRUCTIONS / CPU_CYCLES`
  - IPC > 1：超标量执行效率高
  - IPC < 1：存在停顿或内存瓶颈
- **CPI** (Cycles Per Instruction) = `CPU_CYCLES / INSTRUCTIONS`
  - CPI = 1：理想单发射
  - CPI > 1：存在性能瓶颈
- **前端停顿占比** = `STALL_FRONTEND / CPU_CYCLES * 100%`
- **后端停顿占比** = `STALL_BACKEND / CPU_CYCLES * 100%`

### 2. 分支预测指标

| 事件 | 类型 | 含义 | 分析价值 |
|------|------|------|----------|
| **BRANCH_INSTRUCTIONS** | Hardware | 分支指令数 | 代码分支密度指标 |
| **BRANCH_MISSES** | Hardware | 分支预测失败数 | 预测准确度，影响流水线效率 |

**衍生指标**：
- **分支预测失败率** = `BRANCH_MISSES / BRANCH_INSTRUCTIONS * 100%`
  - < 5%：良好
  - > 10%：可能需要优化分支逻辑或调整代码布局

### 3. 缓存指标

| 事件 | 类型 | 含义 | 分析价值 |
|------|------|------|----------|
| **CACHE_REFERENCES** | Hardware | 缓存访问次数 | 内存访问频率指标 |
| **CACHE_MISSES** | Hardware | 缓存失效次数 | 数据局部性问题程度 |

**衍生指标**：
- **缓存命中率** = `(1 - CACHE_MISSES / CACHE_REFERENCES) * 100%`
  - > 95%：数据局部性好
  - < 80%：存在严重缓存问题，需优化数据布局/访问模式
- **缓存失效率** = `CACHE_MISSES / CACHE_REFERENCES * 100%`

### 4. 系统指标

| 事件 | 类型 | 含义 | 分析价值 |
|------|------|------|----------|
| **CONTEXT_SWITCHES** | Software | 上下文切换次数 | CPU 抢占/调度开销，指示多任务干扰程度 |
| **PAGE_FAULTS** | Software | 页错误次数 | 内存缺页次数，指示内存访问异常或初次访问开销 |

---

## 使用方式

### 方式一：作用域自动采集（推荐）

使用 `AICPU_PMU_SCOPE` 宏，一行代码完成采集：

```cpp
void MyFunction() {
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_SCOPE("MyFunction");
#endif
#endif
    
    // 目标代码...
    
    if (some_condition) {
        return;  // 提前返回也能自动 End + Dump
    }
    
    // 更多代码...
}
```

特点：
- RAII 自动管理，无需手动调用 `End`
- 支持提前 `return` 场景
- 适合函数级或代码块级采集

### 方式二：手动前后打点

使用 `AICPU_PMU_BEGIN` / `AICPU_PMU_END` 宏：

```cpp
void MyFunction() {
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_BEGIN(mySampler);
#endif
#endif
    
    // 目标代码段 A...
    
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_END(mySampler, "SegmentA");
#endif
#endif
    
    // 其他代码段 B（不采集）...
    
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_BEGIN(mySampler2);
#endif
#endif
    
    // 目标代码段 C...
    
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_END(mySampler2, "SegmentC");
#endif
#endif
}
```

特点：
- 精确控制采集范围
- 可在同函数内分段采集
- 需确保 `BEGIN/END` 配对使用

### 方式三：直接使用类接口

```cpp
void MyFunction() {
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    npu::tile_fwk::AicpuPerfEventSampler sampler;
    sampler.Begin();
    
    // 目标代码...
    
    sampler.End();
    DEV_INFO("[AICPU_PMU] MyFunction");
    sampler.Dump();
#endif
#endif
}
```

### 方式四：跨函数采集

使用 `AICPU_PMU_BEGIN_EXTERNAL` / `AICPU_PMU_END_EXTERNAL` 宏，通过指针传递实现跨函数采集：

```cpp
// 头文件中定义共享采样器
class MyController {
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    npu::tile_fwk::AicpuPerfEventSampler* pmuSampler_{nullptr};
#endif
#endif
    
public:
    void StartPmuSampling();
    void EndPmuSampling();
};

// 实现文件
void MyController::StartPmuSampling() {
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    pmuSampler_ = new npu::tile_fwk::AicpuPerfEventSampler();
    AICPU_PMU_BEGIN_EXTERNAL(pmuSampler_);
#endif
#endif
}

void MyController::EndPmuSampling() {
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_END_EXTERNAL(pmuSampler_, "CrossFunction");
    delete pmuSampler_;
    pmuSampler_ = nullptr;
#endif
#endif
}

// 使用示例
void FunctionA(MyController* ctrl) {
    ctrl->StartPmuSampling();
    // 代码段 A...
}

void FunctionB(MyController* ctrl) {
    // 代码段 B（继续采集）...
}

void FunctionC(MyController* ctrl) {
    // 代码段 C（继续采集）...
    ctrl->EndPmuSampling();  // 在任意函数结束采集
}
```

特点：
- 通过指针传递，可在多个函数间延续采集
- 适合跨模块、跨调用层次的性能分析
- 需手动管理采样器生命周期（创建/销毁）

**注意事项**：
- `AicpuPerfEventSampler` 绑定当前线程 TID，仅支持同线程内跨函数采集
- 若跨线程采集，需确保 `BEGIN` 和 `END` 在同一线程执行
- 需要手动管理内存（`new/delete`），避免内存泄漏

### 条件编译说明

所有 PMU 相关代码需包裹在双重条件编译中：

```cpp
#ifdef __DEVICE__           // 仅在设备侧编译
#if AICPU_PMU_EVENT_ENABLE  // PMU 功能开关
    // PMU 采集代码
#endif
#endif
```

当 `AICPU_PMU_EVENT_ENABLE = 0` 时，所有宏展开为空，零性能开销。

---

## 输出报告格式

### 正常输出（PMU 可用）

```
[AICPU_PMU] ExecDyn
[AICPU_PMU] Performance Report
============================================================
Total Running Time: 1234.56 us (1.23 ms)
------------------------------------------------------------
CPU Metrics
------------------------------------------------------------
  CPU Cycles:         1,234,567
  Instructions:       2,345,678
  IPC:                1.90
  CPI:                0.53
  Stall Frontend:     123,456 (10.0%)
  Stall Backend:      234,567 (19.0%)
------------------------------------------------------------
Branch Metrics
------------------------------------------------------------
  Branch Instructions: 456,789
  Branch Misses:       12,345
  Branch Miss Rate:    2.71%
------------------------------------------------------------
Cache Metrics
------------------------------------------------------------
  Cache References:   789,012
  Cache Misses:       23,456
  Cache Hit Rate:     97.03%
------------------------------------------------------------
System Metrics
------------------------------------------------------------
  Context Switches:   0
  Page Faults:        0
============================================================
```

###降级输出（PMU 不可用）

```
[AICPU_PMU] ExecDyn Summary (PMU unavailable)
  Total Running Time: 1234.56 us (1.23 ms)
  Note: PMU events disabled due to permission restrictions
```

---

## 权限与环境要求

### 系统要求

1. **Linux 内核**：支持 `perf_event_open` 系统调用
2. **硬件**：CPU 需具备 PMU 硬件计数器（ARM64 通常具备）
3. **权限**：
   - 非容器环境：通常默认可访问
   - 容器环境：需配置 `--cap-add=PERFMON` 或 `--privileged`
   - 受限环境：自动降级为纯时间测量

### 编译开关

在 `framework/src/machine/utils/device_switch.h` 中配置：

```c
#define AICPU_PMU_EVENT_ENABLE 1  // 启用 PMU 采集
#define AICPU_PMU_EVENT_ENABLE 0  // 禁用 PMU 采集

#define AICPU_PMU_DIRECT_ACCESS 1 // 启用 ARM PMU 寄存器直读模式（零系统调用）
#define AICPU_PMU_DIRECT_ACCESS 0 // 使用默认 perf_event_open 模式
```

---

## 直读寄存器模式（低开销路径）

当 `AICPU_PMU_DIRECT_ACCESS = 1` 且运行在 `aarch64` 架构上时，采样器会切换为 `ArmPmuDirectSampler`，通过 ARMv8 PMUv3 的 `MRS/MSR` 指令直接访问 PMU 寄存器，**完全绕过 `ioctl`/`read` 系统调用**。

### 开销对比

| 操作 | `perf_event_open` 方式 | 直读寄存器方式 |
|------|------------------------|----------------|
| `Begin` | `ioctl(RESET)` + `ioctl(ENABLE)` ≈ 数百 ns | ISB + 若干 MRS ≈ **10~20 ns** |
| `End`   | `ioctl(DISABLE)` + 下次 `read` ≈ 数百 ns | ISB + 若干 MRS ≈ **10~20 ns** |
| 单次读计数器 | `read()` 系统调用 ≈ 200~500 ns | `MRS` 指令 ≈ **2~3 ns** |

对于高频采集或极短代码段（< 1 us），直读方式收益显著。

### 启用前提：内核权限配置

ARMv8 要求内核显式开启 `PMUSERENR_EL0.EN` 位才能让 EL0（用户态）访问 PMU 寄存器。以下任选其一：

**方式 A：Linux 5.17+ 原生开关**

```bash
echo 1 | sudo tee /proc/sys/kernel/perf_user_access
```

**方式 B：加载自定义内核模块**

在每个 CPU 上设置 `PMUSERENR_EL0`：

```c
u64 val = 0x0f;  // EN | SW | CR | ER
asm volatile("msr pmuserenr_el0, %0" :: "r" (val));
```

**方式 C：AICPU 固件侧初始化**

在 device 启动阶段直接写入 `PMUSERENR_EL0`（需特权代码）。

### 运行时行为

- 构造时自动调用 `ProbeAvailable()` 读取 `PMUSERENR_EL0`：
  - 若 `EN=1`：正常工作，`DEV_INFO` 输出 `Direct register access enabled, N hardware counters active`
  - 若 `EN=0`：`DEV_WARN` 提示，`Dump()` 退化为仅打印时间信息
- 自动探测 `PMCR_EL0.N`，适配实际硬件计数器数量（典型值 4~6）
- 开启 `PMCR_LC=1`，周期计数器以 64-bit 工作，避免短任务场景下的溢出

### 支持的事件集（默认 6 个）

| 索引 | ARMv8 Event ID | 含义 |
|------|----------------|------|
| 0 | `0x08` INST_RETIRED | 指令退休数 |
| 1 | `0x04` L1D_CACHE | L1 数据缓存访问 |
| 2 | `0x03` L1D_CACHE_REFILL | L1 数据缓存失效 |
| 3 | `0x12` BR_PRED | 分支预测数 |
| 4 | `0x10` BR_MIS_PRED | 分支预测失败 |
| 5 | `0x24` STALL_BACKEND | 后端停顿周期 |

外加专用周期计数器 `PMCCNTR_EL0`。

### 使用方式

宏接口完全一致，**无需修改业务代码**：

```cpp
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_SCOPE("MyHotPath");
#endif
#endif
```

框架通过 `AicpuPmuSampler` 类型别名自动切换到 `ArmPmuDirectSampler` 或 `AicpuPerfEventSampler`。

### 注意事项

1. **CPU 绑定**：PMU 寄存器是 per-CPU 的，若线程在 `Begin/End` 之间被迁移到其他核，计数将不准确。建议通过 `sched_setaffinity` 绑核，或在采集前后校验 `sched_getcpu()`。
2. **计数器宽度**：通用计数器默认 32-bit，高频事件在长时间段（> 1~4 秒）可能溢出，短任务采集不受影响。
3. **安全性**：开启 EL0 直读后，任何用户进程都能读取 PMU 计数器，生产环境需评估侧信道风险。
4. **Fallback**：运行时若权限不足，会透明退化，但不会自动切换到 `perf_event_open`。如需强兼容，保持 `AICPU_PMU_DIRECT_ACCESS = 0`。

---

## 性能瓶颈诊断指南

### CPU 效率问题

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| IPC < 0.5 | 严重低效 | CPU 利用率极低 | 检查停顿占比、缓存命中率 |
| IPC 0.5~1.0 | 中等效率 | 存在瓶颈 | 分析前端/后端停顿来源 |
| IPC > 1.0 | 高效率 | 超标量执行良好 | 继续保持 |

### 前端瓶颈

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| 前端停顿 > 20% | 取指瓶颈 | 指令供给不足 | 检查代码密度、分支密度 |
| 前端停顿 10%~20% | 中等问题 | 存在取指延迟 | 优化代码布局 |

**典型原因**：
- 分支密度过高，导致取指单元频繁切换路径
- 代码分布分散，cache line 利用率低
- 复杂条件跳转，预测失败导致流水线清空

### 后端瓶颈

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| 后端停顿 > 30% | 执行瓶颈 | 执行单元阻塞严重 | 重点检查缓存命中率 |
| 后端停顿 20%~30% | 中等问题 | 存在执行延迟 | 分析缓存指标 |

**典型原因**：
- 缓存失效导致数据等待
- 内存带宽瓶颈
- 长依赖链导致流水线停顿

### 分支问题

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| 失败率 > 10% | 严重分支问题 | 预测准确度低 | 优化分支逻辑、减少条件判断 |
| 失败率 5%~10% | 中等问题 | 需要改进 | 调整分支布局、使用 likely/unlikely |

### 缓存问题

| 懈标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| 命中率 < 80% | 严重缓存问题 | 数据局部性差 | 优化数据布局、减少随机访问 |
| 命中率 80%~90% | 中等问题 | 存在缓存失效 | 检查数据访问模式 |
| 命中率 > 95% | 良好 | 数据局部性好 | 继续保持 |

**典型优化措施**：
- 数据结构对齐到 cache line（64B）
- 减少指针跳跃访问
- 提升数据访问顺序性
- 减少数据体积，提升 L1/L2 利用率

### 系统干扰

| 懈标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| 上下文切换 > 10 | 系统干扰 | CPU 被抢占频繁 | 减少并发、提升 CPU 亲和性 |
| 页错误 > 100 | 内存问题 | 缺页频繁 | 预分配内存、减少动态分配 |

---

## 最佳实践

### 1. 采集位置选择

推荐采集点：
- 算子入口函数（如 `ExecDyn`）
- 关键计算循环
- 数据搬运密集区域
- 控制流密集区域

不推荐采集点：
- 极短代码段（< 100us，PMU 计数可能不足）
- 包含大量系统调用的代码段（干扰 PMU 计数）

### 2. 多次采集对比

建议对同一代码段多次采集，对比指标稳定性：

```cpp
for (int i = 0; i < 5; i++) {
    AICPU_PMU_SCOPE("Iteration_" + std::to_string(i));
    TargetCode();
}
```

### 3. 与泳道图配合

PMU 采集可与 `PerfBegin/PerfEnd` 泳道图配合使用：

```cpp
PerfBegin(PERF_EVT_MY_OP);
AICPU_PMU_SCOPE("MyOpDetail");
MyOperator();
PerfEnd(PERF_EVT_MY_OP);
```

泳道图提供宏观时间分布，PMU 提供微观 CPU 指标。

### 4. 编译版本选择

建议使用 **Debug 版本** 进行 PMU 分析：
- Debug 版本保留完整符号，便于定位问题代码
- Release 版本可能因优化导致 PMU 指标与实际逻辑不符

---

## 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| PMU 采集实现 | `framework/src/machine/utils/perf_event_sampler.h` | 完整实现代码 |
| 功能开关 | `framework/src/machine/utils/device_switch.h` | `AICPU_PMU_EVENT_ENABLE` 定义 |
| 使用示例 | `framework/src/machine/device/dynamic/device_ctrl.h` | `ExecDyn` 函数中的使用示例 |

---

## 扩展方向

### 1. 支持更多事件类型

当前仅支持 10 个标准事件，可扩展：
- `PERF_TYPE_CACHE`：详细 L1/L2/LLC 缓存事件
- `PERF_TYPE_RAW`：CPU 特定架构事件（需查阅 CPU 手册）
- `PERF_TYPE_TRACEPOINT`：内核跟踪点事件

### 2. 数据持久化

当前仅打印日志，可扩展：
- 输出到文件（JSON/CSV 格式）
- 与 profiling 工具集成
- 支持远程采集与分析

### 3. 多线程采集

当前仅支持单线程采集，可扩展：
- 支持指定目标线程 TID
- 支持多线程同步采集与汇总

---

## 参考资料

- [Linux perf_event_open 文档](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [ARM64 PMU 架构手册](https://developer.arm.com/documentation/)
- [perf 工具使用指南](https://perf.wiki.kernel.org/index.php/Tutorial)