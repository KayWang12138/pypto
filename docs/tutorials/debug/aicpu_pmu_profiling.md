# AICPU PMU 性能采集技术方案

## 概述

本方案提供两种 AICPU 侧的 PMU（Performance Monitoring Unit）性能数据采集能力，用于分析算子在 AICPU 上的执行性能瓶颈：

| 路径 | 编译开关 | 实现方式 | 适用场景 |
|------|----------|----------|----------|
| **perf_event_open 路径** | `AICPU_PMU_EVENT_ENABLE` | Linux syscall + ioctl + read | 兼容性好，通用 Linux 环境 |
| **直读寄存器路径** | `ARM_PMU_DIRECT_ENABLE` | MRS/MSR 指令直接访问 PMU 寄存器 | 极低开销，高频采集场景 |

核心特性：

- **即插即用**：提供宏接口，仅需在目标代码段前后插入打点即可自动采集
- **零侵入设计**：当 PMU 不可用时自动降级为纯时间测量模式
- **多维度分析**：覆盖 CPU、分支、缓存三大类指标
- **事件组采样**：利用 perf event group 机制保证多事件同步采集

---

## 系统架构

### Host 与 NPU Device 环境隔离

```
┌─────────────────────────────────────────────────────────────────────┐
│                         系统架构                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────────────────────────────────────┐          │
│   │            Host 服务器（Linux 内核）                   │          │
│   │                                                      │          │
│   │   • 640 个 ARM64 CPU 核心                            │          │
│   │   • 算子控制代码在这里执行                            │          │
│   │   • perf_event_open 路径可用 ✅                      │          │
│   │   • 直读寄存器需要 PMUSERENR_EL0.EN 设置             │          │
│   └─────────────────────────────────────────────────────┘          │
│                          ↕ PCIe / 通信通道                           │
│   ┌─────────────────────────────────────────────────────┐          │
│   │            NPU Device（独立操作系统）                  │          │
│   │                                                      │          │
│   │   ┌───────────────────────────────────────┐         │          │
│   │   │  AICPU（独立 ARMv8 核心）               │         │          │
│   │   │                                        │         │          │
│   │   │  • 算子执行代码在这里运行 ✅            │         │          │
│   │   │  • 与 Host 环境完全隔离                 │         │          │
│   │   │  • 需要独立设置 PMU 权限                │         │          │
│   │   └───────────────────────────────────────┘         │          │
│   │                                                      │          │
│   │   ┌───────────────────────────────────────┐         │          │
│   │   │  AICore（矩阵计算单元）                 │         │          │
│   │   └───────────────────────────────────────┘         │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**关键注意**：AICPU 和 Host 不共享内核！NPU Device 有自己独立的操作系统。因此在 Host 上加载的内核模块不会影响 AICPU 环境，需要在 NPU Device 内单独处理。

---

## perf_event_open 路径

### 实现原理

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
│   │  ├── 获取当前线程 TID (gettid())             │           │
│   │  ├── 注册 8 个 PMU 事件（事件组）             │           │
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

## 直读寄存器路径（低开销）

### 实现原理

通过 ARMv8 `MRS/MSR` 指令直接访问 PMU 寄存器，完全绕过系统调用：

```c
// 读周期计数器
uint64_t v;
asm volatile("mrs %0, pmccntr_el0" : "=r"(v));

// 写事件类型选择器
asm volatile("msr pmevtyper0_el0, %0" :: "r"(evt));
```

### 开销对比

| 操作 | perf_event_open 方式 | 直读寄存器方式 |
|------|----------------------|----------------|
| `Begin` | `ioctl(RESET)` + `ioctl(ENABLE)` ≈ **数百 ns** | ISB + 若干 MRS ≈ **10~20 ns** |
| `End` | `ioctl(DISABLE)` + `read()` ≈ **数百 ns** | ISB + 若干 MRS ≈ **10~20 ns** |
| 单次读计数器 | `read()` 系统调用 ≈ **200~500 ns** | `MRS` 指令 ≈ **2~3 ns** |

对于高频采集或极短代码段（< 1 us），直读方式收益显著。

### 启用前提：PMUSERENR_EL0 权限配置

ARMv8 要求内核显式开启 `PMUSERENR_EL0.EN` 位才能让 EL0（用户态）访问 PMU 寄存器。

**重要**：由于 NPU Device 是独立操作系统，需要在 **NPU Device 内** 设置权限，而不是 Host 服务器。

#### 方式 A：NPU Device 内加载内核模块

将 `pmu_user_access.ko` 嵌入代码，在 device 启动时自动加载：

```cpp
#include "pmu_user_access_ko_embedded.h"  // 由 embed_ko_to_code.sh 生成
#include "pmu_ko_loader.h"

// 在 device 初始化阶段调用
npu::tile_fwk::PmuInitEmbeddedKo();
```

生成嵌入数据的步骤：

```bash
# 1. 生成 pmu_user_access.ko 内核模块
cd tools/scripts/pmu_user_access
make

# 2. 将 ko 文件转换为 C 数组
./embed_ko_to_code.sh pmu_user_access.ko \
    ../../framework/src/machine/utils/pmu_user_access_ko_embedded.h

# 3. 编译 device 代码（自动包含嵌入数据）
```

#### 方式 B：AICPU 固件侧初始化

在 AICPU 固件启动代码（EL1 特权级）中添加：

```c
void aicpu_pmu_init(void)
{
    // 设置 PMUSERENR_EL0 = 0x0F（允许 EL0 访问 PMU）
    uint64_t val = 0x0F;
    __asm__ volatile(
        "msr pmuserenr_el0, %0\n"
        "isb\n"
        :: "r"(val)
    );
}
```

### 运行时行为

- 构造时自动调用 `ProbeAvailable()` 读取 `PMUSERENR_EL0`：
  - 若 `EN=1`：正常工作，日志输出 `[ARM_PMU_DIRECT] Enabled, N hardware counters active`
  - 若 `EN=0`：日志警告，`Dump()` 退化为仅打印时间信息
- 自动探测 `PMCR_EL0.N`，适配实际硬件计数器数量（典型值 4~6）
- 开启 `PMCR_LC=1`，周期计数器以 64-bit 工作，避免溢出

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

### 注意事项

1. **CPU 绑定**：PMU 寄存器是 per-CPU 的，若线程在 `Begin/End` 之间被迁移到其他核，计数将不准确
2. **计数器宽度**：通用计数器默认 32-bit，长时间段（> 1~4 秒）可能溢出，短任务采集不受影响
3. **安全性**：开启 EL0 直读后，任何用户进程都能读取 PMU 计数器，生产环境需评估侧信道风险
4. **Fallback**：运行时若权限不足，会透明退化，但不会自动切换到 perf_event_open 路径

---

## PMU 事件详解

### perf_event_open 路径（8 个事件）

| 索引 | 事件 | perf 类型 | 含义 |
|------|------|------|------|
| 0 | CPU_CYCLES | `PERF_COUNT_HW_CPU_CYCLES` | CPU 时钟周期数 |
| 1 | INSTRUCTIONS | `PERF_COUNT_HW_INSTRUCTIONS` | 执行指令数 |
| 2 | BRANCH_INSTRUCTIONS | `PERF_COUNT_HW_BRANCH_INSTRUCTIONS` | 分支指令数 |
| 3 | BRANCH_MISSES | `PERF_COUNT_HW_BRANCH_MISSES` | 分支预测失败数 |
| 4 | L1D_CACHE_REFS | `PERF_COUNT_HW_CACHE_L1D (READ, ACCESS)` | L1 数据缓存读访问 |
| 5 | L1D_CACHE_MISSES | `PERF_COUNT_HW_CACHE_L1D (READ, MISS)` | L1 数据缓存读失效 |
| 6 | LL_CACHE_REFS | `PERF_COUNT_HW_CACHE_LL (READ, ACCESS)` | 末级缓存读访问 |
| 7 | LL_CACHE_MISSES | `PERF_COUNT_HW_CACHE_LL (READ, MISS)` | 末级缓存读失效 |

> **注意**：`PERF_COUNT_HW_CACHE_LL` 是 Linux perf 的 last-level cache 标准枚举，在 ARMv8 系统中可能对应 L2 或 L3，取决于具体硬件。

### 衍生指标

| 指标 | 公式 | 含义 |
|------|------|------|
| IPC | `INSTRUCTIONS / CPU_CYCLES` | 每周期执行指令数 |
| CPI | `CPU_CYCLES / INSTRUCTIONS` | 每条指令消耗周期数 |
| Branch Miss Rate | `BRANCH_MISSES / BRANCH_INSTRUCTIONS * 100%` | 分支预测失败率 |
| L1D Cache Miss Rate | `L1D_CACHE_MISSES / L1D_CACHE_REFS * 100%` | L1D 读 miss 率 |
| LL Cache Miss Rate | `LL_CACHE_MISSES / LL_CACHE_REFS * 100%` | 末级 cache 读 miss 率 |

---

## 使用方式

### perf_event_open 路径

使用 `AICPU_PMU_SCOPE` 等宏：

```cpp
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_SCOPE("ExecDyn");
#endif
#endif
```

手动打点：

```cpp
#ifdef __DEVICE__
#if AICPU_PMU_EVENT_ENABLE
    AICPU_PMU_BEGIN(mySampler);
    // 目标代码...
    AICPU_PMU_END(mySampler, "MySegment");
#endif
#endif
```

### 直读寄存器路径

使用 `ARM_PMU_DIRECT_SCOPE` 等宏：

```cpp
#ifdef __DEVICE__
#if ARM_PMU_DIRECT_ENABLE
    ARM_PMU_DIRECT_SCOPE("ExecDyn");
#endif
#endif
```

手动打点：

```cpp
#ifdef __DEVICE__
#if ARM_PMU_DIRECT_ENABLE
    ARM_PMU_DIRECT_BEGIN(mySampler);
    // 目标代码...
    ARM_PMU_DIRECT_END(mySampler, "MySegment");
#endif
#endif
```

---

## 编译开关配置

在 `framework/src/machine/utils/device_switch.h` 中配置：

```c
// perf_event_open 路径开关
#define AICPU_PMU_EVENT_ENABLE 1  // 启用
#define AICPU_PMU_EVENT_ENABLE 0  // 禁用

// 直读寄存器路径开关
#define ARM_PMU_DIRECT_ENABLE 1   // 启用
#define ARM_PMU_DIRECT_ENABLE 0   // 禁用

// device 初始化阶段自动写出并加载内嵌 pmu_user_access.ko
#define PMU_USER_ACCESS_KO_AUTO_LOAD ARM_PMU_DIRECT_ENABLE
```

两套开关相互独立，可同时启用或分别启用。

---

## 输出报告格式

### perf_event_open 路径输出

```text
[AICPU_PMU] ExecDyn
[AICPU_PMU] Performance Report
============================================================
Total Running Time: 426.00 us (0.43 ms)
------------------------------------------------------------
PMU Event Counters
------------------------------------------------------------
  CPU Cycles:         375,656
  Instructions:       349,908
  Branch Instructions:79,604
  Branch Misses:      2,920
  L1D Cache Refs:     120,735
  L1D Cache Misses:   2,488
  LL Cache Refs:      1,234
  LL Cache Misses:    56
------------------------------------------------------------
Derived Metrics
------------------------------------------------------------
  IPC:                0.93
  CPI:                1.07
  Branch Miss Rate:   3.67%
  L1D Cache Hit Rate: 97.94%
  LL Cache Hit Rate:  95.46%
============================================================
```

### 直读寄存器路径输出

```text
[ARM_PMU_DIRECT] ExecDyn
[ARM_PMU_DIRECT] Performance Report (MRS/MSR mode)
============================================================
Total Running Time: 426.00 us (0.43 ms)
Cycles (PMCCNTR):   375,656
------------------------------------------------------------
CPU Metrics
------------------------------------------------------------
  Instructions:       349,908
  IPC:                0.93
  CPI:                1.07
  Stall Backend:      234,567 (19.0%)
------------------------------------------------------------
Branch Metrics
------------------------------------------------------------
  Branches:           45,678
  Branch Misses:      2,920
  Branch Miss Rate:   6.40%
------------------------------------------------------------
Cache Metrics (L1D)
------------------------------------------------------------
  L1D References:     120,735
  L1D Misses:         2,488
  L1D Miss Rate:      2.06%
============================================================
```

---

## 性能瓶颈诊断指南

### CPU 效率问题

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| IPC < 0.5 | 严重低效 | CPU 利用率极低 | 检查缓存命中率、分支预测失败率 |
| IPC 0.5~1.0 | 中等效率 | 存在瓶颈 | 分析缓存和分支指标 |
| IPC > 1.0 | 高效率 | 超标量执行良好 | 继续保持 |

### 分支问题

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| 失败率 > 10% | 严重分支问题 | 预测准确度低 | 优化分支逻辑、减少条件判断 |
| 失败率 5%~10% | 中等问题 | 需要改进 | 调整分支布局、使用 likely/unlikely |
| 失败率 < 5% | 良好 | 分支预测高效 | 继续保持 |

### 缓存问题

| 指标 | 阈值 | 问题诊断 | 建议措施 |
|------|------|----------|----------|
| L1D 命中率 < 80% | 严重缓存问题 | 数据局部性差 | 优化数据布局、减少随机访问 |
| L1D 命中率 80%~90% | 中等问题 | 存在缓存失效 | 检查数据访问模式 |
| L1D 命中率 > 95% | 良好 | 数据局部性好 | 继续保持 |

---

## 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| perf_event_open 实现 | `framework/src/machine/utils/perf_event_sampler.h` | 8 事件采集 |
| 直读寄存器实现 | `framework/src/machine/utils/arm_pmu_direct_sampler.h` | 6 事件采集 |
| KO 加载器 | `framework/src/machine/utils/pmu_ko_loader.h` | 内嵌模块加载 |
| 功能开关 | `framework/src/machine/utils/device_switch.h` | 开关定义 |
| 使用示例 | `framework/src/machine/device/dynamic/device_ctrl.h` | ExecDyn 中的示例 |
| 内核模块 | `tools/scripts/pmu_user_access/` | pmu_user_access.ko |

---

## 参考资料

- [Linux perf_event_open 文档](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [ARM64 PMU 架构手册](https://developer.arm.com/documentation/)
- [perf 工具使用指南](https://perf.wiki.kernel.org/index.php/Tutorial)