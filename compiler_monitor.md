# Compiler Monitor

PyPTO算子编译过程耗时监控与超时检测功能，用于监控 PyPTO 编译过程中的各阶段耗时，并在超时时进行告警或中断。

---

## 概述

Compiler Monitor 提供编译过程的实时进度监控和超时检测功能。它通过后台监控线程定期检查当前编译阶段的执行时间，当超过配置的超时阈值时，可选择抛出异常中断编译或仅输出警告信息。

### 主要特点

- **自动启用** - 导入 `pypto` 时自动启用，无需额外配置
- **实时进度监控** - 定期打印当前编译阶段的执行进度
- **超时检测** - 支持自定义超时阈值，超时后可选择抛出异常或警告
- **线程安全** - 监控在独立线程中运行，不影响主编译流程
- **资源自动管理** - 通过 RAII 模式自动管理监控资源

---

## 架构设计

编译监控特性采用分层架构。**进度打印与超时检测均由 C++ 层 MonitorImpl 的 MonitorLoop 统一实现**；Python 层仅提供配置接口。

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python User Layer                         │
│                  用户代码（编译调用）                              │
└─────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Python Compiler Monitor                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  compiler_monitor.py                                     │   │
│  │  - 配置接口（set_compiler_monitor_options 等）           │   │
│  │  - 调用 C++ 初始化/关闭监控，不负责进度打印与超时检测    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        ↓ pybind11 绑定
┌─────────────────────────────────────────────────────────────────┐
│                      C++ Monitor Layer                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MonitorManager (Singleton)                             │   │
│  │  - 生命周期管理，std::call_once 确保单例初始化             │   │
│  │  - StartStage() / EndStage()                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MonitorImpl                                            │   │
│  │  - 后台监控线程 MonitorLoop（进度打印与超时检测的唯一实现）│   │
│  │  - condition_variable::wait_for 周期性唤醒               │   │
│  │  - 定期打印编译进度、检测超时并执行告警或协作式取消       │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MonitorStageScope (RAII Helper)                        │   │
│  │  - 自动调用 StartStage/EndStage                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                   编译阶段集成点（触发点）                        │
│  1. Python 阶段：Python 侧代码执行与进入 C++ 编译前的准备工作    │
│  2. Pass 流程：pass_manager 中每个 Pass 前后                     │
│  3. CodeGen 阶段：代码生成                                       │
│  4. 生成可执行程序阶段：二进制/可执行文件生成                     │
└─────────────────────────────────────────────────────────────────┘
```

### 超时机制对比

| 机制 | 实现层 | 超时后行为 | 说明 |
|------|--------|------------|------|
| **仅警告** | C++ 层 | 输出告警信息，编译继续 | 不中断编译，需用户手动终止 |
| **协作式取消** | C++ 层 | 在下一检测点抛出异常退出 | 需在编译流程插入检查点，可终止假卡死 |

---

## 超时检测实现模式

超时检测支持两种实现模式，可通过 `timeout_action` 配置选择：

### 模式一：仅警告（Warn Only）

**行为**：检测到超时后，仅输出告警信息，**不终止** C++ 编译流程。编译将继续执行直至完成或用户手动终止。

- **适用场景**：希望了解编译耗时异常，但不希望自动中断编译
- **用户操作**：若需停止，需通过 `Ctrl+C` 等方式手动终止进程
- **配置方式**：`timeout_action="warn"`（也可用 `TimeoutAction.WARN_ONLY` 常量，二者等价）

### 模式二：协作式取消（Cooperative Cancellation）

**行为**：检测到超时后，在 C++ 编译流程的各个阶段之间插入检测点；当超时标志被置位后，**在下一个检测点**抛出异常并终止整个编译过程。

**能解决的问题——“假卡死”**：程序实际仍在正常执行，仅因单阶段耗时过长给人以卡死的错觉。协作式取消可在阶段边界检查超时标志并及时退出。

**无法解决的问题——“真卡死”**：程序真正卡死（如陷入死循环、等待不可达的条件、第三方库阻塞等），检测点永远无法被执行到，编译流程无法被终止。此类情况只能通过用户手动终止（如 `Ctrl+C`）。

| 模式 | 超时后行为 | 能终止假卡死 | 能终止真卡死 |
|------|------------|--------------|--------------|
| 仅警告 | 打印告警，编译继续 | 否（靠用户手动终止） | 否 |
| 协作式取消 | 在下一检测点抛异常退出 | 是 | 否 |

---

## 超时终止实现方案

为防止 C++ 编译流程在某一阶段卡死，需要设计可靠的超时终止机制。Python 与 C++ 在同一进程内执行，超时后的处理采用**告警**或**协作式取消**两种方式，均由 C++ 层实现。

### 协作式取消（Cooperative Cancellation）

#### 核心思路

监控线程**只负责检测超时并设置原子标志**，不抛异常。主编译线程在**检查点**主动检查标志，若发现超时则抛出异常。异常在主编译线程中抛出，可正常通过 pybind11 传播回 Python。

#### 实现要点

围绕核心思路的三个关键步骤分别说明：

---

**步骤一：监控线程——检测超时并设置原子标志（不抛异常）**

- **共享状态**：在 MonitorImpl 中增加原子变量 `cancellation_requested_`，供监控线程写、主编译线程读。
- **检测逻辑**：监控线程按配置的 `interval_sec` 周期唤醒（如 `condition_variable::wait_for`），计算当前阶段耗时与总耗时；当超过 `timeout_sec` 或 `total_timeout_sec` 时，将 `cancellation_requested_` 置位。
- **约束**：监控线程仅负责置位标志并输出告警信息，**不得**在监控线程内抛出异常，否则会导致 `std::terminate`。

---

**步骤二：主编译线程——在检查点主动检查标志**

- **检查接口**：MonitorManager 提供 `CheckCancellation()`，内部读取 `cancellation_requested_`；若已置位，则抛出 `CompilationTimeoutException`（异常在主编译线程中抛出）。
- **检查点位置**：在下列四类集成点调用 `CheckCancellation()`：

| 检查点 | 位置说明 | 调用时机 |
|--------|----------|----------|
| Python 阶段 | Python 侧代码执行与进入 C++ 编译前的准备工作 | 进入 C++ 编译前、阶段边界 |
| Pass 流程 | pass_manager 中每个 Pass | 每个 Pass 执行前、执行后 |
| CodeGen 阶段 | 代码生成 | 阶段开始、阶段结束 |
| 生成可执行程序阶段 | 二进制/可执行文件生成 | 阶段开始、阶段结束 |

- **实现方式**：可利用 `MonitorStageScope` 等 RAII 在阶段进入/退出时调用 `CheckCancellation()`；pass_manager 在每个 Pass 的入口和出口处显式调用。

---

**步骤三：异常传播至 Python**

- **异常类型**：`CompilationTimeoutException` 需在 pybind11 中完成类型注册，以便 C++ 异常能转换为 Python 异常。
- **传播路径**：异常从 C++ 主编译线程抛出，经 pybind11 调用边界传回 Python 解释器，用户可通过 `try/except` 捕获并处理。

#### 局限

协作式取消仅能解决**假卡死**（执行缓慢）。若程序**真卡死**（某段代码完全没有检查点，如第三方库内的死循环或不可达的阻塞），协作式取消无法中断，此时需用户手动终止。

---

## 配置选项

### 核心配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `True` | 是否启用监控 |
| `interval_sec` | int | `30` | 进度打印间隔（秒） |
| `timeout_sec` | int | `600` | 单阶段超时阈值（秒），默认 10 分钟 |
| `total_timeout_sec` | int | `0` | 总编译时间超时阈值（秒），默认 0 表示禁用 |
| `timeout_action` | str | `"throw"` | 超时动作：`"throw"`（协作式取消）或 `"warn"`（仅警告），参见[超时检测实现模式](#超时检测实现模式) |

### TimeoutAction 枚举

```python
from pypto.compiler_monitor import TimeoutAction

TimeoutAction.THROW_EXCEPTION    # 协作式取消：超时后在下一检测点抛出异常终止编译
TimeoutAction.WARN_ONLY          # 仅警告：超时仅输出告警，不终止编译，需用户手动终止
```

### 时钟与时间计算

- `total_start_` - 监控总开始时间（首次 `StartStage()` 时重置）
- `stage_start_` - 当前阶段开始时间
- `last_print_time_` - 上次打印进度的时间
- `python_elapsed_time_` - Python 阶段捕获的时间

打印逻辑：监控线程每隔 `interval_sec` 检查一次，基于 `last_print_time_` 的间隔打印进度（`Total elapsed`）。

---

## 使用方法

### 默认使用（自动启用）

监控在导入 `pypto` 时自动启用，无需额外配置：

```python
import pypto

# 编译代码...
pto_result = pypto.matmul(a, b)
```

**输出示例**：
```
[Compiler Monitor] Stage: Python | Stage elapsed: 30s | Total elapsed: 30s
[Compiler Monitor] Stage: Pass_OoOSchedule | Stage elapsed: 30s | Total elapsed: 3min (180s)
...
[Compiler Monitor] Monitoring stopped | Total elapsed: 5min 15s (315s)
```

仅在校验到达到指定时间间隔时输出进度，以及全部编译完成后输出结束信息；不输出阶段开始、阶段完成的中间提示。

### 自定义配置

```python
import pypto

# 设置较短间隔和超时
pypto.set_compiler_monitor_options(
    interval_sec=10,      # 每 10 秒打印一次
    timeout_sec=20,       # 单阶段超时 20 秒
    timeout_action="warn"  # 超时仅警告
)
```

### 总编译时间超时

```python
import pypto

# 设置单阶段和总体超时
pypto.set_compiler_monitor_options(
    interval_sec=10,
    timeout_sec=60,           # 单阶段最多 60 秒
    total_timeout_sec=300,    # 总编译时间最多 5 分钟
)
```

### 完整配置

```python
import pypto
from pypto.compiler_monitor import StageMode, TimeoutAction

pypto.set_compiler_monitor_options(
    enable=True,
    interval_sec=30,
    timeout_sec=600,
    total_timeout_sec=1800,   # 总编译时间最多 30 分钟
    timeout_action=TimeoutAction.THROW,
)

```

### 超时处理方式配置

超时检测支持两种处理方式，通过 `timeout_action` 配置：

**方式 1：仅告警**（`timeout_action="warn"`）
- 超时后仅输出告警信息，编译继续执行
- 若需停止，需用户手动终止（如 `Ctrl+C`）

```python
pypto.set_compiler_monitor_options(
    timeout_sec=300,
    timeout_action="warn"
)
pto_result = pypto.matmul(a, b)
```

**方式 2：协作式取消**（`timeout_action="throw"`）
- 超时后置位取消标志，在 C++ 编译流程的下一个检测点抛出异常并终止编译
- 可解决假卡死（执行缓慢），无法解决真卡死（死循环等）

```python
pypto.set_compiler_monitor_options(
    timeout_sec=300,
    timeout_action="throw"
)
try:
    pto_result = pypto.matmul(a, b)
except CompilationTimeoutException as e:
    print(f"编译超时: {e}")
```

**超时告警输出示例**：
```
[Compiler Timeout] Stage 'TensorGraphPass' exceeded timeout (300s). Elapsed: 5min 30s
```

**协作式取消异常**：超时后在下一次到达检测点时抛出 `CompilationTimeoutException`，可被 Python 层捕获。

---

## 输出格式

### 进度输出（按时间间隔）

```
[Compiler Monitor] Stage: <当前阶段> | Stage elapsed: <阶段耗时> | Total elapsed: <总耗时>
```

### 编译完成输出

```
[Compiler Monitor] Monitoring stopped | Total elapsed: <总耗时> (<秒数>s)
```

### 超时告警输出

```
[Compiler Timeout] Stage '<阶段名>' exceeded timeout (<阈值>). Elapsed: <实际耗时>
```

---

## 实现原理

### 文件结构

```
python/pypto/
├── compiler_monitor.py      # Python API（配置接口、枚举定义）
├── __init__.py             # 自动初始化监控

python/src/bindings/
└── monitor.cpp             # Python-C++ 绑定

framework/src/interface/compiler_monitor/
├── monitor_manager.h/.cpp  # 单例管理器
├── monitor_impl.h/.cpp     # 监控实现（监控线程循环）
├── monitor_config.h/.cpp   # 阶段配置（COARSE/FINE/CUSTOM）
└── monitor_exception.h     # 异常定义

framework/src/interface/program/
└── recorder.cpp            # 编译阶段集成点之一

framework/src/passes/pass_mgr/
└── pass_manager.cpp        # Pass 流程集成点（每个 Pass 前后）
```

### 自动初始化流程

```
1. 导入 pypto
   └→ __init__.py 被执行
2. __init__.py 调用 initialize_monitor()
   └→ pypto_impl.InitializeMonitor()
3. MonitorManager::Instance() 被调用
   └→ std::call_once 触发 Initialize()
4. 创建 MonitorImpl 并启动监控线程
   └→ 监控开始运行
```

### 超时处理流程

#### 仅告警模式（timeout_action="warn"）

```
1. 用户配置
   └→ pypto.set_compiler_monitor_options(timeout_action="warn")

2. 开始编译（C++ 执行）
   ├→ MonitorManager::Instance().Initialize()
   ├→ 各阶段 StartStage/EndStage
   └→ 监控线程 MonitorLoop 定期检查

3. 超时发生
   ├→ 监控线程检测到 elapsed > timeout_sec
   └→ 输出告警信息到 stderr，编译继续执行

4. 用户操作
   └→ 若需停止，用户需手动终止（如 Ctrl+C）
```

#### 协作式取消模式（timeout_action="throw"）

```
1. 用户配置
   └→ pypto.set_compiler_monitor_options(timeout_action="throw")

2. 开始编译（C++ 执行）
   ├→ MonitorManager::Instance().Initialize()
   ├→ 各阶段 StartStage/EndStage，每个阶段边界为检测点
   └→ 监控线程 MonitorLoop 定期检查

3. 超时发生
   ├→ 监控线程检测到 elapsed > timeout_sec
   └→ 设置 cancellation_requested_ 标志（不抛异常，避免影响监控线程）

4. 主编译线程到达下一个检测点
   ├→ MonitorManager::CheckCancellation() 检测到标志已置位
   └→ 抛出 CompilationTimeoutException

5. 异常传播
   └→ 通过 pybind11 传递到 Python 层，用户可捕获处理
```

**说明**：协作式取消依赖检测点，仅能终止「假卡死」（执行缓慢）。若程序「真卡死」（如死循环中无检查点），无法中断，需用户手动终止。

---

## 编译耗时整体情况打印

会在完成时打印下述各个阶段的总耗时：

| 阶段名称 | 说明 |
|----------|------|
| `Python` | Python 执行阶段（监控启动时默认阶段，从pypto init开始到进入C++编译流程都属于该阶段） |
| `TensorGraphPass` | Tensor 图变换 |
| `TileGraphPass` | Tile 图变换 |
| `BlockGraphPass` | Block 图变换 |
| `CodeGen` | 代码生成 |
| `BinaryGeneration` | 二进制生成 |

---
