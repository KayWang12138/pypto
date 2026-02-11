# Compiler Monitor

编译器监控与超时退出功能，用于监控 PyPTO 编译过程中的各阶段耗时，并在超时时进行告警或中断。

---

## 概述

Compiler Monitor 提供编译过程的实时进度监控和超时检测功能。它通过后台监控线程定期检查当前编译阶段的执行时间，当超过配置的超时阈值时，可选择抛出异常中断编译或仅输出警告信息。

### 主要特点

- **自动启用** - 导入 `pypto` 时自动启用，无需额外配置
- **实时进度监控** - 定期打印当前编译阶段的执行进度
- **超时检测** - 支持自定义超时阈值，超时后可选择抛出异常或警告
- **分层实现** - C++ 层提供基础监控，Python 层提供强制超时中断（`signal.alarm`）
- **线程安全** - 监控在独立线程中运行，不影响主编译流程
- **资源自动管理** - 通过 RAII 模式自动管理监控资源
- **多粒度监控** - 支持粗粒度（COARSE）、细粒度（FINE）和自定义（CUSTOM）监控模式

---

## 架构设计

编译监控特性采用分层架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python User Layer                         │
│                  用户代码（编译调用）                              │
└─────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Python Timeout Guard                           │
│                  (signal.alarm 强制中断)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  compiler_monitor.py / TimeoutGuard                     │   │
│  │  - 使用 signal.alarm() 设置系统定时器                    │   │
│  │  - 超时时 SIGALRM 信号处理器抛出 TimeoutError             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Python Compiler Monitor                      │
│                  (后台进度打印线程)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  compiler_monitor.py / CompilerMonitor                   │   │
│  │  - 独立的后台监控线程                                    │   │
│  │  - 定期打印编译进度（不负责超时控制）                    │   │
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
│  │  - 后台监控线程 MonitorLoop                              │   │
│  │  - condition_variable::wait_for 周期性唤醒               │   │
│  │  - 超时检测与进度打印                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MonitorStageScope (RAII Helper)                        │   │
│  │  - 自动调用 StartStage/EndStage                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                   编译阶段集成点（触发点）                        │
│  - recorder.cpp: TensorGraphPass, UpdateCompileTask             │
│  - pass_manager.cpp: 各个 Pass                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 超时机制对比

| 机制 | 实现层 | 中断方式 | 准确性 | 说明 |
|------|--------|----------|--------|------|
| **C++ 超时检测** | C++ 层 | 检测点抛出异常 | 较低 | 需要代码主动检查，被动检测 |
| **Python signal.alarm** | Python 层 | 内核信号中断 | 高 | 内核管理定时器，主动触发 |

---

## 配置选项

### 核心配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | `True` | 是否启用监控 |
| `interval_sec` | int | `30` | 进度打印间隔（秒） |
| `timeout_sec` | int | `600` | 单阶段超时阈值（秒），默认 10 分钟 |
| `total_timeout_sec` | int | `0` | 总编译时间超时阈值（秒），默认 0 表示禁用 |
| `timeout_action` | str | `"throw"` | 超时动作：`"throw"` 或 `"warn"` |
| `stage_mode` | str | `"coarse"` | 监控粒度：`"coarse"`、`"fine"` 或 `"custom"` |
| `custom_stages` | List[str] | `None` | 自定义阶段名称列表（`stage_mode="custom"` 时必填） |

### TimeoutAction 枚举

```python
from pypto.compiler_monitor import TimeoutAction

TimeoutAction.THROW_EXCEPTION    # 抛出异常中断编译
TimeoutAction.WARN_ONLY          # 输出警告，继续执行
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
[Compiler Monitor] [Stage Started] #1 FrontendParser
[Compiler Monitor] Stage: FrontendParser | Stage elapsed: 30s | Total elapsed: 30s
[Compiler Monitor] FrontendParser completed | Stage elapsed: 45s | Total elapsed: 2min 30s (150s)
[Compiler Monitor] [Stage Started] #2 TensorGraphPass
[Compiler Monitor] Stage: TensorGraphPass | Stage elapsed: 30s | Total elapsed: 3min (180s)
...
[Compiler Monitor] Monitoring stopped | Total elapsed: 5min 15s (315s)
```

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
    stage_mode=StageMode.COARSE,
)

# 设置自定义阶段
pypto.SetMonitorStageMode(
    mode=StageMode.CUSTOM,
    custom_stages="Parse,Optimize,CodeGen,Load"
)
```

### 预设模式

```python
# 静默模式：5 分钟打印一次进度
pypto.set_monitor_quiet_mode()

# 激进模式：30 秒打印一次，10 分钟超时
pypto.set_monitor_aggressive_mode()

# 仅警告模式：超时只警告，不中断
pypto.set_monitor_warn_only(timeout_sec=1800)
```

### 使用 TimeoutGuard 强制超时中断

```python
from pypto.compiler_monitor import TimeoutGuard

# 方式1: 上下文管理器
with TimeoutGuard(timeout_sec=300):
    pto_result = pypto.matmul(a, b)

# 方式2: 装饰器
from pypto.compiler_monitor import timeout_context

@timeout_context(timeout_sec=300)
def run_compilation():
    return pypto.matmul(a, b)
```

**超时输出示例**：
```
[Timeout Guard] Enabled: 300s timeout
[TIMEOUT] Compilation timed out after 300s (actual: 301s)
============================================================
Stack trace:
  File "example.py", line 10, in <module>
    with TimeoutGuard(timeout_sec=300):
  ...
============================================================
```

---

## 输出格式

### 进度输出

```
[Compiler Monitor] Stage: <当前阶段> | Stage elapsed: <阶段耗时> | Total elapsed: <总耗时>
```

### 粗粒度阶段完成输出

```
[Compiler Monitor] <阶段名> completed | Stage elapsed: <阶段耗时> | Total elapsed: <总耗时> (<秒数>s)
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
├── compiler_monitor.py      # Python API（TimeoutGuard、CompilerMonitor）
├── __init__.py             # 自动初始化监控
└── timeout_guard.py        # Python 超时控制（signal.alarm）

python/src/bindings/
└── monitor.cpp             # Python-C++ 绑定

framework/src/interface/compiler_monitor/
├── monitor_manager.h/.cpp  # 单例管理器
├── monitor_impl.h/.cpp     # 监控实现（监控线程循环）
├── monitor_config.h/.cpp   # 阶段配置（COARSE/FINE/CUSTOM）
└── monitor_exception.h     # 异常定义

framework/src/interface/program/
└── recorder.cpp            # 编译阶段集成点

framework/src/passes/pass_mgr/
└── pass_manager.cpp        # Pass 级别集成点
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

### C++ 层核心实现

#### 1. MonitorManager 单例

**位置**: `framework/src/interface/compiler_monitor/monitor_manager.cpp:24-32`

```cpp
MonitorManager& MonitorManager::Instance() {
    static MonitorManager instance;
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        instance.Initialize();
    });
    return instance;
}
```

#### 2. 监控线程循环

**位置**: `framework/src/interface/compiler_monitor/monitor_impl.cpp:178-269`

```cpp
void MonitorImpl::MonitorLoop() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);

        // 等待指定间隔或停止信号
        if (cv_.wait_for(lock, interval_, [this]() { return !running_.load(); })) {
            break;
        }

        // 获取耗时数据
        auto now = std::chrono::steady_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - stage_start_).count();

        // 释放锁后检查超时和打印进度
        lock.unlock();

        // 检查超时（被动检测）
        if (!timeoutThrown_.load() && elapsed > timeout_.count()) {
            timeoutThrown_.store(true);
            if (timeoutAction_ == TimeoutAction::THROW_EXCEPTION) {
                throw CompilationTimeoutException(...);
            }
        }

        // 打印进度
        if (time_since_last_print >= interval_) {
            std::cout << "[Compiler Monitor] Stage: " << stage_name << ...
        }
    }
}
```

#### 3. RAII 阶段跟踪

**位置**: `framework/src/interface/compiler_monitor/monitor_manager.cpp:182-212`

```cpp
MonitorStageScope::MonitorStageScope(const std::string& stageName)
    : stageName_(stageName), active_(false), stageId_(gNextStageId.fetch_add(1)) {
    active_.store(true);
    MonitorManager::Instance().StartStage(stageName_);
}

MonitorStageScope::~MonitorStageScope() {
    if (active_.exchange(false)) {
        MonitorManager::Instance().EndStage();
    }
}
```

### Python 层超时控制

#### signal.alarm 工作原理

```
Python 执行流
    ↓
进入 TimeoutGuard.__enter__
    ↓
signal.signal(SIGALRM, handler)  → 注册自定义信号处理器
    ↓
signal.alarm(timeout_sec)         → 内核设置定时器
    ↓
继续执行编译代码...
    ↓
超时时（timeout_sec 秒后）
    ↓
内核发送 SIGALRM 信号
    ↓
Python 解释器调用 handler(signum, frame)
    ↓
handler 抛出 TimeoutError 异常
    ↓
异常向上传播，中断编译
```

**关键特性**：
- 使用 `signal.alarm()` 系统调用，定时器由内核管理
- 即使 C++ 代码在执行，超时时也能安全中断（在 GIL 释放点）
- 不支持嵌套超时（通过 `_active` 标志防止）
- 退出时恢复原始信号处理器

### 超时退出完整流程

```
1. 用户设置超时保护
   └→ with TimeoutGuard(timeout_sec=300):

2. 进入上下文
   ├→ signal.signal(SIGALRM, handler)
   └→ signal.alarm(300)

3. 开始编译（C++ 执行）
   ├→ MonitorManager::Instance().Initialize()
   ├→ 各阶段 StartStage/EndStage
   └→ 监控线程定期检查

4. 超时发生
   ├→ 内核发送 SIGALRM 信号
   ├→ Python 解释器捕获信号
   └→ 调用 _timeout_handler

5. 处理器抛出异常
   ├→ 打印调用栈
   └→ raise TimeoutError

6. 异常传播
   ├→ 中断 C++ 执行
   └→ 传递到 Python 层

7. 退出上下文
   ├→ signal.alarm(0)
   └→ 恢复原始信号处理器
```

---

## 粗粒度阶段定义

以下阶段会被视为粗粒度阶段，会在完成时打印总耗时：

| 阶段名称 | 说明 |
|----------|------|
| `Python` | Python 执行阶段（监控启动时默认阶段） |
| `FrontendParser` | 前端解析 |
| `TensorGraphPass` | Tensor 图变换 |
| `UpdateCompileTask` | 编译任务更新 |
| `TileGraphPass` | Tile 图变换 |
| `BlockGraphPass` | Block 图变换 |
| `CodeGen` | 代码生成 |
| `BinaryGeneration` | 二进制生成 |

---
