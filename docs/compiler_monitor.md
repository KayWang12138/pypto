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
│  1. Prepare 阶段：从 pypto init 到进入 Program::UpdateCompileTask() 之前 │
│  2. Pass 阶段：CompileFunction 内 runPass(PVC2_OOO) 前后         │
│  3. CodeGen 阶段：backend Execute → GenCode() 整段（见下文定义） │
└─────────────────────────────────────────────────────────────────┘
```

### 编译阶段与监控阶段定义

以下定义用于正确插桩和统计耗时。

#### 1. 多 Function 场景下的阶段统计方式

一次“用户编译”可能对应**多个 Function** 的编译（例如动态控制流下：多个 leaf Tensor Graph function 各自走一遍 CompileFunction + Execute，最后再编译一个 DYNAMIC 的 dyndev）。因此同一“阶段名”会被**多次** StartStage/EndStage。

- **统计语义**：对同一阶段名的多次进入采用**累加耗时**（即该阶段总耗时 = 各次 EndStage 时累计的 stage 耗时之和）。最终输出的耗时统计按**阶段（stage）维度**整体汇总：每个阶段一个总耗时，表示所有 function 在该阶段耗时的累加（例如“Pass 总耗时”= 所有 function 的 Pass 耗时之和）。
- **打印需求**：
  - **总 function 数**：在已知本次编译将处理的 function 总数时（如开始遍历 CompileFunction 前），应上报或打印「本次编译共 N 个 function」；完成时输出中也会包含该总数。
  - **当前进度**：进入每个 function 的 Pass/CodeGen 时，进度输出中应包含「当前正在处理第 k 个 function，共 N 个」（即 k/N），便于用户了解编译进度。
  - **最终耗时**：完成时按 **stage 维度** 打印各阶段总耗时，即每个 stage 一行，数值为该 stage 在所有 function 上的累加耗时，不按 function 拆开。
- **实现要点**：统一使用阶段名 `Pass`、`CodeGen`，在 EndStage 时将本次阶段耗时累加到该阶段名的总计时器。需在能获知「本次编译将处理多少个 function」的入口（如 host_machine 或上层调度处）设置总 function 数；在每次进入 CompileFunction（或 runPass/GenCode 前）设置当前 function 序号，供 MonitorImpl 在进度打印与完成打印中使用。

#### 2. Pass 阶段所指范围

- **Pass 阶段**在监控中特指 **CompileFunction 阶段**：即编译线程对**单个 Function** 执行的一次 `RunPass(Program::GetInstance(), *func, GetPassStrategy())`，其中策略一般为 **PVC2_OOO**（从 RemoveRedundantReshape 到 CodegenPreproc 的整条 Tensor → Tile → Block 链）。插桩位置：`framework/src/interface/machine/host/host_machine.cpp` 的 `CompileFunction()` 内，**`backend.runPass(...)` 调用前后**。

#### 3. CodeGen 与二进制生成合并为一个阶段

- **结论：建议合并为一个阶段（CodeGen）**，从耗时统计的**可实现性**考虑：
  - **分开统计的难点**：静态路径下，源码生成与调用 bisheng 编译（DoCompileCCE）均发生在 `CodeGenCloudNPU::GenCode()` 内部，需在 codegen 层插桩；动态路径下，控制流源码生成与 host/aicpu 二进制、各 leaf 源码与 AICore 二进制**交错执行**，若拆成“CodeGen”和“BinaryGeneration”两阶段，会出现同一次编译中多次切换阶段、顺序依赖实现细节，统计易混乱。
  - **合并的好处**：在 **backend Execute** 中仅对 **GenCode(task, ...)** 调用做**一次** StartStage("CodeGen")、一次 EndStage("CodeGen")，即可得到“从开始生成代码到得到全部二进制”的总耗时，插桩点单一、语义清晰。
- **实现**：在 `framework/src/machine/host/backend.cpp` 的 `Execute()` 内，在调用 `GenCode(...)` **之前** StartStage("CodeGen")，在 `GenCode(...)` **返回之后** EndStage("CodeGen")。无需在 codegen 内部或 CompileDyndevFunction 内再区分“纯代码生成”与“二进制生成”。


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
- **检查点位置**：在下列集成点调用 `CheckCancellation()`：

| 检查点 | 位置说明 | 调用时机 |
|--------|----------|----------|
| Prepare 阶段 | 从 pypto init 到 Program::UpdateCompileTask() 之前（所有 function 共享，不区分） | Program::UpdateCompileTask() 入口 |
| Pass 阶段 | CompileFunction 内 runPass(program, func, PVC2_OOO) | runPass 调用前、调用后（host_machine.cpp） |
| CodeGen 阶段 | backend Execute 内 GenCode() 整段（含代码生成与二进制生成） | GenCode 调用前、调用后（backend.cpp） |

- **实现方式**：可利用 `MonitorStageScope` 等 RAII 在阶段进入/退出时调用 `CheckCancellation()`。Pass 阶段在 host_machine 的 CompileFunction 内包一层；CodeGen 阶段在 backend Execute 内 GenCode 调用前后各调用一次。

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
| `compile_monitor_enable` | bool | `True` | 是否启用监控 |
| `interval_sec` | int | `60` | 当当前子stage耗时已经超过了60s，则进度打印间隔（秒），默认 60（秒），手动设置大于零的数值才有效 |
| `timeout_sec` | int | `0` | 单阶段超时阈值（秒），默认 0 表示禁用 |
| `total_timeout_sec` | int | `600` | 总编译时间超时阈值（秒），默认 10分钟 ，0 表示禁用 |
| `timeout_action` | str | `"warn"` | 超时动作：`"throw"`（协作式取消）或 `"warn"`（仅警告），参见[超时检测实现模式](#超时检测实现模式) |

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
  |__ [Compiler Monitor] Stage: Prepare(processing) | Stashed function: 72 | Stage elapsed: 10.0s | Total elapsed: 10.0s
  ...
  |__ [Compiler Monitor] Stage: Prepare(processing) | Stashed function: 137 | Stage elapsed: 55.0s | Total elapsed: 55.0s
[Compiler Monitor] Stage: Prepare(completed) | Stashed function: 137 | Stage elapsed: 55.6s | Total elapsed: 55.6s
[Compiler Monitor] Function: 1/137 | Stage: Pass(completed) | Stage elapsed: 0.0s | Total elapsed: 55.6s | Func:[TENSOR_LOOP_Nv_TND_Unroll1_PATH0_hiddenfunc0_11]
[Compiler Monitor] Function: 2/137 | Stage: Pass(completed) | Stage elapsed: 1.1s | Total elapsed: 56.7s | Func:[TENSOR_LOOP_S_TND_Unroll6_PATH0_hiddenfunc0_14]
...
[Compiler Monitor] Function: 136/137 | Stage: Pass(completed) | Stage elapsed: 0.0s | Total elapsed: 1min 56s (116s) | Func:[TENSOR_TENSOR_chunk_gated_delta_rule_loop_Unroll1_3]
[Compiler Monitor] | [== WARNING ==] Total elapsed [2min 1s (121s)] exceeded the total time threshold [2min 0s (120s)], you can terminate the process by pressing Ctrl+C !!!
  |__ [Compiler Monitor] Function: 137/137 | Stage: Pass(processing) | Stage elapsed: 10.0s | Total elapsed: 2min 6s (126s) | Func:[TENSOR_chunk_gated_delta_rule_2]
  ...
[Compiler Monitor] | [** WARNING **] Functions: 137/137 | Stage [Pass] elapsed [20.0s] exceeded the current stage total time threshold [20.0s], you can terminate the process by pressing Ctrl+C !!!
  |__ [Compiler Monitor] Function: 137/137 | Stage: Pass(processing) | Stage elapsed: 40.0s | Total elapsed: 2min 36s (156s) | Func:[TENSOR_chunk_gated_delta_rule_2]
[Compiler Monitor] Function: 137/137 | Stage: Pass(completed) | Stage elapsed: 41.9s | Total elapsed: 2min 38s (158s) | Func:[TENSOR_chunk_gated_delta_rule_2]
  |__ [Compiler Monitor] Stage: CodeGen(processing) | Stage elapsed: 10.0s | Total elapsed: 2min 48s (168s)
  ...
[Compiler Monitor] | [** WARNING **] Functions: 137/137 | Stage [CodeGen] elapsed [20.0s] exceeded the current stage total time threshold [20.0s], you can terminate the process by pressing Ctrl+C !!!
  |__ [Compiler Monitor] Stage: CodeGen(processing) | Stage elapsed: 20.0s | Total elapsed: 2min 58s (178s)
  ...
  |__ [Compiler Monitor] Stage: CodeGen(processing) | Stage elapsed: 1min 55s (115s) | Total elapsed: 4min 33s (273s)
[Compiler Monitor] Stage: CodeGen(completed) | Stage elapsed: 1min 56s (116s) | Total elapsed: 4min 35s (275s)
[Compiler Monitor] Compilation finished 137/137 | Total functions: 137
[Compiler Monitor] Stage timing (aggregated by stage):
  CodeGen  117.0s   (sum over 137 functions)
  Pass     102.2s   (sum over 137 functions)
  Prepare  55.6s
[Compiler Monitor] Monitoring stopped | Total elapsed: 4min 35s (275s)
```

仅在校验到达到指定时间间隔时输出进度，以及全部编译完成后输出结束信息；不输出阶段开始、阶段完成的中间提示。

### 自定义配置

```python
import pypto

# 设置较短间隔和超时
@pypto.frontend.jit(
    host_options={"interval_sec": 10,  # 超60s后，每10s打印一次
                  "timeout_sec": 20,   # 单阶段超时20s
                  "timeout_action": "warn"  # 超时仅警告
                  }
)
```

### 总编译时间超时

```python
import pypto

# 设置单阶段和总体超时
@pypto.frontend.jit(
    host_options={"interval_sec": 10,
                  "timeout_sec": 60,        # 单阶段最多 60 秒
                  "total_timeout_sec": 300  # 总编译时间最多 5 分钟
                  }
)
```

### 完整配置

```python
import pypto
from pypto.compiler_monitor import StageMode, TimeoutAction

@pypto.frontend.jit(
    host_options={"compile_monitor_enable": True,
                  "interval_sec": 30,
                  "timeout_sec": 600,
                  "total_timeout_sec": 1800  # 总编译时间最多 30 分钟
                  }
)
```

### 超时处理方式配置

超时检测支持两种处理方式，通过 `timeout_action` 配置：

**方式 1：仅告警**（`timeout_action="warn"`）
- 超时后仅输出告警信息，编译继续执行
- 若需停止，需用户手动终止（如 `Ctrl+C`）

```python
@pypto.frontend.jit(
    host_options={"timeout_sec": 300,
                  "timeout_action": "warn"  # 超时仅警告
                  }
)
pto_result = pypto.matmul(a, b)
```

**方式 2：协作式取消**（`timeout_action="throw"`）
- 超时后置位取消标志，在 C++ 编译流程的下一个检测点抛出异常并终止编译
- 可解决假卡死（执行缓慢），无法解决真卡死（死循环等）

```python
@pypto.frontend.jit(
    host_options={"timeout_sec": 300,
                  "timeout_action": "throw"
                  }
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

当本次编译包含多个 function 时，进度中需体现总 function 数与当前进度：

```
[Compiler Monitor] Functions: <当前第几个>/<总计几个> | Stage: <当前阶段> | Stage elapsed: <阶段耗时> | Total elapsed: <总耗时>
```

- **总计几个**：本次编译将要处理的 function 总数，在已知该数量时即确定并参与后续所有进度输出。
- **当前第几个**：当前正在执行 Pass 或 CodeGen 的 function 序号（从 1 开始计数）。若仅有一个 function 或尚未进入多 function 流程，可省略为单 function 的简化格式（仅 Stage / Stage elapsed / Total elapsed）。

### 编译完成输出

完成时先输出 function 总数与按 stage 汇总的耗时（见 [编译耗时整体情况打印](#编译耗时整体情况打印)），再输出监控结束行：

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

framework/src/interface/machine/host/
└── host_machine.cpp        # Pass 阶段集成点（CompileFunction 内 runPass 前后）

framework/src/machine/host/
└── backend.cpp             # CodeGen 阶段集成点（Execute 内 GenCode 前后）

framework/src/passes/pass_mgr/
└── pass_manager.cpp        # 各 Pass 内部细粒度监控（可选，Pass_<identifier>）
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

在编译完成时打印本次编译的 **function 总数** 以及各阶段的**按 stage 维度汇总**的总耗时。

- **Function 总数**：输出「本次编译共 N 个 function」，与进度中的“当前 k/N”一致。
- **按 stage 维度统计**：每个阶段一行，该阶段的耗时为**所有 function 在该阶段的耗时累加**（不按 function 拆开）。即“Pass 总耗时”= 所有 function 的 Pass 耗时之和，“CodeGen 总耗时”= 所有 function 的 CodeGen 耗时之和。

**示例**（3 个 function 时）：

```
[Compiler Monitor] Compilation finished | Total functions: 3
[Compiler Monitor] Stage timing (aggregated by stage):
  Prepare: 1.2s
  Pass:    12.5s   (sum over 3 functions)
  CodeGen: 8.1s    (sum over 3 functions)
[Compiler Monitor] Monitoring stopped | Total elapsed: 22.0s
```

| 阶段名称 | 说明 | 对应代码位置 |
|----------|------|--------------|
| `Prepare` | Prepare 阶段（从 pypto init 到 Program::UpdateCompileTask() 之前，所有 function 共享） | program.cpp UpdateCompileTask() 入口 |
| `Pass` | CompileFunction 阶段：对**每个**待编译 function 执行 runPass(program, func, PVC2_OOO)（Tensor→Tile→Block 整条链） | host_machine.cpp CompileFunction 内 runPass 前后 |
| `CodeGen` | 对**每个** function 的 GenCode() 整段（含源码生成与二进制生成，含 host 控制流、aicpu 控制流、AICore kernel 的生成与编译）；不单独拆“二进制生成”阶段 | backend.cpp Execute 内 GenCode() 前后 |

说明：不按“TileGraphPass”“BlockGraphPass”再分子阶段，因二者均在 CompileFunction 的同一 runPass(PVC2_OOO) 内顺序执行，统一以 **Pass** 阶段统计即可。CodeGen 与二进制生成合并为一个 **CodeGen** 阶段。

---

## TODO
1. 细粒度各pass的耗时统计与超时监测归一到当前方案；
2. pass的超时阈值默认值按照约定方案根据节点规模设定；
3. 建议第一阶段先按照超时告警、用户手动终止来实现；
