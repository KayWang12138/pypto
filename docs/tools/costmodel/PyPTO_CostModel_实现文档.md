# PyPTO CostModel 实现文档

## 1. 概述

CostModel 是 PyPTO 的性能仿真子系统，用于在 **不依赖真实 NPU 硬件** 的情况下，对 PTO 算子的执行性能进行周期级（cycle-accurate）建模和预测。它模拟 Ascend NPU 的完整硬件层级（Device → AICPU → AI Core → Pipe），支持多种仿真模式，并提供精度验证（Precision Verification）能力。

**核心能力：**

- 周期精确的性能仿真（cycle-accurate simulation）
- 多架构支持（A2A3、A5）
- 子图级别和整图级别的仿真
- Pipeline 级别的指令调度和时序建模
- 精度验证（PvModel）
- 丰富的统计报告输出（JSON、Perfetto、Swim Lane 图表）

---

## 2. 目录结构

```
framework/src/cost_model/
├── simulation/
│   ├── CostModelInterface.h/cpp       # 仿真接口层 — 构建与运行仿真
│   ├── backend.h/cpp                  # C++ 后端代理 — Python 绑定入口
│   ├── cost_model_launcher.h          # 仿真启动器 — 集成运行时系统
│   │
│   ├── arch/                          # 架构相关实现
│   │   ├── A2A3/                      # Ascend A2A3 架构适配
│   │   │   ├── L2CacheImplA2A3.h/cpp  #   L2 Cache 模型
│   │   │   └── PostSimulatorA2A3.h    #   后处理仿真器
│   │   ├── A5/                        # Ascend A5 架构适配
│   │   │   ├── L2CacheImplA5.h/cpp    #   L2 Cache 模型
│   │   │   └── PostSimulatorA5.h      #   后处理仿真器
│   │   ├── GenCalendar/               # Calendar 调度器（时序排布）
│   │   │   └── GenCalendar.h/cpp
│   │   ├── PipeFactory.h/cpp          # Pipe 工厂 — 创建不同类型的 Pipe 仿真器
│   │   ├── PipeSimulatorFast.h/cpp    # 快速 Pipe 仿真引擎（带延迟缓存）
│   │   ├── PipeMachineImpl.h          # Pipe 机器统一接口
│   │   ├── CallPipeImpl.h             # CALL 指令 Pipe 实现
│   │   ├── TileAllocPipeImpl.h        # Tile 内存分配 Pipe 实现
│   │   ├── CacheMachineImpl.h         # Cache 机器实现
│   │   ├── Simulator.h                # 仿真器基类
│   │   └── SimplifiedMemoryAllocator.h # 简化内存分配器
│   │
│   ├── base/                          # 基础框架
│   │   ├── ModelTop.h/cpp             # SimSys — 仿真系统顶层管理
│   │   ├── Machine.h/cpp              # Machine — 机器基类
│   │   ├── Reporter.h/cpp             # 报告生成器
│   │   ├── Config.h                   # 配置基类
│   │   ├── BaseStats.h                # 统计基类
│   │   └── SimObj.h                   # 仿真对象基类
│   │
│   ├── cache/                         # 缓存系统
│   │   ├── CacheMachine.h/cpp         # Cache 机器（L2 Cache 仿真）
│   │   └── FunctionCache.h/cpp        # 函数缓存
│   │
│   ├── common/                        # 公共类型与工具
│   │   ├── CommonType.h               # 核心枚举与数据类型
│   │   ├── CommonData.h               # 公共数据结构（Pid, Tid, OperandType 等）
│   │   ├── CommonTools.h              # 工具函数
│   │   ├── CycleInfo.h                # 周期信息结构
│   │   ├── ISA.h/cpp                  # 指令集架构定义（Tile/TileOp/Function）
│   │   ├── Packet.h/cpp               # 数据包定义（TaskPack, CachePacket 等）
│   │   └── BaseQueue.h                # 仿真队列实现
│   │
│   ├── config/                        # 配置系统
│   │   ├── ModelConfig.h/cpp          # 模型配置
│   │   ├── DeviceConfig.h/cpp         # 设备配置
│   │   ├── CoreConfig.h/cpp           # 核心配置
│   │   ├── AICPUConfig.h/cpp          # AICPU 配置
│   │   ├── PipeConfig.h/cpp           # Pipeline 配置
│   │   ├── CacheConfig.h/cpp          # 缓存配置
│   │   ├── TraceConfig.h/cpp          # 追踪配置
│   │   └── EnvConfig.h/cpp            # 环境配置
│   │
│   ├── emulator/                      # 功能仿真
│   │   └── SoftMemory.h               # 软件内存仿真
│   │
│   ├── machine/                       # 机器层级实现
│   │   ├── DeviceMachine.h/cpp        # 设备级机器
│   │   ├── AICPUMachine.h/cpp         # AICPU 级机器
│   │   ├── CoreMachine.h/cpp          # AI Core 级机器（AIC/AIV/MIXAICORE）
│   │   ├── PipeMachine.h/cpp          # Pipeline 级机器
│   │   ├── Scheduler.h/cpp            # 指令调度器（排序与发射）
│   │   └── SimplifiedMemoryAllocator.h # 内存分配器
│   │
│   ├── pv/                            # 精度验证（Precision Verification）
│   │   ├── PvModel.h                  # PvModel 接口
│   │   ├── PvModelFactory.h/cpp       # PvModel 工厂
│   │   └── PvData.h                   # Pv 数据结构
│   │
│   ├── statistics/                    # 统计与日志
│   │   ├── ModelStats.h/cpp           # 模型级统计
│   │   ├── DeviceStats.h/cpp          # 设备级统计
│   │   ├── AICPUStats.h/cpp           # AICPU 级统计
│   │   ├── CoreStats.h/cpp            # Core 级统计
│   │   ├── CacheStats.h/cpp           # Cache 级统计
│   │   └── TraceLogger.h/cpp          # 追踪日志器
│   │
│   ├── tools/                         # 开发工具
│   │   ├── ParseInput.h/cpp           # 输入解析
│   │   ├── ParseArgs.h                # 命令行参数解析
│   │   └── visualizer.h/cpp           # 可视化工具
│   │
│   └── value/                         # 值计算与状态管理
│       ├── TileCalculator.h/cpp       # Tile 计算器
│       └── TileState.h/cpp            # Tile 状态管理
│
├── simulation_pv/                     # 精度验证仿真实现
│   ├── PvModelImpl.h/cpp              # PvModel 实现
│   ├── PvModelConfig.h/cpp            # Pv 配置
│   └── PvMemAllocator.h/cpp           # Pv 内存分配器
│
python/                                # Python 绑定
├── pypto/cost_model.py                # Python 接口
└── src/bindings/cost_model.cpp        # pybind11 绑定
```

---

## 3. 核心架构

### 3.1 整体分层

```
┌─────────────────────────────────────────────────┐
│              Python 接口层                        │
│  cost_model.py ← pybind11 → cost_model.cpp       │
├─────────────────────────────────────────────────┤
│              启动器层                             │
│  CostModelLauncher (cost_model_launcher.h)        │
│    ├── RunCostModel()     LEAF_FUNCTION 模式      │
│    ├── RunDynCostModel()  NORMAL 模式             │
│    ├── RunTestMode()      功能仿真                │
│    └── RunPvModel()       精度验证                │
├─────────────────────────────────────────────────┤
│              后端代理层                           │
│  xf (backend.h)                       │
│    ├── BuildCostModel()                           │
│    ├── SubmitToCostModel() / SubmitTopo()         │
│    ├── RunCostModel()                             │
│    └── TerminateCostModel()                       │
├─────────────────────────────────────────────────┤
│              仿真接口层                           │
│  CostModelInterface (CostModelInterface.h)        │
│    ├── BuildCostModel()  构建仿真系统             │
│    ├── Submit()          提交函数                 │
│    ├── Run()             执行仿真                 │
│    └── Report()          生成报告                 │
├─────────────────────────────────────────────────┤
│              仿真系统核心                         │
│  SimSys (ModelTop.h) — 仿真系统顶层               │
│    ├── DeviceMachine[]  设备级机器                │
│    ├── AICPUMachine[]   AICPU 级机器              │
│    ├── CoreMachine[]    AI Core 级机器            │
│    ├── PipeMachine[]    Pipeline 级机器           │
│    ├── CacheMachine     L2 Cache                  │
│    ├── FunctionCache    函数缓存                  │
│    ├── GenCalendar      Calendar 调度器           │
│    ├── Scheduler        指令调度器                │
│    └── TraceLogger      追踪日志                  │
└─────────────────────────────────────────────────┘
```

### 3.2 硬件层级映射

CostModel 按照真实 Ascend NPU 的硬件层级进行建模：

```
Device（设备）
 └── AICPU（调度 CPU）
      ├── AI Core 0 (AIC/AIV/MIXAICORE)
      │    ├── Pipe: VECTOR_BMU    (UB 内存分配)
      │    ├── Pipe: MTE_IN        (GM→UB 数据搬入)
      │    ├── Pipe: VECTOR_ALU    (向量计算)
      │    ├── Pipe: MTE_OUT       (UB→GM 数据搬出)
      │    ├── Pipe: CUBE_BMU_L1   (L1 内存分配)
      │    ├── Pipe: CUBE_BMU_L0A  (L0A 内存分配)
      │    ├── Pipe: CUBE_BMU_L0B  (L0B 内存分配)
      │    ├── Pipe: CUBE_BMU_L0C  (L0C 内存分配)
      │    ├── Pipe: MTE1          (L1→L0 数据搬运)
      │    ├── Pipe: CUBE          (矩阵乘)
      │    ├── Pipe: PIPE_S        (标量/控制)
      │    └── Pipe: PIPE_FIX      (定点处理)
      ├── AI Core 1
      │    └── ...
      └── AI Core N
```

### 3.3 关键类关系

```
SimSys ─────────────────────────────────────────
  │ owns
  ├── vector<MachinePtr> machines
  │     ├── DeviceMachine
  │     │     └── vector<AICPUPtr> subMachines
  │     │           └── AICPUMachine
  │     │                 └── vector<shared_ptr<CoreMachine>> subMachines
  │     │                       └── CoreMachine
  │     │                             └── vector<shared_ptr<PipeMachine>> subMachines
  │     │                                   └── PipeMachine
  │     │                                         └── UnifiedPipeMachinePtr pipeImpl
  │     │                                               └── PipeSimulatorFast<PostSimulator>
  │     └── CacheMachine (L2)
  │
  ├── FunctionCache              # hash → Function 映射
  ├── GenCalendar                # Calendar 调度
  ├── TraceLogger                # 追踪日志
  ├── ModelConfig                # 仿真配置
  ├── ModelStats                 # 统计数据
  └── Reporter                   # 报告生成
```

---

## 4. 核心数据结构

### 4.1 Tile（数据块）

`ISA.h:314` — 代表一个数据缓冲区（tensor fragment）。


| 字段        | 类型                | 说明                                    |
| ----------- | ------------------- | --------------------------------------- |
| `magic`     | int                 | 唯一标识符                              |
| `symbol`    | string              | 符号名                                  |
| `dataType`  | DataType            | 数据类型（FP16/FP32/BF16...）           |
| `bufType`   | OperandType         | 缓冲区类型（UB/L1/L0A/L0B/L0C/FIX/DDR） |
| `pipeType`  | CorePipeType        | 所属 Pipeline 类型                      |
| `offset`    | vector\<int\>       | 数据偏移                                |
| `shape`     | vector\<int\>       | 数据形状                                |
| `nodeType`  | NodeType            | 节点类型（LOCAL/INCAST/OUTCAST）        |
| `producer`  | TileOpPtr           | 生产者算子                              |
| `consumers` | vector\<TileOpPtr\> | 消费者算子列表                          |
| `exeInfo`   | ExecuteInfo         | 执行状态信息                            |

### 4.2 TileOp（算子操作）

`ISA.h:361` — 代表一个具体的计算操作。


| 字段        | 类型              | 说明                            |
| ----------- | ----------------- | ------------------------------- |
| `magic`     | int               | 唯一标识符                      |
| `opcode`    | string            | 操作码（如 ADD/MUL/COPY_IN 等） |
| `pipeType`  | CorePipeType      | 所属 Pipeline                   |
| `iOperand`  | vector\<TilePtr\> | 输入操作数                      |
| `oOperand`  | vector\<TilePtr\> | 输出操作数                      |
| `exeInfo`   | ExecuteInfo       | 执行状态                        |
| `operation` | Operation*        | 关联的 PTO Operation            |
| `funcPtr`   | FunctionPtr       | 所属 Function                   |

### 4.3 Function（函数/子图）

`ISA.h:420` — 代表一个可调度的计算单元（叶子函数或组合函数）。


| 字段           | 类型                           | 说明                 |
| -------------- | ------------------------------ | -------------------- |
| `magic`        | int                            | 唯一标识符           |
| `pSgId`        | int                            | 所属子图 ID          |
| `functionHash` | uint64_t                       | 函数哈希             |
| `funcName`     | string                         | 函数名               |
| `machineType`  | MachineType                    | 执行机器类型         |
| `tileOps`      | vector\<TileOpPtr\>            | 包含的算子操作列表   |
| `tiles`        | vector\<TilePtr\>              | 包含的数据块列表     |
| `tileMap`      | map\<int, TilePtr\>            | magic → Tile 映射   |
| `tileOpMap`    | map\<int, TileOpPtr\>          | magic → TileOp 映射 |
| `invoke`       | map\<int, FunctionInvokeInfo\> | 函数调用信息         |
| `hasSchedule`  | bool                           | 是否已有调度信息     |
| `totalCycles`  | uint64_t                       | 总执行周期           |

### 4.4 Task（任务）

`CommonType.h:356` — 代表拓扑调度中的一个可执行任务。


| 字段           | 类型               | 说明             |
| -------------- | ------------------ | ---------------- |
| `seqNo`        | uint64_t           | 序列号           |
| `taskId`       | uint64_t           | 任务 ID          |
| `functionHash` | uint64_t           | 关联函数哈希     |
| `functionName` | string             | 函数名           |
| `predecessors` | vector\<uint64_t\> | 前驱任务         |
| `successors`   | vector\<uint64_t\> | 后继任务         |
| `machineType`  | MachineType        | 目标机器类型     |
| `fixedLatency` | bool               | 是否使用固定延迟 |
| `psgId`        | int                | 子图 ID          |

### 4.5 ExecuteInfo（执行信息）

`ISA.h:280` — Tile 和 TileOp 共享的执行状态。


| 字段          | 类型      | 说明           |
| ------------- | --------- | -------------- |
| `exePipeId`   | int       | 执行 Pipe ID   |
| `latency`     | uint64_t  | 延迟周期数     |
| `issued`      | bool      | 是否已发射     |
| `retired`     | bool      | 是否已完成     |
| `isAllocated` | bool      | 内存是否已分配 |
| `isWritten`   | bool      | 数据是否已写入 |
| `cycleInfo`   | CycleInfo | 周期详细信息   |

---

## 5. 核心枚举

### 5.1 SimMode — 仿真模式

```cpp
enum class SimMode {
    NORMAL = 0,        // 全量仿真：提交拓扑 + 所有叶子函数，模拟完整调度
    EMULATOR,          // 功能仿真：仅验证功能正确性
    LEAF_FUNCTION,     // 叶子函数仿真：仅仿真单个叶子函数的执行周期
    PV_MODEL           // 精度验证：运行 PvModel 验证数值精度
};
```

### 5.2 MachineType — 机器类型

```cpp
enum class MachineType {
    UNKNOWN,
    DEVICE,       // 设备级
    CPU,          // AICPU（调度 CPU）
    AIC,          // AI Core（Cube 核心）
    AIV,          // AI Core（Vector 核心）
    MIXAICORE,    // AI Core（混合核心 = Cube + Vector）
    PIPE,         // Pipeline 级
    CACHE,        // Cache
    HUB,          // Hub 核心
};
```

### 5.3 CorePipeType — Pipeline 类型

```cpp
enum class CorePipeType {
    PIPE_UNKNOW = -1,
    PIPE_TILE_ALLOC = 0,    // Tile 内存分配
    PIPE_VECTOR_BMU,        // Vector BMU（UB 分配）
    PIPE_CUBE_BMU_L1,       // Cube BMU L1
    PIPE_CUBE_BMU_L0A,      // Cube BMU L0A
    PIPE_CUBE_BMU_L0B,      // Cube BMU L0B
    PIPE_CUBE_BMU_L0C,      // Cube BMU L0C
    PIPE_MTE_IN,            // 内存搬运入（GM→UB/L1）
    PIPE_MTE1,              // 内存搬运 L1→L0
    PIPE_VECTOR_ALU,        // 向量计算 ALU
    PIPE_CUBE,              // 矩阵乘（Cube）
    PIPE_MTE_OUT,           // 内存搬出（UB/L1→GM）
    PIPE_S,                 // 标量/控制操作
    PIPE_CALL,              // 函数调用
    PIPE_FIX,               // 定点处理
    TOTAL_CORE_PIPE_TYPE
};
```

### 5.4 NodeType — 节点类型

```cpp
enum class NodeType {
    LOCAL,     // 本地中间数据
    INCAST,    // 子图输入数据
    OUTCAST,   // 子图输出数据
};
```

---

## 6. ISA 指令映射

`ISA.h:35-278` 定义了 `SCHED_CORE_PIPE_TYPE` 映射表，将算子操作码（opcode）映射到对应的 Pipeline 类型。

### 6.1 按 Pipeline 分类的操作


| Pipeline            | 操作                                                                                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PIPE_VECTOR_ALU** | EXP, NEG, SQRT, RSQRT, ADD, SUB, MUL, DIV, CAST, RELU, LN, GATHER, SCATTER, CONCAT, CUM_SUM, TOPK, SORT, WHERE_**, CMP, MAXIMUM, MINIMUM, ABS, FLOOR, CEIL, SIGN, ... |
| **PIPE_CUBE**       | A_MUL_B, A_MULACC_B, A_MUL_Bt, At_MUL_B, At_MUL_Bt, CONV, CONV_ADD, CONCAT_C                                                                                          |
| **PIPE_MTE_IN**     | COPY_IN, UB_COPY_IN, L1_COPY_IN, GATHER_IN_L1, RESHAPE_COPY_IN, TRANSPOSE_MOVEIN, SHMEM_GET                                                                           |
| **PIPE_MTE_OUT**    | COPY_OUT, UB_COPY_OUT, L1_COPY_OUT, L0C_COPY_OUT, RESHAPE_COPY_OUT, TRANSPOSE_MOVEOUT, REMOTE_GATHER, SHMEM_PUT                                                       |
| **PIPE_MTE1**       | L1_TO_L0A, L1_TO_L0B, L1_TO_BT, FIX_COPY_IN, L1_COPY_UB, UB_COPY_L1                                                                                                   |
| **PIPE_S**          | RESHAPE, VIEW, ASSEMBLE, LOAD, SYNC_SRC, SYNC_DST, BAR.V, BAR.M, BAR.ALL, PHASE1, PHASE2, PERMUTE                                                                     |
| **PIPE_CALL**       | CALL                                                                                                                                                                  |

### 6.2 缓冲区分配操作


| 操作      | Pipeline          | 说明         |
| --------- | ----------------- | ------------ |
| UB_ALLOC  | PIPE_VECTOR_BMU   | UB 内存分配  |
| L1_ALLOC  | PIPE_CUBE_BMU_L1  | L1 内存分配  |
| L0A_ALLOC | PIPE_CUBE_BMU_L0A | L0A 内存分配 |
| L0B_ALLOC | PIPE_CUBE_BMU_L0B | L0B 内存分配 |
| L0C_ALLOC | PIPE_CUBE_BMU_L0C | L0C 内存分配 |

### 6.3 机器类型与 Pipeline 映射

```
MACHINE_PIPE_SET 定义：
  AIV:       { VECTOR_BMU, MTE_IN, VECTOR_ALU, MTE_OUT }
  AIC:       { CUBE_BMU_L1, CUBE_BMU_L0A, CUBE_BMU_L0B, CUBE_BMU_L0C,
               MTE_IN, MTE1, CUBE, MTE_OUT }
  MIXAICORE: { CUBE_BMU_L1, CUBE_BMU_L0A, CUBE_BMU_L0B, CUBE_BMU_L0C,
               MTE_IN, MTE1, CUBE, VECTOR_ALU, MTE_OUT }
```

---

## 7. 仿真流程

### 7.1 完整仿真流程（CostModelRunOnce）

```
CostModelLauncher::RunModel()
  │
  ├── 1. 初始化设备与分布式上下文
  │     DeviceInitDistributedContext()
  │     DeviceInitTilingData()
  │     InitKernelInOuts()
  │
  ├── 2. RunCostModel() — Phase 1: LEAF_FUNCTION 仿真
  │     ├── 设置 SimMode = LEAF_FUNCTION
  │     ├── CostModelAgent::SubmitLeafFunctionsToCostModel()
  │     │     └── 遍历所有叶子函数，提交到仿真系统
  │     ├── CostModelAgent::RunCostModel()
  │     │     └── 对每个叶子函数独立仿真，得到 functionTime[hash]
  │     └── CostModelAgent::TerminateCostModel()
  │           └── 收集 functionTime 写入 kArgs.costmodeldata
  │
  ├── 3. RunTestMode() — 功能仿真
  │     └── 多线程执行 DynTileFwkBackendKernelServer
  │           （模拟 AICPU 调度行为）
  │
  ├── 4. RunDynCostModel() — Phase 2: NORMAL 仿真
  │     ├── 设置 SimMode = NORMAL
  │     ├── CostModelAgent::SubmitTopo(dyn_topo.txt)
  │     │     └── 解析拓扑文件，生成任务依赖图
  │     ├── CostModelAgent::SubmitLeafFunctionsToCostModel()
  │     ├── CostModelAgent::RunCostModel()
  │     │     └── 基于拓扑和 functionTime 调度全量仿真
  │     └── CostModelAgent::TerminateCostModel()
  │
  └── 5. RunPvModel() — 精度验证 (需要 BUILD_WITH_CANN)
        ├── PvModelFactory::CreateDyn()
        ├── PvModel::InitPv()
        ├── PvModel::Codegen(function)
        └── 执行精度仿真，CopyTensorFromDev()
```

### 7.2 子图仿真流程（CostModelRunSubgraph）

```
CostModelLauncher::CostModelRunSubgraph(function, pSgId)
  │
  ├── Phase 1: RunSubgraphCostModel() — 叶子函数仿真
  │     ├── SimMode = LEAF_FUNCTION
  │     ├── CostModelAgent::SubmitLeafFunctionsBySubgraph(pSgId)
  │     │     └── 仅提交指定子图的叶子函数
  │     ├── CostModelAgent::RunCostModel()
  │     └── 收集每个函数的 {hash, cycles, name, machine_type}
  │
  └── Phase 2: RunSubgraphDynCostModel() — 拓扑仿真
        ├── SimMode = NORMAL
        ├── CostModelAgent::SubmitSubgraphTopoByPid(path, pSgId)
        │     └── 加载过滤后的子图拓扑
        ├── CostModelAgent::SubmitLeafFunctionsBySubgraph(pSgId)
        ├── CostModelAgent::RunCostModel()
        └── 返回 sim->globalCycles（子图总周期数）
```

### 7.3 仿真系统执行循环

```
CostModelInterface::Run()
  └── sim->Step() 循环
        │
        ├── 更新全局时钟 globalCycles
        ├── 遍历所有 machines
        │     └── machine->Step()
        │           ├── DeviceMachine::Step()
        │           │     └── 处理 submissionQueue → 调度到 AICPU
        │           ├── AICPUMachine::Step()
        │           │     └── 处理任务 → 调度到 CoreMachine
        │           ├── CoreMachine::Step()
        │           │     └── 遍历 PipeMachine → pipeMachine->Step()
        │           └── PipeMachine::Step()
        │                 ├── 从队列取 TileOp
        │                 ├── pipeImpl->Simulate(tileOp)  ← 计算延迟
        │                 ├── 更新 Tile 状态
        │                 └── 通知后续消费者
        │
        ├── 检查终止条件（所有队列空 & 无执行中任务）
        └── 检查死锁
```

---

## 8. Pipe 仿真引擎

### 8.1 PipeSimulatorFast

`arch/PipeSimulatorFast.h` — 高性能 Pipe 仿真引擎，是性能建模的核心。

```cpp
template <typename PostSimulator>
class PipeSimulatorFast : public PipeMachineImpl {
    // Simulate()      — 对 TileOp 进行延迟仿真
    // PostSimulate()  — 后处理仿真（架构相关修正）
    // SimulateForPass()  — Pass 阶段的快速延迟查询
};
```

**工作原理：**

1. 根据 TileOp 的 opcode 和操作数信息，计算指令延迟
2. 利用 `tileopLatencyCacheMp` 缓存已计算的延迟，避免重复计算
3. 通过模板参数 `PostSimulator`（A2A3 或 A5）注入架构特定的后处理

### 8.2 PipeFactory

`arch/PipeFactory.h/cpp` — 工厂模式创建 Pipe 仿真器。

根据 `MachineType` 和 `CorePipeType` 创建对应的 `PipeSimulatorFast` 实例：


| MachineType   | CorePipeType    | 创建的 Pipe 类型                          |
| ------------- | --------------- | ----------------------------------------- |
| AIV           | PIPE_VECTOR_BMU | TileAllocPipeImpl                         |
| AIV           | PIPE_MTE_IN     | PipeSimulatorFast\<PostSimulatorA2A3/A5\> |
| AIV           | PIPE_VECTOR_ALU | PipeSimulatorFast\<PostSimulatorA2A3/A5\> |
| AIV           | PIPE_MTE_OUT    | PipeSimulatorFast\<PostSimulatorA2A3/A5\> |
| AIC/MIXAICORE | PIPE_CUBE_BMU_* | TileAllocPipeImpl / CubeBMUPipeImpl       |
| AIC/MIXAICORE | PIPE_MTE1       | PipeSimulatorFast\<PostSimulatorA2A3/A5\> |
| AIC/MIXAICORE | PIPE_CUBE       | PipeSimulatorFast\<PostSimulatorA2A3/A5\> |

### 8.3 GetCyclesForPass

`arch/PipeSimulatorFast.h` 导出的 C 接口，供编译 Pass 直接查询指令延迟：

```cpp
extern "C" int64_t GetCyclesForPass(
    const std::string& op,
    const std::vector<std::vector<int>>& shape,
    DataType dtype);
```

---

## 9. 调度器

### 9.1 Scheduler（指令调度器）

`machine/Scheduler.h` — 在 CoreMachine 内部对 TileOp 进行排序和发射。

**核心功能：**

- `TileInsertQueue()` — 按 DOM_COUNT 策略排序 Tile 分配顺序
- `TileOpInsertQueue()` — 按依赖关系排序 TileOp 执行顺序
- `SortTile()` — 综合排序，生成调度序列
- `MergeCopyOutGroup()` — 合并连续的 COPY_OUT 操作

### 9.2 GenCalendar（Calendar 调度器）

`arch/GenCalendar/GenCalendar.h` — 基于日历模型的调度器。

在 `SimMode::NORMAL` 模式下使用，模拟多核并行执行的时序排布。支持将任务分配到不同 AI Core 并生成调度日历。

---

## 10. 缓存系统

### 10.1 FunctionCache

`cache/FunctionCache.h` — 存储已解析的 Function 对象。

```cpp
class FunctionCache {
    std::unordered_map<uint64_t, FunctionPtr> cache;  // hash → Function
};
```

- 叶子函数首次解析后缓存，避免重复解析
- 供 NORMAL 模式查询函数执行时间

### 10.2 CacheMachine（L2 Cache）

`cache/CacheMachine.h` — L2 Cache 仿真。

- 模拟 GM（Global Memory）与 L2 Cache 之间的数据交互
- MTE_IN/MTE_OUT 操作会触发 Cache 请求
- 支持数据读写请求和命中/缺失建模

---

## 11. 配置系统

配置类层次：

```
ModelConfig（全局配置）
  ├── DeviceConfig（设备配置：AI Core 数量、频率等）
  │     ├── AICPUConfig（AICPU 配置：AICPU 数量、调度策略）
  │     ├── CoreConfig（Core 配置：核心类型、数量）
  │     └── PipeConfig（Pipeline 配置：各 Pipe 延迟参数）
  ├── CacheConfig（Cache 配置：L2 大小、带宽）
  ├── TraceConfig（追踪配置：日志级别、输出格式）
  └── EnvConfig（环境配置：输出目录等）
```

**关键配置参数：**


| 配置键                        | 说明                      | 默认值     |
| ----------------------------- | ------------------------- | ---------- |
| `KEY_SIM_MODE`                | 仿真模式                  | NORMAL     |
| `KEY_ACCURACY_LEVEL`          | 仿真精度（1=LOW, 2=HIGH） | 1          |
| `KEY_PV_LEVEL`                | 精度验证级别              | PV_NON     |
| `KEY_LOG_LEVEL`               | 日志级别（1~5）           | 3          |
| `KEY_EXECUTE_CYCLE_THRESHOLD` | 最大仿真周期              | UINT64_MAX |
| `KEY_ENABLE_DYN_COST_MODEL`   | 是否启用动态 CostModel    | true       |

---

## 12. 统计与报告

### 12.1 统计层次

```
ModelStats（模型级统计）
  ├── DeviceStats（设备级统计）
  │     ├── AICPUStats（AICPU 统计：任务数、总周期）
  │     └── CoreStats（Core 统计：各 Pipe 利用率）
  └── CacheStats（Cache 统计：命中率、带宽）
```

### 12.2 输出格式


| 格式           | 方法                          | 说明                     |
| -------------- | ----------------------------- | ------------------------ |
| JSON           | `OutputConfig()`              | 配置和统计数据           |
| Perfetto Trace | `OutputPerfettoTrace()`       | Chrome Perfetto 格式追踪 |
| Swim Lane      | `OutputLogForSwimLane()`      | 泳道图数据               |
| Pipe Swim Lane | `OutputLogForPipeSwimLane()`  | Pipeline 级泳道图        |
| Calendar       | `OutputCalendarScheduleCpp()` | Calendar 调度结果        |
| Function Time  | `DumpFunctionExecuteTime()`   | 各函数执行周期           |

### 12.3 TraceLogger

`statistics/TraceLogger.h` — 记录仿真过程中的事件追踪：

- 任务开始/结束
- Pipe 执行开始/结束
- 数据搬入/搬出事件
- 支持输出为 Perfetto 格式用于可视化分析

---

## 13. Python 接口

### 13.1 绑定层

`python/src/bindings/cost_model.cpp` 通过 pybind11 导出两个函数：

```cpp
void BindCostModelRuntime(py::module& m) {
    m.def("CostModelRunOnceDataFromHost", &CostModelRunOnceDataFromHost);
    m.def("CostModelRunSubgraphLine", &CostModelRunSubgraphLine);
}
```

### 13.2 Python API

`python/pypto/cost_model.py` 提供两个公开接口：

#### `_cost_model_run_once_data_from_host(inputs, outputs)`

执行完整的单次仿真（包括 CostModel + TestMode + DynCostModel + PvModel）。

**参数：**

- `inputs`: `List[pypto.Tensor]` — 输入张量
- `outputs`: `List[pypto.Tensor]` — 输出张量

**处理流程：**

1. 自动检测张量是否在 Device 上，如是则先拷贝到 Host
2. 转换为 `DeviceTensorData` 格式
3. 调用 C++ 后端 `CostModelRunOnceDataFromHost()`
4. 如原始数据在 Device 上，将结果拷贝回 Device

#### `_cost_model_run_subgraph_line(inputs, outputs, *, p_sg_id) -> dict`

执行指定子图的仿真，返回各叶子函数周期和子图总周期。

**参数：**

- `inputs`: `List[pypto.Tensor]` — 输入张量
- `outputs`: `List[pypto.Tensor]` — 输出张量
- `p_sg_id`: `int` — 子图 ID

**返回值（dict）：**

```json
{
    "status": "success" | "error",
    "error_msg": "",
    "p_sg_id": 0,
    "subgraph_total_cycles": 12345,
    "functions": [
        {
            "hash": 123456789,
            "cycles": 5000,
            "name": "func_name",
            "machine_type": 4
        }
    ],
    "output_dir": "/path/to/CostModelSimulationOutput"
}
```

---

## 14. 精度验证（PvModel）

### 14.1 概述

PvModel 用于在不依赖真实 NPU 的情况下验证算子的数值精度。它在 Host 端模拟 NPU 的执行行为，计算输出张量并与预期结果对比。

### 14.2 关键组件


| 组件              | 文件                             | 说明                      |
| ----------------- | -------------------------------- | ------------------------- |
| PvModel           | `pv/PvModel.h`                   | PvModel 接口（静态/动态） |
| DynPvModel        | `simulation_pv/PvModelImpl.h`    | 动态 PvModel 实现         |
| PvModelFactory    | `pv/PvModelFactory.h`            | PvModel 工厂              |
| PvMemAllocator    | `simulation_pv/PvMemAllocator.h` | Pv 内存管理               |
| AiCorePvModelImpl | `cost_model_launcher.h`          | AI Core PvModel 实现      |

### 14.3 PvModel 执行流程

```
RunPvModel()
  ├── PvModelFactory::CreateDyn()     创建动态 PvModel
  ├── pv_->InitPv()                   初始化
  ├── pv_->Codegen(function)          从函数生成执行代码
  ├── BuildPvKernelArgs()             构建内核参数
  │     ├── pv_->CopyTensorToDev()    拷贝张量到仿真空间
  │     └── 构建 DevTensorData
  ├── RunTestMode()                   执行仿真
  └── pv_->CopyTensorFromDev()        拷贝结果回来
```

---

## 15. 内存层级建模

CostModel 模拟的 Ascend NPU 内存层级：

```
┌─────────────────────────────────────────┐
│              GM (Global Memory/DDR)     │  ← 最慢，最大
├─────────────────────────────────────────┤
│              L2 Cache                   │  ← CacheMachine 仿真
├─────────────────────────────────────────┤
│              L1 Buffer                  │  ← PIPE_CUBE_BMU_L1 管理
├─────────────────────────────────────────┤
│  L0A Buffer  │  L0B Buffer  │  L0C     │  ← PIPE_CUBE_BMU_L0A/B/C 管理
├─────────────────────────────────────────┤
│              UB (Unified Buffer)        │  ← PIPE_VECTOR_BMU 管理
└─────────────────────────────────────────┘
       ↑ 最快，最小
```

**数据搬运 Pipe：**

- `PIPE_MTE_IN`: GM → UB / L1 (通过 L2 Cache)
- `PIPE_MTE_OUT`: UB / L1 / L0C → GM (通过 L2 Cache)
- `PIPE_MTE1`: L1 → L0A / L0B
- `FIX_COPY_IN`: 定点数据预处理

---

## 16. 辅助组件

### 16.1 MemoryHelper

`cost_model_launcher.h` — 仿真模式下的内存管理辅助类。

- 在仿真模式下（`isTest=true`），所有内存操作在 Host 端用 `malloc` 模拟
- `CopyToDev()` / `CopyFromDev()` — 模拟 Device 内存拷贝
- `AllocDev()` / `AllocZero()` — 模拟 Device 内存分配

### 16.2 HostAgentStub

`cost_model_launcher.h` — 单例模式的 Host 内存管理器。

- 管理 Host 端分配的所有内存
- 在 `Finalize()` 时统一释放
- 避免仿真过程中的内存泄漏

### 16.3 TileCalculator / TileState

`value/TileCalculator.h` / `value/TileState.h` — Tile 计算和状态管理。

- 管理 Tile 的内存分配和数据生命周期
- 跟踪 Tile 的读写引用计数
- 协调与 Pipeline 执行的数据流

---

## 17. 设计特点

1. **分层解耦**：仿真系统与真实运行时共享相同的 Function/Tile/TileOp 数据结构，通过 CostModelAgent 适配
2. **模式切换**：通过 `SimMode` 枚举在 LEAF_FUNCTION/NORMAL/PV_MODEL 间切换，支持不同粒度的仿真
3. **架构适配**：通过模板参数 `PostSimulator` 和 `PipeFactory` 工厂模式支持多代 Ascend 架构
4. **延迟缓存**：`PipeSimulatorFast` 缓存已计算的指令延迟，避免重复计算
5. **Calendar 调度**：NORMAL 模式使用日历模型模拟多核并行执行
6. **灵活输出**：支持 JSON、Perfetto、泳道图等多种输出格式，便于性能分析
