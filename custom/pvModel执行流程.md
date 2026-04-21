# PvModel 执行流程知识库

> 本文档用于记录 PvModel 仿真系统的执行逻辑、过程、输入输出及相关原理。
> 随着对代码的理解逐步深入，本文档将持续完善。

---

## 重要说明 ⚠️

### 架构选择

本文档 **仅关注 A5 架构**，A2/A3 架构不再重点关注。

- **入口函数：** `CreateDynPvModelImplA5` (`PvModelFactory.cpp`)
- **忽略架构：** A2、A3 的相关实现代码不在分析范围内

### 实现选择

PvModel 存在两套实现架构：

| 实现类 | 状态 | 分析重点 | 说明 |
|--------|------|----------|------|
| **PvModelImpl** | ❌ 不再维护 | **忽略** | 静态 PvModel 实现，通过 `std::system(cmd)` 命令调用外部测试程序，已废弃 |
| **DynPvModelImpl** | ✅ 当前架构 | **重点关注** | 动态 PvModel 实现，通过 `libpem_davinci.so` 动态库获取仿真能力 |

**静态 PvModel (PvModelImpl) 废弃原因：**
- 通过外部命令调用测试框架 (`PvModel%s --gtest_filter=...`)
- 执行效率低，无法与框架深度集成
- 已停止演进和维护

**动态 PvModel (DynPvModelImpl) 优势：**
- 直接调用 PEM 模拟器 API
- 执行效率高，与框架深度集成
- 支持动态算子仿真
- 是当前及未来的主流架构

**本文档后续所有分析均针对 `DynPvModelImpl` 实现。**

---

## 目录

1. [概述](#1-概述)
2. [执行上下文](#2-执行上下文)
3. [核心流程](#3-核心流程)
4. [关键函数详解](#4-关键函数详解)
5. [输入输出分析](#5-输入输出分析)
6. [原理与设计](#6-原理与设计)
7. [待深入挖掘的问题](#7-待深入挖掘的问题)

---

## 1. 概述

### 1.1 PvModel 是什么？

PvModel 是 PyPTO 框架中的精确性能仿真模型，用于在无 NPU 硬件环境下模拟算子执行，获取精确的性能数据和执行结果。

### 1.2 核心文件位置

| 文件 | 路径 | 职责 | 分析重点 |
|------|------|------|----------|
| PvModelFactory.cpp | `framework/src/cost_model/simulation/pv/PvModelFactory.cpp` | PvModel 工厂类，创建 DynPvModelImpl 实例 | ✅ 重点 |
| PvModelImpl.h | `framework/src/cost_model/simulation_pv/PvModelImpl.h` | **DynPvModelImpl** 实现类定义 | ✅ 重点 |
| PvModelImpl.cpp | `framework/src/cost_model/simulation_pv/PvModelImpl.cpp` | **DynPvModelImpl** 实现类 | ✅ 重点 |
| cost_model_launcher.h | `framework/src/cost_model/simulation/cost_model_launcher.h` | CostModel 启动器，调用 PvModel | ✅ 重点 |

> ⚠️ **注意：** `PvModelImpl.h/cpp` 中同时包含 `PvModelImpl`（静态版本）和 `DynPvModelImpl`（动态版本）两个类的实现。分析时需区分：
> - `PvModelImpl<SystemConfig, CaseConfig>` → 静态版本，**忽略**
> - `DynPvModelImpl<SystemConfig, CaseConfig>` → 动态版本，**重点关注**

### 1.3 动态库依赖

| 动态库 | 来源 | 作用 |
|--------|------|------|
| `libtile_fwk_simulation_pv.so` | 当前项目编译 (`framework/src/cost_model/simulation_pv/CMakeLists.txt`) | PvModel 实现库 |
| `libpem_davinci.so` | CANN (`$ASCEND_HOME_PATH/toolkit/tools/simulator/<arch>/lib/`) | PvModel 仿真引擎（PEM 模拟器） |

---

## 2. 执行上下文

### 2.1 上游调用

PvModel 的执行入口位于 `CostModelLauncher::RunModel()` 函数：

```cpp
// cost_model_launcher.h:207-229
void RunModel(const std::vector<RawTensorDataPtr> &inputs, 
              const std::vector<RawTensorDataPtr> &outputs) {
    // ... 前置初始化 ...
    RunCostModel(&kArgs);              // 运行 CostModel 获取时间成本
    RunTestMode(&kArgs, DEVICE_MAX_AICPU_NUM);  // 第一次仿真执行
    RunDynCostModel();                 // 动态 CostModel 分析拓扑
#ifdef BUILD_WITH_CANN
    RunPvModel(kArgs, inputs, outputs); // PvModel 精确仿真
#endif
}
```

**上游触发条件：**
- `BUILD_WITH_CANN` 宏生效（需要 `ASCEND_HOME_PATH` 环境变量）
- 运行模式为仿真模式 (`CFG_RUN_MODE_SIM`)

### 2.2 下游输出

PvModel 执行完成后：

| 输出 | 说明 |
|------|------|
| 仿真结果数据 | 输入/输出 tensor 的执行结果（通过 `CopyFromDev` 拷贝回主机） |
| 二进制文件 | `.bin` 文件（纯机器指令）存放在临时目录 |
| 性能数据 | 通过 PEM 模拟器获取的精确性能指标 |

### 2.3 完整调用链

```
用户代码
    │
    ▼
Program::GetInstance().Execute() / assemble()
    │
    ▼
CostModelLauncher::CostModelRunOnce()
    │
    ▼
CostModelLauncher::RunDynamic()
    │
    ▼
CostModelLauncher::RunModel()
    ├─ RunCostModel()         → CostModel 时间成本分析
    ├─ RunTestMode()          → 第一次仿真执行
    ├─ RunDynCostModel()      → 动态拓扑分析
    │
    ▼ [BUILD_WITH_CANN]
RunPvModel()
    ├─ InitPv()               → 加载 PEM 模拟器动态库
    ├─ Codegen()              → 生成 CCE 并编译为 .bin
    ├─ BuildPvKernelArgs()    → 构建仿真参数
    ├─ RunTestMode()          → 第二次仿真执行（使用 PvModel 参数）
    ├─ SetDevPtr()            → 设置设备指针
    └─ CopyFromDev()          → 拷贝结果回主机
```

---

## 3. 核心流程

### 3.1 RunPvModel 函数流程

```cpp
// cost_model_launcher.h:341-361
void RunPvModel(DeviceKernelArgs &kArgs, 
                const std::vector<RawTensorDataPtr> &inputs, 
                const std::vector<RawTensorDataPtr> &outputs) {
    // 1. 初始化 PvModel
    pv_ = CostModel::PvModelFactory::CreateDyn();
    pv_->InitPv();
    
    // 2. 创建 AiCore 模型实现
    model_ = std::make_shared<AiCorePvModelImpl>(pv_);
    
    // 3. Codegen - 生成 CCE 并编译
    pv_->Codegen(function_);
    
    // 4. 构建仿真参数
    BuildPvKernelArgs(kArgs, inputs, outputs);
    
    // 5. 执行仿真
    RunTestMode(&kArgs, maxCpuNum);
    
    // 6. 拷贝结果
    SetDevPtr(inputs, outputs);
    CopyFromDev(inputs, outputs);
}
```

### 3.2 各步骤详解

#### Step 1: InitPv()

**职责：** 加载 PEM 模拟器动态库，获取仿真 API 函数指针。

**关键代码：**
```cpp
// PvModelImpl.h:307-328
void InitPv() {
    auto archType = Platform::Instance().GetSoc().GetNPUArch();
    const char *ascendHome = std::getenv("ASCEND_HOME_PATH");
    std::string soPath = std::string(ascendHome) + 
        "/toolkit/tools/simulator/" + archTypeStr + "/lib/libpem_davinci.so";
    
    void *handle = dlopen(soPath.c_str(), RTLD_LAZY);
    
    // 加载 PEM 模拟器 API
    pv_init_ = (PvInitFunc)load_symbol(handle, "pv_init");
    pv_launch_sub_core_ = (PvLaunchSubCoreFunc)load_symbol(handle, "pv_launch_sub_core");
    pv_step_ = (PvStepFunc)load_symbol(handle, "pv_step");
    pv_mem_write_ = (PvMemWriteFunc)load_symbol(handle, "pv_mem_write");
    pv_mem_read_ = (PvMemReadFunc)load_symbol(handle, "pv_mem_read");
    pv_reg_write_ = (PvRegWriteFunc)load_symbol(handle, "pv_reg_write");
    pv_set_toml_ = (PvSetTomalFunc)load_symbol(handle, "set_toml");
}
```

**加载的 API：**

| API | 作用 |
|-----|------|
| `pv_init` | 初始化模拟器 |
| `pv_launch_sub_core` | 加载二进制并启动子核执行 |
| `pv_step` | 单步执行仿真 |
| `pv_mem_write` | 写入模拟器内存 |
| `pv_mem_read` | 读取模拟器内存 |
| `pv_reg_write` | 写入模拟器寄存器 |
| `set_toml` | 设置配置文件 |

#### Step 2: Codegen()

**职责：** 为每个叶子函数生成 CCE 源码，编译为二进制文件。

**详细流程见 [4.1 Codegen 函数详解](#41-codegen-函数详解)**

#### Step 3: BuildPvKernelArgs()

**职责：** 构建 PvModel 仿真所需的 Kernel 参数。

**关键代码：**
```cpp
// cost_model_launcher.h:363-413
void BuildPvKernelArgs(DeviceKernelArgs &kArgs, ...) {
    // 1. 设置设备参数
    devProg->devArgs.nrAicpu = 6;
    devProg->devArgs.nrValidAic = 24;
    
    // 2. 分配内存地址
    devProg->devArgs.runtimeDataRingBufferAddr = pv_->AllocWorkspaceDev(DEV_ARGS_SIZE);
    devProg->workspaceSize = devProg->memBudget.Total();
    
    // 3. 构建 tensor 信息
    buildInouts(inputs, dataPtr);
    buildInouts(outputs, dataPtr);
    
    // 4. 设置参数地址
    kArgs.inputs = (int64_t *)pv_->CopyToDev(tensorInfo.data(), tensorSize);
    kArgs.workspace = (int64_t *)pv_->AllocWorkspaceDev(devProg->workspaceSize);
    kArgs.cfgdata = (int64_t *)pv_->CopyToDev(devProgData.data(), devProgData.size());
    kArgs.aicoreModel = model_.get();  // 关联 AiCorePvModelImpl
}
```

#### Step 4: RunTestMode()

**职责：** 启动多线程仿真引擎，模拟 AICPU 核心执行。

**关键代码：**
```cpp
// cost_model_launcher.h:440-474
void RunTestMode(DeviceKernelArgs *kArgs, int maxCpuNum) {
    std::vector<std::thread> aicpus(maxCpuNum);
    
    for (int i = 0; i < threadNum; i++) {
        aicpus[i] = std::thread([&]() {
            if ((devProg->devArgs.enableCtrl == 0) && 
                (uint32_t)tidx == devProg->devArgs.scheCpuNum) {
                // 调度线程
                PyptoKernelCtrlServer(kArgs);
            } else {
                // 执行线程
                DynTileFwkBackendKernelServer(kArgs);
            }
        });
    }
    
    // 等待所有线程完成
    for (int i = 0; i < threadNum; i++) {
        aicpus[i].join();
    }
}
```

**线程角色：**

| 线程类型 | 数量 | 职责 |
|----------|------|------|
| 调度线程 (`PyptoKernelCtrlServer`) | 1 | 任务调度、分发 |
| 执行线程 (`DynTileFwkBackendKernelServer`) | N | 执行具体 kernel 任务 |

#### Step 5: CopyFromDev()

**职责：** 将仿真结果从模拟器内存拷贝回主机内存。

---

## 4. 关键函数详解

### 4.1 Codegen 函数详解

**位置：** `PvModelImpl.h:344-399`

#### 输入

| 输入项 | 来源 | 说明 |
|--------|------|------|
| `func` | `npu::tile_fwk::Function*` | PyPTO Function 对象 |
| `leafDict` | `func->GetDyndevAttribute()->funcGroup.devRootList` | 所有叶子函数 |
| `leafFuncAttr->binPath` | 叶子函数属性 | 原始 `.o` 文件路径 |
| `ASCEND_HOME_PATH` | 环境变量 | CANN 安装路径 |
| `PTO_TILE_LIB_CODE_PATH` | 环境变量（可选） | PTO-ISA 库路径 |

#### 输出

| 输出项 | 说明 |
|--------|------|
| `srcPath` | `*_pvmodel.cpp` - 添加了 PvModel 入口的源文件 |
| `objPath` | `*.o` - 编译生成的目标文件 |
| `binPath` | `*.bin` - 提取 `.text` 段的纯二进制文件 |
| `cceBin` | 存储每个叶子函数的编译信息映射 |

#### 流程图

```
Codegen(func)
    │
    ├─ 1. 收集叶子函数
    │   └─ 遍历 devRootList → 提取所有 leafFunction
    │
    ├─ 2. 遍历每个叶子函数
    │   │
    │   ├─ 2.1 复制源文件
    │   │   orgSrcPath = binPath[:-1] + "cpp"    // .bin → .cpp (假设 .bin 后缀改为 .cpp)
    │   │   srcPath = binPath[:-2] + "_pvmodel.cpp"
    │   │   CopyFile(orgSrcPath, srcPath)
    │   │
    │   ├─ 2.2 AddKernelEntry()
    │   │   • 添加函数声明: extern "C" [aicore] void KernelName(...)
    │   │   • 添加入口函数: PvModelKernelEntry(funcData, opAttrOffset)
    │   │
    │   ├─ 2.3 编译 CCE
    │   │   bisheng -c -O3 -g -x cce -std=c++17
    │   │       -D__AIC__/-D__AIV__
    │   │       -D__DAV_V310
    │   │       --cce-aicore-only
    │   │       --cce-aicore-arch=<arch>
    │   │       -I<include_paths>
    │   │       -o objPath srcPath
    │   │
    │   ├─ 2.4 生成二进制
    │   │   llvm-objcopy -O binary -j .text objPath binPath
    │   │
    │   └─ 2.5 存储编译结果
    │   cceBin[hash] = PvModelCceBin(psgId, hash, coreType, srcPath, binPath)
    │
    └─ 3. 特殊处理 DummyFunction
        cceBin[hash] = PvModelCceBin(..., CoreType::HUB)
```

#### 源文件来源分析

**binPath 的来源链路：**

```
Pass阶段                    CodeGen阶段                  PvModel阶段
    │                           │                            │
    ▼                           ▼                            ▼
subgraph_to_function       GenCode()                   pv_->Codegen()
    │                           │                            │
    │                      GenFuncBody()                   │
    │                      (生成CCE源码)                    │
    │                           │                            │
    │                      GenCodeToBinaryTask()            │
    │                      (生成.cpp + 编译为.o)             │
    │                           │                            │
    │                      UpdateSubFunc()                  │
    │                      (设置binPath=".o")               │
    │                           │                            │
    ▼                           ▼                            ▼
创建空的                binPath被设置             从binPath推断
LeafFuncAttribute        为.o文件路径              .cpp路径并重新编译
```

**关键代码路径：**

1. **Pass 阶段创建空属性：**
   ```cpp
   // subgraph_to_function.cpp:438
   leafFunc->SetLeafFuncAttribute(std::make_shared<LeafFuncAttribute>());
   ```

2. **CodeGen 阶段生成源码和编译：**
   ```cpp
   // codegen_cloudnpu.cpp:252-257
   GenFuncBodyBefore(subFuncPair, topFunc, compileInfo, leafKernelFunc);
   GenFuncBody(*subFunc, topFunc, leafKernelFunc);  // 生成源码
   GenFuncEnd(leafKernelFunc);
   GenCodeToBinaryTask(leafKernelFunc, compileInfo, "");  // 编译
   ```

3. **CodeGen 阶段设置 binPath：**
   ```cpp
   // codegen_cloudnpu.cpp:287
   attr->binPath = compileInfo.GetBinAbsPath();  // 设置为 .o 文件路径
   ```

#### AddKernelEntry 的作用

**添加的代码：**

```cpp
// 函数声明
extern "C" [aicore] void KernelName(CoreFuncParam* param, 
    int64_t GMStackBase, 
    __gm__ int64_t *hcclContext, 
    __gm__ GMTensorInfo* oriAddrParam);

// PvModel 入口函数
extern "C" __global__ [aicore] void PvModelKernelEntry(
    __gm__ npu::tile_fwk::DynFuncData *funcData, 
    __gm__ uint64_t *opAttrOffset) {
    CoreFuncParam param = {
        funcData, 
        &funcData->opAttrs[opAttrOffset[0]], 
        funcData->exprTbl
    };
    KernelName(&param, 
        funcData->stackWorkSpaceAddr, 
        (__gm__ int64_t *)funcData->startArgs->commContexts, 
        (__gm__ GMTensorInfo*)NULL);
}
```

**作用：**

| 作用 | 说明 |
|------|------|
| **统一调用入口** | PvModel 需要一个标准化的入口函数名 |
| **参数适配** | 将 `DynFuncData` 转换为原始 kernel 的 `CoreFuncParam` |
| **仿真框架对接** | PEM 模拟器从 `PvModelKernelEntry` 开始执行 |

**如果不添加会怎样：**
- PEM 模拟器无法找到入口点
- 参数传递失败
- 仿真无法执行

#### coreType 的决定逻辑

**定义位置：** `Function::IsCube()` (`function.cpp:283-291`)

```cpp
bool Function::IsCube() const {
    for (const auto &oper : Operations()) {
        if ((oper.HasAttr(OpAttributeKey::isCube) && 
             oper.GetBoolAttribute(OpAttributeKey::isCube))
            || oper.GetOpcode() == Opcode::OP_L1_COPY_IN_CONV) {
            return true;  // 有 Cube 算子
        }
    }
    return false;  // Vector 算子
}
```

**coreType 决定规则：**

| 核类型 | 条件 | 说明 |
|--------|------|------|
| **AIC (Cube)** | 子图中有 `isCube=true` 的算子 | 矩阵运算核 |
| **AIV (Vector)** | 子图中没有 Cube 算子 | 向量运算核 |
| **AICPU** | 特殊算子（控制流等） | AI CPU 核 |
| **MIX** | 同时包含 AIC 和 AIV | 需要 Pass 拆分 |

**设置代码：**
```cpp
// codegen_cloudnpu.cpp:290-291
CoreType coreType = compileInfo.IsCube() ? CoreType::AIC : CoreType::AIV;
attr->coreType = coreType;
```

#### 文件类型说明

| 文件类型 | 说明 | 生成方式 |
|----------|------|----------|
| `.cpp` | CCE 源代码 | `GenFuncBody()` 生成 |
| `.o` | 目标文件 | `bisheng` 编译输出 |
| `.bin` | 纯二进制 | `llvm-objcopy` 提取 `.text` 段 |

**为什么需要 .bin：**
- `.o` 包含符号表、重定位信息等额外数据
- PEM 模拟器只需要纯机器指令
- `llvm-objcopy -O binary -j .text` 提取纯净的指令码

#### 为什么 CodeGenCloudNPU 不返回 .bin 路径

**原因：**

| 目标 | 说明 |
|------|------|
| 正常 NPU 运行 | 只需要 `.o`，由运行时加载 |
| PvModel 仿真 | 需要 `.bin`，由 PEM 模拟器加载 |

`CodeGenCloudNPU::GenCode` 的设计是为正常编译流程：
- 生成 `.cpp` 源码
- 编译为 `.o` 目标文件
- 存储路径到 `LeafFuncAttribute`

`.bin` 是 PvModel 特有需求，由 `pv_->Codegen()` 额外生成。

#### llvm-objcopy 命令说明

```bash
llvm-objcopy -O binary -j .text objPath binPath
```

| 参数 | 作用 |
|------|------|
| `-O binary` | 输出格式为纯二进制 |
| `-j .text` | 只提取 `.text` 段（代码段） |
| `objPath` | 输入的目标文件 |
| `binPath` | 输出的二进制文件 |

---

## 5. 输入输出分析

### 5.1 RunPvModel 总体输入输出

**输入：**

| 输入 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `kArgs` | `DeviceKernelArgs&` | 上层传递 | Kernel 参数结构 |
| `inputs` | `vector<RawTensorDataPtr>` | 用户数据 | 输入 tensor |
| `outputs` | `vector<RawTensorDataPtr>` | 用户数据 | 输出 tensor |
| `function_` | `Function*` | 成员变量 | PyPTO Function 对象 |
| `ASCEND_HOME_PATH` | 环境变量 | 系统环境 | CANN 安装路径 |

**输出：**

| 输出 | 类型 | 说明 |
|------|------|------|
| `inputs/outputs` 数据更新 | `vector<RawTensorDataPtr>` | 仿真执行结果 |
| `cceBin` | `unordered_map` | 编译信息映射 |
| `.bin` 文件 | 文件 | 存放于临时目录 |

### 5.2 Codegen 输入输出

见 [4.1 Codegen 函数详解](#41-codegen-函数详解)

### 5.3 BuildPvKernelArgs 输入输出

**输入：**

| 输入 | 说明 |
|------|------|
| `kArgs` | 待填充的 Kernel 参数 |
| `inputs/outputs` | Tensor 数据列表 |
| `devProgBinary` | 编译后的程序二进制 |

**输出：**

填充后的 `kArgs` 结构：

| 字段 | 内容 |
|------|------|
| `inputs` | 输入 tensor 地址 |
| `outputs` | 输出 tensor 地址 |
| `workspace` | 工作空间地址 |
| `cfgdata` | 配置数据地址 |
| `aicoreModel` | AiCorePvModelImpl 实例 |

---

## 6. 原理与设计

### 6.1 为什么需要 PvModel？

| 场景 | 说明 |
|------|------|
| 无 NPU 环境 | 在开发机上仿真算子执行 |
| 精确性能分析 | 获取指令级性能数据 |
| 结果验证 | 在无硬件时验证算子逻辑正确性 |

### 6.2 为什么 RunTestMode 被调用两次？

**当前代码行为：**

```cpp
void RunModel(...) {
    RunTestMode(&kArgs, DEVICE_MAX_AICPU_NUM);  // 第一次
    // ...
    RunPvModel(...);
        // ...
        RunTestMode(&kArgs, maxCpuNum);          // 第二次
}
```

**问题分析：**

| 调用 | 参数来源 | CPU 数量 | 目的（推测） |
|------|----------|----------|--------------|
| 第一次 | `MemoryHelper` 构建 | `DEVICE_MAX_AICPU_NUM` | CostModel 分析前的数据准备 |
| 第二次 | `BuildPvKernelArgs` 构建 | 6 | PvModel 精确仿真 |

**待确认：**
- 是否为设计遗留？
- 第一次是否应该在某些条件下跳过？
- 两次仿真是否执行相同逻辑？

### 6.3 PEM 模拟器原理（待挖掘）

`libpem_davinci.so` 是华为提供的 AiCore 精确模拟器：

| API | 功能（推测） |
|-----|--------------|
| `pv_init` | 初始化模拟器状态 |
| `pv_launch_sub_core` | 加载 `.bin` 到模拟器内存 |
| `pv_step` | 单步执行指令 |
| `pv_mem_write/read` | 模拟器内存访问 |
| `pv_reg_write` | 寄存器设置 |

**待深入：**
- PEM 内部架构
- 指令级仿真流程
- 性能数据采集机制

### 6.4 AiCorePvModelImpl 的作用

```cpp
// cost_model_launcher.h:144-165
class AiCorePvModelImpl : public CostModel::AiCoreModel {
    std::shared_ptr<CostModel::DynPvModel> pv_;
    
    void SendTask(int coreIdx, uint64_t taskId) {
        pv_->Run(data, coreIdx, FuncID(taskId), TaskID(taskId));
    }
};
```

**作用：**
- 作为 `AiCoreModel` 的具体实现
- 被 `DynTileFwkBackendKernelServer` 调用执行任务
- 代理调用 `DynPvModel::Run()` 执行仿真

### 6.5 DynPvModelImpl::Run() 流程（待深入）

> ⚠️ **重要：** 此处的 `Run()` 和 `RunModel()` 是 **DynPvModelImpl** 动态版本的方法，与静态版本 `PvModelImpl` 通过 cmd 命令调用的 `RunModel()` 完全不同。

**动态版本 Run() 流程：**

```cpp
// PvModelImpl.cpp:467-496 (DynPvModelImpl::Run)
void DynPvModelImpl::Run(DynFuncData *funcdata, int coreId, int funcId, int taskId) {
    // 1. 查找 CCE 二进制
    auto cceIter = cceBin.find(psgId);
    
    // 2. 准备执行环境
    SetUp(cce, data, opAttrOffset, dir, &dupData);
    
    // 3. 执行仿真 - 直接调用 PEM API
    RunModel();  // 动态版本：调用 libpem_davinci.so 的 API
    
    // 4. 恢复结果
    TearDown();
}
```

**与静态版本的对比：**

| 版本 | RunModel 实现 | 执行方式 |
|------|---------------|----------|
| **PvModelImpl (静态)** | `std::system(cmd)` 调用外部程序 | 废弃，忽略 |
| **DynPvModelImpl (动态)** | 调用 PEM 模拟器 API (`pv_launch_sub_core`, `pv_step` 等) | 当前架构，重点分析 |

**待深入（动态版本）：**
- `SetUp` 如何准备执行环境（内存布局、参数构建）
- `RunModel` 如何调用 PEM API (`pv_launch_sub_core`, `pv_step`)
- `TearDown` 如何通过 `pv_mem_read` 读取仿真结果

---

## 7. 待深入挖掘的问题

### 7.1 高优先级问题

| 问题 | 重要性 | 当前理解状态 | 备注 |
|------|--------|--------------|------|
| DynPvModelImpl::Run 详细流程 | 高 | 需要逐行分析 | 动态版本核心流程 |
| PEM 模拟器 API 调用机制 | 高 | 仅知 API 名称，调用细节未知 | `pv_launch_sub_core`、`pv_step` 等 |
| SetUp/TearDown 内存交互 | 高 | 知道存在，具体流程未知 | PEM 内存读写 |
| RunTestMode 两次调用是否合理 | 高 | 存疑，需要确认设计意图 | 可能是历史遗留 |

### 7.2 中优先级问题

| 问题 | 重要性 | 当前理解状态 | 备注 |
|------|--------|--------------|------|
| DynTileFwkBackendKernelServer 实现 | 中 | 知道入口，内部逻辑未知 | 任务执行入口 |
| PyptoKernelCtrlServer 实现 | 中 | 调度逻辑未知 | 任务调度入口 |
| MemoryHelper vs PvModel 内存管理差异 | 中 | 需要对比分析 | 两种内存管理模式 |
| DynFuncData 结构详解 | 中 | 字段含义部分已知 | 仿真数据传递结构 |

### 7.3 低优先级问题

| 问题 | 重要性 | 当前理解状态 | 备注 |
|------|--------|--------------|------|
| PvMemAllocator 内存分配策略 | 低 | 知道存在，细节未知 | 地址分配逻辑 |
| DataMap hostPtr/devPtr 映射关系 | 低 | 需要理解地址转换 | 主机/设备地址映射 |
| A5 架构特有的编译选项 | 低 | 需要确认 | `-D__DAV_V310` 是否适用于 A5 |

> ⚠️ **已移除的问题：** 静态版本 `PvModelImpl` 的 `PvModel%s` 命令调用逻辑已废弃，不再分析。

---

## 附录

### A. 架构版本对比

#### PvModelImpl vs DynPvModelImpl

| 特性 | PvModelImpl (静态) | DynPvModelImpl (动态) |
|------|-------------------|----------------------|
| **状态** | ❌ 已废弃，不再维护 | ✅ 当前架构，持续演进 |
| **分析重点** | **忽略** | **重点关注** |
| **执行方式** | `std::system(cmd)` 调用外部测试程序 | 直接调用 `libpem_davinci.so` API |
| **命令示例** | `cd xxx/ && PvModel%s --gtest_filter=... --spec=spec.toml` | `pv_launch_sub_core(pc, bin_file, ...)` |
| **效率** | 低（进程切换开销） | 高（直接 API 调用） |
| **集成度** | 低（外部程序） | 高（框架内集成） |
| **适用场景** | 旧架构静态算子 | 新架构动态算子 |

#### A2/A3 vs A5 架构

| 特性 | A2/A3 | A5 |
|------|-------|-----|
| **分析重点** | **忽略** | **重点关注** |
| **入口函数** | - | `CreateDynPvModelImplA5` |
| **架构宏** | - | 需确认具体宏定义 |
| **编译选项** | - | 需确认 `--cce-aicore-arch` 参数 |

### B. 关键数据结构

#### DeviceKernelArgs

```cpp
struct DeviceKernelArgs {
    int64_t *inputs;      // 输入 tensor 地址列表
    int64_t *outputs;     // 输出 tensor 地址列表
    int64_t *workspace;   // 工作空间地址
    int64_t *cfgdata;     // 配置数据地址
    CostModel::AiCoreModel* aicoreModel;  // AiCore 模型实例
    CostModel::ModelData* costmodeldata;  // CostModel 数据
    // ...
};
```

#### LeafFuncAttribute

```cpp
struct LeafFuncAttribute {
    std::string kernelName;       // kernel 函数名
    std::string binPath;          // .o 文件路径
    std::string kernelDeclare;    // kernel 声明
    CoreType coreType;            // 核类型 (AIC/AIV/AICPU)
    AIVCore aivCore;              // AIV 核位置
    int32_t mixId;                // MIX 子图 ID
    // ...
};
```

#### PvModelCceBin

```cpp
struct PvModelCceBin {
    uint32_t psgId;              // 子图 ID
    uint64_t funcHash;           // 函数哈希
    npu::tile_fwk::CoreType coreType;  // 核类型
    std::string srcPath;         // 源文件路径
    std::string binPath;         // 二进制文件路径
};
```

### B. 编译命令详解

#### bisheng 编译命令

```bash
bisheng -c -O3 -g -x cce -std=c++17 \
    -D__AIC__/-D__AIV__ \
    -D__DAV_V310 \
    --cce-aicore-only \
    --cce-aicore-arch=<arch> \
    -I<ASCEND_HOME_PATH>/include/tilefwk \
    -I<ASCEND_HOME_PATH>/include/tileop \
    -I<PTO_TILE_LIB_CODE_PATH>/include \
    -o <objPath> <srcPath>
```

| 参数 | 作用 |
|------|------|
| `-c` | 只编译不链接 |
| `-O3` | 最高优化级别 |
| `-g` | 生成调试信息 |
| `-x cce` | 指定源语言为 CCE |
| `-std=c++17` | C++17 标准 |
| `-D__AIC__/__AIV__` | 核类型宏定义 |
| `-D__DAV_V310` | 架构版本宏 |
| `--cce-aicore-only` | 只编译 AiCore 代码 |
| `--cce-aicore-arch` | 目标架构 |

#### llvm-objcopy 命令

```bash
llvm-objcopy -O binary -j .text <objPath> <binPath>
```

---

## 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-04-17 | 初稿创建，包含 RunPvModel、Codegen、AddKernelEntry 等核心分析 |
| 2026-04-17 | **重要更新：** 明确仅关注 A5 架构和 DynPvModelImpl 动态实现，静态 PvModelImpl 已废弃忽略 |

---

*本文档将持续更新，后续将深入挖掘 PEM 模拟器 API 调用机制、DynPvModelImpl::Run() 详细流程等问题。*

> ⚠️ **分析原则：** 所有后续分析均针对 **A5 架构 + DynPvModelImpl 动态实现**，静态版本和旧架构代码不再纳入分析范围。