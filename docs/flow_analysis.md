# 代码流程梳理

> 📊 **流程图**: 详细的流程图请参考 [flow_diagram.md](./flow_diagram.md)

## 1. EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData
**功能**: 在仿真模式下构建控制流缓存（Control Flow Cache）

### 流程概览
```
初始化控制流缓存记录模式 → 初始化Kernel参数 → 执行仿真启动 → 地址重定位 → 激活缓存
```

### 详细步骤

#### 阶段1: 初始化控制流缓存记录 (97-99行)
- **设置记录标志**: `devProg->controlFlowCache.isRecording = true`
- **重置任务计数**: `deviceTaskCount = 0`
- **重置缓存偏移**: `cacheDataOffset = 0`
- **目的**: 启用控制流缓存的记录模式，用于捕获执行过程中的控制流信息

#### 阶段2: 初始化Kernel参数 (100-103行)
- **创建Kernel参数**: `DeviceKernelArgs kArgs`
- **初始化Tiling数据**: 
  - 调用 `DeviceLauncher::DeviceInitTilingData(EmulationMemoryUtils(), kArgs, devProgData, config, nullptr)`
  - 使用 `EmulationMemoryUtils()` (仿真内存工具，使用主机内存)
- **初始化输入输出**:
  - 调用 `DeviceLauncher::DeviceInitKernelInOuts(EmulationMemoryUtils(), kArgs, inputList, outputList, ...)`
  - 准备输入输出张量数据

#### 阶段3: 执行仿真启动 (104行)
- **调用**: `EmulationLaunchOnce(kArgs)`
- **内部流程**:
  1. 初始化后端内核服务器: `DynTileFwkBackendKernelServerInit(&kArgs)`
  2. 创建多个AICPU线程（根据 `devProg->devArgs.nrAicpu`）
  3. 每个线程:
     - 设置CPU亲和性
     - 执行 `DynTileFwkBackendKernelServer(&kArgs)` (AICPU内核服务器)
  4. 等待所有线程完成
- **目的**: 在主机上模拟NPU执行，记录控制流信息

#### 阶段4: 地址重定位 (106-112行)
- **停止记录**: `isRecording = false`
- **获取上下文工作空间地址**: `contextWorkspaceAddr`
- **执行地址重定位**:
  1. `IncastOutcastAddrReloc`: 重定位输入/输出cast地址（从记录地址到0，表示缓存地址）
  2. `RuntimeAddrRelocWorkspace`: 重定位运行时工作空间地址
  3. `RuntimeAddrRelocProgram`: 重定位运行时程序地址（从devProg地址到0）
  4. `TaskAddrRelocWorkspace`: 重定位任务工作空间地址
  5. `TaskAddrRelocProgram`: 重定位任务程序地址（从devProg地址到0）
- **目的**: 将执行时记录的地址转换为缓存中的相对地址，以便后续重用

#### 阶段5: 激活缓存 (113-114行)
- **重置启动状态**: `devProg->ResetFromLaunch()`
- **激活缓存**: `controlFlowCache.isActivated = true`
- **目的**: 标记缓存已构建完成，可以用于后续执行

---

## 2. DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData
**功能**: 在NPU设备上执行一次动态启动

### 流程概览
```
环境初始化 → 捕获模式设置 → ACL初始化 → 缓存操作符处理 → Kernel参数初始化 → 
注册Kernel二进制 → 动态启动 → 性能分析 → 同步
```

### 详细步骤

#### 阶段1: 环境初始化 (129-134行)
- **设置运行类型**: `config::SetRunDataOption(KEY_RUNTYPE, "npu")`
- **设置Kernel二进制数据**: 
  - `DeviceRunner::SetBinData(function->GetDyndevAttribute()->kernelBinary)`
- **目的**: 配置运行环境和Kernel数据

#### 阶段2: 捕获模式设置 (135-143行)
- **设置捕获流**: `SetCaptureStream(aicoreStream, aicpuStream, isCapture)`
  - 检查是否需要捕获模型执行
  - 设置AICore和AICPU流
- **更改捕获模式**: 如果需要捕获，调用 `ChangeCaptureMode()` 设置为RELAXED模式
- **设置捕获标志**: `DeviceRunner::Get().SetCaptureFlag(isCapture)`
- **设置性能分析函数**: `GetHostProfInstance().SetProfFunction(function)`
- **目的**: 配置模型捕获和性能分析

#### 阶段3: ACL初始化 (146-149行)
- **初始化ACL**: `aclInit(nullptr)`
- **错误处理**: 如果初始化失败且不是重复初始化错误，返回错误码
- **目的**: 初始化Ascend Computing Language运行时

#### 阶段4: 缓存操作符处理 (151-158行)
- **检查缓存操作符**: 如果 `cachedOperator == nullptr`
- **检查Kernel重用模式**: `DeviceRunCacheKernelEnable(function)`
- **获取缓存的配置数据地址**: `DeviceRunCacheKernelGet(function)`
- **设置缓存操作符**: 将获取的地址设置到 `cachedOperatorData`
- **目的**: 处理Kernel重用缓存，避免重复加载

#### 阶段5: Kernel参数初始化 (159-165行)
- **检查设备ID**: `CheckDeviceId()`
- **创建Kernel参数**: `DeviceKernelArgs kArgs`
- **填充设备信息**: `DeviceLauncherConfigFillDeviceInfo(config)`
- **初始化Tiling数据**: 
  - `DeviceInitTilingData(DeviceMemoryUtils(), kArgs, devProgBinary, config, cachedOperator)`
  - 使用 `DeviceMemoryUtils()` (设备内存工具，使用NPU设备内存)
- **设置缓存Kernel**: `DeviceRunCacheKernelSet(function, (uint8_t *)kArgs.cfgdata)`
- **初始化输入输出**: `DeviceInitKernelInOuts(DeviceMemoryUtils(), kArgs, inputList, outputList, ...)`
- **目的**: 准备在NPU上执行所需的Kernel参数

#### 阶段6: 注册Kernel二进制 (166-170行)
- **注册Kernel二进制**: `DeviceRunner::Get().RegisterKernelBin(&(*reinterpret_cast<rtBinHandle *>(CachedOperator::GetBinHandleHolder(cachedOperator))))`
- **错误处理**: 如果注册失败，返回错误
- **目的**: 将Kernel二进制注册到设备运行时

#### 阶段7: 动态启动 (171-174行)
- **执行动态启动**: `DeviceRunner::Get().DynamicLaunch(aicpuStream, nullptr, aicoreStream, 0, &kArgs, config.blockdim, config.aicpuNum)`
- **内部流程**:
  1. 初始化AICPU SO二进制（首次）
  2. 准备设备参数（DeviceArgs）
  3. 复制参数到设备内存
  4. 执行准备阶段（RunPrepare）
  5. 启动Kernel:
     - `launchDynamicAiCpuInit`: 初始化AICPU
     - `launchDynamicAiCpu`: 启动AICPU任务
     - `launchDynamicAiCore`: 启动AICore任务
- **错误处理**: 如果启动失败，返回错误
- **目的**: 在NPU上实际执行计算任务

#### 阶段8: 性能分析 (175-178行)
- **运行性能分析**: `RunWithProfile(aicoreStream, aicpuStream)`
- **错误处理**: 如果性能分析失败，返回错误
- **目的**: 收集和报告性能数据

#### 阶段9: 同步 (179-181行)
- **条件同步**: 如果 `streamSynchronize == true`
- **执行同步**: `DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, nullptr, aicoreStream)`
- **目的**: 等待所有流上的任务完成

---

## 关键差异对比

| 特性 | EmulationLauncher | DeviceLauncher |
|------|------------------|----------------|
| **执行环境** | 主机CPU（仿真） | NPU设备 |
| **内存管理** | EmulationMemoryUtils (主机内存) | DeviceMemoryUtils (设备内存) |
| **主要目的** | 构建控制流缓存 | 实际执行计算 |
| **线程模型** | 多线程AICPU仿真 | 设备流执行 |
| **地址处理** | 需要地址重定位 | 直接使用设备地址 |
| **缓存操作** | 记录和构建缓存 | 使用缓存（如果启用） |
| **性能分析** | 无 | 支持性能分析 |
| **同步机制** | 线程join | 流同步 |

## 共同点

1. **都使用相同的Kernel参数初始化流程**:
   - `DeviceInitTilingData`
   - `DeviceInitKernelInOuts`

2. **都处理相同的输入输出数据结构**:
   - `DeviceTensorData` 列表

3. **都使用相同的配置结构**:
   - `DeviceLauncherConfig`

4. **都操作相同的程序数据结构**:
   - `DevAscendProgram`

## 调用关系

```
用户代码
  ├─→ EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData
  │     └─→ EmulationLaunchOnce (仿真执行)
  │           └─→ DynTileFwkBackendKernelServer (AICPU仿真)
  │
  └─→ DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData
        └─→ DeviceRunner::DynamicLaunch (NPU执行)
              ├─→ launchDynamicAiCpuInit
              ├─→ launchDynamicAiCpu
              └─→ launchDynamicAiCore
```
