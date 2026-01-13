# 代码执行流程图

## 1. EmulationLauncher::BuildControlFlowCacheWithEmulationTensorData 流程图

```mermaid
flowchart TD
    Start([开始: BuildControlFlowCacheWithEmulationTensorData]) --> Init1[初始化控制流缓存记录模式]
    Init1 --> SetRecording[设置 isRecording = true<br/>deviceTaskCount = 0<br/>cacheDataOffset = 0]
    SetRecording --> InitKArgs[创建 DeviceKernelArgs kArgs]
    InitKArgs --> InitTiling[DeviceInitTilingData<br/>使用 EmulationMemoryUtils<br/>主机内存]
    InitTiling --> InitIO[DeviceInitKernelInOuts<br/>初始化输入输出张量]
    InitIO --> EmulationLaunch[调用 EmulationLaunchOnce]
    
    EmulationLaunch --> InitServer[DynTileFwkBackendKernelServerInit<br/>初始化后端内核服务器]
    InitServer --> CheckInit{初始化成功?}
    CheckInit -->|失败| ReturnError1[返回错误]
    CheckInit -->|成功| CreateThreads[创建多个AICPU线程<br/>数量: devProg->devArgs.nrAicpu]
    
    CreateThreads --> ThreadLoop[对每个线程]
    ThreadLoop --> SetAffinity[设置CPU亲和性]
    SetAffinity --> SetName[设置线程名称 aicputN]
    SetName --> RunServer[DynTileFwkBackendKernelServer<br/>执行AICPU内核服务器]
    RunServer --> JoinThreads[等待所有线程完成]
    
    JoinThreads --> CheckResult{所有线程成功?}
    CheckResult -->|失败| ReturnError2[返回错误]
    CheckResult -->|成功| StopRecording[停止记录<br/>isRecording = false]
    
    StopRecording --> GetAddr[获取 contextWorkspaceAddr]
    GetAddr --> Reloc1[IncastOutcastAddrReloc<br/>重定位输入/输出cast地址]
    Reloc1 --> Reloc2[RuntimeAddrRelocWorkspace<br/>重定位运行时工作空间]
    Reloc2 --> Reloc3[RuntimeAddrRelocProgram<br/>重定位运行时程序地址]
    Reloc3 --> Reloc4[TaskAddrRelocWorkspace<br/>重定位任务工作空间]
    Reloc4 --> Reloc5[TaskAddrRelocProgram<br/>重定位任务程序地址]
    
    Reloc5 --> Reset[devProg->ResetFromLaunch<br/>重置启动状态]
    Reset --> Activate[设置 isActivated = true<br/>激活缓存]
    Activate --> ReturnSuccess[返回成功]
    
    ReturnError1 --> End1([结束])
    ReturnError2 --> End1
    ReturnSuccess --> End1
    
    style Start fill:#e1f5ff
    style End1 fill:#ffe1f5
    style EmulationLaunch fill:#fff4e1
    style Reloc1 fill:#e1ffe1
    style Reloc2 fill:#e1ffe1
    style Reloc3 fill:#e1ffe1
    style Reloc4 fill:#e1ffe1
    style Reloc5 fill:#e1ffe1
    style Activate fill:#ffe1f5
```

## 2. DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData 流程图

```mermaid
flowchart TD
    Start([开始: DeviceLaunchOnceWithDeviceTensorData]) --> EnvInit[环境初始化]
    EnvInit --> SetRunType[设置运行类型: npu]
    SetRunType --> SetBinData[设置Kernel二进制数据]
    SetBinData --> CaptureStream[设置捕获流<br/>SetCaptureStream]
    
    CaptureStream --> CheckCapture{需要捕获?}
    CheckCapture -->|是| ChangeMode[更改捕获模式为RELAXED]
    CheckCapture -->|否| SetFlag[设置捕获标志和性能分析函数]
    ChangeMode --> SetFlag
    
    SetFlag --> ACLInit[ACL初始化<br/>aclInit]
    ACLInit --> CheckACL{初始化成功?}
    CheckACL -->|失败| ReturnError1[返回错误]
    CheckACL -->|成功| CheckCache{cachedOperator == nullptr?}
    
    CheckCache -->|是| CheckReuse{启用Kernel重用?}
    CheckReuse -->|是| GetCache[获取缓存的配置数据地址]
    GetCache --> SetCache[设置 cachedOperator]
    CheckReuse -->|否| CheckDevice
    CheckCache -->|否| CheckDevice[检查设备ID]
    SetCache --> CheckDevice
    
    CheckDevice --> CreateKArgs[创建 DeviceKernelArgs kArgs]
    CreateKArgs --> FillDeviceInfo[填充设备信息<br/>DeviceLauncherConfigFillDeviceInfo]
    FillDeviceInfo --> InitTiling[DeviceInitTilingData<br/>使用 DeviceMemoryUtils<br/>设备内存]
    InitTiling --> SetCacheKernel[DeviceRunCacheKernelSet<br/>设置缓存Kernel]
    SetCacheKernel --> InitIO[DeviceInitKernelInOuts<br/>初始化输入输出]
    
    InitIO --> RegisterBin[注册Kernel二进制<br/>RegisterKernelBin]
    RegisterBin --> CheckReg{注册成功?}
    CheckReg -->|失败| ReturnError2[返回错误: Register kernel bin failed]
    CheckReg -->|成功| DynamicLaunch[执行动态启动<br/>DynamicLaunch]
    
    DynamicLaunch --> InitAicpuSo[初始化AICPU SO二进制<br/>首次执行]
    InitAicpuSo --> PrepareArgs[准备设备参数 DeviceArgs]
    PrepareArgs --> CopyArgs[复制参数到设备内存]
    CopyArgs --> RunPrepare[执行准备阶段 RunPrepare]
    RunPrepare --> LaunchInit[launchDynamicAiCpuInit<br/>初始化AICPU]
    LaunchInit --> LaunchCpu[launchDynamicAiCpu<br/>启动AICPU任务]
    LaunchCpu --> LaunchCore[launchDynamicAiCore<br/>启动AICore任务]
    
    LaunchCore --> CheckLaunch{启动成功?}
    CheckLaunch -->|失败| ReturnError3[返回错误]
    CheckLaunch -->|成功| RunProfile[运行性能分析<br/>RunWithProfile]
    
    RunProfile --> CheckProfile{性能分析成功?}
    CheckProfile -->|失败| ReturnError4[返回错误]
    CheckProfile -->|成功| CheckSync{需要同步?}
    
    CheckSync -->|是| Sync[执行流同步<br/>DynamicLaunchSynchronize]
    CheckSync -->|否| ReturnSuccess[返回成功]
    Sync --> ReturnSuccess
    
    ReturnError1 --> End1([结束])
    ReturnError2 --> End1
    ReturnError3 --> End1
    ReturnError4 --> End1
    ReturnSuccess --> End1
    
    style Start fill:#e1f5ff
    style End1 fill:#ffe1f5
    style DynamicLaunch fill:#fff4e1
    style LaunchInit fill:#e1ffe1
    style LaunchCpu fill:#e1ffe1
    style LaunchCore fill:#e1ffe1
    style RunProfile fill:#ffe1f5
    style Sync fill:#ffe1f5
```

## 3. 两个流程的对比图

```mermaid
flowchart LR
    subgraph Emulation[仿真模式 - BuildControlFlowCache]
        E1[初始化缓存记录] --> E2[初始化Kernel参数<br/>主机内存]
        E2 --> E3[多线程仿真执行]
        E3 --> E4[地址重定位]
        E4 --> E5[激活缓存]
    end
    
    subgraph Device[NPU模式 - DeviceLaunchOnce]
        D1[环境初始化] --> D2[捕获模式设置]
        D2 --> D3[ACL初始化]
        D3 --> D4[缓存操作符处理]
        D4 --> D5[初始化Kernel参数<br/>设备内存]
        D5 --> D6[注册Kernel二进制]
        D6 --> D7[动态启动执行]
        D7 --> D8[性能分析]
        D8 --> D9[流同步]
    end
    
    E5 -.缓存数据.-> D4
    
    style Emulation fill:#e1f5ff
    style Device fill:#ffe1f5
    style E3 fill:#fff4e1
    style D7 fill:#fff4e1
```

## 4. 详细执行流程对比

```mermaid
graph TB
    subgraph Common[共同部分]
        C1[DeviceInitTilingData<br/>初始化Tiling数据]
        C2[DeviceInitKernelInOuts<br/>初始化输入输出]
    end
    
    subgraph EmulationPath[仿真路径]
        E1[EmulationMemoryUtils<br/>主机内存] --> C1
        C1 --> C2
        C2 --> E2[EmulationLaunchOnce<br/>多线程仿真]
        E2 --> E3[地址重定位<br/>5个重定位操作]
        E3 --> E4[激活缓存]
    end
    
    subgraph DevicePath[设备路径]
        D1[DeviceMemoryUtils<br/>设备内存] --> C1
        C1 --> C2
        C2 --> D2[RegisterKernelBin<br/>注册二进制]
        D2 --> D3[DynamicLaunch<br/>设备执行]
        D3 --> D4[RunWithProfile<br/>性能分析]
        D4 --> D5[DynamicLaunchSynchronize<br/>流同步]
    end
    
    style Common fill:#e1ffe1
    style EmulationPath fill:#e1f5ff
    style DevicePath fill:#ffe1f5
    style E2 fill:#fff4e1
    style D3 fill:#fff4e1
```

## 5. 内存和线程模型对比

```mermaid
graph LR
    subgraph EmulationMem[仿真模式内存模型]
        EM1[主机内存<br/>EmulationMemoryUtils] --> EM2[多线程AICPU仿真<br/>线程数: nrAicpu]
        EM2 --> EM3[地址重定位<br/>执行地址 → 缓存地址]
    end
    
    subgraph DeviceMem[设备模式内存模型]
        DM1[设备内存<br/>DeviceMemoryUtils] --> DM2[设备流执行<br/>aicpuStream + aicoreStream]
        DM2 --> DM3[直接使用设备地址<br/>无需重定位]
    end
    
    style EmulationMem fill:#e1f5ff
    style DeviceMem fill:#ffe1f5
    style EM2 fill:#fff4e1
    style DM2 fill:#fff4e1
```

## 6. 完整调用链

```mermaid
flowchart TD
    User[用户代码] --> Choice{选择执行模式}
    
    Choice -->|构建缓存| Emulation[EmulationLauncher::<br/>BuildControlFlowCacheWithEmulationTensorData]
    Choice -->|实际执行| Device[DeviceLauncher::<br/>DeviceLaunchOnceWithDeviceTensorData]
    
    Emulation --> E1[DeviceInitTilingData<br/>EmulationMemoryUtils]
    E1 --> E2[DeviceInitKernelInOuts]
    E2 --> E3[EmulationLaunchOnce]
    E3 --> E4[DynTileFwkBackendKernelServerInit]
    E4 --> E5[多线程<br/>DynTileFwkBackendKernelServer]
    E5 --> E6[地址重定位<br/>5个操作]
    E6 --> E7[激活缓存]
    
    Device --> D1[环境初始化]
    D1 --> D2[DeviceInitTilingData<br/>DeviceMemoryUtils]
    D2 --> D3[DeviceInitKernelInOuts]
    D3 --> D4[RegisterKernelBin]
    D4 --> D5[DeviceRunner::DynamicLaunch]
    D5 --> D6[launchDynamicAiCpuInit]
    D6 --> D7[launchDynamicAiCpu]
    D7 --> D8[launchDynamicAiCore]
    D8 --> D9[RunWithProfile]
    D9 --> D10[DynamicLaunchSynchronize]
    
    E7 -.缓存数据.-> D4
    
    style User fill:#e1f5ff
    style Emulation fill:#fff4e1
    style Device fill:#ffe1f5
    style E6 fill:#e1ffe1
    style D5 fill:#e1ffe1
```
