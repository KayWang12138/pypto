# DeviceLaunchOnceWithDeviceTensorData rt 接口调用时序图

本文档使用时序图展示 `DeviceLaunchOnceWithDeviceTensorData` 函数中所有 rt 接口的调用顺序和关系。

## 完整时序图

```mermaid
sequenceDiagram
    participant Main as DeviceLaunchOnceWithDeviceTensorData
    participant SetCapture as SetCaptureStream
    participant CheckDev as CheckDeviceId
    participant InitTiling as DeviceInitTilingData
    participant InitIO as DeviceInitKernelInOuts
    participant RegBin as DeviceRunner::RegisterKernelBin
    participant DynLaunch as DeviceRunner::DynamicLaunch
    participant RunPrep as RunPrepare
    participant RunSync as RunPreSync
    participant DynKernel as DynamicKernelLaunch
    participant LaunchCore as launchDynamicAiCore
    participant LaunchCpu as launchDynamicAiCpu
    participant DynSync as DynamicLaunchSynchronize
    participant RT as Runtime (rt接口)

    Note over Main: 阶段0-2: 初始化和配置
    Main->>Main: 阶段1: 设置配置
    Main->>Main: 阶段2: 设置二进制数据

    Note over Main,RT: 阶段3: 流捕获设置
    Main->>SetCapture: SetCaptureStream(aicoreStream, aicpuStream)
    alt isCapture == true
        SetCapture->>RT: rtStreamAddToModel(aicpuStream, rtModel)
        RT-->>SetCapture: return
    end
    SetCapture-->>Main: return

    Note over Main,RT: 阶段4-6: 捕获模式和缓存处理
    Main->>Main: 阶段4: 设置捕获模式
    Main->>Main: 阶段5: ACL初始化
    Main->>Main: 阶段6: 缓存操作符处理

    Note over Main,RT: 阶段7: 检查设备ID
    Main->>CheckDev: CheckDeviceId()
    CheckDev->>RT: rtGetDevice(&devId)
    RT-->>CheckDev: return devId
    CheckDev-->>Main: return

    Note over Main,RT: 阶段8: 初始化Tiling数据
    Main->>InitTiling: DeviceInitTilingData(...)
    InitTiling->>RT: rtMemcpy(dev, size, host, size, H2D)
    RT-->>InitTiling: return
    InitTiling-->>Main: return

    Note over Main,RT: 阶段9: 初始化内核输入输出
    Main->>InitIO: DeviceInitKernelInOuts(...)
    InitIO->>RT: rtMemcpy(dev, size, host, size, H2D)
    RT-->>InitIO: return
    InitIO-->>Main: return

    Note over Main,RT: 阶段10: 注册内核二进制
    Main->>RegBin: RegisterKernelBin(&hdl)
    RegBin->>RT: rtRegisterAllKernel(&binary, hdl)
    RT-->>RegBin: return binHdl
    RegBin-->>Main: return

    Note over Main,RT: 阶段11: 动态启动
    Main->>DynLaunch: DynamicLaunch(aicpuStream, aicoreStream, ...)

    Note over DynLaunch,RT: 11.1: 准备阶段
    DynLaunch->>RunPrep: RunPrepare()
    loop for each core
        RunPrep->>RT: rtMemcpy(sharedBuffer, perfData, H2D)
        RT-->>RunPrep: return
    end
    RunPrep-->>DynLaunch: return

    Note over DynLaunch,RT: 11.2: 预同步阶段
    DynLaunch->>RunSync: RunPreSync(aicpuStream, aicoreStream)
    RunSync->>RT: aclrtCreateEventExWithFlag(&event)
    RT-->>RunSync: return event
    RunSync->>RT: aclrtRecordEvent(event, aicoreStream)
    RT-->>RunSync: return
    RunSync->>RT: aclrtStreamWaitEvent(aicpuStream, event)
    RT-->>RunSync: return
    RunSync-->>DynLaunch: return

    Note over DynLaunch,RT: 11.3: 内核参数拷贝
    DynLaunch->>DynKernel: DynamicKernelLaunch(...)
    DynKernel->>RT: rtMemcpy(cfgdata, localArgs, H2D)
    RT-->>DynKernel: return
    DynKernel-->>DynLaunch: return

    Note over DynLaunch,RT: 11.4: 启动AICore内核
    DynLaunch->>LaunchCore: launchDynamicAiCore(aicoreStream, kernelArgs)
    LaunchCore->>RT: rtKernelLaunchWithHandleV2(binHdl, tilingKey, blockDim, rtArgs, aicoreStream, cfg)
    RT-->>LaunchCore: return (异步)
    LaunchCore-->>DynLaunch: return

    Note over DynLaunch,RT: 11.5: 启动AICPU内核
    DynLaunch->>LaunchCpu: launchDynamicAiCpu(aicpuStream, kArgs)
    LaunchCpu->>RT: rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, rtArgs, aicpuStream)
    RT-->>LaunchCpu: return (异步)
    LaunchCpu-->>DynLaunch: return

    DynLaunch-->>Main: return

    Note over Main,RT: 阶段12: 性能分析运行
    Main->>Main: RunWithProfile(...)

    Note over Main,RT: 阶段13: 流同步
    alt streamSynchronize == true
        Main->>DynSync: DynamicLaunchSynchronize(aicpuStream, aicoreStream)
        DynSync->>RT: rtStreamSynchronize(aicoreStream)
        RT-->>DynSync: return (等待完成)
        DynSync->>RT: rtStreamSynchronize(aicpuStream)
        RT-->>DynSync: return (等待完成)
        DynSync-->>Main: return
    end

    Note over Main: 阶段14: 函数结束
    Main->>Main: 输出总耗时
```

## 简化版时序图（仅 rt 接口）

```mermaid
sequenceDiagram
    participant Main as DeviceLaunchOnceWithDeviceTensorData
    participant RT as Runtime (rt接口)

    Note over Main,RT: 阶段3: 流捕获
    Main->>RT: rtStreamAddToModel(aicpuStream, rtModel)
    RT-->>Main: return

    Note over Main,RT: 阶段7: 设备检查
    Main->>RT: rtGetDevice(&devId)
    RT-->>Main: return devId

    Note over Main,RT: 阶段8: Tiling数据拷贝
    Main->>RT: rtMemcpy(dev, size, host, size, H2D)
    RT-->>Main: return

    Note over Main,RT: 阶段9: 输入输出数据拷贝
    Main->>RT: rtMemcpy(dev, size, host, size, H2D)
    RT-->>Main: return

    Note over Main,RT: 阶段10: 注册内核
    Main->>RT: rtRegisterAllKernel(&binary, hdl)
    RT-->>Main: return binHdl

    Note over Main,RT: 阶段11.1: 准备性能数据
    loop for each core
        Main->>RT: rtMemcpy(sharedBuffer, perfData, H2D)
        RT-->>Main: return
    end

    Note over Main,RT: 阶段11.2: 事件同步
    Main->>RT: aclrtCreateEventExWithFlag(&event)
    RT-->>Main: return event
    Main->>RT: aclrtRecordEvent(event, aicoreStream)
    RT-->>Main: return
    Main->>RT: aclrtStreamWaitEvent(aicpuStream, event)
    RT-->>Main: return

    Note over Main,RT: 阶段11.3: 内核参数拷贝
    Main->>RT: rtMemcpy(cfgdata, localArgs, H2D)
    RT-->>Main: return

    Note over Main,RT: 阶段11.4: 启动AICore内核
    Main->>RT: rtKernelLaunchWithHandleV2(binHdl, tilingKey, blockDim, rtArgs, aicoreStream, cfg)
    RT-->>Main: return (异步启动)

    Note over Main,RT: 阶段11.5: 启动AICPU内核
    Main->>RT: rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, rtArgs, aicpuStream)
    RT-->>Main: return (异步启动)

    Note over Main,RT: 阶段13: 流同步
    Main->>RT: rtStreamSynchronize(aicoreStream)
    RT-->>Main: return (等待AICore完成)
    Main->>RT: rtStreamSynchronize(aicpuStream)
    RT-->>Main: return (等待AICPU完成)
```

## 按接口类型分类的时序图

### 内存管理接口 (rtMemcpy)

```mermaid
sequenceDiagram
    participant Main as DeviceLaunchOnceWithDeviceTensorData
    participant RT as Runtime

    Note over Main,RT: 阶段8: Tiling数据
    Main->>RT: rtMemcpy(devTiling, hostTiling, H2D)
    
    Note over Main,RT: 阶段9: 输入输出数据
    Main->>RT: rtMemcpy(devIO, hostIO, H2D)
    
    Note over Main,RT: 阶段11.1: 性能数据
    loop for each core
        Main->>RT: rtMemcpy(sharedBuffer, perfData, H2D)
    end
    
    Note over Main,RT: 阶段11.3: 内核参数
    Main->>RT: rtMemcpy(cfgdata, localArgs, H2D)
```

### 内核启动接口

```mermaid
sequenceDiagram
    participant Main as DeviceLaunchOnceWithDeviceTensorData
    participant RT as Runtime

    Note over Main,RT: 阶段10: 注册内核
    Main->>RT: rtRegisterAllKernel(&binary, hdl)
    RT-->>Main: binHdl

    Note over Main,RT: 阶段11.4: 启动AICore
    Main->>RT: rtKernelLaunchWithHandleV2(binHdl, tilingKey, blockDim, rtArgs, aicoreStream, cfg)
    Note right of RT: 异步执行

    Note over Main,RT: 阶段11.5: 启动AICPU
    Main->>RT: rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, rtArgs, aicpuStream)
    Note right of RT: 异步执行
```

### 流同步接口

```mermaid
sequenceDiagram
    participant Main as DeviceLaunchOnceWithDeviceTensorData
    participant RT as Runtime

    Note over Main,RT: 阶段3: 流捕获
    Main->>RT: rtStreamAddToModel(aicpuStream, rtModel)

    Note over Main,RT: 阶段11.2: 事件同步
    Main->>RT: aclrtCreateEventExWithFlag(&event)
    Main->>RT: aclrtRecordEvent(event, aicoreStream)
    Main->>RT: aclrtStreamWaitEvent(aicpuStream, event)

    Note over Main,RT: 阶段13: 流同步
    Main->>RT: rtStreamSynchronize(aicoreStream)
    Note right of RT: 阻塞等待AICore完成
    Main->>RT: rtStreamSynchronize(aicpuStream)
    Note right of RT: 阻塞等待AICPU完成
```

## 并行执行时序图

```mermaid
sequenceDiagram
    participant Main as DeviceLaunchOnceWithDeviceTensorData
    participant AICore as AICore Stream
    participant AICPU as AICPU Stream
    participant RT as Runtime

    Note over Main,RT: 阶段11: 并行启动内核

    par AICore 内核启动
        Main->>AICore: rtKernelLaunchWithHandleV2(...)
        AICore->>RT: 执行内核
        Note over AICore: 异步执行中...
    and AICPU 内核启动
        Main->>AICPU: rtAicpuKernelLaunchExWithArgs(...)
        AICPU->>RT: 执行内核
        Note over AICPU: 异步执行中...
    end

    Note over Main,RT: 阶段13: 同步等待

    par 等待AICore完成
        Main->>AICore: rtStreamSynchronize(aicoreStream)
        AICore-->>Main: 完成
    and 等待AICPU完成
        Main->>AICPU: rtStreamSynchronize(aicpuStream)
        AICPU-->>Main: 完成
    end
```

## 关键时间点说明

1. **阶段3**: 流捕获设置（可选，仅在捕获模式激活时）
2. **阶段7**: 设备ID检查（确保设备可用）
3. **阶段8-9**: 数据准备（将主机数据拷贝到设备）
4. **阶段10**: 内核注册（注册内核二进制，获取句柄）
5. **阶段11**: 内核启动（并行启动 AICore 和 AICPU）
   - 11.1: 准备性能数据
   - 11.2: 事件同步（确保流顺序）
   - 11.3: 拷贝内核参数
   - 11.4: 启动 AICore 内核（异步）
   - 11.5: 启动 AICPU 内核（异步）
6. **阶段13**: 流同步（等待所有任务完成）

## 注意事项

1. **异步执行**: `rtKernelLaunchWithHandleV2` 和 `rtAicpuKernelLaunchExWithArgs` 是异步调用，立即返回
2. **阻塞同步**: `rtStreamSynchronize` 是阻塞调用，会等待流中所有任务完成
3. **内存拷贝**: 所有 `rtMemcpy` 调用都是同步的，拷贝完成后才返回
4. **事件同步**: `aclrtStreamWaitEvent` 用于在流之间建立依赖关系
5. **并行执行**: AICore 和 AICPU 内核可以并行执行，最后通过流同步等待两者都完成

## 性能优化建议

1. **减少内存拷贝**: 尽量复用设备内存，减少 H2D 拷贝次数
2. **异步执行**: 利用异步特性，在数据拷贝的同时准备其他资源
3. **流管理**: 合理使用多个流实现并行执行
4. **事件同步**: 使用事件而非流同步可以减少不必要的等待
