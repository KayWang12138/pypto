# DeviceLaunchOnceWithDeviceTensorData 中涉及的 rt 接口调用分析

本文档详细梳理 `DeviceLaunchOnceWithDeviceTensorData` 函数中所有涉及 Runtime (rt) 接口调用的地方。

## 一、直接调用的 rt 接口

### 1. 阶段3: SetCaptureStream → rtStreamAddToModel

**位置**: `device_launcher.cpp:94`

```cpp
int DeviceLauncher::SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream, bool &isCapture)
{
    // ...
    if (isCapture) {
        rtError_t ret = rtStreamAddToModel(aicpuStream, rtModel);
        if (ret != 0) {
            ALOG_ERROR_F("rtStreamAddToModel failed, return[%d]", ret);
            return -1;
        }
    }
}
```

**接口**: `rtStreamAddToModel`
- **功能**: 将流添加到捕获模型中
- **参数**: `aicpuStream` (流句柄), `rtModel` (模型句柄)
- **调用时机**: 当检测到流捕获模式激活时

---

## 二、间接调用的 rt 接口（通过 DeviceRunner）

### 2. 阶段10: RegisterKernelBin → rtRegisterAllKernel

**位置**: `device_runner.cpp:821`

```cpp
int DeviceRunner::RegisterKernelBin(void **hdl) {
    // ...
    rtDevBinary_t binary{.magic = RT_DEV_BINARY_MAGIC_ELF, .version = 0, .data = bin, .length = binSize};
    int rc = rtRegisterAllKernel(&binary, hdl);
    // ...
}
```

**接口**: `rtRegisterAllKernel`
- **功能**: 注册内核二进制到运行时
- **参数**: `binary` (二进制数据), `hdl` (输出句柄)
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `DeviceRunner::RegisterKernelBin`

---

### 3. 阶段11: DynamicLaunch → rtKernelLaunchWithHandleV2

**位置**: `device_runner.cpp:515`

```cpp
int DeviceRunner::launchDynamicAiCore(rtStream_t aicoreStream, AstKernelArgs *kernelArgs) {
    // ...
    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;
    return rtKernelLaunchWithHandleV2(binHdl_, tilingKey, blockDim_, &rtArgs, nullptr, aicoreStream, &cfg);
}
```

**接口**: `rtKernelLaunchWithHandleV2`
- **功能**: 启动 AICore 内核
- **参数**: 
  - `binHdl_`: 内核二进制句柄
  - `tilingKey`: Tiling 键值
  - `blockDim_`: 块维度
  - `&rtArgs`: 内核参数
  - `aicoreStream`: AICore 流
  - `&cfg`: 任务配置
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `DeviceRunner::DynamicLaunch` → `launchDynamicAiCore`

---

### 4. 阶段11: DynamicLaunch → rtAicpuKernelLaunchExWithArgs

**位置**: `device_runner.cpp:535`

```cpp
int DeviceRunner::launchDynamicAiCpu(rtStream_t aicpuStream, AstKernelArgs *kArgs) {
    // ...
    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum_, &rtArgs, nullptr, aicpuStream, 0);
}
```

**接口**: `rtAicpuKernelLaunchExWithArgs`
- **功能**: 启动 AICPU 内核
- **参数**: 
  - `KERNEL_TYPE_AICPU_KFC`: 内核类型
  - `"AST_DYN_AICPU"`: 内核名称
  - `aicpuNum_`: AICPU 数量
  - `&rtArgs`: 内核参数
  - `aicpuStream`: AICPU 流
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `DeviceRunner::DynamicLaunch` → `launchDynamicAiCpu`

---

### 5. 阶段13: DynamicLaunchSynchronize → rtStreamSynchronize

**位置**: `device_runner.cpp:490-491`

```cpp
int DeviceRunner::DynamicLaunchSynchronize(rtStream_t aicpuStream, rtStream_t ctrlStream, rtStream_t aicoreStream) {
    int rcAicore = rtStreamSynchronize(aicoreStream);
    int rcAicpu = rtStreamSynchronize(aicpuStream);
    // ...
}
```

**接口**: `rtStreamSynchronize`
- **功能**: 同步流，等待流中所有任务完成
- **参数**: `aicoreStream` / `aicpuStream` (流句柄)
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `DeviceRunner::DynamicLaunchSynchronize`

---

## 三、间接调用的 rt 接口（通过其他辅助函数）

### 6. 阶段7: CheckDeviceId → rtGetDevice

**位置**: `runtime.h:94`

```cpp
inline void CheckDeviceId() {
    int32_t devId = 0;
    int32_t getDeviceResult = rtGetDevice(&devId);
    // ...
}
```

**接口**: `rtGetDevice`
- **功能**: 获取当前设备 ID
- **参数**: `&devId` (输出设备 ID)
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `CheckDeviceId`

---

### 7. 阶段8/9: DeviceInitTilingData / DeviceInitKernelInOuts → rtMemcpy

**位置**: 通过 `DeviceMemoryUtils` 间接调用

**接口**: `rtMemcpy`
- **功能**: 内存拷贝（主机到设备或设备到主机）
- **调用场景**:
  - 初始化 Tiling 数据时拷贝到设备
  - 初始化内核输入输出时拷贝张量数据
- **常见调用**:
  - `RT_MEMCPY_HOST_TO_DEVICE`: 主机到设备
  - `RT_MEMCPY_DEVICE_TO_HOST`: 设备到主机

---

### 8. 阶段11: DynamicLaunch → RunPrepare → rtMemcpy

**位置**: `device_runner.cpp:580`

```cpp
int DeviceRunner::RunPrepare() {
    for (uint32_t i = 0; i < args_.nrAic + args_.nrAiv; i++) {
        rtMemcpy((reinterpret_cast<uint8_t *>(args_.sharedBuffer + sizeof(uint64_t) * SHAK_BUF_DFX_DATA_INDEX)) + i * SHARED_BUFFER_SIZE,
            sizeof(uint64_t),
            reinterpret_cast<uint8_t *>(&perfData_[i]),
            sizeof(uint64_t),
            RT_MEMCPY_HOST_TO_DEVICE);
    }
}
```

**接口**: `rtMemcpy`
- **功能**: 拷贝性能数据到设备共享缓冲区
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `DeviceRunner::DynamicLaunch` → `RunPrepare`

---

### 9. 阶段11: DynamicLaunch → RunPreSync → aclrtEvent 相关接口

**位置**: `device_runner.cpp:596-610`

```cpp
int DeviceRunner::RunPreSync(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    aclrtEvent event;
    int rc = aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC);
    // ...
    rc = aclrtRecordEvent(event, aicoreStream);
    // ...
    rc = aclrtStreamWaitEvent(aicpuStream, event);
    // ...
}
```

**接口**: 
- `aclrtCreateEventExWithFlag`: 创建事件（ACL 接口，底层可能调用 rt 接口）
- `aclrtRecordEvent`: 记录事件
- `aclrtStreamWaitEvent`: 流等待事件

**注意**: 这些是 ACL 接口，但底层可能调用 rt 接口。

---

### 10. 阶段11: DynamicLaunch → DynamicKernelLaunch → rtMemcpy

**位置**: `device_runner.cpp:751`

```cpp
int DeviceRunner::DynamicLaunch(...) {
    // ...
    int rc = rtMemcpy(kernelArgs->cfgdata, sizeof(localArgs), &localArgs, sizeof(localArgs), RT_MEMCPY_HOST_TO_DEVICE);
    // ...
}
```

**接口**: `rtMemcpy`
- **功能**: 拷贝内核参数到设备
- **调用路径**: `DeviceLaunchOnceWithDeviceTensorData` → `DeviceRunner::DynamicLaunch` → `DynamicKernelLaunch`

---

## 四、rt 接口调用汇总表

| 阶段 | rt 接口 | 调用位置 | 功能 | 直接/间接 |
|------|---------|----------|------|-----------|
| 3 | `rtStreamAddToModel` | `SetCaptureStream` | 将流添加到捕获模型 | 间接 |
| 7 | `rtGetDevice` | `CheckDeviceId` | 获取设备 ID | 间接 |
| 8/9 | `rtMemcpy` | `DeviceInitTilingData` / `DeviceInitKernelInOuts` | 内存拷贝 | 间接 |
| 10 | `rtRegisterAllKernel` | `RegisterKernelBin` | 注册内核二进制 | 间接 |
| 11 | `rtKernelLaunchWithHandleV2` | `launchDynamicAiCore` | 启动 AICore 内核 | 间接 |
| 11 | `rtAicpuKernelLaunchExWithArgs` | `launchDynamicAiCpu` | 启动 AICPU 内核 | 间接 |
| 11 | `rtMemcpy` | `RunPrepare` | 拷贝性能数据 | 间接 |
| 11 | `rtMemcpy` | `DynamicKernelLaunch` | 拷贝内核参数 | 间接 |
| 11 | `aclrtCreateEventExWithFlag` | `RunPreSync` | 创建事件 | 间接 |
| 11 | `aclrtRecordEvent` | `RunPreSync` | 记录事件 | 间接 |
| 11 | `aclrtStreamWaitEvent` | `RunPreSync` | 流等待事件 | 间接 |
| 13 | `rtStreamSynchronize` | `DynamicLaunchSynchronize` | 同步流 | 间接 |

---

## 五、rt 接口分类

### 5.1 流管理接口
- `rtStreamAddToModel`: 流捕获相关
- `rtStreamSynchronize`: 流同步

### 5.2 内核启动接口
- `rtKernelLaunchWithHandleV2`: AICore 内核启动
- `rtAicpuKernelLaunchExWithArgs`: AICPU 内核启动

### 5.3 内存管理接口
- `rtMemcpy`: 内存拷贝（主机↔设备）
- `rtMemset`: 内存设置（在初始化阶段可能调用）

### 5.4 设备管理接口
- `rtGetDevice`: 获取设备 ID
- `rtGetLogicDevIdByUserDevId`: 获取逻辑设备 ID（可能间接调用）

### 5.5 内核注册接口
- `rtRegisterAllKernel`: 注册内核二进制

### 5.6 事件管理接口（ACL 层，底层可能调用 rt）
- `aclrtCreateEventExWithFlag`: 创建事件
- `aclrtRecordEvent`: 记录事件
- `aclrtStreamWaitEvent`: 流等待事件

---

## 六、调用流程图

```
DeviceLaunchOnceWithDeviceTensorData
│
├─ 阶段3: SetCaptureStream
│   └─ rtStreamAddToModel
│
├─ 阶段7: CheckDeviceId
│   └─ rtGetDevice
│
├─ 阶段8/9: DeviceInitTilingData / DeviceInitKernelInOuts
│   └─ rtMemcpy (通过 DeviceMemoryUtils)
│
├─ 阶段10: RegisterKernelBin
│   └─ rtRegisterAllKernel
│
├─ 阶段11: DynamicLaunch
│   ├─ RunPrepare
│   │   └─ rtMemcpy (性能数据)
│   ├─ RunPreSync
│   │   ├─ aclrtCreateEventExWithFlag
│   │   ├─ aclrtRecordEvent
│   │   └─ aclrtStreamWaitEvent
│   ├─ DynamicKernelLaunch
│   │   └─ rtMemcpy (内核参数)
│   ├─ launchDynamicAiCore
│   │   └─ rtKernelLaunchWithHandleV2
│   └─ launchDynamicAiCpu
│       └─ rtAicpuKernelLaunchExWithArgs
│
└─ 阶段13: DynamicLaunchSynchronize
    └─ rtStreamSynchronize (aicoreStream, aicpuStream)
```

---

## 七、注意事项

1. **ACL vs RT 接口**: 
   - 部分接口是 ACL 层接口（如 `aclrtCreateEventExWithFlag`），但底层实现可能调用 rt 接口
   - 直接 rt 接口通常以 `rt` 开头

2. **错误处理**: 
   - 所有 rt 接口调用都应该检查返回值
   - 返回 `RT_ERROR_NONE` (0) 表示成功

3. **流同步**: 
   - `rtStreamSynchronize` 是阻塞调用，会等待流中所有任务完成
   - 在性能分析场景中可能需要多次同步

4. **内存拷贝方向**: 
   - `RT_MEMCPY_HOST_TO_DEVICE`: 主机到设备
   - `RT_MEMCPY_DEVICE_TO_HOST`: 设备到主机

5. **内核启动**: 
   - AICore 和 AICPU 使用不同的启动接口
   - AICore 使用 `rtKernelLaunchWithHandleV2`
   - AICPU 使用 `rtAicpuKernelLaunchExWithArgs`

---

## 八、相关文件

- `framework/src/machine/runtime/device_launcher.cpp`: 主函数实现
- `framework/src/machine/runtime/device_runner.cpp`: DeviceRunner 实现，包含大部分 rt 接口调用
- `framework/src/machine/runtime/runtime.h`: CheckDeviceId 等辅助函数
- `framework/src/machine/runtime/device_memory_utils.h`: 内存管理相关接口
