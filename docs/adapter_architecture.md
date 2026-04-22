# PyPTO Adapter 机制架构分析

## 1. 概述

Adapter 模块是 PyPTO 框架与华为 CANN（Compute Architecture for Neural Networks）运行时库之间的**动态加载抽象层**。它将 PyPTO 与 CANN 具体实现解耦，使框架能在有无 CANN 硬件的环境中均可运行。

**两种工作模式：**

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| **CANN 在线** | 编译时定义 `BUILD_WITH_CANN`，运行时库存在 | 通过 `dlopen`/`dlsym` 从 CANN 共享库动态加载函数 |
| **Stub 回退** | 编译时未定义 `BUILD_WITH_CANN`，或运行时库不存在 | 调用无操作（no-op）的 stub 函数，返回成功值 |

## 2. 分层架构

```
┌─────────────────────────────────────────────────┐
│               调用方代码                          │
│  (machine/runtime, distributed_context, ...)     │
└────────────────────┬────────────────────────────┘
                     │ 调用公共 API
                     v
┌─────────────────────────────────────────────────┐
│            API 层 (api/*.h, api/*.cpp)            │
│  公共 C++ 接口，类型安全的函数分发                  │
│  ┌──────────────┐    ┌──────────────────────┐    │
│  │ #ifdef       │    │ Stub 回退             │    │
│  │ BUILD_WITH   │ 或 │ stubs/*.cpp           │    │
│  │ _CANN 路径   │    │ 无操作实现             │    │
│  └──────┬───────┘    └──────────────────────┘    │
│         │                                        │
│         v                                        │
│  AdapterManager.Instance().GetXxxAdapter()       │
│         .GetFunction(EnumValue)                  │
└────────────────────┬────────────────────────────┘
                     │ dlsym 函数指针查找
                     v
┌─────────────────────────────────────────────────┐
│       Manager 层 (manager/)                      │
│  ┌──────────────┐  ┌──────────────────┐          │
│  │ AdapterMgr   │  │ CannAdapter<T>   │          │
│  │ 单例，持有   │──│ 模板类，管理     │          │
│  │ 6 个适配器   │  │ 函数指针数组     │          │
│  └──────────────┘  └────────┬─────────┘          │
│                             │ dlopen/dlsym        │
│                    ┌────────┴─────────┐          │
│                    │ PluginHandler    │          │
│                    │ RAII 封装        │          │
│                    │ dlopen/dlclose   │          │
│                    └──────────────────┘          │
└────────────────────┬────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────┐
│       CANN 共享库 (.so)                          │
│  libascendcl.so, libruntime.so, libascend_hal.so│
│  libhccl.so, libprofapi.so, libascend_dump.so   │
└─────────────────────────────────────────────────┘
```

**五层结构：** 类型定义(Types) → PluginHandler → CannAdapter → AdapterManager → API → Stubs

## 3. 核心组件详解

### 3.1 PluginHandler（dlopen/dlsym RAII 封装）

**文件：** `manager/plugin_handler.h`, `manager/plugin_handler.cpp`

对 POSIX `dlopen`/`dlsym`/`dlclose` 的轻量 RAII 封装。

| 方法 | 实现 |
|------|------|
| `OpenHandler(libName)` | `dlopen(libName, RTLD_NOW \| RTLD_GLOBAL)`，失败时打印 WARNING 日志 |
| `CloseHandler()` | `dlclose(handler_)`，置空 |
| `GetFunction(funcName)` | `dlsym(handler_, funcName)`，失败时打印 INFO 日志 |

**dlopen 标志：** `RTLD_NOW | RTLD_GLOBAL` — 立即解析所有符号（非延迟），且解析的符号对后续加载的库可用。

### 3.2 CannAdapter 模板类

**文件：** `manager/cann_adapter.h`

类模板，以枚举类型 `EnumType` 为参数。每个适配器域（ACL、Runtime、HAL 等）各有特化实例。

```cpp
template<typename EnumType>
class CannAdapter {
    bool isInit_;
    PluginHandler libHandler_;
    std::array<void*, static_cast<size_t>(EnumType::Bottom)> functions_;
};
```

**关键约束：** 每个枚举类型必须有 `Bottom` 哨兵值作为最后一个枚举项，用于确定函数指针数组大小。

**核心方法：**

| 方法 | 说明 |
|------|------|
| `Initialize(libName, funcNameMap)` | 打开共享库，解析所有函数符号。返回 true 表示成功或已初始化 |
| `GetFunction(func)` | 返回指定枚举值对应的原始函数指针，越界返回 nullptr |

**Initialize() 流程：**
1. `isInit_` 已为 true → 直接返回 true（幂等）
2. `libHandler_.OpenHandler(libName)` — dlopen，失败则返回 false
3. `InitFunctions(funcNameMap)` — 遍历映射表，逐个 dlsym 解析。**单个符号解析失败仅记录日志，不阻止整体初始化**
4. 置 `isInit_ = true`

**设计要点：** 支持部分加载 — 如果 dlopen 成功但某些 dlsym 失败，适配器仍被视为"已初始化"。缺失的函数槽位为 nullptr，API 层调用时自然回退到 stub。

### 3.3 AdapterManager（单例）

**文件：** `manager/adapter_manager.h`, `manager/adapter_manager.cpp`

Meyer's 单例，持有全部 6 个适配器实例。

**初始化顺序：** ACL → Adump → HAL → HCCL → Msprof → Runtime

每个适配器的 `Initialize()` 独立执行，失败仅记录 INFO 日志，不影响其他适配器加载。

## 4. 六个适配器域

### 汇总表

| 适配器 | 共享库 | 枚举类型 | 函数数量 | 职责 |
|--------|--------|---------|---------|------|
| ACL | `libascendcl.so` | `AclFunc` | 15 | ACL 初始化/设备管理/内存拷贝/事件/流 |
| Runtime | `libruntime.so` | `RuntimeFunc` | 23 | 内存分配/设备管理/内核加载/流管理 |
| HAL | `libascend_hal.so` | `HalFunc` | 3 | 硬件抽象层（内存控制/资源映射） |
| HCCL | `libhccl.so` | `HcclFunc` | 7 | 分布式通信（通信子/拓扑） |
| Msprof | `libprofapi.so` | `MsprofFunc` | 6 | 性能分析（时间戳/上报/回调） |
| Adump | `libascend_dump.so` | `AdumpFunc` | 2 | 数据 Dump（开关/Tensor 导出） |

**总计：** 6 个共享库，56 个公共 API 函数

### 各域函数符号映射

#### ACL（15 个函数）

| 枚举值 | CANN 符号 | 公共 API |
|--------|----------|---------|
| `Init` | `aclInit` | `AclInit` |
| `Finalize` | `aclFinalize` | `AclFinalize` |
| `RtMemcpy` | `aclrtMemcpy` | `AclRtMemcpy` |
| `RtSetDevice` | `aclrtSetDevice` | `AclRtSetDevice` |
| `RtResetDevice` | `aclrtResetDevice` | `AclRtResetDevice` |
| `RtCreateEvent` | `aclrtCreateEvent` | `AclRtCreateEvent` |
| `RtRecordEvent` | `aclrtRecordEvent` | `AclRtRecordEvent` |
| `RtCreateEventExWithFlag` | `aclrtCreateEventExWithFlag` | `AclRtCreateEventExWithFlag` |
| `RtStreamWaitEvent` | `aclrtStreamWaitEvent` | `AclRtStreamWaitEvent` |
| `RtGetStreamResLimit` | `aclrtGetStreamResLimit` | `AclRtGetStreamResLimit` |
| `RtGetStreamAttribute` | `aclrtGetStreamAttribute` | `AclRtGetStreamAttribute` |
| `RtCacheLastTaskOpInfo` | `aclrtCacheLastTaskOpInfo` | `AclRtCacheLastTaskOpInfo` |
| `RtSetExceptionInfoCallback` | `aclrtSetExceptionInfoCallback` | `AclRtSetExceptionInfoCallback` |
| `MdlRICaptureGetInfo` | `aclmdlRICaptureGetInfo` | `AclMdlRICaptureGetInfo` |
| `MdlRICaptureThreadExchangeMode` | `aclmdlRICaptureThreadExchangeMode` | `AclMdlRICaptureThreadExchangeMode` |

#### Runtime（23 个函数）

| 枚举值 | CANN 符号 | 类别 |
|--------|----------|------|
| `Malloc` | `rtMalloc` | 内存管理 |
| `Memset` | `rtMemset` | 内存管理 |
| `Memcpy` | `rtMemcpy` | 内存管理 |
| `MemcpyAsync` | `rtMemcpyAsync` | 内存管理 |
| `Free` | `rtFree` | 内存管理 |
| `SetDevice` | `rtSetDevice` | 设备管理 |
| `GetDevice` | `rtGetDevice` | 设备管理 |
| `GetSocSpec` | `rtGetSocSpec` | 设备管理 |
| `GetSocVersion` | `rtGetSocVersion` | 设备管理 |
| `GetAiCpuCount` | `rtGetAiCpuCount` | 设备管理 |
| `GetL2CacheOffset` | `rtGetL2CacheOffset` | 设备管理 |
| `GetLogicDevIdByUserDevId` | `rtGetLogicDevIdByUserDevId` | 设备管理 |
| `FuncGetByName` | `rtsFuncGetByName` | 内核管理 |
| `BinaryLoadFromFile` | `rtsBinaryLoadFromFile` | 内核管理 |
| `StreamCreate` | `rtStreamCreate` | 流管理 |
| `StreamDestroy` | `rtStreamDestroy` | 流管理 |
| `StreamAddToModel` | `rtStreamAddToModel` | 流管理 |
| `StreamSynchronize` | `rtStreamSynchronize` | 流管理 |
| `DevBinaryUnRegister` | `rtDevBinaryUnRegister` | 内核注册 |
| `RegisterAllKernel` | `rtRegisterAllKernel` | 内核注册 |
| `LaunchCpuKernel` | `rtsLaunchCpuKernel` | 内核启动 |
| `KernelLaunchWithHandleV2` | `rtKernelLaunchWithHandleV2` | 内核启动 |
| `AicpuKernelLaunchExWithArgs` | `rtAicpuKernelLaunchExWithArgs` | 内核启动 |

#### HAL（3 个函数）

| 枚举值 | CANN 符号 |
|--------|----------|
| `MemCtl` | `halMemCtl` |
| `ResMap` | `halResMap` |
| `GetDeviceInfoByBuff` | `halGetDeviceInfoByBuff` |

#### HCCL（7 个函数）

| 枚举值 | CANN 符号 |
|--------|----------|
| `GetCommName` | `HcclGetCommName` |
| `GetL0TopoTypeEx` | `HcomGetL0TopoTypeEx` |
| `GetCommHandleByGroup` | `HcomGetCommHandleByGroup` |
| `GetRootInfo` | `HcclGetRootInfo` |
| `CommDestroy` | `HcclCommDestroy` |
| `CommInitRootInfo` | `HcclCommInitRootInfo` |
| `AllocComResourceByTiling` | `HcclAllocComResourceByTiling` |

> **注意：** HCCL 在 CANN Mobile 下不可用，使用 `#if defined(BUILD_WITH_CANN) && !defined(BUILD_WITH_CANN_MOBILE)` 条件排除。

#### Msprof（6 个函数）

| 枚举值 | CANN 符号 |
|--------|----------|
| `SysCycleTime` | `MsprofSysCycleTime` |
| `GetHashId` | `MsprofGetHashId` |
| `ReportApi` | `MsprofReportApi` |
| `ReportCompactInfo` | `MsprofReportCompactInfo` |
| `ReportAdditionalInfo` | `MsprofReportAdditionalInfo` |
| `RegisterCallback` | `MsprofRegisterCallback` |

#### Adump（2 个函数）

| 枚举值 | CANN 符号 | 备注 |
|--------|----------|------|
| `GetDumpSwitch` | `_ZN3Adx18AdumpGetDumpSwitchENS_8DumpTypeE` | C++ name mangling |
| `DumpTensorV2` | `_ZN3Adx17AdumpDumpTensorV2ERKSsS1_RKSt6vectorINS_12TensorInfoV2ESaIS3_EEPv` | C++ name mangling |

> **注意：** Adump 使用 C++ mangled 符号名（`Adx` 命名空间），是唯一需要数据转换层（`ConvertTensorInfos`）的适配器。

## 5. 统一分发模式

每个 API 函数遵循完全相同的分发逻辑。以 `AclRtSetDevice` 为例：

```cpp
AclError AclRtSetDevice(int32_t deviceId)
{
#ifdef BUILD_WITH_CANN
    void *func = AdapterManager::Instance().GetAclAdapter().GetFunction(AclFunc::RtSetDevice);
    if (func != nullptr) {
        aclError(*aclFunc)(int32_t) = reinterpret_cast<aclError(*)(int32_t)>(func);
        return aclFunc(deviceId);
    }
#endif
    return StubRtSetDevice(deviceId);
}
```

**分发流程：**

```
调用方: AclRtSetDevice(0)
  │
  ├─ 编译时 BUILD_WITH_CANN 已定义？
  │    ├─ 是 → AdapterManager.GetAclAdapter().GetFunction(RtSetDevice)
  │    │        ├─ 函数指针非空 → reinterpret_cast 后调用 CANN 原生函数
  │    │        └─ 函数指针为空 → 回退到 Stub
  │    └─ 否 → 直接调用 Stub
  │
  v
CANN: aclrtSetDevice(0)    或    Stub: return ACLRT_SUCCESS
```

## 6. 类型抽象体系

每个适配器域定义三层类型：

| 层级 | 文件 | 示例 | 说明 |
|------|------|------|------|
| **公共类型** | `api/xxx_define.h` | `AclError`, `RtStream` | PyPTO 暴露给调用方的类型 |
| **适配器类型** | `manager/types/xxx_adapter_types.h` | `AclFunc`, `kAclLibName` | 枚举值、库名、符号映射 |
| **CANN 原生类型** | CANN 头文件 | `aclError`, `aclrtStream` | 仅在 `#ifdef BUILD_WITH_CANN` 内使用 |

**编译时类型安全：** 每个 `*_api.cpp` 文件底部包含 `static_assert` 校验，确保 PyPTO 公共类型与 CANN 原生类型的二进制兼容性（枚举值相等、结构体大小相同）。

## 7. Stub 层

所有 stub 遵循统一模式：记录 DEBUG 日志 → 抑制未使用参数警告 → 返回域对应的成功值。

| Stub 文件 | 成功返回值 | 特殊行为 |
|-----------|-----------|---------|
| `acl_stubs.cpp` | `ACLRT_SUCCESS` (0) | `StubRtGetStreamResLimit` 设置 `*value = 20`；`StubRtGetStreamAttribute` 设置 `cacheOpInfoSwitch = 1` |
| `runtime_stubs.cpp` | `RT_SUCCESS` (0) | 不写入输出参数 |
| `hal_stubs.cpp` | `HAL_ERROR_NONE` (0) | 不写入输出参数 |
| `hcomm_stubs.cpp` | `HCOMM_SUCCESS` (0) | 不写入输出参数 |
| `msprof_stubs.cpp` | `0` | 所有函数返回 0 |
| `adump_stubs.cpp` | `0` | `GetDumpSwitch` 返回 0（dump 关闭） |

**重要：** 大部分 stub 不会初始化输出参数。调用方在 stub 模式下需注意输出参数可能未初始化。

## 8. 容错机制（三层防线）

### 第一层：编译时守卫（`#ifdef BUILD_WITH_CANN`）

- 无 CANN 编译时：所有 API 函数编译为纯 stub 路径
- 无需 CANN 头文件、无需链接 CANN 库
- HCCL 额外守卫：`#if defined(BUILD_WITH_CANN) && !defined(BUILD_WITH_CANN_MOBILE)`

### 第二层：运行时容错（dlopen 失败）

- CANN 编译但库不在时：`dlopen` 失败 → `Initialize()` 返回 false → `GetFunction()` 返回 nullptr → API 回退到 stub
- **支持部分失败：** dlopen 成功但部分 dlsym 失败时，缺失函数回退 stub，其余正常工作

### 第三层：编译时 ABI 校验（static_assert）

- 验证 PyPTO 类型与 CANN 类型的枚举值、结构体大小一致
- 编译期捕获 ABI 不匹配，而非运行时

## 9. 构建配置

**目标：** `tile_fwk_adapter`（共享库）

**编译定义（PUBLIC，传递给消费者）：**
- `BUILD_WITH_CANN` — 当 CANN 可用且非 UTest 编译时定义
- `BUILD_WITH_CANN_MOBILE` — 当使用 CANN Mobile 编译时定义

**Include 目录（PUBLIC，仅 BUILD_WITH_CANN 时）：**
- `${ASCEND_CANN_PACKAGE_PATH}/pkg_inc`
- `${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling`
- `${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime`
- `${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime`

**链接目录（PUBLIC，仅 BUILD_WITH_CANN 时）：**
- `${ASCEND_CANN_PACKAGE_PATH}/lib64`
- `/usr/local/Ascend/driver/lib64/driver`
- `${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib`

**链接库（PRIVATE）：**
- `dl`（POSIX 动态加载）、`c_sec`（安全库）、`intf_pub_cxx17`、`tile_fwk_intf_pub`

## 10. 完整文件清单

```
framework/src/adapter/
├── CMakeLists.txt                          # 构建配置
├── manager/
│   ├── adapter_manager.h                   # 单例声明
│   ├── adapter_manager.cpp                 # 单例实现
│   ├── cann_adapter.h                      # CannAdapter<T> 模板
│   ├── plugin_handler.h                    # dlopen RAII 声明
│   ├── plugin_handler.cpp                  # dlopen RAII 实现
│   └── types/
│       ├── acl_adapter_types.h             # ACL 枚举/库名/符号映射
│       ├── runtime_adapter_types.h         # Runtime 枚举/库名/符号映射
│       ├── hal_adapter_types.h             # HAL 枚举/库名/符号映射
│       ├── hccl_adapter_types.h            # HCCL 枚举/库名/符号映射
│       ├── msprof_adapter_types.h          # Msprof 枚举/库名/符号映射
│       └── adump_adapter_types.h           # Adump 枚举/库名/符号映射
├── api/
│   ├── acl_define.h / acl_api.h / acl_api.cpp
│   ├── runtime_define.h / runtime_api.h / runtime_api.cpp
│   ├── hal_define.h / hal_api.h / hal_api.cpp
│   ├── hcomm_define.h / hcomm_api.h / hcomm_api.cpp
│   ├── msprof_define.h / msprof_api.h / msprof_api.cpp
│   └── adump_define.h / adump_api.h / adump_api.cpp
└── stubs/
    ├── acl_stubs.h / acl_stubs.cpp
    ├── runtime_stubs.h / runtime_stubs.cpp
    ├── hal_stubs.h / hal_stubs.cpp
    ├── hcomm_stubs.h / hcomm_stubs.cpp
    ├── msprof_stubs.h / msprof_stubs.cpp
    └── adump_stubs.h / adump_stubs.cpp
```
