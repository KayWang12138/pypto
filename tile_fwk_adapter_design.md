# tile_fwk_adapter 模块需求设计文档

## 1. 项目概述

### 1.1 项目背景

PyPTO (Python Parallel Tensor Operation) 框架在运行时需要调用华为昇腾 CANN (Compute Architecture for Neural Networks) 底层库提供的功能，包括运行时管理、算子执行、硬件抽象、分布式通信和性能分析等。为了解耦框架与底层库的直接依赖，提升框架的可测试性和可移植性，需要引入一个统一的适配器层来封装 CANN 库的动态加载和接口调用。

### 1.2 项目目标

- **解耦依赖**：将框架与 CANN 底层库解耦，实现动态加载
- **统一接口**：提供统一的适配器接口，隐藏底层实现细节
- **支持测试**：支持在非 CANN 环境下通过桩函数进行单元测试
- **灵活配置**：支持编译时决定是否链接 CANN 库

## 2. 需求分析

### 2.1 功能需求

| 需求编号 | 需求描述 | 优先级 |
|---------|---------|--------|
| FR-001 | 支持 Runtime 库 (libruntime.so) 的动态加载和接口封装 | 高 |
| FR-002 | 支持 ACL 库 (libacl.so) 的动态加载和接口封装 | 高 |
| FR-003 | 支持 HCCL 库 (libhcom.so) 的动态加载和接口封装 | 高 |
| FR-004 | 支持 Msprof 库 (libmsprof.so) 的动态加载和接口封装 | 中 |
| FR-005 | 支持 HAL 库的动态加载和接口封装 | 高 |
| FR-006 | 提供桩函数层，支持无 CANN 环境下的测试 | 高 |
| FR-007 | 支持编译时条件编译，可选是否链接 CANN 库 | 中 |

### 2.2 非功能需求

| 需求编号 | 需求描述 | 指标 |
|---------|---------|------|
| NFR-001 | 接口调用性能 | 动态调用开销 < 5% |
| NFR-002 | 代码覆盖率 | 新增代码单元测试覆盖率 > 80% |
| NFR-003 | 可维护性 | 遵循 SOLID 原则，支持扩展新适配器类型 |

## 3. 系统设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PyPTO Framework                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Runtime  │ │   ACL    │ │   HAL    │ │  HCCL    │ │  Msprof  │ │
│  │   API    │ │   API    │ │   API    │ │   API    │ │   API    │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
├───────┼────────────┼────────────┼────────────┼────────────┼───────┤
│       │            │            │            │            │       │
│  ┌────▼────────────▼────────────▼────────────▼────────────▼─────┐ │
│  │                      Adapter Manager                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │ │
│  │  │ Runtime  │ │   ACL    │ │   HAL    │ │  HCCL    │ ...    │ │
│  │  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │        │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │ │
│  └───────┼────────────┼────────────┼────────────┼────────────────┘ │
├──────────┼────────────┼────────────┼────────────┼──────────────────┤
│          │            │            │            │                  │
│  ┌───────▼────────────▼────────────▼────────────▼─────────────────┐ │
│  │              Plugin Handler (dlopen/dlsym)                      │ │
│  └───────────────────────┬─────────────────────────────────────────┘ │
├──────────────────────────┼───────────────────────────────────────────┤
│  ┌───────────────────────▼─────────────────────────────────────────┐ │
│  │  libruntime.so  libacl.so  libhal.so  libhcom.so  libmsprof.so │ │
│  │                      (CANN Libraries)                           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块组成

#### 3.2.1 API 层 (`adapter/api/`)

提供面向框架的 C++ 接口封装：

| 文件 | 功能描述 |
|------|----------|
| `runtime_api.h/cpp` | Runtime 运行时接口封装（内存管理、流管理、内核启动等） |
| `runtime_define.h` | Runtime 数据类型和结构体定义 |
| `acl_api.h/cpp` | ACL 接口封装（设备初始化、内存拷贝、事件管理等） |
| `acl_define.h` | ACL 数据类型和结构体定义 |
| `hal_api.h/cpp` | HAL 硬件抽象层接口封装（内存控制、资源映射、设备信息查询） |
| `hal_define.h` | HAL 数据类型和结构体定义 |
| `hcomm_api.h/cpp` | HCCL 分布式通信接口封装 |
| `hcomm_define.h` | HCCL 数据类型和结构体定义 |
| `msprof_api.h/cpp` | Msprof 性能分析接口封装 |
| `msprof_define.h` | Msprof 数据类型和结构体定义 |

#### 3.2.2 管理层 (`adapter/manager/`)

实现适配器的核心管理逻辑：

| 文件 | 功能描述 |
|------|----------|
| `adapter_manager.h/cpp` | 单例管理器，统一管理五类适配器实例（ACL/HAL/HCCL/Msprof/Runtime） |
| `cann_adapter.h` | 模板类 `CannAdapter<T>`，实现通用适配器逻辑 |
| `plugin_handler.h/cpp` | 动态库加载处理器（封装 dlopen/dlsym/dlclose） |
| `types/*_adapter_types.h` | 各类适配器的函数枚举和名称映射 |

#### 3.2.3 桩函数层 (`adapter/stubs/`)

提供无 CANN 环境下的测试桩：

| 文件 | 功能描述 |
|------|----------|
| `runtime_stubs.h/cpp` | Runtime 接口桩函数实现 |
| `acl_stubs.h/cpp` | ACL 接口桩函数实现 |
| `hal_stubs.h/cpp` | HAL 接口桩函数实现 |
| `hcomm_stubs.h/cpp` | HCCL 接口桩函数实现 |
| `msprof_stubs.h/cpp` | Msprof 接口桩函数实现 |

### 3.3 核心类设计

#### 3.3.1 CannAdapter 模板类

```cpp
template<typename EnumType>
class CannAdapter {
public:
    CannAdapter() : isInit_(false) {
        functions_.fill(nullptr);
    }
    ~CannAdapter() {
        libHandler_.CloseHandler();
        functions_.fill(nullptr);
        isInit_ = false;
    }
    bool Initialize(const std::string &libName, 
                    const std::map<EnumType, std::string> &funcNameMap);
    bool Initialize(const std::map<EnumType, std::string> &funcNameMap);
    void *GetFunction(const EnumType func) const;
private:
    bool isInit_;
    PluginHandler libHandler_;
    std::array<void*, static_cast<size_t>(EnumType::Bottom)> functions_;
};
```

**设计要点**：
- 模板类设计，支持不同类型的函数枚举
- 使用 `std::array` 存储函数指针，保证 O(1) 访问性能
- 支持两种初始化方式：带动态库路径和不带动态库路径（桩函数模式）
- 析构时自动释放资源

#### 3.3.2 AdapterManager 单例类

```cpp
class AdapterManager {
public:
    static AdapterManager& Instance();
    const CannAdapter<AclFunc>& GetAclAdapter() const;
    const CannAdapter<HalFunc>& GetHalAdapter() const;
    const CannAdapter<HcclFunc>& GetHcclAdapter() const;
    const CannAdapter<MsprofFunc>& GetMsprofAdapter() const;
    const CannAdapter<RuntimeFunc>& GetRuntimeAdapter() const;
private:
    AdapterManager();
    ~AdapterManager();
    CannAdapter<AclFunc> aclAdapter_;
    CannAdapter<HalFunc> halAdapter_;
    CannAdapter<HcclFunc> hcclAdapter_;
    CannAdapter<MsprofFunc> msprofAdapter_;
    CannAdapter<RuntimeFunc> runtimeAdapter_;
};
```

**初始化逻辑**：
- `AclAdapter`：动态加载 `libacl.so`
- `HalAdapter`：使用函数映射表初始化（不依赖具体动态库，走桩函数或静态链接）
- `HcclAdapter`：动态加载 `libhcom.so`
- `MsprofAdapter`：动态加载 `libmsprof.so`
- `RuntimeAdapter`：动态加载 `libruntime.so`

#### 3.3.3 PluginHandler 类

```cpp
class PluginHandler {
public:
    PluginHandler();
    ~PluginHandler();
    bool OpenHandler(const std::string &libName);
    void CloseHandler();
    void* GetFunction(const std::string &funcName) const;
private:
    void *handler_;  // dlopen 返回的句柄
};
```

## 4. 接口设计

### 4.1 Runtime API 接口

| 接口名称 | 功能描述 |
|----------|----------|
| `RuntimeMalloc` | 设备内存分配 |
| `RuntimeMemset` | 设备内存初始化 |
| `RuntimeMemcpy` / `RuntimeMemcpyAsync` | 内存拷贝（同步/异步） |
| `RuntimeFree` | 释放设备内存 |
| `RuntimeSetDevice` / `RuntimeGetDevice` | 设备设置/获取 |
| `RuntimeGetSocSpec` / `RuntimeGetSocVersion` / `RuntimeGetAiCpuCount` | 芯片规格查询 |
| `RuntimeStreamCreate` / `RuntimeStreamDestroy` / `RuntimeStreamSynchronize` | 流管理 |
| `RuntimeKernelLaunchWithHandleV2` | 内核启动 |
| `RuntimeLaunchCpuKernel` | CPU 内核启动 |
| `RuntimeAicpuKernelLaunchExWithArgs` | AI CPU 内核启动 |

### 4.2 ACL API 接口

| 接口名称 | 功能描述 |
|----------|----------|
| `AclInit` / `AclFinalize` | ACL 初始化/反初始化 |
| `AclRtMemcpy` | ACL 运行时内存拷贝 |
| `AclRtSetDevice` / `AclRtResetDevice` | 设备管理 |
| `AclRtCreateEvent` / `AclRtRecordEvent` / `AclRtStreamWaitEvent` | 事件管理 |
| `AclRtSetExceptionInfoCallback` | 异常回调注册 |
| `AclMdlRICaptureGetInfo` / `AclMdlRICaptureThreadExchangeMode` | 模型重放捕获 |

### 4.3 HAL API 接口

| 接口名称 | 功能描述 |
|----------|----------|
| `HalMemCtl` | 内存控制 |
| `HalResMap` | 资源映射（AICORE、L2BUFF、C2C 等） |
| `HalGetDeviceInfoByBuff` | 通过缓冲区获取设备信息 |

### 4.4 HCCL API 接口

| 接口名称 | 功能描述 |
|----------|----------|
| `HcommGetCommName` | 获取通信域名称 |
| `HcommGetL0TopoTypeEx` | 获取拓扑类型 |
| `HcommGetCommHandleByGroup` | 通过组名获取通信句柄 |
| `HcommGetRootInfo` | 获取根节点信息 |
| `HcommCommInitRootInfo` / `HcommCommDestroy` | 通信域初始化/销毁 |
| `HcommAllocComResourceByTiling` | 分配通信资源 |

### 4.5 Msprof API 接口

| 接口名称 | 功能描述 |
|----------|----------|
| `MspfSysCycleTime` | 获取系统周期时间 |
| `MspfGetHashId` | 获取哈希 ID |
| `MspfReportApi` | 上报 API 调用信息 |
| `MspfReportCompactInfo` | 上报紧凑信息 |
| `MspfReportAdditionalInfo` | 上报附加信息 |
| `MspfRegisterCallback` | 注册回调函数 |

## 5. 编译配置

```cmake
add_library(tile_fwk_adapter SHARED)
target_sources(tile_fwk_adapter PRIVATE 
    api/*.cpp
    manager/*.cpp
    stubs/*.cpp)

target_include_directories(tile_fwk_adapter
    PUBLIC
        $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc>>
        $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
        $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime>>
        $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime>>
        $<BUILD_INTERFACE:$<$<BOOL:${BUILD_WITH_CANN}>:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(tile_fwk_adapter
    PUBLIC
        $<BUILD_INTERFACE:$<$<AND:$<BOOL:${BUILD_WITH_CANN}>,$<NOT:$<BOOL:${ENABLE_UTEST}>>>:BUILD_WITH_CANN>>
)
```

## 6. 测试策略

### 6.1 单元测试

- **测试文件**: `framework/tests/ut/machine/src/test_adapter_api.cpp`
- **测试内容**: 验证适配器初始化、函数加载、桩函数行为

### 6.2 集成测试

- 验证框架各模块通过适配器正确调用 CANN 功能
- 验证无 CANN 环境下桩函数正常工作

### 6.3 测试桩迁移

原测试桩文件从 `framework/tests/ut/stubs/` 迁移至 `framework/src/adapter/stubs/`，统一纳入适配器模块管理，移除旧 `framework/tests/ut/stubs/CMakeLists.txt`。

## 7. 风险评估

| 风险编号 | 风险描述 | 影响 | 缓解措施 |
|---------|---------|------|----------|
| R-001 | 动态加载性能开销 | 中 | 函数指针缓存，O(1) 访问 |
| R-002 | 库版本兼容性 | 中 | 严格定义函数签名，运行时检查 |
| R-003 | 多线程安全 | 低 | AdapterManager 单例初始化线程安全（Meyers' Singleton） |
| R-004 | 桩函数覆盖不全 | 低 | 逐步完善桩函数实现 |

## 8. 交付物

| 交付物 | 路径 | 说明 |
|--------|------|------|
| 源代码 | `framework/src/adapter/` | 适配器模块完整源码（126 个新增/修改文件） |
| 头文件 | `framework/include/tilefwk/pypto_fwk_log.h` | 对外暴露的日志头文件 |
| 单元测试 | `framework/tests/ut/machine/src/test_adapter_api.cpp` | 新增测试用例 |
| 构建配置 | `framework/src/adapter/CMakeLists.txt` | 模块构建配置 |

## 9. 变更记录

| 版本 | 日期 | 作者 | 变更内容 |
|------|------|------|----------|
| v1.0 | 2026-04-02 | arthur_morgan_hw | 初始版本，添加 tile_fwk_adapter 模块，支持 ACL/HAL/HCCL/Msprof/Runtime 五类适配器 |

---

**Commit ID**: `50b7ffab724ba2ae6f85aa9326152c6002a11650`

**变更统计**: 126 个文件变更，+4051/-1659 行代码
