# PyPTO 管理机制详解（Manager/Registry/Factory）

> **适用对象：** 想要深入理解PyPTO框架管理机制的开发者  
> **学习时间：** 40-60分钟  
> **前置知识：** 已阅读[核心概念](../02-core/01-concepts.md)和[Interface模块](../02-core/04-interface.md)  
> **学习目标：** 理解PyPTO框架中的各种管理机制、注册表机制和工厂模式

## 概述

PyPTO框架采用多种设计模式来管理系统组件，主要包括：
- **Manager模式**：单例管理器，统一管理特定资源
- **Registry模式**：注册表机制，支持动态注册和查找
- **Factory模式**：工厂模式，根据上下文创建对象

这些机制共同构成了PyPTO框架的核心基础设施。

**相关文档：**
- [关键机制列表](01-key-mechanisms-list.md) - 所有机制总览
- [Interface模块](../02-core/04-interface.md) - 接口和管理机制
- [Machine模块](../02-core/08-machine.md) - 运行时管理机制

---

## 目录

1. [Manager机制](#manager机制)
2. [Registry机制](#registry机制)
3. [Factory机制](#factory机制)
4. [机制关系与协作](#机制关系与协作)
5. [最佳实践](#最佳实践)

---

## Manager机制

### 概述

Manager机制采用单例模式，为特定资源提供全局访问入口。所有Manager都通过 `GetInstance()` 或 `Instance()` 方法获取单例实例。

### 1. Program机制（程序管理机制）

**定义**：全局程序单例，管理所有函数和编译状态

**核心职责**：
- 管理函数映射表（`functionmap_`）
- 管理当前函数栈（`currentFunctionPtr_`）
- 管理动态函数上下文（`currentDynamicFunctionPtr_`）
- 提供张量槽位管理器（`TensorSlotManager`）
- 提供函数缓存（`FunctionCache`）

**关键API**：
```cpp
// 获取单例
static Program &GetInstance();

// 函数管理
void BeginFunction(const std::string &funcName, ...);
Function *GetCurrentFunction();
void SetCurrentFunction(Function *function);
Function *GetFunctionByMagicName(const std::string &magicName);

// 张量槽位管理
std::shared_ptr<TensorSlotManager> GetTensorSlotManager();

// 动态函数管理
Function *GetCurrentDynamicFunction() const;
void SetCurrentDynamicFunction(Function *dynFunc);
```

**代码位置**：
- 头文件：`framework/src/interface/program/program.h`
- 实现文件：`framework/src/interface/program/program.cpp`

**使用场景**：
- 编译流程中访问当前函数
- 函数调用时查找被调用函数
- 管理函数层级关系
- 管理张量槽位映射

**文档位置**：[Interface模块](../02-core/04-interface.md)

---

### 2. ConfigManager机制（配置管理机制）

**定义**：全局配置管理系统，管理编译选项、运行时配置、Pass参数

**核心职责**：
- 管理配置键值对
- 支持配置作用域（Scope）
- 提供配置验证
- 支持配置持久化

**关键组件**：

#### 2.1 ConfigManager（旧版）

**代码位置**：
- 头文件：`framework/src/interface/configs/config_manager.h`
- 实现文件：`framework/src/interface/configs/config_manager.cpp`

**关键API**：
```cpp
static ConfigManager* GetInstance();
void SetConfig(const std::string &key, const std::string &value);
std::string GetConfig(const std::string &key, const std::string &defaultValue = "");
```

#### 2.2 ConfigManagerNg（新版）

**代码位置**：
- 头文件：`framework/src/interface/configs/config_manager_ng.h`
- 实现文件：`framework/src/interface/configs/config_manager_ng.h`

**关键特性**：
- 支持配置作用域（Scope）
- 类型安全的配置访问
- 配置验证和默认值

**关键API**：
```cpp
static ConfigManagerNg &GetInstance();
std::shared_ptr<ConfigScope> CurrentScope();
void SetConfig(const std::string &key, const Json &value);
Json GetConfig(const std::string &key, const Json &defaultValue = Json());
```

#### 2.3 Python侧配置缓存（_CachedOptions）

**代码位置**：`python/pypto/config.py`

**关键特性**：
- Python API的配置接口
- 配置缓存和同步
- 支持配置前缀分组

**关键API**：
```python
def set_codegen_options(**options):
    """设置代码生成选项"""
    
def set_pass_options(**options):
    """设置Pass选项"""
    
def set_host_options(**options):
    """设置Host选项"""
```

**使用场景**：
- 设置编译选项（codegen、pass、host等）
- 运行时配置（run_mode、log_level等）
- Pass参数配置

**文档位置**：[Interface模块](../02-core/04-interface.md)

---

### 3. PlatformManager机制（平台管理机制）

**定义**：硬件平台信息和管理系统

**核心职责**：
- 管理硬件平台信息
- 提供内存限制查询
- 管理设备信息
- 提供硬件特性查询

**关键API**：
```cpp
static PlatformManager &Instance();
Platform &GetPlatform();
Die &GetDie();
```

**代码位置**：
- 头文件：`framework/src/machine/platform/platform_manager.h`
- 实现文件：`framework/src/machine/platform/platform_manager.cpp`

**使用场景**：
- Pass优化时查询内存限制
- 代码生成时获取硬件特性
- 运行时获取设备信息

**文档位置**：[Machine模块](../02-core/08-machine.md)

---

### 4. OpInfoManager机制（操作信息管理机制）

**定义**：操作类型和信息的全局管理器

**核心职责**：
- 管理操作码到操作类型的映射
- 管理Tiling键值
- 管理操作函数名
- 管理控制流二进制缓冲区

**关键API**：
```cpp
static OpInfoManager &GetInstance();
const std::string &GetOpType();
uint64_t GetOpTilingKey();
std::string &GetOpFuncName();
std::vector<uint8_t> &GetControlBuffer();
std::string &GetCustomOpJsonPath();
```

**代码位置**：
- 头文件：`framework/src/interface/utils/op_info_manager.h`
- 实现文件：`framework/src/interface/utils/op_info_manager.cpp`

**使用场景**：
- 代码生成时获取操作类型
- 编译控制流时管理二进制缓冲区
- AICPU操作注册时获取操作信息

**文档位置**：[Interface模块](../02-core/04-interface.md)

---

### 5. CacheManager机制（缓存管理机制）

**定义**：编译结果的持久化缓存管理器

**核心职责**：
- 管理二进制缓存
- 管理缓存目录
- 支持缓存模式（CacheMode）
- 缓存文件的读写

**关键API**：
```cpp
static CacheManager &Instance();
bool LoadCache(const std::string &cacheKey, ...);
bool SaveCache(const std::string &cacheKey, ...);
```

**代码位置**：
- 头文件：`framework/src/machine/cache_manager/cache_manager.h`
- 实现文件：`framework/src/machine/cache_manager/cache_manager.cpp`

**缓存模式**：
- `CacheMode::DISABLE`：禁用缓存
- `CacheMode::READ_ONLY`：只读缓存
- `CacheMode::WRITE_ONLY`：只写缓存
- `CacheMode::READ_WRITE`：读写缓存

**使用场景**：
- 编译结果缓存
- 二进制文件缓存
- 加速重复编译

**文档位置**：[编译阶段功能详解](02-compile-stage.md)

---

### 6. LoggerManager机制（日志管理机制）

**定义**：日志系统的全局管理器

**核心职责**：
- 管理日志级别
- 统一日志接口
- 日志输出控制

**关键API**：
```cpp
static LoggerManager &GetInstance();
void SetLogLevel(int level);
void Log(int level, const std::string &message);
```

**代码位置**：
- 头文件：`framework/src/interface/utils/log.h`
- 实现文件：`framework/src/interface/utils/log.cpp`

**日志级别**：
- `ALOG_DEBUG`：调试信息
- `ALOG_INFO`：一般信息
- `ALOG_WARN`：警告信息
- `ALOG_ERROR`：错误信息
- `ALOG_FATAL`：致命错误

**使用场景**：
- 编译过程日志
- 运行时日志
- 调试信息输出

**文档位置**：[Interface模块](../02-core/04-interface.md)

---

### 7. AicoreManager机制（AICore管理机制）

**定义**：AICore设备的管理系统

**核心职责**：
- 管理AICore资源
- 任务调度
- 性能分析

**代码位置**：
- 头文件：`framework/src/machine/device/aicore_manager.h`
- 实现文件：`framework/src/machine/device/aicore_manager.cpp`

**使用场景**：
- AICore任务执行
- 性能分析
- 资源管理

---

### 8. AicpuTaskManager机制（AICPU任务管理机制）

**定义**：AICPU任务的管理系统

**核心职责**：
- 管理AICPU任务
- 后端服务管理
- 任务执行

**代码位置**：
- 头文件：`framework/src/machine/device/aicpu_task_manager.h`
- 实现文件：`framework/src/machine/device/aicpu_task_manager.cpp`

**使用场景**：
- AICPU任务提交
- 后端服务调用
- 任务状态管理

---

### 9. WrapManager机制（包装管理机制）

**定义**：动态函数包装的管理系统

**核心职责**：
- 管理控制流包装
- 函数调用包装
- 运行时包装

**代码位置**：
- 头文件：`framework/src/machine/device/dynamic/wrap_manager.h`
- 实现文件：`framework/src/machine/device/dynamic/wrap_manager.cpp`

**使用场景**：
- 动态函数包装
- 控制流执行
- 运行时调用

---

### 10. BackendServerHandleManager机制（后端服务句柄管理机制）

**定义**：后端服务句柄的管理系统

**核心职责**：
- 管理AICPU后端服务连接
- 句柄复用
- 连接池管理

**代码位置**：
- 头文件：`framework/src/machine/device/machine_interface/pypto_aicpu_interface.h`

**使用场景**：
- AICPU后端服务调用
- 连接管理
- 性能优化

---

## Registry机制

### 概述

Registry机制采用注册表模式，支持动态注册和查找。所有Registry都通过 `GetInstance()` 获取单例实例。

### 1. PassRegistry机制（Pass注册表机制）

**定义**：Pass的注册表系统，支持动态注册和创建Pass

**核心职责**：
- 注册Pass创建函数
- 根据Pass名称创建Pass实例
- 管理Pass注册信息

**关键API**：
```cpp
static PassRegistry &GetInstance();
void RegisterPass(PassName passName, PassCreateFn createFn);
std::unique_ptr<Pass> CreatePass(PassName passName);
```

**代码位置**：
- 头文件：`framework/src/passes/pass_mgr/pass_registry.h`
- 实现文件：`framework/src/passes/pass_mgr/pass_registry.cpp`

**注册方式**：
```cpp
// 自动注册宏
REGISTER_PASS(PassName::InferMemoryConflict, InferMemoryConflict);

// 手动注册
PassRegistry::GetInstance().RegisterPass(
    PassName::MyCustomPass,
    []() { return std::make_unique<MyCustomPass>(); }
);
```

**使用场景**：
- Pass自动注册
- Pass动态创建
- Pass插件扩展

**文档位置**：[Passes模块](../02-core/09-passes.md)

---

### 2. InferShapeRegistry机制（形状推断注册表机制）

**定义**：形状推断函数的注册表系统

**核心职责**：
- 注册操作的形状推断函数
- 根据操作码查找形状推断函数
- 调用形状推断函数

**关键API**：
```cpp
static InferShapeRegistry &GetInstance();
void RegisterInferShapeFunc(Opcode opcode, InferShapeFunc func);
void CallInferShapeFunc(Operation *op);
```

**代码位置**：
- 头文件：`framework/src/interface/operation/op_infer_shape_impl.h`
- 实现文件：`framework/src/interface/operation/op_infer_shape_impl.cpp`

**注册方式**：
```cpp
// 自动注册宏
REGISTER_INFER_SHAPE(Opcode::OP_ADD, InferShapeAdd);

// 手动注册
InferShapeRegistry::GetInstance().RegisterInferShapeFunc(
    Opcode::OP_MY_OP,
    [](Operation *op) { /* 形状推断逻辑 */ }
);
```

**使用场景**：
- 操作形状推断
- Pass优化时验证形状
- 图转换时推导形状

**文档位置**：[Operation类技术文档](../02-core/06-operation.md#形状推断机制)

---

### 3. TiledFuncRegistry机制（Tiled函数注册表机制）

**定义**：Tiled函数的注册表系统

**核心职责**：
- 注册操作的Tiled实现
- 根据操作码查找Tiled函数
- 调用Tiled函数

**代码位置**：
- 头文件：`framework/src/interface/operation/operation_common.h`

**使用场景**：
- Tiled函数注册
- 代码生成时调用Tiled函数
- 操作实现扩展

---

## Factory机制

### 概述

Factory机制采用工厂模式，根据上下文创建合适的对象。所有Factory都通过静态方法创建对象。

### 1. CodeGenFactory机制（代码生成工厂机制）

**定义**：代码生成器的工厂模式

**核心职责**：
- 根据上下文创建合适的代码生成器
- 管理代码生成器类型
- 提供统一的代码生成接口

**关键API**：
```cpp
static std::shared_ptr<CodeGenCCE> GetCodeGenCCE(const CodeGenCtx &ctx);
```

**代码位置**：
- 头文件：`framework/src/codegen/codegen_factory.h`
- 实现文件：`framework/src/codegen/codegen_factory.cpp`

**使用场景**：
- 代码生成时创建代码生成器
- 根据编译上下文选择生成器
- 支持多种代码生成后端

**文档位置**：[Codegen模块](../02-core/10-codegen.md)

---

## 机制关系与协作

### 机制调用关系图

```
┌─────────────────────────────────────────────────────────┐
│              管理机制调用关系图                            │
└─────────────────────────────────────────────────────────┘

Program::GetInstance()
  ├─→ GetTensorSlotManager() → TensorSlotManager
  ├─→ GetFunctionCache() → FunctionCache
  └─→ GetCurrentFunction() → Function*

ConfigManagerNg::GetInstance()
  ├─→ SetConfig() → 配置存储
  └─→ GetConfig() → 配置读取

PlatformManager::Instance()
  └─→ GetPlatform() → Platform

PassManager::Instance()
  └─→ PassRegistry::GetInstance() → CreatePass()
      └─→ Pass::RunOnFunction()

CodeGenFactory::GetCodeGenCCE()
  └─→ CodeGenCCE::GenCode()
      └─→ SymbolManager
```

### 典型使用流程

#### 1. 编译流程中的管理机制使用

```
1. Program::GetInstance().BeginFunction()
   └─→ 创建Function，设置currentFunctionPtr_

2. PassManager::Instance().RunPass()
   └─→ PassRegistry::GetInstance().CreatePass()
       └─→ Pass::RunOnFunction()

3. CodeGenFactory::GetCodeGenCCE()
   └─→ CodeGenCCE::GenCode()
       └─→ SymbolManager

4. PlatformManager::Instance().GetDie()
   └─→ 查询内存限制

5. CacheManager::Instance().SaveCache()
   └─→ 保存编译结果
```

#### 2. 运行时管理机制使用

```
1. Program::GetInstance().GetCurrentDynamicFunction()
   └─→ 获取当前动态函数

2. OpInfoManager::GetInstance().GetControlBuffer()
   └─→ 获取控制流二进制

3. AicpuTaskManager
   └─→ 提交AICPU任务

4. LoggerManager::GetInstance().Log()
   └─→ 输出日志
```

---

## 最佳实践

### 1. 单例使用

**正确做法**：
```cpp
// 获取单例
auto &program = Program::GetInstance();
auto *func = program.GetCurrentFunction();

// 配置管理
auto &config = ConfigManagerNg::GetInstance();
config.SetConfig("key", value);
```

**错误做法**：
```cpp
// 不要缓存单例指针（可能导致悬空指针）
Program *program = &Program::GetInstance();  // 不推荐

// 不要直接访问私有成员
program->functionmap_;  // 错误：私有成员
```

### 2. Registry注册

**正确做法**：
```cpp
// 使用自动注册宏
REGISTER_PASS(PassName::MyPass, MyPass);

// 在静态初始化时注册
static bool registered = []() {
    PassRegistry::GetInstance().RegisterPass(...);
    return true;
}();
```

**错误做法**：
```cpp
// 不要在运行时动态注册（可能导致竞态条件）
void SomeFunction() {
    PassRegistry::GetInstance().RegisterPass(...);  // 不推荐
}
```

### 3. Factory使用

**正确做法**：
```cpp
// 根据上下文创建对象
CodeGenCtx ctx;
auto codegen = CodeGenFactory::GetCodeGenCCE(ctx);
codegen->GenCode(function);
```

**错误做法**：
```cpp
// 不要直接实例化（应该通过Factory）
auto codegen = std::make_shared<CodeGenCCE>(ctx);  // 不推荐
```

### 4. 配置管理

**正确做法**：
```cpp
// 使用作用域管理配置
auto scope = ConfigManagerNg::GetInstance().CurrentScope();
scope->SetConfig("key", value);
// 作用域结束时自动恢复
```

**错误做法**：
```cpp
// 不要直接修改全局配置（应该使用作用域）
ConfigManagerNg::GetInstance().SetConfig("key", value);  // 可能影响其他代码
```

---

## 总结

PyPTO框架的管理机制通过以下方式协同工作：

1. **Manager机制**：提供全局资源管理，使用单例模式确保唯一性
2. **Registry机制**：支持动态注册和查找，实现插件化架构
3. **Factory机制**：根据上下文创建对象，实现灵活的创建策略

这些机制共同构成了PyPTO框架的核心基础设施，为编译和执行流程提供了统一的管理接口。

**关键要点**：
- 所有Manager都是单例，通过 `GetInstance()` 或 `Instance()` 访问
- Registry支持动态注册，通常在静态初始化时注册
- Factory根据上下文创建对象，提供统一的创建接口
- 配置管理支持作用域，避免配置冲突

**相关文档**：
- [关键机制列表](01-key-mechanisms-list.md)
- [Interface模块](../02-core/04-interface.md)
- [Passes模块](../02-core/09-passes.md)
- [Codegen模块](../02-core/10-codegen.md)

