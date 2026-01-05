# PyPTO 关键变量和结构详解

> **适用对象：** 想要深入理解内部实现的开发者、调试专家  
> **学习时间：** 40-60分钟  
> **前置知识：** 已阅读核心模块文档  
> **学习目标：** 理解关键数据结构、环境变量和配置参数

## 概述

本文档详细解释 PyPTO 工程中的关键变量、数据结构、配置参数和核心类。理解这些关键元素有助于深入掌握 PyPTO 的内部工作机制。

**文档内容：**
- 📊 **数据结构**：Function、Operation、LogicalTensor等核心结构
- ⚙️ **环境变量**：GLOBAL_LOG_LEVEL、ASCEND_HOME_PATH等
- 🔧 **配置参数**：Pass配置、运行时选项
- 🎯 **核心类**：关键类的成员变量和方法

**相关文档：**
- [架构设计总结](11-architecture-design.md) - 理解架构设计理念
- [模块关系图](13-module-relationships.md) - 理解模块间关系
- [API 使用总结](12-api-reference.md) - 了解 API 使用方式

---

## 目录

- [核心数据结构](#核心数据结构)
- [配置系统](#配置系统)
- [编译相关结构](#编译相关结构)
- [运行时结构](#运行时结构)
- [关键变量](#关键变量)
- [类型定义](#类型定义)
- [内存管理](#内存管理)
- [调试结构](#调试结构)

---

## 核心数据结构

### 1. Function 类

**位置（以源码为准）**: `framework/src/interface/function/function.h`

**作用**: PyPTO 的核心数据结构，表示一个完整的计算函数，包含所有操作和张量信息。

**Python 侧对应封装：**
- `python/pypto/functions.py::Function`（对 `pypto_impl` 的薄封装，提供 `Dump/DumpSSA/DumpJsonFile` 等调试接口）

```cpp
class Function {
public:
    // 核心属性
    std::string name_;                    // 函数名称
    GraphType graph_type_;               // 图类型 (TENSOR_GRAPH, TILE_GRAPH, etc.)
    std::vector<std::unique_ptr<Operation>> operations_;  // 操作序列
    std::map<std::string, std::unique_ptr<LogicalTensor>> tensorMap_;  // 张量映射

    // 输入输出
    std::vector<std::unique_ptr<LogicalTensor>> inCasts_, outCasts_;  // 输入输出张量

    // 编译信息
    std::unique_ptr<CompileInfo> compile_info_;  // 编译结果信息

    // 方法
    Status AddOperation(std::unique_ptr<Operation> op);  // 添加操作
    const std::vector<std::unique_ptr<Operation>>& GetOperations() const;  // 获取操作列表
    LogicalTensor* AddInput(const Shape& shape, DataType dtype);  // 添加输入
    LogicalTensor* AddOutput(const Shape& shape, DataType dtype);  // 添加输出
    Status SortOperations();  // 拓扑排序操作
    uint64_t ComputeHash() const;  // 计算函数哈希
};
```

**关键属性说明**:
- **`operations_`**: 存储函数中的所有操作，按拓扑序排列
- **`tensorMap_`**: 名称到张量的映射，管理所有中间张量
- **`graph_type_`**: 当前图的抽象层次 (Tensor → Tile → Block → Execute)
- **`inCasts_/outCasts_`**: 函数的输入输出接口定义

### 2. Operation 类

**位置**: `framework/include/pypto/operation.h`

**作用**: 表示计算图中的一个操作节点，包含操作类型、输入输出和属性信息。

```cpp
class Operation {
public:
    // 核心属性
    Opcode opcode_;                    // 操作码 (ADD, MUL, MATMUL, etc.)
    std::vector<Operand> iOperands_;   // 输入操作数
    std::vector<Operand> oOperands_;   // 输出操作数
    OpAttribute opAttribute_;          // 操作属性

    // 构造和析构
    Operation(Opcode opcode, const std::vector<Operand>& inputs,
              const std::vector<Operand>& outputs, OpAttribute attr = {});

    // 操作数管理
    Status ReplaceInputOperand(size_t index, Operand newOperand);
    const std::vector<Operand>& GetInputOperands() const;
    const std::vector<Operand>& GetOutputOperands() const;

    // 属性访问
    OpAttribute GetAttribute() const;
    void SetAttribute(OpAttribute attr);
};
```

**关键属性说明**:
- **`opcode_`**: 定义操作类型 (数学运算、内存操作、控制流等)
- **`iOperands_/oOperands_`**: 操作的输入输出操作数
- **`opAttribute_`**: 操作特定的属性 (如数据类型、计算精度等)

### 3. LogicalTensor 类

**位置**: `framework/include/pypto/logical_tensor.h`

**作用**: 表示函数中的逻辑张量，管理张量的形状、类型和生命周期信息。

```cpp
class LogicalTensor {
public:
    // 核心属性
    uint64_t magic_;              // 唯一标识符
    Shape shape_;                 // 张量形状
    DataType dtype_;              // 数据类型
    std::string name_;            // 张量名称

    // 内存信息
    bool isParameter_;            // 是否为参数张量
    MemoryType memoryType_;       // 内存类型 (HOST, DEVICE, SHARED)

    // 引用信息
    std::vector<Operation*> producers_;    // 生产者操作
    std::vector<Operation*> consumers_;    // 消费者操作

    // 构造方法
    LogicalTensor(const Shape& shape, DataType dtype, const std::string& name = "");

    // 属性访问
    const Shape& GetShape() const { return shape_; }
    DataType GetDataType() const { return dtype_; }
    const std::string& GetName() const { return name_; }
    uint64_t GetMagic() const { return magic_; }

    // 形状操作
    Status Reshape(const Shape& newShape);
    Status SetDynamicShape(bool isDynamic);

    // 内存操作
    Status SetMemoryType(MemoryType type);
    MemoryType GetMemoryType() const;
};
```

**关键属性说明**:
- **`magic_`**: 张量的唯一标识符，用于符号管理和代码生成
- **`shape_`**: 张量的维度信息，支持动态形状
- **`producers_/consumers_`**: 跟踪张量的数据流依赖关系

### 4. Operand 结构体

**作用**: 表示操作的操作数，连接操作和张量。

```cpp
struct Operand {
    LogicalTensor* tensor_;      // 引用的张量
    bool isOutput_;              // 是否为输出操作数
    uint32_t index_;             // 在操作中的索引

    // 构造方法
    Operand(LogicalTensor* tensor, bool isOutput, uint32_t index = 0);

    // 比较操作符 (用于排序和查找)
    bool operator==(const Operand& other) const;
    bool operator<(const Operand& other) const;
};
```

**设计意义**:
- 将操作和张量解耦，便于图变换和优化
- 支持同一张量被多个操作使用或产生

---

## 配置系统

### 1. 配置管理器 (_CachedOptions)

**位置**: `python/pypto/config.py`

**作用**: Python 端的配置缓存和管理器，提供统一的配置接口。

```python
class _CachedOptions:
    def __init__(self):
        self._options = pypto_impl.GetOptions()  # 从 C++ 获取配置

    def reset(self):
        """重置配置缓存"""
        self._options = pypto_impl.GetOptions()

    def set_options(self, prefix, options):
        """设置带前缀的配置选项"""
        for name, value in options.items():
            key = f"{prefix}.{name}"
            if key in self._options and value is not None:
                pypto_impl.SetOption(key, value)
                self._options[key] = value

    def __getitem__(self, key):
        """获取配置值"""
        return self._options[key]

    def __setitem__(self, key, value):
        """设置配置值"""
        self._options[key] = value
        pypto_impl.SetOption(key, value)
```

**全局实例**:
```python
_pto_options = _CachedOptions()
```

### 2. 配置选项类 (_Options)

**作用**: 提供上下文管理和装饰器模式的配置接口。

```python
class _Options:
    """Configuration options class, supports context manager and decorator modes"""

    INIT_FIELDS = [
        "name", "codegen_options", "host_options", "pass_options",
        "runtime_options", "verify_options", "debug_options",
        "vec_tile_shapes", "cube_tile_shapes", "matrix_size"
    ]

    PREFIX_MAP = {
        "codegen_options": "codegen.",
        "host_options": "host.",
        "pass_options": "pass.",
        "runtime_options": "runtime.",
        "verify_options": "verify.",
        "debug_options": "debug.",
    }

    def __init__(self, **kwargs):
        # 初始化各个配置域
        for field in self.INIT_FIELDS:
            setattr(self, field, kwargs.get(field, {}))

    def __enter__(self):
        """上下文管理器入口"""
        self._old_options = {}
        for field in self.INIT_FIELDS:
            options = getattr(self, field)
            if options:
                prefix = self.PREFIX_MAP.get(field, "")
                self._old_options[field] = _pto_options.get_options(prefix)
                _pto_options.set_options(prefix, options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        for field, old_options in self._old_options.items():
            prefix = self.PREFIX_MAP.get(field, "")
            _pto_options.set_options(prefix, old_options)
```

### 3. Tile 形状配置

#### VecTile 配置

```python
class VecTile:
    """Vector 计算的 Tile 形状配置"""
    def __init__(self, height=64, width=512):
        self.height = height    # 每次处理的行数
        self.width = width      # 每次处理的列数
```

#### CubeTile 配置

```python
class CubeTile:
    """Cube 计算的 Tile 形状配置"""
    def __init__(self, m, k, n, set_l1_tile=False):
        """
        CubeTile for matmul operation
        m[0], k[0], n[0] for L0 Cache
        m[1], k[1], n[1] for L1 Cache
        """
        self.m = m  # M 维度分块
        self.k = k  # K 维度分块
        self.n = n  # N 维度分块
        self.set_l1_tile = set_l1_tile
```

**配置方法**:
```python
# 设置 Vector Tile
pypto.set_vec_tile_shapes(height=64, width=512)

# 设置 Cube Tile
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
```

---

## 编译相关结构

### 1. CompileInfo 结构体

**作用**: 存储编译结果信息，传递给运行时系统。

```cpp
struct CompileInfo {
    // 工作空间信息
    uint64_t workspaceSize_;           // 工作空间大小
    std::vector<InvokeParaOffset> paramOffsets_;  // 参数偏移

    // 函数信息
    std::vector<Function*> functions_; // 编译的函数列表

    // 符号信息
    std::map<uint64_t, std::string> symbolTable_;  // 符号表

    // 调试信息
    std::string sourceCode_;           // 生成的源代码
    std::vector<std::string> warnings_; // 编译警告
};
```

### 2. SymbolicExpressionTable

**作用**: 管理符号表达式的表格，用于运行时符号求值。

```cpp
class SymbolicExpressionTable {
public:
    // 表达式存储
    std::vector<uint64_t*> expressions_;  // 表达式指针数组
    std::map<std::string, uint64_t> symbolMap_;  // 符号映射

    // 方法
    Status AddExpression(uint64_t* expr);
    uint64_t* GetExpression(size_t index) const;
    Status SetSymbolValue(const std::string& name, uint64_t value);
    uint64_t GetSymbolValue(const std::string& name) const;
};
```

### 3. PassManager 类

**位置**: `framework/src/passes/pass_mgr/pass_manager.h`

**作用**: 管理所有编译优化 Pass 的执行。

```cpp
class PassManager {
public:
    // Pass 注册
    Status RegisterPass(std::unique_ptr<Pass> pass);

    // Pass 执行
    Status RunPass(Function *func, PassType type);
    Status RunPassChain(Function *func, const std::vector<PassType>& types);

    // Pass 查询
    std::vector<Pass*> GetPasses(PassType type) const;
    bool HasPass(PassType type, const std::string& name) const;

private:
    // Pass 存储: 类型 -> Pass 列表
    std::map<PassType, std::vector<std::unique_ptr<Pass>>> passes_;
};
```

---

## 运行时结构

### 1. DeviceAgentTask 结构体

**作用**: 封装设备执行任务的所有必要信息。

```cpp
struct DeviceAgentTask {
    // 编译信息
    MachineCompileInfo compileInfo_;    // 编译时计算的信息

    // 数据缓冲区
    std::vector<void*> inputBuffers_;   // 输入数据缓冲区
    std::vector<void*> outputBuffers_;  // 输出数据缓冲区
    std::vector<size_t> bufferSizes_;   // 缓冲区大小

    // 执行参数
    std::vector<RuntimeArg> runtimeArgs_;  // 运行时参数

    // 回调函数
    TaskCompletionCallback callback_;    // 任务完成回调

    // 调试信息
    std::string taskName_;              // 任务名称
    uint64_t taskId_;                   // 任务 ID
};
```

### 2. RuntimeAgent 类

**作用**: 管理运行时的资源分配和任务执行。

```cpp
class RuntimeAgent {
public:
    // 内存管理
    Status AllocateMemory(size_t size, MemoryType type, void** ptr);
    Status DeallocateMemory(void* ptr);

    // 数据传输
    Status CopyToDevice(void* dst, const void* src, size_t size);
    Status CopyFromDevice(void* dst, const void* src, size_t size);

    // 任务执行
    Status ExecuteTask(DeviceAgentTask* task);
    Status WaitForTask(DeviceAgentTask* task);

    // 流管理
    Status CreateStream(Stream** stream);
    Status DestroyStream(Stream* stream);
    Status SynchronizeStream(Stream* stream);

private:
    // 资源池
    std::vector<void*> memoryPool_;     // 内存池
    std::vector<Stream*> streamPool_;   // 流池
    std::mutex resourceMutex_;          // 资源锁
};
```

### 3. MachineAgent 类

**位置**: `framework/src/machine/runtime/machine_agent.h`

**作用**: 设备任务的代理，负责任务准备和执行。

```cpp
class MachineAgent {
public:
    // 任务处理
    Status AgentProc(DeviceAgentTask *task);

    // 准备阶段
    Status PrepareWorkSpace(DeviceAgentTask *task);
    Status PrepareInvokeEntry(DeviceAgentTask *task);

    // 执行阶段
    Status ExecuteOnDevice(DeviceAgentTask *task);
    Status HandleExecutionResult(DeviceAgentTask *task);

private:
    // 依赖组件
    std::unique_ptr<RuntimeAgent> runtimeAgent_;
    std::unique_ptr<DeviceMachine> deviceMachine_;
};
```

---

## 关键变量

### 1. 全局配置变量

#### Python 端全局变量

```python
# python/pypto/config.py
_pto_options = _CachedOptions()  # 全局配置缓存

# 默认配置值
DEFAULT_VEC_TILE_HEIGHT = 64
DEFAULT_VEC_TILE_WIDTH = 512
DEFAULT_CUBE_TILE_M = [128, 128]
DEFAULT_CUBE_TILE_K = [128, 128]
DEFAULT_CUBE_TILE_N = [128, 128]
```

#### C++ 端全局变量

```cpp
// framework/src/interface/configs/config_manager.cpp
static ConfigManager* g_config_manager = nullptr;  // 全局配置管理器

// 编译选项
static bool g_enable_debug = false;
static bool g_enable_profiling = false;
static int g_optimization_level = 2;
```

### 2. 常量定义

#### 数据类型常量

```cpp
// framework/include/pypto/data_types.h
enum class DataType {
    UNKNOWN = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    INT32 = 3,
    INT16 = 4,
    INT8 = 5,
    UINT8 = 6,
    BOOL = 7
};

// 数据类型大小映射
static const std::map<DataType, size_t> DATA_TYPE_SIZES = {
    {DataType::FLOAT32, 4},
    {DataType::FLOAT16, 2},
    {DataType::INT32, 4},
    {DataType::INT16, 2},
    {DataType::INT8, 1},
    {DataType::UINT8, 1},
    {DataType::BOOL, 1}
};
```

#### 操作码定义

```cpp
// framework/include/pypto/opcodes.h
enum class Opcode {
    // 数学运算
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    MATMUL = 5,

    // 激活函数
    RELU = 100,
    SIGMOID = 101,
    TANH = 102,
    SOFTMAX = 103,

    // 内存操作
    LOAD = 200,
    STORE = 201,
    COPY = 202,

    // 控制流
    LOOP = 300,
    CONDITIONAL = 301,
    CALL = 302
};
```

### 3. 环境变量

#### 编译时环境变量

```bash
# 构建配置
CMAKE_BUILD_TYPE=Release          # 构建类型
BUILD_WITH_CANN=ON                 # 启用 CANN 支持
BUILD_PYTHON=ON                    # 构建 Python 绑定
BUILD_TESTS=ON                     # 构建测试

# 路径配置
ASCEND_HOME_PATH="/usr/local/Ascend"  # Ascend 工具链路径
LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib:$LD_LIBRARY_PATH"

# 编译器选项
CXX=g++-9                          # C++ 编译器
CC=gcc-9                           # C 编译器
```

#### 运行时环境变量

```bash
# 日志配置
GLOBAL_LOG_LEVEL=0                 # 全局日志级别 (0=DEBUG, 3=ERROR)
ASCEND_GLOBAL_LOG_LEVEL=0          # Ascend 日志级别

# 设备配置
CUDA_VISIBLE_DEVICES=0,1           # GPU 设备可见性
ASCEND_VISIBLE_DEVICES=0           # Ascend 设备可见性

# 内存配置
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # PyTorch CUDA 内存配置
```

---

## 类型定义

### 1. 基础类型定义

```cpp
// framework/include/pypto/types.h
using Shape = std::vector<int64_t>;          // 张量形状类型
using Magic = uint64_t;                      // 张量魔数类型
using DeviceId = int;                        // 设备 ID 类型
using StreamId = uint64_t;                   // 流 ID 类型

// 函数指针类型
using KernelFunction = void(*)(void**);      // 内核函数指针
using TaskCallback = void(*)(void*);         // 任务回调函数
using MemoryAllocator = void*(*)(size_t);    // 内存分配器函数
```

### 2. 枚举类型定义

#### GraphType 枚举

```cpp
enum class GraphType {
    TENSOR_GRAPH = 0,     // 张量级图 (最高抽象)
    TILE_GRAPH = 1,       // Tile 级图 (硬件感知)
    BLOCK_GRAPH = 2,      // 块级图 (并行执行)
    EXECUTE_GRAPH = 3     // 执行级图 (最终调度)
};
```

#### MemoryType 枚举

```cpp
enum class MemoryType {
    HOST = 0,             // 主机内存
    DEVICE = 1,           // 设备内存
    SHARED = 2,           // 共享内存
    PINNED = 3            // 页锁定内存
};
```

#### PassType 枚举

```cpp
enum class PassType {
    TENSOR_GRAPH_PASS = 0,    // 张量图优化
    TILE_GRAPH_PASS = 1,      // Tile 图优化
    BLOCK_GRAPH_PASS = 2,     // 块图优化
    EXECUTE_GRAPH_PASS = 3     // 执行图优化
};
```

---

## 内存管理

### 1. 内存分配器

```cpp
// framework/include/pypto/memory_allocator.h
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;

    // 内存分配
    virtual void* Allocate(size_t size, MemoryType type) = 0;
    virtual void Deallocate(void* ptr) = 0;

    // 内存信息
    virtual size_t GetTotalMemory(MemoryType type) const = 0;
    virtual size_t GetFreeMemory(MemoryType type) const = 0;
    virtual size_t GetUsedMemory(MemoryType type) const = 0;

    // 内存对齐
    virtual size_t GetAlignment(MemoryType type) const = 0;

protected:
    // 对齐计算
    size_t AlignSize(size_t size, size_t alignment) const {
        return (size + alignment - 1) & ~(alignment - 1);
    }
};
```

### 2. 张量内存布局

```cpp
// framework/include/pypto/tensor_layout.h
struct TensorLayout {
    // 基本信息
    Shape shape_;                    // 张量形状
    DataType dtype_;                 // 数据类型
    size_t element_size_;            // 元素大小

    // 内存布局
    size_t total_size_;              // 总字节数
    std::vector<size_t> strides_;    // 各维度步长
    size_t alignment_;               // 内存对齐要求

    // 计算方法
    Status ComputeStrides();         // 计算步长
    Status ComputeTotalSize();       // 计算总大小
    size_t GetOffset(const std::vector<size_t>& indices) const;  // 计算偏移
};
```

### 3. 内存池管理

```cpp
// framework/include/pypto/memory_pool.h
class MemoryPool {
public:
    // 池管理
    Status Initialize(size_t pool_size, MemoryType type);
    Status Finalize();

    // 内存操作
    void* Allocate(size_t size);
    void Deallocate(void* ptr);

    // 池信息
    size_t GetPoolSize() const { return pool_size_; }
    size_t GetUsedSize() const { return used_size_; }
    size_t GetFreeSize() const { return pool_size_ - used_size_; }

    // 碎片整理
    Status Defragment();

private:
    MemoryType type_;                // 内存类型
    size_t pool_size_;               // 池总大小
    size_t used_size_;               // 已使用大小
    std::vector<MemoryBlock> free_blocks_;     // 空闲块列表
    std::unordered_map<void*, MemoryBlock> allocations_;  // 分配映射
};
```

---

## 调试结构

### 1. 调试信息结构体

```cpp
// framework/include/pypto/debug_info.h
struct DebugInfo {
    // 基本信息
    std::string function_name_;       // 函数名称
    std::string file_name_;           // 文件名
    int line_number_;                 // 行号

    // 执行信息
    uint64_t execution_id_;           // 执行 ID
    std::chrono::time_point<std::chrono::steady_clock> start_time_;  // 开始时间
    std::chrono::time_point<std::chrono::steady_clock> end_time_;    // 结束时间

    // 性能指标
    size_t memory_used_;              // 内存使用
    size_t instructions_executed_;    // 执行指令数
    double execution_time_ms_;        // 执行时间 (毫秒)

    // 错误信息
    Status status_;                   // 执行状态
    std::string error_message_;       // 错误信息

    // 调试数据
    std::map<std::string, std::string> debug_data_;  // 调试键值对
};
```

### 2. 日志系统结构

```cpp
// framework/include/pypto/logger.h
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

class Logger {
public:
    static Logger& Instance();

    // 日志输出
    void Log(LogLevel level, const std::string& message);
    void LogF(LogLevel level, const char* format, ...);

    // 配置
    void SetLogLevel(LogLevel level);
    void SetLogFile(const std::string& filename);
    void EnableConsoleOutput(bool enable);

private:
    LogLevel current_level_;
    std::ofstream log_file_;
    bool console_output_;
    std::mutex log_mutex_;
};
```

### 3. 性能分析

性能分析使用系统工具（perf、gprof 等），PyPTO 不提供内置的性能分析器。

---

## 总结

PyPTO 的关键变量和结构构成了框架的核心抽象层：

### 数据流层次

1. **Function**: 顶层函数抽象，包含完整的计算图
2. **Operation**: 计算图节点，定义具体计算操作
3. **LogicalTensor**: 张量抽象，管理数据和依赖关系
4. **Operand**: 操作数连接，解耦操作和张量

### 配置管理层次

1. **_CachedOptions**: Python 配置缓存
2. **_Options**: 上下文配置管理
3. **Tile 配置**: 硬件优化参数
4. **Pass 配置**: 编译优化选项

### 编译执行层次

1. **PassManager**: 优化 Pass 管理
2. **CompileInfo**: 编译结果封装
3. **DeviceAgentTask**: 执行任务封装
4. **RuntimeAgent**: 运行时资源管理

### 关键设计理念

- **分层抽象**: 从高层 Tensor 到低层硬件的渐进式抽象
- **插件化架构**: Pass、操作符的动态注册和管理
- **配置驱动**: 统一的配置管理系统
- **资源管理**: 智能的内存和设备资源管理

理解这些关键变量和结构是掌握 PyPTO 内部机制的基础，对于开发高级功能、调试复杂问题和优化性能都至关重要。

---

## 检索索引（结构体/变量名→所在模块→用途）

| 结构体/变量名 | 所在模块 | 用途 | 文档位置 |
|-------------|---------|------|---------|
| `Function` | `framework/src/interface/function/` | 函数级 IR 核心抽象 | [核心数据结构](#1-function-类) |
| `Operation` | `framework/include/pypto/operation.h` | 计算图操作节点 | [核心数据结构](#2-operation-类) |
| `LogicalTensor` | `framework/src/interface/tensor/` | 逻辑张量表示 | [核心数据结构](#3-logicaltensor-类) |
| `Shape` | `framework/src/interface/tensor/` | 张量形状 | [类型定义](#类型定义) |
| `DataType` | `framework/src/interface/tensor/` | 数据类型枚举 | [类型定义](#类型定义) |
| `MachineCompileInfo` | `framework/src/machine/host/` | 机器编译信息 | [编译相关结构](#编译相关结构) |
| `RuntimeAgent` | `framework/src/machine/runtime/` | 运行时代理 | [运行时结构](#运行时结构) |
| `GLOBAL_LOG_LEVEL` | 环境变量 | 全局日志级别 | [关键变量](#关键变量) |
| `ASCEND_HOME_PATH` | 环境变量 | CANN 安装路径 | [关键变量](#关键变量) |
| `TILE_FWK_DEVICE_ID` | 环境变量 | 设备 ID | [关键变量](#关键变量) |

**注意：** 以上索引仅列出部分关键结构体和变量，完整列表请参考各章节详细说明。

---

**相关文档：**
- [架构设计总结](11-architecture-design.md) - 理解整体架构设计
- [模块关系图](13-module-relationships.md) - 理解模块协作关系
- [API 使用总结](12-api-reference.md) - 了解外部使用接口

