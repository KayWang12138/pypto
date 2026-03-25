# 精度工具支持通信算子实现计划（多进程版本）

## 1. 需求分析

### 1.1 目标
为PyPTO精度工具添加对通信类算子的支持，使用**真正的多进程间通信**，使精度工具能够验证包含通信算子的模型。

### 1.2 用户选择的方案
- **启动方式**：用户手动启动多进程（使用mpirun或类似工具）
- **同步机制**：共享内存 + 原子操作
- **结果收集**：每个rank独立验证
- **Golden数据**：每个rank独立golden

### 1.3 支持的通信算子
- `OP_SHMEM_PUT` - 发送数据到远程共享内存
- `OP_SHMEM_GET` - 从远程共享内存获取数据
- `OP_SHMEM_PUT_UB2GM` - UB直接到远程共享内存
- `OP_SHMEM_GET_GM2UB` - 远程共享内存到UB
- `OP_SHMEM_SIGNAL` - 发送信号
- `OP_SHMEM_WAIT_UNTIL` - 等待信号
- `OP_SHMEM_SET` - 清零共享内存

### 1.4 关键挑战
1. **真正的跨进程共享内存**：需要使用POSIX共享内存API（shm_open, mmap）
2. **跨进程原子操作**：需要使用std::atomic或futex实现
3. **进程间同步**：实现signal的跨进程通知机制
4. **共享内存命名**：需要唯一标识共享内存区域
5. **资源清理**：确保进程异常退出时正确清理共享内存

---

## 2. 架构设计

### 2.1 整体架构
```
┌─────────────────────────────────────────────────────────────┐
│                    Rank 0 进程                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│  │ Interpreter  │───▶│ ShmemManager │───▶│ SharedMemory ││
│  │             │    │             │    │   (POSIX)    ││
│  └──────────────┘    └──────────────┘    └──────────────┘│
└─────────────────────────────────────────────────────────────┘
                           │
                           │ 跨进程共享内存
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Rank 1 进程                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│  │ Interpreter  │───▶│ ShmemManager │───▶│ SharedMemory ││
│  │             │    │             │    │   (POSIX)    ││
│  └──────────────┘    └──────────────┘    └──────────────┘│
└─────────────────────────────────────────────────────────────┘
                           │
                           │ ...
                           │
┌─────────────────────────────────────────────────────────────┐
│                    Rank N 进程                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│  │ Interpreter  │───▶│ ShmemManager │───▶│ SharedMemory ││
│  │             │    │    (本地)    │    │   (POSIX)    ││
│  └──────────────┘    └──────────────┘    └──────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

#### 2.2.1 ShmemManager（共享内存管理器）
**职责**：
- 创建和销毁跨进程共享内存区域
- 管理多个通信组的共享内存
- 提供跨进程原子操作接口
- 提供跨进程同步机制

**接口设计**：
```cpp
class ShmemManager {
public:
    struct ShmemRegion {
        void* dataPtr;           // 数据区域指针
        void* signalPtr;         // 信号区域指针
        size_t dataSize;         // 数据区域大小
        size_t signalSize;       // 信号区域大小
        std::string shmName;     // 共享内存名称
        int dataShmFd;          // 数据区域文件描述符
        int signalShmFd;        // 信号区域文件描述符
        int worldSize;           // 进程数
        int creatorRank;         // 创建者rank ID
    };

    // 获取单例
    static ShmemManager* GetInstance();

    // 创建共享内存区域（所有进程调用）
    ShmemRegion* CreateRegion(const std::string& groupName, int worldSize,
                             int rankId, DataType dataType,
                             const std::vector<int64_t>& shape);

    // 销毁共享内存区域（仅创建者调用）
    void DestroyRegion(const std::string& groupName);

    // 获取指定rank的数据指针
    void* GetDataPtr(const std::string& groupName, int rankId,
                     const std::vector<int64_t>& offset);

    // 获取指定rank的信号指针
    void* GetSignalPtr(const std::string& groupName, int srcRank,
                       int dstRank, const std::vector<int64_t>& offset);

    // 跨进程原子操作：SET
    void AtomicSet(void* addr, int32_t value);

    // 跨进程原子操作：ADD
    void AtomicAdd(void* addr, int32_t value);

    // 跨进程等待信号满足条件
    void WaitUntil(void* addr, int32_t expectedValue, bool resetSignal);

    // 清理所有共享内存（异常退出时调用）
    void CleanupAll();

private:
    std::map<std::string, ShmemRegion> regions_;
    std::mutex mutex_;
};
```

#### 2.2.2 RankInfo（进程信息管理）
**职责**：
- 获取当前进程的rank ID
- 获取world size
- 提供进程间协调接口

**接口设计**：
```cpp
class RankInfo {
public:
    // 获取单例
    static RankInfo* GetInstance();

    // 初始化（从环境变量或torch.distributed获取）
    void Initialize();

    // 获取当前rank ID
    int GetRankId() const { return rankId_; }

    // 获取world size
    int GetWorldSize() const { return worldSize_; }

    // 生成唯一的共享内存名称
    std::string GenerateShmemName(const std::string& groupName) const;

private:
    int rankId_ = -1;
    int worldSize_ = 1;
    std::string jobId_;  // 用于生成唯一的共享内存名称
};
```

#### 2.2.3 精度工具算子实现
**新增文件**：`framework/src/interface/interpreter/calc_distributed.cpp`

**实现函数**：
```cpp
// OP_SHMEM_PUT
void ExecuteOpShmemPut(ExecuteOperationContext* ctx);

// OP_SHMEM_GET
void ExecuteOpShmemGet(ExecuteOperationContext* ctx);

// OP_SHMEM_PUT_UB2GM
void ExecuteOpShmemPutUb2Gm(ExecuteOperationContext* ctx);

// OP_SHMEM_GET_GM2UB
void ExecuteOpShmemGetGm2Ub(ExecuteOperationContext* ctx);

// OP_SHMEM_SIGNAL
void ExecuteOpShmemSignal(ExecuteOperationContext* ctx);

// OP_SHMEM_WAIT_UNTIL
void ExecuteOpShmemWaitUntil(ExecuteOperationContext* ctx);

// OP_SHMEM_SET
void ExecuteOpShmemSet(ExecuteOperationContext* ctx);
```

---

## 3. 详细实现步骤

### 阶段一：基础设施搭建

#### 步骤1.1：实现RankInfo
**文件**：`framework/src/interface/interpreter/rank_info.h`（新建）
**文件**：`framework/src/interface/interpreter/rank_info.cpp`（新建）

**实现内容**：
```cpp
class RankInfo {
public:
    static RankInfo* GetInstance() {
        static RankInfo instance;
        return &instance;
    }

    void Initialize() {
        // 1. 尝试从环境变量获取
        if (const char* rankEnv = std::getenv("OMPI_COMM_WORLD_RANK")) {
            rankId_ = std::atoi(rankEnv);
        } else if (const char* rankEnv = std::getenv("PMI_RANK")) {
            rankId_ = std::atoi(rankEnv);
        }

        if (const char* sizeEnv = std::getenv("OMPI_COMM_WORLD_SIZE")) {
            worldSize_ = std::atoi(sizeEnv);
        } else if (const char* sizeEnv = std::getenv("PMI_SIZE")) {
            worldSize_ = std::atoi(sizeEnv);
        }

        // 2. 获取job ID（用于生成唯一的共享内存名称）
        if (const char* jobId = std::getenv("SLURM_JOB_ID")) {
            jobId_ = jobId;
        } else {
            jobId_ = "pypto_" + std::to_string(getpid());
        }
    }

    int GetRankId() const { return rankId_; }
    int GetWorldSize() const { return worldSize_; }

    std::string GenerateShmemName(const std::string& groupName) const {
        return "/pypto_shmem_" + jobId_ + "_" + groupName;
    }

private:
    RankInfo() = default;
    int rankId_ = -1;
    int worldSize_ = 1;
    std::string jobId_;
};
```

#### 步骤1.2：实现ShmemManager
**文件**：`framework/src/interface/interpreter/shmem_manager.h`（新建）
**文件**：`framework/src/interface/interpreter/shmem_manager.cpp`（新建）

**实现内容**：
```cpp
class ShmemManager {
public:
    struct ShmemRegion {
        void* dataPtr;
        void* signalPtr;
        size_t dataSize;
        size_t signalSize;
        std::string dataShmName;
        std::string signalShmName;
        int dataShmFd;
        int signalShmFd;
        int worldSize;
        int creatorRank;
        bool isCreator;
    };

    static ShmemManager* GetInstance() {
        static ShmemManager instance;
        return &instance;
    }

    ShmemRegion* CreateRegion(const std::string& groupName, int worldSize,
                             int rankId, DataType dataType,
                             const std::vector<int64_t>& shape) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 检查是否已存在
        if (regions_.find(groupName) != regions_.end()) {
            return &regions_[groupName];
        }

        ShmemRegion region;
        region.worldSize = worldSize;
        region.creatorRank = rankId;
        region.isCreator = (rankId == 0);  // rank 0 创建共享内存

        // 计算数据区域大小
        size_t elementSize = GetDataTypeSize(dataType);
        size_t dataElementCount = worldSize;
        for (auto dim : shape) {
            dataElementCount *= dim;
        }
        region.dataSize = dataElementCount * elementSize;

        // 计算信号区域大小
        size_t signalElementCount = worldSize * worldSize;
        for (auto dim : shape) {
            signalElementCount *= dim;
        }
        region.signalSize = signalElementCount * sizeof(int32_t);

        // 生成共享内存名称
        RankInfo* rankInfo = RankInfo::GetInstance();
        region.dataShmName = rankInfo->GenerateShmemName(groupName + "_data");
        region.signalShmName = rankInfo->GenerateShmemName(groupName + "_signal");

        // 创建共享内存
        if (region.isCreator) {
            // rank 0 创建共享内存
            region.dataShmFd = shm_open(region.dataShmName.c_str(),
                                       O_CREAT | O_RDWR, 0666);
            ftruncate(region.dataShmFd, region.dataSize);

            region.signalShmFd = shm_open(region.signalShmName.c_str(),
                                         O_CREAT | O_RDWR, 0666);
            ftruncate(region.signalShmFd, region.signalSize);
        } else {
            // 其他进程打开共享内存
            region.dataShmFd = shm_open(region.dataShmName.c_str(), O_RDWR, 0);
            region.signalShmFd = shm_open(region.signalShmName.c_str(), O_RDWR, 0);
        }

        // 映射共享内存
        region.dataPtr = mmap(nullptr, region.dataSize,
                              PROT_READ | PROT_WRITE, MAP_SHARED,
                              region.dataShmFd, 0);
        region.signalPtr = mmap(nullptr, region.signalSize,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                region.signalShmFd, 0);

        // 初始化信号区域（仅创建者）
        if (region.isCreator) {
            memset(region.signalPtr, 0, region.signalSize);
        }

        // 同步：确保所有进程都完成了映射
        MPI_Barrier(MPI_COMM_WORLD);

        regions_[groupName] = region;
        return &regions_[groupName];
    }

    void DestroyRegion(const std::string& groupName) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = regions_.find(groupName);
        if (it == regions_.end()) {
            return;
        }

        ShmemRegion& region = it->second;

        // 仅创建者销毁共享内存
        if (region.isCreator) {
            munmap(region.dataPtr, region.dataSize);
            munmap(region.signalPtr, region.signalSize);
            close(region.dataShmFd);
            close(region.signalShmFd);
            shm_unlink(region.dataShmName.c_str());
            shm_unlink(region.signalShmName.c_str());
        }

        regions_.erase(it);
    }

    void* GetDataPtr(const std::string& groupName, int rankId,
                     const std::vector<int64_t>& offset) {
        auto it = regions_.find(groupName);
        ASSERT(ExecuteOperationScene::SHMEM_REGION_NOT_FOUND,
               it != regions_.end());

        ShmemRegion& region = it->second;

        // 计算偏移量
        size_t byteOffset = rankId * (region.dataSize / region.worldSize);
        for (size_t i = 0; i < offset.size(); ++i) {
            byteOffset += offset[i];
        }

        return static_cast<char*>(region.dataPtr) + byteOffset;
    }

    void* GetSignalPtr(const std::string& groupName, int srcRank, int dstRank,
                       const std::vector<int64_t>& offset) {
        auto it = regions_.find(groupName);
        ASSERT(ExecuteOperationScene::SHMEM_REGION_NOT_FOUND,
               it != regions_.end());

        ShmemRegion& region = it->second;

        // 计算偏移量（信号矩阵：[srcRank, dstRank, ...]）
        size_t byteOffset = (srcRank * region.worldSize + dstRank) *
                           (region.signalSize / (region.worldSize * region.worldSize));
        for (size_t i = 0; i < offset.size(); ++i) {
            byteOffset += offset[i];
        }

        return static_cast<char*>(region.signalPtr) + byteOffset;
    }

    void AtomicSet(void* addr, int32_t value) {
        std::atomic<int32_t>* atomicAddr = static_cast<std::atomic<int32_t>*>(addr);
        atomicAddr->store(value, std::memory_order_release);
    }

    void AtomicAdd(void* addr, int32_t value) {
        std::atomic<int32_t>* atomicAddr = static_cast<std::atomic<int32_t>*>(addr);
        atomicAddr->fetch_add(value, std::memory_order_acq_rel);
    }

    void WaitUntil(void* addr, int32_t expectedValue, bool resetSignal) {
        std::atomic<int32_t>* atomicAddr = static_cast<std::atomic<int32_t>*>(addr);

        // 轮询等待（可以优化为futex）
        while (atomicAddr->load(std::memory_order_acquire) != expectedValue) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        if (resetSignal) {
            atomicAddr->store(0, std::memory_order_release);
        }
    }

    void CleanupAll() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& [groupName, region] : regions_) {
            if (region.isCreator) {
                munmap(region.dataPtr, region.dataSize);
                munmap(region.signalPtr, region.signalSize);
                close(region.dataShmFd);
                close(region.signalShmFd);
                shm_unlink(region.dataShmName.c_str());
                shm_unlink(region.signalShmName.c_str());
            }
        }

        regions_.clear();
    }

private:
    ShmemManager() = default;
    ~ShmemManager() {
        CleanupAll();
    }

    std::map<std::string, ShmemRegion> regions_;
    std::mutex mutex_;
};
```

#### 步骤1.3：更新CMakeLists.txt
**文件**：`framework/src/interface/interpreter/CMakeLists.txt`

**修改内容**：
```cmake
# 添加新文件
list(APPEND INTERPRETER_SRCS
    ...
    rank_info.cpp
    shmem_manager.cpp
    calc_distributed.cpp
)
```

### 阶段二：通信算子实现

#### 步骤2.1：实现OP_SHMEM_PUT
**文件**：`framework/src/interface/interpreter/calc_distributed.cpp`（新建）

**实现逻辑**：
```cpp
void ExecuteOpShmemPut(ExecuteOperationContext* ctx) {
    // 1. 解析参数
    auto inputData = ctx->ioperandDataViewList->at(0);
    auto shmemData = ctx->ioperandDataViewList->at(1);
    auto outputData = ctx->ooperandInplaceDataViewList->at(0);

    // 2. 获取算子属性
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    int dstRankId = ctx->op->GetIntAttribute("dst_rank");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    std::vector<int64_t> shape = ctx->op->GetIntListAttribute("shape");
    AtomicType atomicType = ctx->op->GetEnumAttribute<AtomicType>("atomic_type");

    // 3. 获取共享内存管理器
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* dstAddr = shmemManager->GetDataPtr(groupName, dstRankId, offset);

    // 4. 执行数据拷贝
    void* srcAddr = inputData->GetData()->data();
    size_t byteSize = GetByteSize(inputData->GetDataType(), shape);

    if (atomicType == AtomicType::ADD) {
        // 原子加操作
        AtomicAdd(dstAddr, srcAddr, byteSize, inputData->GetDataType());
    } else {
        // 普通拷贝
        memcpy(dstAddr, srcAddr, byteSize);
    }

    // 5. 设置输出（pred token）
    if (outputData != nullptr) {
        memcpy(outputData->GetData()->data(), srcAddr, byteSize);
    }
}
```

#### 步骤2.2：实现OP_SHMEM_GET
**实现逻辑**：
```cpp
void ExecuteOpShmemGet(ExecuteOperationContext* ctx) {
    // 1. 解析参数
    auto outputData = ctx->ooperandInplaceDataViewList->at(0);
    auto shmemData = ctx->ioperandDataViewList->at(0);

    // 2. 获取算子属性
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    int srcRankId = ctx->op->GetIntAttribute("src_rank");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    std::vector<int64_t> shape = ctx->op->GetIntListAttribute("shape");
    AtomicType atomicType = ctx->op->GetEnumAttribute<AtomicType>("atomic_type");

    // 3. 获取共享内存管理器
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* srcAddr = shmemManager->GetDataPtr(groupName, srcRankId, offset);

    // 4. 执行数据拷贝
    void* dstAddr = outputData->GetData()->data();
    size_t byteSize = GetByteSize(outputData->GetDataType(), shape);

    if (atomicType == AtomicType::ADD) {
        // 原子加操作
        AtomicAdd(dstAddr, srcAddr, byteSize, outputData->GetDataType());
    } else {
        // 普通拷贝
        memcpy(dstAddr, srcAddr, byteSize);
    }
}
```

#### 步骤2.3：实现OP_SHMEM_SIGNAL
**实现逻辑**：
```cpp
void ExecuteOpShmemSignal(ExecuteOperationContext* ctx) {
    // 1. 解析参数
    auto shmemSignal = ctx->ioperandDataViewList->at(0);

    // 2. 获取算子属性
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    int dstRankId = ctx->op->GetIntListAttribute("dst_pe")[0];
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    int32_t signalValue = ctx->op->GetIntAttribute("signal");
    AtomicType atomicType = ctx->op->GetEnumAttribute<AtomicType>("sig_op");

    // 3. 获取共享内存管理器
    RankInfo* rankInfo = RankInfo::GetInstance();
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* signalAddr = shmemManager->GetSignalPtr(
        groupName, rankInfo->GetRankId(), dstRankId, offset);

    // 4. 执行信号操作
    if (atomicType == AtomicType::ADD) {
        shmemManager->AtomicAdd(signalAddr, signalValue);
    } else {
        shmemManager->AtomicSet(signalAddr, signalValue);
    }
}
```

#### 步骤2.4：实现OP_SHMEM_WAIT_UNTIL
**实现逻辑**：
```cpp
void ExecuteOpShmemWaitUntil(ExecuteOperationContext* ctx) {
    // 1. 解析参数
    auto shmemSignal = ctx->ioperandDataViewList->at(0);

    // 2. 获取算子属性
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    std::vector<int64_t> offset = ctx->op->GetIntListAttribute("offset");
    int32_t expectedValue = ctx->op->GetIntAttribute("cmp_value");
    bool resetSignal = ctx->op->GetBoolAttribute("clear_signal");

    // 3. 获取共享内存管理器
    RankInfo* rankInfo = RankInfo::GetInstance();
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* signalAddr = shmemManager->GetSignalPtr(
        groupName, rankInfo->GetRankId(), rankInfo->GetRankId(), offset);

    // 4. 执行等待操作
    shmemManager->WaitUntil(signalAddr, expectedValue, resetSignal);
}
```

#### 步骤2.5：实现OP_SHMEM_SET
**实现逻辑**：
```cpp
void ExecuteOpShmemSet(ExecuteOperationContext* ctx) {
    // 1. 解析参数
    auto shmemData = ctx->ioperandDataViewList->at(0);

    // 2. 获取算子属性
    std::string groupName = ctx->op->GetStringAttribute("group_name");
    std::vector<int64_t> offset = ctx->op->->GetIntListAttribute("offset");
    std::vector<int64_t> shape = ctx->op->GetIntListAttribute("shape");

    // 3. 获取共享内存管理器
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    void* dataAddr = shmemManager->GetDataPtr(groupName, 0, offset);

    // 4. 清零内存
    size_t byteSize = GetByteSize(shmemData->GetDataType(), shape);
    memset(dataAddr, 0, byteSize);
}
```

#### 步骤2.6：注册通信算子
**文件**：`framework/src/interface/interpreter/calc_distributed.cpp`

**添加注册代码**：
```cpp
REGISTER_CALC_OP(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ExecuteOpShmemPut);
REGISTER_CALC_OP(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ExecuteOpShmemGet);
REGISTER_CALC_OP(OP_SHMEM_PUT_UB2GM, Opcode::OP_SHMEM_PUT_UB2GM, ExecuteOpShmemPutUb2Gm);
REGISTER_CALC_OP(OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_GET_GM2UB, ExecuteOpShmemGetGm2Ub);
REGISTER_CALC_OP(OP_SHMEM_SIGNAL, Opcode::OP_SHMEM_SIGNAL, ExecuteOpShmemSignal);
REGISTER_CALC_OP(OP_SHMEM_WAIT_UNTIL, Opcode::OP_SHMEM_WAIT_UNTIL, ExecuteOpShmemWaitUntil);
REGISTER_CALC_OP(OP_SHMEM_SET, Opcode::OP_SHMEM_SET, ExecuteOpShmemSet);
```

### 阶段三：集成与测试

#### 步骤3.1：初始化RankInfo
**文件**：`framework/src/interface/interpreter/function.cpp`

**添加初始化逻辑**：
```cpp
void FunctionFrame::Initialize() {
    // ... 现有初始化代码 ...

    // 初始化RankInfo
    RankInfo::GetInstance()->Initialize();

    // 注册异常退出清理函数
    std::atexit([]() {
        ShmemManager::GetInstance()->CleanupAll();
    });
}
```

#### 步骤3.2：编写多进程测试用例
**文件**：`framework/tests/st/distributed/precision/test_allreduce_precision.cpp`（新建））

**测试场景**：
```cpp
TEST(AllReducePrecision, AllReduceAddRmsNorm) {
    // 1. 获取rank信息
    RankInfo* rankInfo = RankInfo::GetInstance();
    int rankId = rankInfo->GetRankId();
    int worldSize = rankInfo->GetWorldSize();

    // 2. 准备输入数据（每个rank有独立的golden）
    std::vector<float> inputData = GenerateInputData(rankId, worldSize);
    std::vector<float> goldenOutput = GenerateGoldenOutput(rankId, worldSize);

    // 3. 创建共享内存区域
    ShmemManager* shmemManager = ShmemManager::GetInstance();
    auto region = shmemManager->CreateRegion("allreduce", worldSize, rankId,
                                             DT_FP32, {64, 128});

    // 4. 执行AllReduce操作
    // Put数据到所有rank的共享内存
    for (int dstRank = 0; dstRank < worldSize; ++dstRank) {
        ExecuteOpShmemPut(/* 参数 */);
    }

    // 发送信号
    for (int dstRank = 0; dstRank < worldSize; ++dstRank) {
        ExecuteOpShmemSignal(/* 参数 */);
    }

    // 等待所有信号
    ExecuteOpShmemWaitUntil(/* 参数 */);

    // Get所有rank的数据并累加
    std::vector<float> result(64 * 128, 0.0f);
    for (int srcRank = 0; srcRank < worldSize; ++srcRank) {
        std::vector<float> temp(64 * 128);
        ExecuteOpShmemGet(/* 参数 */);
        // 累加到result
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += temp[i];
        }
    }

    // 5. 验证结果
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], goldenOutput[i]);
    }

    // 6. 清理（仅rank 0）
    if (rankId == 0) {
        shmemManager->DestroyRegion("allreduce");
    }
}
```

#### 步骤3.3：编写Python测试脚本
**文件**：`framework/tests/st/distributed/precision/run_allreduce_test.py`（新建）

**测试脚本**：
```python
#!/usr/bin/env python3
import subprocess
import sys

def run_allreduce_precision_test():
    world_size = 4

    # 使用mpirun启动多进程测试
    cmd = [
        "mpirun",
        "-n", str(world_size),
        "./tile_fwk_stest_distributed",
        "run",
        "--gtest_filter=AllReducePrecision.AllReduceAddRmsNorm",
        "--frontend=cpp"
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)

    if result.returncode != 0:
        print(f"\nTest failed with return code: {result.returncode}")
        sys.exit(1)
    else:
        print("\nTest passed!")
        sys.exit(0)

if __name__ == "__main__":
    run_allreduce_precision_test()
```

### 阶段四：文档与优化

#### 步骤4.1：编写设计文档
**文件**：`docs/design/precision_tool_communication_support.md`（新建）

**文档内容**：
1. 设计概述
2. 架构图
3. 核心组件说明
4. 多进程通信机制
5. 共享内存管理
6. 原子操作实现
7. 使用示例
8. 已知限制

#### 步骤4.2：更新用户文档
**文件**：`docs/tutorials/debug/precision.md`

**添加内容**：
```markdown
## 通信算子精度验证

精度工具现已支持通信算子的精度验证，使用**真正的多进程间通信**。

### 启动方式

使用mpirun或类似工具启动多进程：

```bash
# 4个进程测试
mpirun -n 4 python test_allreduce.py

# 或使用Python multiprocessing
python test_allreduce.py
```

### 基本用法

```python
@pypto.frontend.jit(verify_options=verify_options)
def allreduce_kernel(input_tensor, weight_tensor):
    # 创建共享内存
    shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor(
        group_name="tp",
        n_pes=8,
        dtype=pypto.DT_FP32,
        shape=[64, 128]
    )

    # 清空共享内存
    pypto.distributed.shmem_clear(shmem_data, [64, 128], [0, 0], is_signal=False)
    pypto.distributed.shmem_clear(shmem_signal, [8, 8, 64, 128], [0, 0, 0, 0], is_signal=True)

    # AllReduce操作
    for rank in range(8):
        put_out = pypto.distributed.shmem_put(
            input_tensor,
            [0, 0],
            shmem_data,
            rank,
            put_op=pypto.AtomicType.ADD
        )
        pypto.distributed.shmem_signal(
            shmem_signal,
            rank,
            1,
            [1, 1] + [64, 128],
            [rank, rank, 0, 0],
            sig_op=pypto.AtomicType.ADD,
            pred=[put_out]
        )

    # 等待所有rank完成
    wait_out = pypto.distributed.shmem_wait_until(
        shmem_signal,
        pypto.OpType.EQ,
        8,
        [1, 1] + [64, 128],
        [my_pe, my_pe, 0, 0],
        clear_signal=True
    )

    # 获取AllReduce结果
    result = pypto.experimental.shmem_load(shmem_data, my_pe, [64, 128], [0, 0])

    return result
```

### 注意事项

1. **多进程启动**：必须使用mpirun或类似工具启动多进程
2. **共享内存清理**：仅rank 0负责创建和销毁共享内存
3. **同步机制**：使用跨进程原子操作和轮询等待
4. **Golden数据**：每个rank有独立的golden输入和输出
5. **验证结果**：每个rank独立验证，输出各自的验证报告
```

#### 步骤4.3：性能优化
**优化方向**：
1. **信号等待优化**：使用futex代替轮询（Linux特有）
2. **数据拷贝优化**：对于大张量，使用多线程拷贝
3. **原子操作优化**：批量处理原子操作
4. **内存对齐优化**：确保共享内存区域正确对齐

---

## 4. 验证计划

### 4.1 单元测试
- [ ] RankInfo初始化和rank获取
- [ ] ShmemManager创建和销毁（多进程）
- [ ] ShmemPut/ShmemGet数据正确性（多进程）
- [ ] ShmemSignal/ShmemWaitUntil同步正确性（多进程）
- [ ] 原子操作（ADD）正确性（多进程）
- [ ] 多通信组管理（多进程）
- [ ] 边界条件测试

### 4.2 集成测试
- [ ] AllReduce算子精度验证（多进程）
- [ ] AllGather算子精度验证（多进程）
- [ ] ReduceScatter算子精度验证（多进程）
- [ ] MoE分布式通信精度验证（多进程）
- [ ] 组合通信算子精度验证（多进程）

### 4.3 性能测试
- [ ] 不同数据规模的性能测试（多进程）
- [ ] 不同world size的性能测试（多进程）
- [ ] 内存占用测试（多进程）

### 4.4 回归测试
- [ ] 现有精度工具测试用例
- [ ] 现有通信算子测试用例

---

## 5. 风险与缓解

### 5.1 技术风险
| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 共享内存大小受限 | 高 | 中 | 实现分块处理机制 |
| 原子操作跨进程问题 | 高 | 中 | 使用std::atomic确保跨进程可用 |
| 信号等待死锁 | 高 | 低 | 添加超时机制 |
| 共享内存命名冲突 | 中 | 低 | 使用job ID生成唯一名称 |
| 进程异常退出资源泄漏 | 高 | 中 | 注册atexit清理函数 |

### 5.2 兼容性风险
| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 不同系统POSIX shm差异 | 中 | 低 | 使用标准POSIX API |
| std::atomic跨进程兼容性 | 高 | 中 | 验证并测试跨进程原子操作 |
| MPI环境变量差异 | 中 | 低 | 支持多种MPI实现 |

---

## 6. 时间估算

| 阶段 | 任务 | 预估时间 |
|------|------|----------|
| 阶段一 | 基础设施搭建 | 4天 |
| 阶段二 | 通信算子实现 | 5天 |
| 阶段三 | 集成与测试 | 5天 |
| 阶段四 | 文档与优化 | 2天 |
| **总计** | | **16天** |

---

## 7. 交付物

1. **代码文件**：
   - `framework/src/interface/interpreter/rank_info.h/cpp`
   - `framework/src/interface/interpreter/shmem_manager.h/cpp`
   - `framework/src/interface/interpreter/calc_distributed.cpp`

2. **测试文件**：
   - `framework/tests/st/distributed/precision/test_allreduce_precision.cpp`
   - `framework/tests/st/distributed/precision/run_allreduce_test.py`

3. **文档文件**：
   - `docs/design/precision_tool_communication_support.md`
   - 更新 `docs/tutorials/debug/precision.md`

4. **构建配置**：
   - 更新 `framework/src/interface/interpreter/CMakeLists.txt`

---

## 8. 后续优化方向

1. **Futex优化**：使用Linux futex优化信号等待
2. **性能分析工具**：添加通信算子性能分析功能
3. **可视化工具**：可视化通信模式和共享内存使用情况
4. **更多通信原语**：支持更多分布式通信原语
5. **自动调优**：根据通信模式自动优化共享内存配置

---

## 9. 参考资源

1. **现有代码**：
   - `framework/src/interface/tileop/distributed/tileop_shmem.h`
   - `framework/src/interface/operation/distributed/shmem_operation_impl.cpp`
   - `python/pypto/op/distributed.py`

2. **测试用例**：
   - `models/glm_v4_5/glm_matmul_allreduce_add_rmsnorm.py`
   - `framework/tests/st/distributed/ops/src/test_allreduce.cpp`

3. **文档**：
   - `docs/tutorials/debug/precision.md`
   - `docs/api/distributed.md`

4. **多进程参考**：
   - `models/glm_v4_5/utils/distributed_config.py`
   - `framework/tests/st/distributed/framework/src/test_framework_init.cpp`

---

## 10. 总结

本计划通过以下方式实现精度工具对通信算子的支持：

1. **真正的多进程通信**：使用POSIX共享内存实现跨进程通信
2. **原子操作支持**：使用std::atomic实现跨进程原子操作
3. **进程信息管理**：通过RankInfo获取rank ID和world size
4. **共享内存管理**：ShmemManager管理跨进程共享内存的生命周期
5. **独立验证**：每个rank独立进行精度验证

该方案具有以下优势：
- ✅ 使用真正的多进程通信，模拟真实分布式环境
- ✅ 支持跨进程原子操作和信号同步
- ✅ 每个rank独立验证，符合用户需求
- ✅ 易于扩展和维护
- ✅ 不影响现有精度工具功能

预期在16天内完成开发和测试，确保精度工具能够准确验证包含通信算子的模型。
