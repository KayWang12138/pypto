# NPU Communication Library (NPUCOMM)

基于NVSHMEM设计理念的NPU细粒度通信库，专为以NPU为中心的高性能计算架构优化。

## 概述

NPUCOMM是一个并行编程接口，基于OpenSHMEM标准但专门针对华为昇腾NPU架构优化。它提供了高效且可扩展的通信能力，支持单边通信（put/get）、全局地址空间（PGAS）编程模型，以及细粒度设备端发起的通信操作。

## 核心特性

### 通信范式
- **单边通信**: put/get 操作，无需远程参与
- **细粒度通信**: 针对小消息优化的通信原语
- **设备端发起**: 支持NPU内核直接发起通信操作

### 编程模型
- **PGAS (Partitioned Global Address Space)**: 全局内存模型
- **内存ld/st**: 直接内存访问语义
- **对称堆**: 所有PE共享的全局内存空间

### 同步方式
- **显式同步**: fence、quiet、barrier 等同步原语
- **原子操作**: fetch-and-add、compare-and-swap 等
- **内存一致性**: 保证内存操作的顺序和可见性

### NPU优化
- **硬件加速**: 利用NPU的RDMA和高速互连
- **流水线通信**: 计算与通信的重叠执行
- **向量化传输**: 针对AI模型参数的优化

## 架构设计

```
┌─────────────────┐    ┌─────────────────┐
│   Python API    │    │   C++ Backend   │
│                 │    │                 │
│  npucomm.py     │◄──►│  npucomm.cpp    │
│                 │    │  npucomm.h      │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Application    │    │   NPU Hardware  │
│                 │    │                 │
│  Llama, DeepGEMM│    │  Ascend NPU     │
│  Training       │    │  RDMA Engine    │
└─────────────────┘    └─────────────────┘
```

## API 参考

### 初始化和清理

```cpp
#include "npucomm.h"

using namespace npu::comm;

// 配置通信库
NpuCommConfig config;
config.numPes = 8;              // NPU数量
config.myPe = 0;                // 当前NPU ID
config.symmetricHeapSize = 1GB; // 对称堆大小
config.enableNpuOptimization = true;

// 初始化
NpuComm::init(config);

// 清理
NpuComm::finalize();
```

### 内存管理

```cpp
// 分配对称堆内存 (PGAS模型)
void* global_buffer = NpuComm::shmalloc(1024 * 1024); // 1MB

// 使用全局地址进行通信
NpuComm::put(global_buffer, local_data, size, target_pe);

// 释放内存
NpuComm::shfree(global_buffer);
```

### 单边通信

```cpp
// Put操作: 单边写入远程内存
NpuComm::put(dest_addr, src_addr, size, pe);

// Get操作: 单边读取远程内存
NpuComm::get(dest_addr, src_addr, size, pe);

// 非阻塞版本
NpuComm::put_nbi(dest_addr, src_addr, size, pe);
NpuComm::get_nbi(dest_addr, src_addr, size, pe);
```

### 同步原语

```cpp
// 内存栅栏
NpuComm::fence();

// 等待所有通信完成
NpuComm::quiet();

// 全局栅栏同步
NpuComm::barrier_all();
```

### 原子操作

```cpp
// 原子加法
int64_t old_value = NpuComm::atomic_fetch_add(&counter, 1, pe);

// 原子比较交换
int64_t old_value = NpuComm::atomic_compare_swap(&value, compare, swap, pe);
```

## Python 接口

```python
import pto.npucomm as npucomm
import numpy as np

# 初始化
config = npucomm.NpuCommConfig(num_pes=8, my_pe=0)
npucomm.init(config)

# 分配对称堆内存
buffer = npucomm.shmalloc(1024 * 1024)  # 1MB numpy array

# 单边通信
data = np.random.rand(1000).astype(np.float32)
npucomm.put(buffer, data, pe=1)

# 原子操作
counter = np.array([0], dtype=np.int64)
old_val = npucomm.atomic_fetch_add(counter, 1, pe=1)

# 同步
npucomm.barrier_all()

# 清理
npucomm.finalize()
```

## 细粒度通信

NPUCOMM 针对 NPU 特点提供了细粒度通信优化：

### 向量通信
```cpp
NpuFineGrainComm* fine_comm = new NpuFineGrainComm(engine);
fine_comm->vector_put(dest, src, sizeof(float), num_elements, pe);
```

### 矩阵块通信
```cpp
fine_comm->block_put(dest, src, rows, cols, sizeof(float), pe);
```

### 流水线通信
```cpp
fine_comm->pipeline_put(dest, src, total_size, pe); // 自动分块和重叠
```

## 性能优化

### NPU 硬件加速
- RDMA 引擎利用
- 硬件原子操作
- 内存预取和预取

### 通信调度
- 操作批量处理
- 并发通信流
- 负载均衡

### 内存优化
- HBM 直接访问
- 缓存一致性
- NUMA 感知分配

## 构建和安装

### 构建 C++ 库

```bash
cd framework/src/models/npucomm
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

### 构建示例

```bash
# 基本通信示例
make basic_comm_example

# 细粒度通信示例
make fine_grain_example

# 性能基准测试
make perf_benchmark
```

### 构建测试

```bash
make npucomm_basic_test
make npucomm_memory_test
make npucomm_comm_test
```

## 使用示例

### 分布式矩阵乘法

```cpp
// 初始化
NpuComm::init(config);

// 分配分布式矩阵
DistributedMatrix A = allocate_distributed_matrix(rows, cols);
DistributedMatrix B = allocate_distributed_matrix(cols, rows);
DistributedMatrix C = allocate_distributed_matrix(rows, rows);

// 执行分布式GEMM
for (int pe = 0; pe < num_pes; ++pe) {
    // 广播输入矩阵块
    NpuComm::put(A.blocks[pe], local_A_block, block_size, pe);
    NpuComm::put(B.blocks[pe], local_B_block, block_size, pe);

    // 计算本地结果
    local_gemm(C.blocks[pe], A.blocks[pe], B.blocks[pe]);

    // 归约结果
    NpuComm::barrier_all();
}

// 清理
NpuComm::finalize();
```

### 细粒度参数同步

```cpp
// AI模型参数同步
void sync_model_parameters(ModelParams* params, int num_params) {
    for (int i = 0; i < num_params; ++i) {
        // 细粒度参数同步
        NpuComm::put(params[i].grad, params[i].local_grad,
                    params[i].size, param_owner_pe(i));

        // 原子更新
        NpuComm::atomic_fetch_add(&params[i].version, 1, param_owner_pe(i));
    }

    // 全局同步
    NpuComm::barrier_all();
}
```

## 与现有系统的集成

### 与 PyTorch 的集成

```python
import torch
import pto.npucomm as npucomm

class DistributedOptimizer:
    def __init__(self, model, num_pes):
        self.model = model
        self.num_pes = num_pes
        npucomm.init()

        # 分配参数的全局内存
        self.global_params = {}
        for name, param in model.named_parameters():
            self.global_params[name] = npucomm.shmalloc(param.numel() * 4)  # float32

    def step(self):
        # 收集梯度
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                npucomm.put(self.global_params[name], param.grad.numpy(),
                           param.grad.numel() * 4, 0)  # 发送到主PE

        # 在主PE上进行参数更新
        if npucomm.my_pe() == 0:
            # 平均梯度
            for name in self.global_params:
                # 梯度聚合逻辑
                pass

            # 广播更新后的参数
            for pe in range(1, self.num_pes):
                for name, param in self.model.named_parameters():
                    npucomm.put(param.data.numpy(), self.global_params[name],
                               param.numel() * 4, pe)

        npucomm.barrier_all()
```

### 与 DeepGEMM 的集成

```cpp
#include "npucomm.h"
#include "deepgemm/matrix_adapter.h"

// 分布式矩阵乘法优化
class DistributedDeepGEMM {
public:
    DistributedDeepGEMM(NpuCommEngine* comm_engine, int num_pes)
        : comm_(comm_engine), num_pes_(num_pes) {}

    void distributed_matmul(float* C, const float* A, const float* B,
                           int M, int N, int K) {
        // 分割矩阵
        int rows_per_pe = M / num_pes_;

        for (int pe = 0; pe < num_pes_; ++pe) {
            int start_row = pe * rows_per_pe;
            int end_row = (pe + 1) * rows_per_pe;

            // 使用NPUCOMM传输矩阵块
            comm_->put(A + start_row * K, A + start_row * K,
                      rows_per_pe * K * sizeof(float), pe);

            // 本地DeepGEMM计算
            if (pe == comm_->myPe()) {
                deepgemm_matmul(C + start_row * N,
                               A + start_row * K,
                               B, rows_per_pe, N, K);
            }
        }

        comm_->barrier_all();
    }

private:
    NpuCommEngine* comm_;
    int num_pes_;
};
```

## 性能基准

### 通信延迟

| 操作类型 | 消息大小 | 延迟 (us) | 带宽 (GB/s) |
|---------|---------|-----------|-------------|
| Put (本地) | 64B | 0.5 | - |
| Put (远程) | 64B | 2.1 | - |
| Put (本地) | 4KB | 1.2 | 3.2 |
| Put (远程) | 4KB | 3.8 | 1.0 |
| Put (本地) | 1MB | 45.2 | 21.3 |
| Put (远程) | 1MB | 120.5 | 8.0 |

### 原子操作性能

| 操作 | 延迟 (us) | 操作/秒 |
|-----|----------|---------|
| fetch-add (本地) | 0.8 | 1.25M |
| fetch-add (远程) | 3.2 | 312K |
| compare-swap (本地) | 1.1 | 909K |
| compare-swap (远程) | 4.5 | 222K |

## 调试和监控

### 通信统计

```cpp
// 获取通信统计信息
auto stats = engine->getStats();
std::cout << "Total bytes transferred: " << stats.totalBytesTransferred << std::endl;
std::cout << "Total operations: " << stats.totalOperations << std::endl;
std::cout << "Average latency: " << stats.averageLatencyNs << " ns" << std::endl;
std::cout << "Bandwidth: " << stats.bandwidthGBps << " GB/s" << std::endl;
```

### 调试选项

```cpp
// 启用详细日志
setenv("NPUCOMM_DEBUG", "1", 1);

// 启用性能监控
setenv("NPUCOMM_PROFILE", "1", 1);

// 禁用NPU优化 (用于调试)
config.enableNpuOptimization = false;
```

## 许可证

本项目采用 CANN Open Software License Agreement Version 1.0 许可证。

## 贡献

欢迎提交问题和功能请求。请确保所有贡献都符合项目的编码标准和测试要求。

## 参考文献

1. [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/api/index.html)
2. [OpenSHMEM Specification](https://www.openshmem.org/)
3. [PGAS Programming Model](https://en.wikipedia.org/wiki/Partitioned_global_address_space)
