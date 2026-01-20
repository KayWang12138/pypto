# get_block_idx() 函数实现说明

本文档详细说明 `get_block_idx()` 函数的性质、作用和实现方式。

## 1. 函数性质

`get_block_idx()` 是 **NPU 编译器内置函数（Compiler Built-in/Intrinsic）**，类似于 CUDA 的 `blockIdx.x`。

**关键特征：**
- ✅ **编译器内置**：由 NPU 编译器（CCE）直接提供
- ✅ **运行时值**：返回当前执行 core 的逻辑索引
- ✅ **无需定义**：在代码中直接使用，无需包含头文件
- ✅ **硬件相关**：值来自硬件运行时状态

## 2. 函数签名和返回值

```cpp
uint32_t get_block_idx();
```

**返回值：**
- 当前 AICore 的逻辑 block 索引（从 0 开始）
- 范围：`[0, blockDim - 1]`，其中 `blockDim` 是启动 kernel 时指定的 core 数量

## 3. 在 gemm_performance_kernel.cpp 中的使用

```57:64:pto-isa/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp
    uint32_t mIterIdx = get_block_idx() % mIter; // get current launch core idx
    uint32_t nIterIdx = get_block_idx() / mIter;
    uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
    uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
    uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
    currentSrc0 = src0 + gmOffsetA;
    currentSrc1 = src1 + gmOffsetB;
```

**作用：**
- 计算当前 core 在 2D 网格中的位置（mIterIdx, nIterIdx）
- 根据位置计算数据偏移量
- 实现 SPMD（Single Program, Multiple Data）执行模型

## 4. 执行模型：SPMD

**SPMD（Single Program, Multiple Data）模式：**

```
所有 24 个 Core 同时启动，执行相同的 kernel 代码
  ↓
每个 Core 调用 get_block_idx() 获取自己的索引
  ↓
根据索引计算自己负责的数据区域
  ↓
并行处理不同的数据块
```

**示例：**
```cpp
// 假设 blockDim = 24, mIter = 4, nIter = 6
// 则 mIter * nIter = 4 * 6 = 24，正好匹配 24 个 core

Core 0: get_block_idx() = 0  → mIterIdx=0, nIterIdx=0  → 处理 (0,0) tile
Core 1: get_block_idx() = 1  → mIterIdx=1, nIterIdx=0  → 处理 (1,0) tile
Core 2: get_block_idx() = 2  → mIterIdx=2, nIterIdx=0  → 处理 (2,0) tile
...
Core 23: get_block_idx() = 23 → mIterIdx=3, nIterIdx=5  → 处理 (3,5) tile
```

## 5. 编译器层面的处理

### 5.1 内核入口函数

在 `aicore.ascpp` 中可以看到：

```275:281:pypto_zimo/framework/src/machine/kernel/aicore.ascpp
extern "C" __global__ __aicore__ void KERNEL_ENTRY(__OPTYPE__, __TILINGKEY__)(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
#if defined(__AIV__) and defined(__MIX__)
    aicore_blockIdx = get_block_idx() * get_subblockdim() + get_subblockid() + get_block_num();
#else
    aicore_blockIdx = get_block_idx();
#endif
```

**说明：**
- `get_block_idx()` 在 kernel 入口时被调用
- 结果存储在 `aicore_blockIdx` 中
- 用于索引每个 core 的共享缓冲区

### 5.2 编译器内置函数

`get_block_idx()` 是编译器内置函数，类似于：

**CUDA 类比：**
```cpp
// CUDA
int blockIdx = blockIdx.x;  // 编译器内置

// NPU (PTO)
uint32_t blockIdx = get_block_idx();  // 编译器内置
```

## 6. 硬件实现

`get_block_idx()` 的值来自 NPU 硬件的运行时状态：

1. **启动时分配**：当 kernel 启动时（通过 `<<<blockDim, ...>>>`），硬件调度器为每个 core 分配一个 block 索引
2. **运行时读取**：每个 core 通过硬件寄存器或特殊指令读取自己的 block 索引
3. **编译器生成**：编译器将 `get_block_idx()` 转换为读取硬件状态的内联代码

**类比：**
- 类似于 CPU 的 `pthread_self()` 或 OpenMP 的 `omp_get_thread_num()`
- 但这是在硬件层面实现的，无需软件调度

## 7. 相关函数

### 7.1 `get_subblockdim()` 和 `get_subblockid()`

在某些场景中（如 MIX 模式），还会使用：

```cpp
// 虚拟 block 索引
auto vid = get_block_idx() * get_subblockdim() + get_subblockid();
```

**用途：** 当每个物理 core 有多个子任务时使用。

### 7.2 `get_coreid()`

```cpp
int coreId = get_coreid();  // 获取物理 core ID
```

**区别：**
- `get_block_idx()`：逻辑 block 索引（可能跨多个物理 core）
- `get_coreid()`：物理 core ID（硬件标识）

## 8. 实际应用示例

### 8.1 GEMM 多核切分

```cpp
// gemm_performance_kernel.cpp
constexpr uint32_t mIter = m / singleCoreM;  // 例如：6144 / 1536 = 4

uint32_t mIterIdx = get_block_idx() % mIter;  // 当前 core 的 m 索引 (0-3)
uint32_t nIterIdx = get_block_idx() / mIter;  // 当前 core 的 n 索引 (0-5)

// 计算数据偏移
uint64_t gmOffsetA = mIterIdx * singleCoreM * k;     // A 矩阵的偏移
uint64_t gmOffsetB = nIterIdx * k * singleCoreN;     // B 矩阵的偏移
uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;  // C 矩阵的偏移
```

**网格布局（24 个 core）：**
```
nIterIdx
  0    1    2    3    4    5
  ┌────┴────┴────┴────┴────┐
0 │ 0   1   2   3   4   5  │
  │                        │
1 │ 6   7   8   9  10  11  │
  │                        │
2 │12  13  14  15  16  17  │ mIterIdx
  │                        │
3 │18  19  20  21  22  23  │
  └────────────────────────┘

每个 core 负责处理一个 [singleCoreM, singleCoreN] 的 C tile
```

### 8.2 数据访问模式

```cpp
// 每个 core 访问自己负责的数据区域
currentSrc0 = src0 + gmOffsetA;  // Core 0: src0 + 0
                                 // Core 1: src0 + singleCoreM * k
                                 // Core 2: src0 + 2 * singleCoreM * k
                                 // ...

currentSrc1 = src1 + gmOffsetB;  // Core 0: src1 + 0
                                 // Core 6: src1 + k * singleCoreN
                                 // ...

currentDst = out + gmOffsetC;    // Core 0: out + 0
                                 // Core 1: out + singleCoreM * n
                                 // ...
```

## 9. 为什么需要 get_block_idx()？

### 9.1 SPMD 执行模式

在 SPMD 模式下，所有 core 执行相同的代码，但处理不同的数据：

```cpp
// 所有 core 都执行这段代码
uint32_t myBlockIdx = get_block_idx();  // 但每个 core 得到不同的值
uint32_t myDataOffset = myBlockIdx * dataSize;  // 计算不同的数据偏移
// 处理 myDataOffset 到 myDataOffset + dataSize 的数据
```

### 9.2 数据并行性

- **任务并行**：不同 core 处理不同的任务（MPMD）
- **数据并行**：所有 core 执行相同任务，但处理不同数据（SPMD）

`get_block_idx()` 是实现数据并行的关键。

## 10. 与 CUDA 的对比

| 特性 | CUDA | NPU (PTO) |
|------|------|-----------|
| **Block 索引** | `blockIdx.x` | `get_block_idx()` |
| **Thread 索引** | `threadIdx.x` | `get_subblockid()`（在某些场景） |
| **Grid 维度** | `gridDim.x` | 通过 `blockDim` 参数传入 |
| **函数性质** | 编译器内置变量 | 编译器内置函数 |

## 11. 查看方式

### 11.1 无法查看源代码

`get_block_idx()` 是编译器内置函数，**没有 C++ 源代码实现**。

**原因：**
- 编译器在编译时将 `get_block_idx()` 替换为读取硬件状态的代码
- 具体实现是编译器内部的，不暴露给用户

### 11.2 查看编译后的代码

可以查看编译后的汇编代码来了解它如何工作：

1. **编译时的转换**：编译器将 `get_block_idx()` 转换为读取硬件寄存器的指令
2. **运行时行为**：每个 core 读取自己的 block 索引值

### 11.3 相关文档

- **PTO 编程模型：** `pto-isa/docs/coding/ProgrammingModel.md`
- **执行模型：** `pto-isa/docs/mkdocs/src/manual/02-machine-model.md`

## 12. 总结

### 12.1 函数性质

- ✅ **编译器内置函数**：由 NPU 编译器（CCE）提供
- ✅ **无需定义**：直接在代码中使用
- ✅ **运行时值**：返回当前 core 的逻辑 block 索引

### 12.2 作用

- ✅ **SPMD 执行模型**：所有 core 执行相同代码，通过 `get_block_idx()` 区分
- ✅ **数据切分**：根据 block 索引计算数据偏移，实现数据并行
- ✅ **多核协同**：24 个 core 并行处理不同数据块

### 12.3 实现方式

- **编译时**：编译器将 `get_block_idx()` 转换为读取硬件状态的代码
- **运行时**：每个 core 从硬件寄存器读取自己的 block 索引
- **无法查看**：没有 C++ 源代码，是编译器内部的实现

### 12.4 使用示例

```cpp
// 计算当前 core 在 2D 网格中的位置
uint32_t mIterIdx = get_block_idx() % mIter;
uint32_t nIterIdx = get_block_idx() / mIter;

// 根据位置计算数据偏移
uint64_t offset = mIterIdx * rowStride + nIterIdx * colStride;
```

**关键点：**
- `get_block_idx()` 的值在运行时由硬件确定
- 每个 core 得到不同的值（0 到 blockDim-1）
- 用于实现 SPMD 数据并行计算

