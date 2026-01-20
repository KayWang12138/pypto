# NPU AICore 存储架构详解

本文档详细解释 NPU 中 24 个 AICore 的存储架构，特别是 L0 和 L1 存储的分配关系。

## 1. 核心结论

### 1.1 L0 存储：**一对一**（每个 Core 独立）

**每个 AICore 都有自己独立的 L0 存储**，包括：
- **L0A**：用于存储矩阵 A 的数据（64 KB）
- **L0B**：用于存储矩阵 B 的数据（64 KB）
- **L0C**：用于存储矩阵 C 的结果（256 KB）

### 1.2 L1 存储：**一对一**（每个 Core 独立）

**每个 AICore 都有自己独立的 L1 存储**（1 MB）。

### 1.3 L2 存储：**共享**（所有 Core 共享）

**L2 存储是所有 AICore 共享的**（32 MB，Ascend910A）。

## 2. 存储层次结构

```
GM (Global Memory / DDR)
    ↓
L2 Cache (共享，32 MB)
    ↓
L1 Cache (每个 Core 独立，1 MB × 24 = 24 MB)
    ↓
L0 Cache (每个 Core 独立)
    ├── L0A (64 KB)
    ├── L0B (64 KB)
    └── L0C (256 KB)
```

## 3. 平台配置示例

### 3.1 Ascend910A（32 个 AICore）

```ini
[SoCInfo]
ai_core_cnt=32
l2_size=33554432  # 32 MB，共享

[AICoreSpec]
l0_a_size=65536    # 64 KB，每个 Core 独立
l0_b_size=65536    # 64 KB，每个 Core 独立
l0_c_size=262144   # 256 KB，每个 Core 独立
l1_size=1048576    # 1 MB，每个 Core 独立
```

### 3.2 A3 平台（24 个 AICore）

从 `gemm_performance_kernel.cpp` 可以看到：
```cpp
constexpr uint32_t blockDim = 24;  // 24 个 AICore
```

每个 Core 的存储配置（参考 Ascend910A）：
- L0A: 64 KB
- L0B: 64 KB
- L0C: 256 KB
- L1: 1 MB

## 4. 存储访问模式

### 4.1 数据流

```
GM → TLOAD → L1 (每个 Core 独立) → TEXTRACT → L0A/L0B (每个 Core 独立)
                                                      ↓
                                              TMATMUL → L0C (每个 Core 独立)
                                                      ↓
                                              TSTORE → GM
```

### 4.2 多核并行

在 `gemm_performance_kernel.cpp` 中：

```cpp
// 每个 Core 通过 get_block_idx() 获取自己的索引
uint32_t mIterIdx = get_block_idx() % mIter;  // 当前 Core 的 m 索引
uint32_t nIterIdx = get_block_idx() / mIter;  // 当前 Core 的 n 索引

// 每个 Core 计算自己的数据偏移
uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
```

**关键点：**
- 每个 Core 独立访问自己的 L0/L1 存储
- 每个 Core 处理不同的数据块（通过 `get_block_idx()` 区分）
- 所有 Core 共享 GM 和 L2

## 5. 存储大小总结

### 5.1 每个 AICore 的存储

| 存储类型 | 大小 | 用途 | 是否独立 |
|---------|------|------|---------|
| L0A | 64 KB | 存储矩阵 A 的 tile | ✅ 独立 |
| L0B | 64 KB | 存储矩阵 B 的 tile | ✅ 独立 |
| L0C | 256 KB | 存储矩阵 C 的结果 | ✅ 独立 |
| L1 | 1 MB | 中间缓存，用于 TLOAD/TSTORE | ✅ 独立 |
| UB | 256 KB | Unified Buffer，向量计算 | ✅ 独立 |

### 5.2 共享存储

| 存储类型 | 大小 | 用途 | 是否共享 |
|---------|------|------|---------|
| L2 | 32 MB (Ascend910A) | 所有 Core 共享的二级缓存 | ✅ 共享 |
| GM/DDR | 可变 | 全局内存 | ✅ 共享 |

## 6. 为什么是一对一？

### 6.1 性能考虑

1. **低延迟访问**：每个 Core 独立访问自己的 L0/L1，避免竞争和同步开销
2. **并行计算**：24 个 Core 可以同时进行不同的计算，互不干扰
3. **数据局部性**：每个 Core 处理的数据块在空间上连续，L0/L1 独立可以更好地利用局部性

### 6.2 硬件设计

从平台配置文件可以看到：
- `AICoreSpec` 中定义了**每个 Core** 的存储大小
- `SoCInfo` 中定义了**整个 SoC** 的 L2 大小

这说明：
- L0/L1 是**每个 Core 的硬件资源**
- L2 是**整个 SoC 的共享资源**

## 7. 实际应用示例

在 `gemm_performance_kernel.cpp` 中：

```cpp
// 24 个 Core 并行执行
template <...>
__global__ AICORE void GemmPerformance(...) {
    // 每个 Core 获取自己的索引
    uint32_t mIterIdx = get_block_idx() % mIter;
    uint32_t nIterIdx = get_block_idx() / mIter;
    
    // 每个 Core 使用自己的 L0/L1 存储
    TileMatA aMatTile[BUFFER_NUM];  // 存储在 L1
    TileMatB bMatTile[BUFFER_NUM];   // 存储在 L1
    LeftTile aTile[BUFFER_NUM];      // 存储在 L0A
    RightTile bTile[BUFFER_NUM];      // 存储在 L0B
    ResTile cTile;                    // 存储在 L0C
    
    // 每个 Core 独立执行计算
    RunGemmE2E(...);
}
```

**关键点：**
- 24 个 Core 同时启动，每个 Core 执行相同的 kernel 代码
- 每个 Core 通过 `get_block_idx()` 获取不同的索引
- 每个 Core 使用自己独立的 L0/L1 存储
- 所有 Core 共享 GM 和 L2

## 8. 总结

### 8.1 存储分配关系

| 存储层级 | 分配方式 | 说明 |
|---------|---------|------|
| **L0** | **一对一** | 每个 AICore 有独立的 L0A/L0B/L0C |
| **L1** | **一对一** | 每个 AICore 有独立的 L1（1 MB） |
| **L2** | **共享** | 所有 AICore 共享 L2（32 MB） |
| **GM** | **共享** | 所有 AICore 共享全局内存 |

### 8.2 关键结论

1. ✅ **L0 存储是一对一的**：每个 AICore 都有自己独立的 L0A/L0B/L0C
2. ✅ **L1 存储是一对一的**：每个 AICore 都有自己独立的 L1（1 MB）
3. ✅ **L2 存储是共享的**：所有 AICore 共享 L2（32 MB）
4. ✅ **24 个 AICore 并行工作**：每个 Core 独立访问自己的 L0/L1，互不干扰

### 8.3 性能影响

- **优势**：每个 Core 独立访问，无竞争，低延迟
- **限制**：每个 Core 的 L0/L1 大小有限，需要合理规划 tile 大小
- **优化**：通过合理的 tile 切分和双缓冲，可以最大化利用每个 Core 的存储

