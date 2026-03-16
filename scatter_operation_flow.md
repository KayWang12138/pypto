# Scatter 操作流程总结

## 一、修改背景

由于直接操作 GM 会导致性能问题，需要使用 TLOAD/TSTORE 指令进行 GM 访问。

## 二、修改内容

### 1. 添加临时缓冲区管理（参考 hypot.h）

```cpp
template <typename T>
struct ScatterTmpBuffers {
    __ubuf__ void *buf0;
    __ubuf__ void *buf1;
    __ubuf__ typename T::Type *buf0Typed;
    __ubuf__ typename T::Type *buf1Typed;
};

template <typename T, typename TTmp>
TILEOP ScatterTmpBuffers<T> InitScatterTmpBuffers(TTmp tmpbuf, size_t elementCount) {
    uint64_t dataSizeBytes = elementCount * sizeof(typename T::Type);
    const uint32_t ALIGNMENT = 32;
    uint64_t alignedSizeBytes = (dataSizeBytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ uint8_t *basePtr = reinterpret_cast<__ubuf__ uint8_t *>(tmpbufAddr);

    ScatterTmpBuffers<T> buffers;
    buffers.buf0Typed = reinterpret_cast<__ubuf__ typename T::Type *>(basePtr);
    buffers.buf0 = reinterpret_cast<__ubuf__ void *>(basePtr);
    __ubuf__ uint8_t *ptrBuf1 = basePtr + alignedSizeBytes;

    buffers.buf1Typed = reinterpret_cast<__ubuf__ typename T::Type *>(ptrBuf1);
    buffers.buf1 = reinterpret_cast<__ubuf__ void *>(ptrBuf1);

    return buffers;
}
```

**特点**：
- 32 字节对齐，提高访问效率
- 从大的临时缓冲区中划分出两个子缓冲区
- 支持泛型数据类型

### 2. 修改新版 Tscatter 函数

**函数签名**：
```cpp
template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3, typename C>
TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 temp, C coordinate)
```

**参数说明**：
- T0 dst: GM 上的目标 tensor
- T1 src1: UB 上的 indices tensor
- T2 src2: UB 上的 src tensor
- T3 temp: UB 上的临时缓冲区（大缓冲区）
- C coordinate: 用于计算 GM 偏移的坐标

## 三、操作流程

### 步骤 1：初始化临时缓冲区

```cpp
auto buffers = InitScatterTmpBuffers<T0>(temp, i3Shape * i4Shape);
```

**作用**：
- 从传入的大临时缓冲区中划分出两个对齐的子缓冲区
- buf0 和 buf1 分别用于存储中间结果
- 32 字节对齐，提高访问效率

### 步骤 2：计算 GM 基地址和偏移

```cpp
auto dstBaseOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);
__gm__ DstType *dstBaseAddr = reinterpret_cast<__gm__ DstType *>(dst.GetAddr()) + dstBaseOffset;
__ubuf__ IdxType *idxBaseAddr = reinterpret_cast<__ubuf__ IdxType *>(src1.GetAddr());
__ubuf__ SrcType *srcBaseAddr = reinterpret_cast<__ubuf__ SrcType *>(src2.GetAddr());
```

**作用**：
- 使用 coordinate 计算当前 tile 在 GM 中的偏移
- 获取各 tensor 的基地址指针

### 步骤 3：遍历所有元素

```cpp
for (LoopVar i = 0; i < i0Shape; ++i) {
    for (LoopVar j = 0; j < i1Shape; ++j) {
        for (LoopVar k = 0; k < i2Shape; ++k) {
            for (LoopVar l = 0; l < i3Shape; ++l) {
                for (LoopVar m = 0; m < i4Shape; ++m) {
                    // 处理每个元素
                }
            }
        }
    }
}
```

### 步骤 4：读取 index 和 srcVal

```cpp
IdxType index = *(idxBaseAddr + i * i0Stride + j * i1Stride + k * i2Stride + l * i3Stride + m);
SrcType srcVal = *(srcBaseAddr + i * s0Stride + j * s1Stride + k * s2Stride + l * s3Stride + m);
```

### 步骤 5：计算 GM 偏移

```cpp
int64_t localOffset = 0;
if constexpr (axis == 0) {
    localOffset = static_cast<int64_t>(index) * d0Stride + j * d1Stride + k * d2Stride + l * d3Stride + m;
} else if constexpr (axis == 1) {
    localOffset = i * d0Stride + static_cast<int64_t>(index) * d1Stride + k * d2Stride + l * d3Stride + m;
} else if constexpr (axis == 2) {
    localOffset = i * d0Stride + j * d1Stride + static_cast<int64_t>(index) * d2Stride + l * d3Stride + m;
} else if constexpr (axis == 3) {
    localOffset = i * d0Stride + j * d1Stride + k * d2Stride + static_cast<int64_t>(index) * d3Stride + m;
} else {
    localOffset = i * d0Stride + j * d1Stride + k * d2Stride + l * d3Stride + static_cast<int64_t>(index);
}
```

**作用**：
- 根据 axis 和 index 计算目标在 GM 中的偏移
- 支持任意 axis（0-4）

### 步骤 6：根据 scatterMode 执行不同操作

#### 6.1 scatterMode = 0 (UPDATE)

```cpp
__gm__ DstType *dstAddr = dstBaseAddr + localOffset;

DstTileType dstTile(1, 1);
SrcTileType srcTile(1, 1);

pto::TASSIGN(srcTile, srcVal);

GlobalData dstGlobal(dstAddr, pto::Shape(1, 1, 1, 1, 1), pto::Stride(1, 1, 1, 1, 1));

pto::TASSIGN(dstTile, srcTile);
pto::TSTORE(dstGlobal, dstTile);
```

**流程**：
1. 构造 dstTile 和 srcTile（1x1）
2. 将 srcVal 赋值给 srcTile
3. 构造 GlobalData（GM 上的 1x1 区域）
4. 使用 TSTORE 将 srcTile 写入 GM

#### 6.2 scatterMode = 1 (ADD)

```cpp
__gm__ DstType *dstAddr = dstBaseAddr + localOffset;

DstTileType dstTile(1, 1);
SrcTileType srcTile(1, 1);

pto::TASSIGN(srcTile, srcVal);

GlobalData dstGlobal(dstAddr, pto::Shape(1, 1, 1, 1, 1), pto::Stride(1, 1, 1, 1, 1));

pto::TLOAD(dstTile, dstGlobal);  // 从 GM 读取当前值
pto::TADD(dstTile, dstTile, srcTile);  // 累加：dstTile = dstTile + srcTile
pto::TSTORE(dstGlobal, dstTile);  // 写回 GM
```

**流程**：
1. 构造 dstTile 和 srcTile（1x1）
2. 将 srcVal 赋值给 srcTile
3. 构造 GlobalData（GM 上的 1x1 区域）
4. 使用 TLOAD 从 GM 读取当前值到 dstTile
5. 使用 TADD 计算 dstTile = dstTile + srcTile
6. 使用 TSTORE 将 dstTile 写回 GM

#### 6.3 scatterMode = 2 (MULTIPLY)

```cpp
__gm__ DstType *dstAddr = dstBaseAddr + localOffset;

DstTileType dstTile(1, 1);
SrcTileType srcTile(1, 1);

pto::TASSIGN(srcTile, srcVal);

GlobalData dstGlobal(dstAddr, pto::Shape(1, 1, 1, 1, 1), pto::Stride(1, 1, 1, 1, 1));

pto::TLOAD(dstTile, dstGlobal);  // 从 GM 读取当前值
pto::TMUL(dstTile, dstTile, srcTile);  // 乘法：dstTile = dstTile * srcTile
pto::TSTORE(dstGlobal, dstTile);  // 写回 GM
```

**流程**：
1. 构造 dstTile 和 srcTile（1x1）
2. 将 srcVal 赋值给 srcTile
3. 构造 GlobalData（GMGM 上的 1x1 区域）
4. 使用 TLOAD 从 GM 读取当前值到 dstTile
5. 使用 TMUL 计算 dstTile = dstTile * srcTile
6. 使用 TSTORE 将 dstTile 写回 GM

## 四、关键改进点

### 1. 使用 TLOAD/TSTORE 替代直接 GM 访问

**改进前**：
```cpp
dstBaseAddr[localOffset] = srcVal;  // 直接写 GM
dstBaseAddr[localOffset] += srcVal;  // 直接读改 GM
```

**改进后**：
```cpp
pto::TSTORE(dstGlobal, srcTile);  // 使用 TSTORE 写 GM
pto::TLOAD(dstTile, dstGlobal);  // 使用 TLOAD 读 GM
pto::TADD(dstTile, dstTile, srcTile);  // 计算
pto::TSTORE(dstGlobal, dstTile);  // 使用 TSTORE 写 GM
```

**优势**：
- TLOAD/TSTORE 是 NPU 专用指令，性能更好
- 支持流水线操作
- 更好的内存访问模式

### 2. 临时缓冲区管理

**特点**：
- 32 字节对齐，提高访问效率
- 从大缓冲区中划分两个子缓冲区
- 支持泛型数据类型
- 参考 hypot.h 的实现

### 3. 支持 GM 偏移计算

**特点**：
- 使用 coordinate 计算当前 tile 在 GM 中的偏移
- 支持动态 shape 和 offset
- 支持任意 axis（0-4）

### 4. 完整的 scatterMode 支持

- **scatterMode = 0 (UPDATE)**：直接写入
- **scatterMode = 1 (ADD)**：累加
- **scatterMode = 2 (MULTIPLY)**：乘法

## 五、代码位置

### 修改的文件

1. **scatter.h** (`/mnt/workspace/gitCode/cann/pypto/framework/src/interface/tileop/vector/scatter.h`)
   


**Operand 映射**：
- ID0: dstTile (GM, DST_IDX)
- ID1: tmpBuffer (UB, TMP_IDX)
- ID2: selfTile (UB, SRC0_IDX) - 不加载，跳过
- ID3: idxTile (UB, SRC1_IDX)
- ID4: srcTile (UB, SRC2_IDX)

## 六、参考实现

### hypot.h

**临时缓冲区管理**：
- 32 字节对齐
- 从大缓冲区划分两个子缓冲区
- 支持泛型数据类型

### gather.h

**TLOAD 使用**：
```cpp
pto::TLOAD(dstTile, srcGlobal);  // 从 GM 加载到 UB
```

### index_outcast.h

**TSTORE 使用**：
```cpp
pto::TSTORE(dstGlobal, srcTile);  // 从 UB 存储到 GM
```


