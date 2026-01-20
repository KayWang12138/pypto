# TMATMUL 和 TMATMUL_ACC 底层实现指南

本文档详细说明 `TMATMUL` 和 `TMATMUL_ACC` 操作的底层实现位置和调用链。

## 1. 调用链概览

```
用户代码 (gemm_performance_kernel.cpp)
    ↓
TMATMUL / TMATMUL_ACC (pto_instr.hpp)
    ↓
TMATMUL_IMPL / TMATMUL_ACC_IMPL (a2a3/TMatmul.hpp)
    ↓
TMatmul 函数 (a2a3/TMatmul.hpp)
    ↓
mad 函数 (硬件内联函数)
    ↓
NPU 硬件指令 (MAD)
```

## 2. 实现位置详解

### 2.1 用户接口层

**文件：** `pto-isa/include/pto/common/pto_instr.hpp`

```302:315:pto-isa/include/pto/common/pto_instr.hpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
  return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
  WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
  return {};
}
```

**作用：**
- 提供用户友好的 C++ 模板接口
- 处理事件同步（`TSYNC`）
- 通过 `MAP_INSTR_IMPL` 宏映射到具体实现

### 2.2 平台实现层

**文件：** `pto-isa/include/pto/npu/a2a3/TMatmul.hpp`

#### TMATMUL_IMPL

```88:97:pto-isa/include/pto/npu/a2a3/TMatmul.hpp
PTO_INTERNAL void TMATMUL_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    CheckDynamicMad(m, k, n);
    TMatmul<TileRes, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, kDirectionAlign);
}
```

#### TMATMUL_ACC_IMPL

```100:110:pto-isa/include/pto/npu/a2a3/TMatmul.hpp
PTO_INTERNAL void TMATMUL_ACC_IMPL(
    TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    CheckDynamicMad(m, k, n);
    TMatmul<TileRes, TileLeft, TileRight, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, kDirectionAlign);
}
```

**关键点：**
- `TMATMUL`：`cmatrixInitVal = true`（初始化 C 矩阵）
- `TMATMUL_ACC`：`cmatrixInitVal = false`（累加到 C 矩阵）
- `cmatrixSource` 都是 `false`（不使用 C 作为源）

### 2.3 硬件接口层

**文件：** `pto-isa/include/pto/npu/a2a3/TMatmul.hpp`

```29:40:pto-isa/include/pto/npu/a2a3/TMatmul.hpp
template <typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource, bool cmatrixInitVal>
__tf__ AICORE void TMatmul(typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n, bool kDirectionAlign)
{
    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);
    if (m == 1) {
        m = 16; // avoid gemv mode, if m is 1, the gemv mode will be used in a3
    }
    mad(c, a, b, m, k, n, 0, kDirectionAlign, cmatrixSource, cmatrixInitVal);
}
```

**关键元素：**

1. **`__tf__`**：标记为 Tile Function，由编译器特殊处理
2. **`AICORE`**：AICore 函数属性
3. **`__cc__`、`__ca__`、`__cb__`**：内存位置限定符
   - `__cc__`：L0C（C 矩阵）
   - `__ca__`：L0A（A 矩阵）
   - `__cb__`：L0B（B 矩阵）
4. **`__cce_get_tile_ptr`**：获取 tile 的数据指针
5. **`mad`**：**硬件矩阵乘法指令的内联函数**

### 2.4 硬件指令层

**`mad` 函数**是 NPU 硬件指令的内联函数（intrinsic），由编译器直接生成硬件指令。

**函数签名（概念）：**
```cpp
void mad(
    __cc__ CType *c,        // 输出：L0C 中的 C 矩阵
    __ca__ AType *a,        // 输入：L0A 中的 A 矩阵
    __cb__ BType *b,        // 输入：L0B 中的 B 矩阵
    uint16_t m,             // M 维度
    uint16_t k,             // K 维度
    uint16_t n,             // N 维度
    uint16_t init_val,      // 初始值（通常为 0）
    bool kDirectionAlign,   // K 方向对齐标志
    bool cmatrixSource,     // 是否使用 C 作为源（ACC 时 false）
    bool cmatrixInitVal     // 是否初始化 C（TMATMUL=true, TMATMUL_ACC=false）
);
```

**数学语义：**
- **TMATMUL** (`cmatrixInitVal = true`)：
  ```
  C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]
  ```

- **TMATMUL_ACC** (`cmatrixInitVal = false`)：
  ```
  C_out[i,j] = C_in[i,j] + Σ(k=0 to K-1) A[i,k] * B[k,j]
  ```

## 3. 关键宏和函数说明

### 3.1 `MAP_INSTR_IMPL` 宏

**定义位置：** `pto-isa/include/pto/common/pto_instr.hpp`

```cpp
#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)
```

**作用：**
- 将 `TMATMUL(...)` 映射到 `TMATMUL_IMPL(...)`
- 将 `TMATMUL_ACC(...)` 映射到 `TMATMUL_ACC_IMPL(...)`

### 3.2 `__cce_get_tile_ptr` 函数

**作用：** 获取 tile 对象的实际数据指针，指向 L0A/L0B/L0C 的内存地址。

### 3.3 `mad` 函数

**性质：** 编译器内置函数（compiler intrinsic）

**实现：** 由 NPU 编译器（CCE）直接生成硬件指令，不是普通的 C++ 函数。

**硬件指令：** 生成 NPU 的 MAD（Matrix Multiply-Accumulate）硬件指令。

## 4. 实际使用示例

在 `gemm_performance_kernel.cpp` 中：

```27:35:pto-isa/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp
template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile, uint32_t k)
{
    if (k == 0) {
        TMATMUL(cTile, aTile, bTile);      // 第一次：初始化
    } else {
        TMATMUL_ACC(cTile, cTile, aTile, bTile);  // 后续：累加
    }
}
```

**执行流程：**
1. `k == 0`：调用 `TMATMUL` → 初始化 C = A * B
2. `k > 0`：调用 `TMATMUL_ACC` → 累加 C = C + A * B

## 5. 查看实现的路径

### 5.1 C++ 接口层
- **文件：** `pto-isa/include/pto/common/pto_instr.hpp`
- **行号：** 302-315

### 5.2 A2A3 平台实现层
- **文件：** `pto-isa/include/pto/npu/a2a3/TMatmul.hpp`
- **行号：**
  - `TMATMUL_IMPL`：88-97
  - `TMATMUL_ACC_IMPL`：100-110
  - `TMatmul` 函数：29-40

### 5.3 文档
- **TMATMUL：** `pto-isa/docs/isa/TMATMUL.md`
- **TMATMUL_ACC：** `pto-isa/docs/isa/TMATMUL_ACC.md`

### 5.4 硬件指令层

**`mad` 函数：**
- **性质：** 编译器内置函数（intrinsic）
- **实现：** 由 NPU 编译器（CCE）在编译时生成硬件指令
- **硬件指令：** NPU 的 MAD（Matrix Multiply-Accumulate）指令
- **查看方式：** 
  - 编译后的汇编代码
  - NPU 指令集文档（不在代码仓库中，属于硬件文档）

## 6. 关键区别

| 操作 | cmatrixInitVal | cmatrixSource | 数学公式 |
|------|---------------|---------------|----------|
| **TMATMUL** | `true` | `false` | `C = A * B` |
| **TMATMUL_ACC** | `false` | `false` | `C_out = C_in + A * B` |

## 7. 内存位置

操作必须在以下内存位置：

- **A 矩阵（aTile）：** L0A（`__ca__`）
- **B 矩阵（bTile）：** L0B（`__cb__`）
- **C 矩阵（cTile）：** L0C（`__cc__`）

这些都是每个 AICore 独立的 L0 缓存。

## 8. 总结

**完整的调用链：**

```
TMATMUL(cTile, aTile, bTile)
  → TMATMUL_IMPL(cTile, aTile, bTile)  [a2a3/TMatmul.hpp:88]
    → TMatmul(..., cmatrixInitVal=true)  [a2a3/TMatmul.hpp:39]
      → mad(c, a, b, m, k, n, ...)  [硬件内联函数]
        → NPU MAD 硬件指令 [编译时生成]
```

**查看建议：**

1. **用户接口：** `pto-isa/include/pto/common/pto_instr.hpp`
2. **平台实现：** `pto-isa/include/pto/npu/a2a3/TMatmul.hpp`
3. **文档：** `pto-isa/docs/isa/TMATMUL.md` 和 `TMATMUL_ACC.md`
4. **硬件指令：** `mad` 函数是编译器内置函数，查看编译后的汇编代码或硬件文档

