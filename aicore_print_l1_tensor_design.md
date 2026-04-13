# AICore Print L1 Tensor 打印方案设计

## 1 背景与问题

### 1.1 当前 API 覆盖情况

`aicore_print.h` 当前提供以下 tensor 打印 API：

| API | 地址空间 | 指针类型 | 可直接打印？ |
|-----|---------|---------|------------|
| `AiCorePrintGmTensor` | GM | `__gm__ const T*` | ✓ Scalar 可直接读写 GM ring buffer |
| `AiCorePrintUbTensor` | UB | `__ubuf__ const T*` | ✓ Scalar 可直接读写 GM ring buffer |
| `AiCorePrintL1Tensor` | L1 | `__cbuf__ const T*` | **✗ Scalar 无法将 L1 数据直接写入 GM ring buffer** |

### 1.2 核心问题：L1 数据无法直接写入 print ring buffer

当前 print 机制的工作流程：

```
AICore Scalar Processor:
  data[i]                                    → 从源地址空间读取值到寄存器
  Encode(type, val_bytes, fmt_bytes)         → 将值逐字节写入 GM ring buffer
  Sync()                                     → dcci 刷 cache line 到 GM
```

这对 GM 和 UB 数据可行，因为 Scalar Processor 可以同时读源数据、写 GM ring buffer。但 **L1（`__cbuf__`）是 Cube Engine 专用的数据缓冲区**，数据必须通过 DMA 引擎（`copy_cbuf_to_gm`）搬运到 GM，不能由 Scalar Processor 直接按字节搬移。

因此，打印 L1 tensor 必须经过 **L1 → GM 暂存 → 打印** 的三步流程。

### 1.3 实际场景

以下为编译生成的 kernel 代码片段（`kernel_aicore/TENSOR_s0_..._aic.cpp`）：

```cpp
// L1 缓冲区声明
float __cbuf__ *L1_S0_E4096 = (float __cbuf__ *)get_imm(0x0);   // Cube Engine L1 缓冲区
float *L1_S0_E4096_T = (float *)get_imm(0x0);                    // TileTensor 使用

// TileTensor 构造
L1TileTensorFP32Dim2_0 l1Tensor_0((uint64_t)L1_S0_E4096_T, (Shape2Dim(dim0, dim1)));

// 从 GM 加载数据到 L1（ND2NZ 分形格式）
TLoad<CopyInMode::ND2NZ, PaddingMode::NO_PADDING>(l1Tensor_0, gmTensor_1, ...);
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

// ❌ 当前只能打印 GM 数据，无法打印 L1 数据
AiCorePrintGmTensor(param->ctx, (__gm__ float*)gmTensor_1.GetAddr(), 1024, 0);

// ✗ 无法做到：
// AiCorePrintL1Tensor(param->ctx, (__cbuf__ float*)l1Tensor_0.GetAddr(), 1024, 0, staging, "l1_data");
```

---

## 2 关键技术约束

### 2.1 L1 → GM 的 DMA 搬运机制

L1 到 GM 的数据搬运通过 DMA 引擎 `copy_cbuf_to_gm` 实现（定义于 `arch32/dynamic/cube_dyn.h`）：

```cpp
// 底层 DMA intrinsic（CCE 编译器内置）
copy_cbuf_to_gm(__gm__ GMT* dst, __cbuf__ L1T* src,
                int sid, uint16_t nBurst, uint16_t lenBurst,
                uint16_t srcStride, uint16_t dstStride);
```

**参数说明**：
- DMA 以 32 字节（`BLOCK_SIZE`）为最小搬运单位
- `nBurst`：突发传输次数
- `lenBurst`：每次突发的长度（32B 块数，或 fallback 模式下的字节数）
- 运行于 `PIPE_MTE3` 流水线

**高层封装**（动态版本，运行时参数）：

```cpp
// tileop/arch32/dynamic/cube_dyn.h:675
template <typename GMT, typename L1T>
TILEOP void DynL1CopyOutND(
    __gm__ GMT* dst, __cbuf__ L1T* src,
    unsigned TShape0, unsigned TShape1,     // 源形状
    unsigned GmShape0, unsigned GmShape1,   // GM 目标形状
    int reserved)
```

### 2.2 GM 暂存缓冲区来源：复用 Workspace + rawTensorDesc 机制

AICore 内核无法动态分配 GM 内存。所有 GM 缓冲区必须由 Host 端预分配。

**方案选择**：本方案**复用现有的 workspace + `rawTensorDesc` 机制**传递 staging buffer 地址，无需扩展 `shakeBuffer` 或修改 Host 侧调度架构。

**现有 rawTensorDesc 机制回顾**（`aicore_runtime.h`）：

```cpp
struct DevRawTensorDesc {
    uint32_t location;       // 0=LOCAL(workspace), 1=INCAST, 2=OUTCAST
    uint32_t offsetOrIndex;  // workspace 偏移 or tensor 数组索引
};

// kernel 内获取 tensor 地址
INLINE uint64_t GetTensorAddr(CoreFuncParam* ctx, int idx)
{
    auto func = ctx->funcData;
    auto desc = &func->rawTensorDesc[ctx->opAttrs[idx]];
    if (desc->location == RAW_TENSOR_LOCATION_LOCAL)
        return func->workspaceAddr + desc->offsetOrIndex;  // workspace 偏移
    else
        return func->rawTensorAddr[desc->offsetOrIndex] & RAW_TENSOR_ADDR_MASK;
}
```

**方案核心思路**：

1. **CodeGen 阶段**：当检测到 kernel 中存在 L1 tensor 打印调用时，在 workspace 中分配一块 GM 暂存区域（staging slot）。
2. **将 staging slot 描述为一个 `location=LOCAL` 的 rawTensor**，通过已有的 `rawTensorDesc` 数组传递给 kernel。
3. **Kernel 内通过 `GetTensorAddr(param, staging_idx)` 获取 staging 地址**，传入 `AiCorePrintL1Tensor` API。

**优势**：
- **Host 侧零改动**：无需修改 `aicpu_common.h`、`aicore_hal.h`、`aicore_manager.h`、`device_sche.h` 等调度层文件。
- **按需分配**：只有实际包含 L1 打印调用的 kernel 才分配 staging，未使用时零开销。
- **大小灵活**：staging 大小由 CodeGen 根据最大待打印 L1 tensor 尺寸计算，不受固定值限制。
- **复用成熟机制**：`rawTensorDesc` + `workspaceAddr` 已在 kernel 中广泛使用（所有 local tensor 都走这个路径），稳定可靠。

### 2.3 流水线同步

DMA 搬运（`PIPE_MTE3`）与 Scalar 打印（`PIPE_S`）是不同流水线，需要通过 `set_flag` / `wait_flag` 同步：

```
DMA 搬运（PIPE_MTE3）         Scalar 打印（PIPE_S）
┌──────────────────┐          ┌──────────────────┐
│ copy_cbuf_to_gm  │          │                  │
│ L1 → GM staging  │          │                  │
└────────┬─────────┘          │                  │
         │                     │                  │
    set_flag(MTE3→S) ────────→│                  │
         │                     │  wait_flag       │
         │                     │  AiCorePrintGm   │
         │                     │  GM staging → ring│
         │                     └──────────────────┘
```

---

## 3 整体流程

```
┌─────────────── AICore Kernel ───────────────┐
│                                               │
│  0. 获取 staging buffer 地址                  │
│     GetTensorAddr(param, staging_idx)         │
│     → workspaceAddr + desc.offsetOrIndex      │
│                                               │
│  1. DMA: L1 data → GM staging buffer         │
│     copy_cbuf_to_gm(staging, l1_data, ...)   │
│                                               │
│  2. Sync: set_flag(MTE3→S) + wait_flag       │
│                                               │
│  3. Print: AiCorePrintGmTensorNamed(staging)  │
│     Scalar 逐值读 GM staging → 编码到 ring buf│
│                                               │
└───────────────────────────────────────────────┘
         │
         │ Host reads ring buffer
         ▼
┌─────────────── Host Side ────────────────────┐
│  DumpAicoreLog() → logger_.Read(buf)         │
│  → DEV_INFO("core-0 tensor 'l1_data' ...")   │
└───────────────────────────────────────────────┘
```

---

## 4 CodeGen 阶段变更

### 4.1 Workspace 中分配 staging slot

当 CodeGen 检测到 kernel 中存在 L1 tensor 打印调用（即调用了 `AiCorePrintL1Tensor`）时，在 workspace 中预留一块连续的 GM 暂存区域：

```
Workspace Layout:
┌──────────────────────────────────┐
│  ... 其他 local tensor slots ...  │
├──────────────────────────────────┤
│  L1_PRINT_STAGING slot           │  ← 新增，仅在存在 L1 打印时分配
│  (大小 = max_elem_count * sizeof(T)) │
├──────────────────────────────────┤
│  ... 后续 slots ...               │
└──────────────────────────────────┘
```

**分配策略**：
- `max_elem_count` = 所有 L1 打印调用中 `(end - begin)` 的最大值。
- 若无法静态确定（如打印范围为运行时变量），使用保守上界（如 L1 总容量 / sizeof(T)）。
- Staging buffer 大小向上取整到 32 字节对齐（`copy_cbuf_to_gm` 的最小搬运单位）。

### 4.2 rawTensorDesc 注册

将 staging slot 注册为一个 `location=LOCAL` 的 rawTensor：

```cpp
// CodeGen 在构造 DynFuncData 时，追加一个 rawTensorDesc 条目：
DevRawTensorDesc stagingDesc;
stagingDesc.location = RAW_TENSOR_LOCATION_LOCAL;  // 0
stagingDesc.offsetOrIndex = stagingSlotOffset;      // workspace 内偏移

// 对应的 opAttrs 中引用此 staging desc 的索引
// 例如 staging_desc_index = rawTensorDescSize - 1（追加在末尾）
```

### 4.3 生成 staging 地址获取代码

在生成的 kernel 代码中，staging 地址通过标准 `GetTensorAddr` 获取：

```cpp
// 由 CodeGen 自动生成的代码
constexpr int STAGING_RAW_TENSOR_IDX = <codegen_assigned_index>;
__gm__ float* staging = (__gm__ float*)GetTensorAddr(param, STAGING_RAW_TENSOR_IDX);

// 然后传入 L1 打印 API
AiCorePrintL1Tensor(param->ctx, (__cbuf__ float*)l1Tensor_0.GetAddr(),
                     1024, 0, staging, "l1_after_load");
```

### 4.4 无 L1 打印时的行为

当 kernel 中不存在任何 L1 打印调用时，**不分配 staging slot**，workspace 大小、rawTensorDesc 数组均无变化，零开销。

---

## 5 Kernel 侧 API 设计

### 5.1 L1 原始 DMA 搬运函数

```cpp
// aicore_print.h 新增（在 #if IS_AICORE 块内）

/**
 * 将 L1 数据通过 DMA 搬运到 GM 暂存缓冲区。
 * 内部使用 copy_cbuf_to_gm，支持任意元素类型和数量。
 *
 * @param dst       GM 目标地址
 * @param src       L1 源地址
 * @param count     元素数量
 *
 * @note DMA 最小搬运单位为 32 字节，不足部分会向上取整，
 *       拷贝的数据可能比请求的略多（尾部填充字节），不影响正确性。
 */
template <typename T>
__aicore__ void L1RawCopyToGM(__gm__ T* dst, __cbuf__ const T* src, int64_t count)
{
    int64_t totalBytes = count * sizeof(T);
    if (totalBytes == 0) return;

    uint16_t nBurst;
    uint16_t lenBurst;

    if (totalBytes >= 32) {
        // 正常模式：以 32B 块为单位的突发传输
        nBurst = 1;
        lenBurst = (totalBytes + 31) / 32;  // 向上取整到 32B 边界
    } else {
        // Fallback 模式：直接按字节传输（与 DynL1CopyOutND 一致）
        nBurst = 1;
        lenBurst = static_cast<uint16_t>(totalBytes);
    }

    copy_cbuf_to_gm(dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*srcStride*/, 0 /*dstStride*/);
}
```

### 5.2 Copy-then-Print API

```cpp
/**
 * 打印 L1 (__cbuf__) tensor 数据（Copy-then-Print 方式）。
 *
 * 完整流程：
 *   1. DMA 搬运：L1 data → GM staging buffer
 *   2. 流水线同步：等待 MTE3 完成
 *   3. 逐值打印：从 GM staging 读值，编码到 print ring buffer
 *
 * @param ctx       LogContext 指针（来自 param->ctx）
 * @param data      L1 数据指针（__cbuf__ 地址空间）
 * @param end       打印结束索引（不包含）
 * @param begin     打印起始索引（默认 0）
 * @param staging   GM 暂存缓冲区（通过 workspace + rawTensorDesc 传入）
 * @param name      tensor 名称（__gm__ 字符串字面量）
 *
 * @note staging 缓冲区大小必须 >= (end - begin) * sizeof(T) 字节。
 * @note L1 数据经 ND2NZ 加载后为分形格式，索引为 L1 物理线性偏移。
 * @note 应在 TLoad + wait_flag 完成后调用。
 */
template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data,
                                 int64_t end, int64_t begin,
                                 __gm__ T* staging, __gm__ const char* name)
{
    int64_t count = end - begin;
    if (count <= 0) return;

    // Step 1: DMA L1 → GM staging
    L1RawCopyToGM(staging, data + begin, count);

    // Step 2: 同步 MTE3 → S（等待 DMA 完成）
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

    // Step 3: 从 GM staging 打印（复用现有 GM 打印 API）
    AiCorePrintGmTensorNamed<T>(ctx, staging, count, 0, name);
}

// 4 参数重载（无名称，走基本打印路径）
template <typename T>
INLINE void AiCorePrintL1Tensor(LogContext* ctx, __cbuf__ const T* data,
                                 int64_t end, int64_t begin,
                                 __gm__ T* staging)
{
    int64_t count = end - begin;
    if (count <= 0) return;

    L1RawCopyToGM(staging, data + begin, count);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);

    AiCorePrintGmTensor<T>(ctx, staging, count, 0);
}
```

---

## 6 使用示例

### 6.1 基本用法（Named，带名称和索引）

```cpp
extern "C" [aicore] void kernel_func(CoreFuncParam* param, ...) {
    float __cbuf__ *L1_S0_E4096 = (float __cbuf__ *)get_imm(0x0);
    // ...
    TLoad<CopyInMode::ND2NZ>(l1Tensor_0, gmTensor_1, ...);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // 通过 rawTensorDesc 获取 workspace 中的 staging buffer
    // staging_idx 由 CodeGen 在生成此调用时自动分配
    __gm__ float* staging = (__gm__ float*)GetTensorAddr(param, staging_idx);

    // 打印 L1 数据
    AiCorePrintL1Tensor(param->ctx,
        (__cbuf__ float*)l1Tensor_0.GetAddr(),  // L1 源
        1024, 0,                                  // 范围
        staging,                                  // GM 暂存
        "l1_after_load");                         // 名称
}
```

**输出**：
```
core-0 tensor 'l1_after_load', range=[0, 1024)
core-0 l1_after_load[0] 1.000000
core-0 l1_after_load[1] 2.000000
...
core-0 l1_after_load[1023] 1024.000000
```

### 6.2 无名称版本

```cpp
__gm__ float* staging = (__gm__ float*)GetTensorAddr(param, staging_idx);
AiCorePrintL1Tensor(param->ctx,
    (__cbuf__ float*)l1Tensor_0.GetAddr(),
    256, 0,
    staging);
```

**输出**：
```
core-0 tensor data, range=[0, 256)
core-0 1.000000
core-0 2.000000
...
```

### 6.3 打印子范围

```cpp
__gm__ float* staging = (__gm__ float*)GetTensorAddr(param, staging_idx);
AiCorePrintL1Tensor(param->ctx,
    (__cbuf__ float*)l1Tensor_0.GetAddr(),
    1024, 512,      // 仅打印 [512, 1024)
    staging,
    "l1_partial");
```

**输出**：
```
core-0 tensor 'l1_partial', range=[0, 512)
core-0 l1_partial[0] 513.000000
core-0 l1_partial[1] 514.000000
...
```

### 6.4 使用原始 `__cbuf__` 指针

```cpp
float __cbuf__ *L1_buf = (float __cbuf__ *)get_imm(0x0);

// TLoad 完成后 ...

__gm__ float* staging = (__gm__ float*)GetTensorAddr(param, staging_idx);
AiCorePrintL1Tensor(param->ctx,
    L1_buf,             // 直接使用 __cbuf__ 指针
    256, 0,
    staging,
    "l1_raw");
```

### 6.5 多次打印复用同一 staging buffer

当 kernel 中需要对多个 L1 tensor 或同一 tensor 的不同范围多次打印时，可复用同一个 staging buffer（只要 staging 大小 >= 每次打印的数据量）：

```cpp
__gm__ float* staging = (__gm__ float*)GetTensorAddr(param, staging_idx);

// 打印第一个 L1 tensor（1024 个元素）
AiCorePrintL1Tensor(param->ctx,
    (__cbuf__ float*)l1Tensor_0.GetAddr(), 1024, 0,
    staging, "l1_tensor_0");

// 打印第二个 L1 tensor（复用 staging，512 个元素）
AiCorePrintL1Tensor(param->ctx,
    (__cbuf__ float*)l1Tensor_1.GetAddr(), 512, 0,
    staging, "l1_tensor_1");
```

**注意**：多次打印会串行执行（每次都经过 DMA + sync），因此 staging buffer 在不同调用间天然不冲突。

---

## 7 暂存缓冲区传递机制

### 7.1 通过 Workspace + rawTensorDesc 传递

**完全复用现有 tensor 传递管线，无需扩展 shakeBuffer**：

```
CodeGen 阶段                           Kernel 运行时
┌─────────────────────┐                ┌──────────────────────────────────┐
│ Workspace 分配       │                │                                  │
│ ┌─────────────────┐ │                │  GetTensorAddr(param, idx)       │
│ │ local tensor 0  │ │   DynFuncData  │  → desc = rawTensorDesc[idx]    │
│ │ local tensor 1  │ │───────────────→│  → desc.location == LOCAL       │
│ │ ...             │ │  rawTensorDesc │  → workspaceAddr + desc.offset  │
│ │ L1 staging slot │ │   数组         │                                  │
│ │ (32B 对齐)      │ │                │  返回 __gm__ T* staging 地址    │
│ └─────────────────┘ │                └──────────────────────────────────┘
└─────────────────────┘
         │
         │ 与已有 print buffer 传递独立，互不干扰
         ▼
┌─────────────────────┐                ┌──────────────────────────────────┐
│ Host 分配            │   shakeBuffer  │ Kernel 使用                      │
│ print buffer        │───[5]─────────→│ AicoreLogger ring buffer         │
│ (16KB/core)         │                │ Init(buffer, 16384)              │
└─────────────────────┘                └──────────────────────────────────┘
```

**关键差异**：
- print buffer（ring buffer）通过 `shakeBuffer[5]` 传递（不变）。
- L1 staging buffer 通过 workspace + `rawTensorDesc` 传递（**新增，不占 shakeBuffer**）。

---

## 8 代码变更清单

| 文件 | 变更项 | 类型 | 改动范围 |
|------|--------|------|---------|
| `aicore_print.h` | 新增 `L1RawCopyToGM()` | 新增函数 | Kernel 侧 |
| `aicore_print.h` | 新增 `AiCorePrintL1Tensor()` (2 个重载) | 新增 API | Kernel 侧 |
| CodeGen（workspace 分配逻辑） | 检测 L1 打印调用，在 workspace 中分配 staging slot | 新增逻辑 | 编译侧 |
| CodeGen（rawTensorDesc 构造） | 将 staging slot 注册为 `location=LOCAL` 的 rawTensor | 新增逻辑 | 编译侧 |
| CodeGen（kernel 代码生成） | 生成 `GetTensorAddr(param, staging_idx)` 获取 staging 地址 | 修改生成代码 | 编译侧 |

**对比原方案的变更范围缩减**：

| 文件 | 原方案 | 本方案 |
|------|--------|--------|
| `aicpu_common.h` | 新增 `SHAK_BUF_L1_STAGING_INDEX`、`L1_PRINT_STAGING_SIZE` | **不变** |
| `device_sche.h` | `AicoreLogManager` 新增 staging 分配 | **不变** |
| `aicore_manager.h` | `SendDevTaskModel` 传递 staging 地址 | **不变** |
| `aicore_hal.h` | `InitTaskData` 新增 staging 参数 | **不变** |
| `aicore_entry.h` | `ExecuteContext` 新增 `l1StagingBuf` | **不变** |
| `aicore_print.h` | 新增 `L1RawCopyToGM()`、`AiCorePrintL1Tensor()` | 新增（同原方案） |
| CodeGen | 无 | 新增 staging 分配 + rawTensorDesc 注册 |

---

## 9 注意事项

### 9.1 ND2NZ 分形格式

当 L1 数据通过 `TLoad<CopyInMode::ND2NZ>` 加载时，数据在 L1 中以 **Nz 分形格式**存储。打印的索引均为 L1 物理线性偏移，不对应原始逻辑 ND 坐标。

**建议**：
- 验证 TLoad 物理写入是否正确 → 直接用本 API
- 验证逻辑值是否正确 → 打印 GM 源数据（`AiCorePrintGmTensor`）

### 9.2 DMA 对齐

`copy_cbuf_to_gm` 的最小搬运单位为 32 字节（`BLOCK_SIZE`）。当总字节数不足 32 时，进入 fallback 模式按字节搬运；当总字节数 >= 32 但非 32 整数倍时，实际拷贝量会略多于请求量（尾部填充）。这不影响数据正确性，但尾部的填充字节是无意义的。

### 9.3 流水线事件 ID

DMA 同步使用 `set_flag(PIPE_MTE3, PIPE_S, EVENT_IDx)`。需确保 `EVENT_IDx` 不与当前 kernel 中其他同步操作冲突。建议使用较大的 EVENT_ID（如 `EVENT_ID7` 或通过参数传入），或由 API 内部管理。

### 9.4 Buffer 容量

受 ring buffer 容量限制（16KB）。Named 路径下每种元素类型的容量参考：

| 元素类型 | 每值开销 + END | staging 最小需求 | 16KB ring buffer 可打印 |
|---------|-------------|----------------|-----------------------|
| float32 | 13 + 1 = 14B | count × 4B（32B 对齐） | ⌊(16360-26)/14⌋ = 1166 个 |
| float16 | 11 + 1 = 12B | count × 2B（32B 对齐） | ⌊(16360-26)/12⌋ = 1361 个 |
| bfloat16_t | 11 + 1 = 12B | count × 2B（32B 对齐） | ⌊(16360-26)/12⌋ = 1361 个 |
| float8_t | 10 + 1 = 11B | count × 1B（32B 对齐） | ⌊(16360-26)/11⌋ = 1485 个 |
| int64 | 17 + 1 = 18B | count × 8B（32B 对齐） | ⌊(16360-26)/18⌋ = 907 个 |

如果 L1 数据量超过 ring buffer 容量，ring buffer 会覆盖最早记录。Named 路径下每条数据自带 `name[index]` 前缀，即使头部丢失仍可定位。

### 9.5 TileTensor 地址获取

| TileTensor 特化 | `GetAddr()` 返回值 | 转换方式 |
|----------------|-------------------|---------|
| `TileTensor<T, LA, Hardware::GM>` | `T*` | `(__gm__ T*)tensor.GetAddr()` |
| `TileTensor<T, LA, Hardware::L1>` | `uint64_t` | `(__cbuf__ T*)tensor.GetAddr()` |
| `TileTensor<T, LA, Hardware::UB>` | `uint64_t` | `(__ubuf__ T*)tensor.GetAddr()` |

### 9.6 打印时机

L1 打印应在数据加载完成（`TLoad` 之后、`wait_flag` 之后）进行，确保 L1 数据已就绪：

```cpp
TLoad<CopyInMode::ND2NZ>(l1Tensor_0, gmTensor_1, ...);
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);  // 确保 L1 数据就绪

// ✓ 此时可以安全打印 L1 数据
__gm__ float* staging = (__gm__ float*)GetTensorAddr(param, staging_idx);
AiCorePrintL1Tensor(param->ctx,
    (__cbuf__ float*)l1Tensor_0.GetAddr(), count, 0, staging, "l1_data");
```

### 9.7 Staging Buffer 大小计算

CodeGen 阶段需计算 staging buffer 大小，策略如下：

1. **静态可确定打印范围**：`staging_size = max(所有 (end - begin) * sizeof(T))`，向上取整到 32 字节。
2. **打印范围为运行时变量**：保守使用 L1 缓冲区总大小作为 staging_size。
3. **同一 kernel 中多次 L1 打印**：所有调用共享一个 staging buffer，大小取最大值（因为是串行执行的，见 6.5 节）。

### 9.8 Workspace 容量影响

L1 staging buffer 占用 workspace 空间。对于典型的调试场景（打印 1024 个 float32），仅需 4KB staging。对于大 tensor 调试（如打印整个 L1 缓冲区），staging 大小可能达到数十 KB。需在 CodeGen 阶段验证 workspace 总大小不超过硬件限制，必要时给出编译告警。

---

## 10 方案对比与选型理由

### 10.1 备选方案对比

| 维度 | 方案 A：shakeBuffer 扩展 | **方案 D（本方案）：workspace + rawTensorDesc** |
|------|------------------------|----------------------------------------------|
| Host 侧改动 | 改 aicpu_common.h、aicore_hal.h、aicore_manager.h、device_sche.h | **零改动** |
| Kernel 入口改动 | 改 aicore_entry.h（ExecuteContext） | **零改动** |
| Kernel API 改动 | 新增 aicore_print.h 函数 | 新增 aicore_print.h 函数（同） |
| CodeGen 改动 | 无 | 新增 staging slot 分配 + rawTensorDesc 注册 |
| 内存效率 | 每个 core 固定 4KB，无论是否使用 | 按需分配，用多少给多少 |
| 大小灵活性 | 固定 4KB，需 Python 配置接口 | 由 CodeGen 自动计算 |
| 传递通道 | 扩展 shakeBuffer 数组（`int64_t[8]` 已满，需改结构体） | 复用已有 rawTensorDesc 机制 |
| 实现复杂度 | Host + Kernel 双侧改动 | Kernel 侧 + CodeGen 侧，Host 侧零改动 |
| shakeBuffer 溢出风险 | `KernelArgs.shakeBuffer[8]` 已满，扩展需改 `SHARED_BUFFER_SIZE` | 无此风险 |

### 10.2 选型理由

1. **最小改动原则**：Host 调度层是共享基础设施，修改影响面广、回归成本高。本方案将变更收敛在 Kernel API 层和 CodeGen 层，符合最小改动原则。
2. **复用成熟机制**：`rawTensorDesc` + `workspaceAddr` 已在 kernel 中广泛使用（所有 local tensor 都走这个路径），稳定可靠。
3. **按需分配**：staging 仅在实际需要时分配，不浪费 GM 内存。
4. **避免 shakeBuffer 扩展风险**：`KernelArgs` 结构体受 `SHARED_BUFFER_SIZE = 512` 限制，`shakeBuffer[8]` 已全部占用，扩展需要修改 `SHARED_BUFFER_SIZE` 常量并验证对所有调度路径的影响。
