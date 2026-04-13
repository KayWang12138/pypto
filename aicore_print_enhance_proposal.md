# AICore Print 增强方案：支持数据名与索引输出

## 1 现状问题分析

### 1.1 多阶段编码的根因

当前 `AiCoreLogF(ctx, "%f\n", val)` 的调用链产生两阶段编码，是 **printf 变参 API 设计的必然结果**：

```
AiCoreLogF(ctx, "%f\n", val)
  ├── __AiCorePrint(ctx, &fmt, val)   ← 阶段一：处理值参数
  │     → PrintFp32(fmt, val)
  │     → EncodeFloatType → Encode(FP32, val_bytes, 4, "%f\n", 2)
  │       编码内容: [FP32][valLen=4][float 4字节][fmtLen=3]["%f\0"]  = 12 字节
  │       *fmt 推进 2 位，指向 "\n"
  │
  └── ctx->Print(ctx, fmt)             ← 阶段二：处理尾部剩余文本
        → Encode(NORMAL, "\n", 1, "\n", 1) + Encode(END) + Sync()
          编码内容: [NORMAL][valLen=1]["\n"][fmtLen=2]["\n\0"][END]  = 9+1 字节
```

**为什么必须两阶段？**

对于变参调用 `AiCoreLogF(ctx, "%ld, %ld, %f\n", a, b, c)`：
- 阶段一执行 3 次（每个参数一次），每次消费一个 `%X` 格式说明符
- 阶段二执行 1 次，编码尾部剩余的 `"\n"` 并写入 END 标记

两阶段拆分是**按格式说明符逐段解析**的自然结果，不是冗余设计。

### 1.2 两阶段编码内容相似的原因

两阶段使用完全相同的 `Encode(NodeTy, val, valLen, fmt, fmtLen)` 模板：

```
[type:1B][valLen:2B][val:N字节][fmtLen:2B][fmt字符串+'\0']
```

| 对比项 | 阶段一（值编码） | 阶段二（尾部编码） |
|--------|-----------------|-------------------|
| type | FP32 | NORMAL |
| val | float 原始字节 (4B) | 字面文本字节 (1B "\n") |
| fmt | "%f" (3B) | "\n" (2B) |

**统一编码格式**使得 Host 侧 `Read()` 解码器可以用同一套逻辑处理所有类型：读 type → 读 val → 读 fmt → `snprintf(buf, fmt, val)` → 拼接输出。

### 1.3 能否合并为一阶段编码？

| 方案 | 编码方式 | 问题 |
|------|---------|------|
| 当前两阶段 | 每段 `[type][val][fmt]` | 灵活支持变参，但开销高 |
| 合并一阶段 | 在 AICore 端预先 snprintf 成完整字符串再编码 | ① AICore 无 snprintf 实现 ② 完整字符串可能很长，浪费 buffer |
| 紧凑编码（新 NodeTy） | 新增专用的紧凑类型，省去 fmt 字段 | **可行，已采用** |

---

## 2 需求：输出数据名与索引

### 2.1 当前输出格式

```
tensor data, range=[0, 1024)
1.000000
2.000000
3.000000
...
1024.000000
```

**缺陷**：
1. 无 tensor 名称，多个 tensor 打印时无法区分
2. 无索引，当 buffer 溢出导致数据丢失时，无法判断输出的值对应哪个 index

### 2.2 期望输出格式

```
tensor 'weight', range=[0, 1024)
weight[0] 1.000000
weight[1] 2.000000
weight[2] 3.000000
...
weight[1023] 1024.000000
```

---

## 3 方案设计

### 3.1 方案对比

#### 方案 A：直接修改格式串（简单但开销大）

```cpp
AiCoreLogF(ctx, "[%ld] %f\n", i, data[i]);
```

**每值开销**：INT 编码(18B) + FP32 编码(12B) + NORMAL+END(10B) = **40 字节/值**。

1024 个 float → ~41KB，即使 64KB buffer 也较紧张。不推荐。

#### 方案 B：紧凑编码（已采用）

核心思路：为批量 tensor 打印场景新增**专用紧凑编码类型**，将 index + value 合并编码，省去 fmt 字段。同时引入 `TENSOR_HEADER` 类型编码 tensor 名称。

1024 个 float → ~14.4KB，16KB buffer 即可容纳。

---

## 4 已实现的详细设计

### 4.1 新增编码类型

```cpp
// aicore_print.h:115
enum NodeTy { END, NORMAL, FP32, INT, CHAR, STRING, POINTER, BF16, FP16,
              TENSOR_HEADER, INDEXED_FP32, INDEXED_INT64, INDEXED_BF16, INDEXED_FP16 };
```

### 4.2 编码格式定义

#### TENSOR_HEADER（头部元信息）

```
┌──────────┬───────────────┬─────────────────┬────────────────┬──────────────┬──────────────┐
│ type: 1B │ nameLen: 2B   │ name: N字节+'\0' │ begin: 8B      │ end: 8B      │
│ (uint8)  │ (short LE)    │ (字符串)         │ (int64 LE)     │ (int64 LE)   │
└──────────┴───────────────┴─────────────────┴────────────────┴──────────────┘

示例 name="weight": 1 + 2 + 7 + 8 + 8 = 26 字节
```

Host 侧解码后缓存 tensor 名称到 `lastTensorName_`，供后续 INDEXED 数据行复用。

#### INDEXED 类型（紧凑数据行）

```
┌──────────┬───────────────┬────────────────┐
│ type: 1B │ index: 8B     │ value: 2~8B    │
│ (uint8)  │ (int64 LE)    │ (原始字节 LE)   │
└──────────┴───────────────┴────────────────┘

INDEXED_FP32:  1 + 8 + 4 = 13 字节
INDEXED_INT64: 1 + 8 + 8 = 17 字节
INDEXED_BF16:  1 + 8 + 2 = 11 字节
INDEXED_FP16:  1 + 8 + 2 = 11 字节
```

每条数据行紧跟 1 字节 `END` 标记，完整的一行数据：

| 类型 | 编码开销 + END | 当前开销 + END | 节省 |
|------|---------------|---------------|------|
| float32 | 13 + 1 = **14** | 21 + 1 = 22 | **36%** |
| int64 | 17 + 1 = **18** | 22 + 1 = 23 | **22%** |
| bf16 | 11 + 1 = **12** | 12 + 1 = 13 | **8%** |
| fp16 | 11 + 1 = **12** | 12 + 1 = 13 | **8%** |

### 4.3 AICore 侧实现

#### 4.3.1 AicoreLogger 新增公开方法

```cpp
// aicore_print.h:215-251 — 三个新方法

// 编码 tensor 头部（名称 + range）
__aicore__ void EncodeTensorHeader(__gm__ const char* name, int64_t begin, int64_t end);

// 编码单条紧凑数据行（index + value，无 fmt 字段）
__aicore__ void EncodeIndexed(NodeTy ty, int64_t index, const uint8_t* val, short valLen);

// 写入 END 标记
__aicore__ void EncodeEnd();
```

#### 4.3.2 环形缓冲区溢出跳过逻辑

`Encode(uint8_t)` 在 buffer 满时跳过旧记录，已扩展支持新类型（`aicore_print.h:579-612`）：

```cpp
__aicore__ void Encode(uint8_t val)
{
    if (head_ == tail_ + size_) {
        while (Read<uint8_t>(tail_) != END) {
            auto segType = Read<uint8_t>(tail_);
            tail_++;
            switch (segType) {
                case TENSOR_HEADER: {
                    auto nl = Read<short>(tail_);
                    tail_ += sizeof(short) + nl;   // nameLen + name+'\0'
                    tail_ += 8 + 8;                 // begin + end
                    break;
                }
                case INDEXED_FP32:  tail_ += 8 + 4; break;
                case INDEXED_INT64: tail_ += 8 + 8; break;
                case INDEXED_BF16:
                case INDEXED_FP16:  tail_ += 8 + 2; break;
                default:                               // 原有类型
                    tail_ += Read<short>(tail_) + sizeof(short);
                    tail_ += Read<short>(tail_) + sizeof(short);
                    break;
            }
        }
        tail_++;
    }
    volatile __gm__ uint8_t* p = &data_[head_++ % size_];
    *p = val;
}
```

#### 4.3.3 __AiCorePrintTensorImpl 带 name 的重载

新增函数重载，与原有 4 参数版本共存，保持向后兼容（`aicore_print.h:725-761`）：

```cpp
template <typename T, typename PtrT>
INLINE void __AiCorePrintTensorImpl(LogContext* ctx, PtrT data, int64_t end,
                                    int64_t begin, __gm__ const char* name)
{
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);

    // 1. 写入 TENSOR_HEADER + END + Sync
    logger->EncodeTensorHeader(name, begin, end);
    logger->EncodeEnd();
    logger->Sync();

    // 2. 逐元素写入 INDEXED_* + END + Sync
    for (int64_t i = begin; i < end; ++i) {
        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(data[i]);
            logger->EncodeIndexed(INDEXED_FP32, i, ...);
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(data[i]);
            logger->EncodeIndexed(INDEXED_INT64, i, ...);
        } else if constexpr (std::is_pointer_v<ElemT>) {
            // pointer 重用 INDEXED_INT64
        }
        // bf16 → INDEXED_BF16, fp16 → INDEXED_FP16
        logger->EncodeEnd();
        logger->Sync();
    }
}
```

**关键设计**：`name` 作为第 5 个参数，与原 4 参数签名（`begin` 有默认值）不冲突，C++ 重载决议正确。

### 4.4 外部 API

保持原有 API 不变，新增 `*Named` 变体：

```cpp
// 原有 API（向后兼容，走旧的 AiCoreLogF 编码路径）
template <typename T>
INLINE void AiCorePrintGmTensor(LogContext* ctx, __gm__ const T* data, int64_t end, int64_t begin = 0);

template <typename T>
INLINE void AiCorePrintUbTensor(LogContext* ctx, __ubuf__ const T* data, int64_t end, int64_t begin = 0);

// 新增 Named API（走紧凑编码路径）
template <typename T>
INLINE void AiCorePrintGmTensorNamed(LogContext* ctx, __gm__ const T* data,
                                      int64_t end, int64_t begin, __gm__ const char* name);

template <typename T>
INLINE void AiCorePrintUbTensorNamed(LogContext* ctx, __ubuf__ const T* data,
                                      int64_t end, int64_t begin, __ubuf__ const char* name);
```

**调用示例**：

```cpp
__gm__ const char weightName[] = "weight";
AiCorePrintGmTensorNamed<float>(param->ctx, gmTensor, 1024, 0, weightName);
```

### 4.5 Host 侧 Read 解码

#### 4.5.1 新增 lastTensorName_ 成员

```cpp
// aicore_print.h:654-656 — 仅 Host 侧
#ifdef __TILE_FWK_HOST__
    std::string lastTensorName_;
#endif
```

#### 4.5.2 Read() 方法改造

`Read()` 方法（`aicore_print.h:343-478`）采用两级 switch 结构：

1. **外层 switch**：按 `type` 分发到 TENSOR_HEADER / INDEXED_* / default 三大分支
2. **内层 switch**（default 分支）：原有 valLen+fmtLen 解码逻辑不变

```
Read() 读取流程：

┌── 读取 type 字节
│
├── TENSOR_HEADER ──→ 读 nameLen(2B) + name(NB) + begin(8B) + end(8B)
│                     缓存 name 到 lastTensorName_
│                     输出: "tensor 'weight', range=[0, 1024)\n"
│
├── INDEXED_FP32 ───→ 读 index(8B) + float(4B)
│                     输出: "weight[0] 1.000000\n"
│
├── INDEXED_INT64 ──→ 读 index(8B) + int64(8B)
│                     输出: "weight[0] 42\n"
│
├── INDEXED_BF16 ───→ 读 index(8B) + bf16(2B) → DecodeBf16
│                     输出: "weight[0] 1.500000\n"
│
├── INDEXED_FP16 ───→ 读 index(8B) + fp16(2B) → DecodeF16
│                     输出: "weight[0] 1.500000\n"
│
├── END ────────────→ 结束当前行，return size
│
└── default ────────→ 原有逻辑（valLen+val+fmtLen+fmt）
```

**关键**：INDEXED 类型无 valLen/fmtLen 字段，`tail_` 按固定长度跳过；原有类型走 `valLen + fmtLen` 变长跳过逻辑不变。

---

## 5 容量评估

### 5.1 打印 1024 个 float32 的 buffer 需求

以 name="weight"（6 字符）为例，TENSOR_HEADER 开销 = 1 + 2 + 7 + 8 + 8 = 26 字节。

| 方案 | 每值开销 | 1024 值 + header | 所需最小 buffer |
|------|---------|-----------------|----------------|
| 当前（无 index） | 21 + 1 = 22 | ~22.6 KB | 32KB |
| 方案 A（加 index，现有编码） | 40 + 1 = 41 | ~42 KB | 48KB+ |
| **已实现方案（紧凑编码，含 name+index）** | **13 + 1 = 14** | **~14.4 KB** | **16KB** |

### 5.2 不同 buffer 大小的容量极限

| Buffer 大小 | 紧凑编码可容纳 float 数量 | 当前方案可容纳数量 | 容量提升 |
|-------------|------------------------|------------------|---------|
| 16KB (16360) | ⌊(16360-26)/14⌋ = **1166** | ⌊(16360-81)/22⌋ = 740 | **+58%** |
| 32KB (32744) | ⌊(32744-26)/14⌋ = **2337** | ⌊(32744-81)/22⌋ = 1487 | **+57%** |
| 64KB (65512) | ⌊(65512-26)/14⌋ = **4677** | ⌊(65512-81)/22⌋ = 2977 | **+57%** |

---

## 6 输出示例

### 6.1 原有 API 输出（不变）

```cpp
AiCorePrintGmTensor<float>(ctx, data, 1024, 0);
```

```
tensor data, range=[0, 1024)
1.000000
2.000000
...
1024.000000
```

### 6.2 Named API 正常输出

```cpp
AiCorePrintGmTensorNamed<float>(ctx, data, 1024, 0, "weight");
```

```
tensor 'weight', range=[0, 1024)
weight[0] 1.000000
weight[1] 2.000000
weight[2] 3.000000
...
weight[1023] 1024.000000
```

### 6.3 Named API buffer 溢出时

即使头部和早期数据被环形缓冲区覆盖，剩余数据仍携带 name[index] 前缀：

```
weight[524] 525.000000
weight[525] 526.000000
...
weight[1023] 1024.000000
```

**即使头部丢失，每条数据的 `name[index]` 前缀仍可精确定位其在原始 tensor 中的位置**。

---

## 7 代码变更清单

| 文件 | 变更项 | 行号 |
|------|--------|------|
| `aicore_print.h` | NodeTy 枚举新增 TENSOR_HEADER + 4 个 INDEXED 类型 | :115-116 |
| `aicore_print.h` | AicoreLogger::EncodeTensorHeader() | :215-234 |
| `aicore_print.h` | AicoreLogger::EncodeIndexed() | :236-246 |
| `aicore_print.h` | AicoreLogger::EncodeEnd() | :248-251 |
| `aicore_print.h` | AicoreLogger::Encode(uint8_t) 溢出跳过逻辑扩展 | :579-612 |
| `aicore_print.h` | AicoreLogger::Read() 两级 switch 解码 | :343-478 |
| `aicore_print.h` | AicoreLogger::lastTensorName_ 成员 (host only) | :654-656 |
| `aicore_print.h` | __AiCorePrintTensorImpl 5 参数重载（紧凑编码路径） | :725-761 |
| `aicore_print.h` | AiCorePrintGmTensorNamed() API | :769-774 |
| `aicore_print.h` | AiCorePrintUbTensorNamed() API | :783-788 |

---

## 8 设计决策与注意事项

### 8.1 为什么 Named 走独立编码路径而非复用 AiCoreLogF？

| 对比 | AiCoreLogF 路径 | 紧凑编码路径 |
|------|----------------|-------------|
| 每值开销 | 22 字节（含 fmt 字段） | 14 字节（无 fmt 字段） |
| 含 index | 需额外 INT 编码 (+18B) | 内置 (0 额外) |
| 含 name | 每行重复编码 | 仅 header 编码一次 |
| buffer 利用率 | 低 | 高 |

AiCoreLogF 是通用 printf 通道，fmt 字段对已知格式的批量 tensor 数据是冗余的。

### 8.2 向后兼容性

- 原有 `AiCorePrintGmTensor` / `AiCorePrintUbTensor` 4 参数 API **签名与行为完全不变**
- 新增类型枚举值在原有值之后追加，不影响已有编码/解码逻辑
- `Encode(uint8_t)` 和 `Read()` 通过 switch 的 default 分支保持对原有类型的兼容

### 8.3 name 参数传递

`name` 是 `__gm__ const char*`（或 `__ubuf__ const char*`），指向 device GM 上的字符串常量。调用方式：

```cpp
__gm__ const char tensorName[] = "weight";   // 字符串常量在 GM 上
AiCorePrintGmTensorNamed<float>(ctx, data, 1024, 0, tensorName);
```

需确保字符串在 kernel 执行期间地址有效。

### 8.4 Sync 频率

当前实现每条数据行调用一次 `Sync()`（cache line flush）。如需进一步减少 AICore 侧开销，可改为每 N 条 Sync 一次，但需权衡：若 kernel 异常终止，未 Sync 的数据将丢失。

### 8.5 多 tensor 打印

`lastTensorName_` 在每次遇到 `TENSOR_HEADER` 时更新，因此交替打印多个 tensor 时能正确切换名称：

```
tensor 'weight', range=[0, 4)
weight[0] 1.000000
weight[1] 2.000000
weight[2] 3.000000
weight[3] 4.000000
tensor 'bias', range=[0, 4)
bias[0] 0.100000
bias[1] 0.200000
bias[2] 0.300000
bias[3] 0.400000
```

---

## 9 扩展：支持 FP8 数据打印（E4M3、E5M2、E8M0）

### 9.1 背景

昇腾 AI 处理器上存在多种 8-bit 浮点类型，需要明确区分。

#### 9.1.1 类型体系梳理

源码中涉及 FP8 相关的类型定义分散在两套头文件中，需要区分上层框架类型与底层 AICore 类型：

**底层 AICore 层**（`framework/include/tilefwk/data_type.h`）：

| 枚举值 | C 类型名 | 位宽 | CANN type | 位布局 | 说明 |
|--------|----------|------|-----------|--------|------|
| `DT_FP8` | `float8_t` | 8 bit | 28 | 1+4+3 | 昇腾编译器内置 float8（内部为 E4M3 格式） |
| `DT_FP8E4M3` | `float8_e4m3_t` | 8 bit | 36 | 1+4+3 | 显式 E4M3FN 格式 |
| `DT_FP8E5M2` | `float8_e5m2_t` | 8 bit | 35 | 1+5+2 | 显式 E5M2 格式，支持 Inf/NaN |
| `DT_FP8E8M0` | `float8_e8m0_t` | 8 bit | 37 | 1+7+0 | 纯指数格式（无尾数位），用于缩放因子 |

#### 9.1.2 `DT_FP8` 与显式格式类型的关系

`DT_FP8`（`float8_t`）是昇腾编译器内置的原生 float8 类型，拥有独立的枚举值和 CANN type code（28）。`DT_FP8E4M3`（`float8_e4m3_t`）、`DT_FP8E5M2`（`float8_e5m2_t`）、`DT_FP8E8M0`（`float8_e8m0_t`）是显式格式的 FP8 子类型，拥有各自的枚举值和 CANN type code（36/35/37）。

| 对比项 | `float8_t`（`DT_FP8`） | `float8_e4m3_t`（`DT_FP8E4M3`） | `float8_e5m2_t`（`DT_FP8E5M2`） | `float8_e8m0_t`（`DT_FP8E8M0`） |
|--------|------------------------|--------------------------------|--------------------------------|--------------------------------|
| 枚举值 | `DT_FP8` | `DT_FP8E4M3` | `DT_FP8E5M2` | `DT_FP8E8M0` |
| CANN type | 28 | 36 | 35 | 37 |
| 位布局 | 1+4+3 | 1+4+3 | 1+5+2 | 1+7+0 |

虽然在 `aicore_print.h` 中 `float8_t` 被定义为 `float8_e4m3_t` 的 typedef（在特定平台守卫内），且在 `fp8_convert.cpp` 中 `DT_FP8` 和 `DT_FP8E4M3` 都路由到相同的 E4M3 解码逻辑，但它们在框架层面是独立的类型。

#### 9.1.3 本方案的范围

本方案**只支持三种显式格式类型**：`float8_e4m3_t`（`DT_FP8E4M3`）、`float8_e5m2_t`（`DT_FP8E5M2`）、`float8_e8m0_t`（`DT_FP8E8M0`），**不支持** `float8_t`（`DT_FP8`）。

原因：
1. 三种显式格式类型覆盖了 E4M3、E5M2、E8M0 三种编码格式，功能上完全覆盖 `DT_FP8`（`DT_FP8` 内部就是 E4M3 格式）
2. 三种显式格式类型在 `data_type.h` 中拥有独立的 `DTYPE_DESC` 映射，是框架层的正式类型
3. 显式格式类型精度语义更清晰，用户在打印时能准确知道数据的编码格式

#### 9.1.4 平台可用性

三种显式格式类型仅在 `__DAV_V310` 平台上可用。需要引入编译期宏守卫：

```cpp
#ifdef __DAV_V310
#define __FP8_EXPLICIT_TYPES_AVAILABLE__ 1
#endif
```

这三种类型的所有 AICore 侧代码（类型分支、回调注册、编码）均需包裹在 `#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__` 中。

**重要注意事项**：
1. **类型定义守卫**：`data_type.h` 中的 `float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t` 类型定义也需要平台守卫，避免在非 V310 平台上编译错误
2. **编译期检查**：在使用这些类型前，应该有编译期检查确保平台支持
3. **运行时保护**：如果平台不支持，应该提供明确的错误信息而不是 aicore error

当前 `aicore_print.h` 的 `NodeTy` 枚举和 INDEXED 编码均未覆盖这三种 FP8 类型，需要扩展。

### 9.2 三种 FP8 格式详解与 BF16 对比

#### 9.2.1 位布局对比

```
BF16:     [S][EEEEEEEE][MMMMMMMMMMMMMMMM]     16 bit, bias=127
FP8 E4M3: [S][EEEE][MMM]                       8 bit, bias=7
FP8 E5M2: [S][EEEEE][MM]                       8 bit, bias=15
FP8 E8M0: [S][EEEEEEE]                         8 bit, bias=63 (无尾数位)
```

| 对比项 | BF16 | FP8 E4M3 | FP8 E5M2 | FP8 E8M0 |
|--------|------|----------|----------|----------|
| 字节大小 | 2 | 1 | 1 | 1 |
| 符号位 | 1 | 1 | 1 | 1 |
| 指数位 | 8 | 4 | 5 | 7 |
| 尾数位 | 7 | 3 | 2 | **0** |
| 指数偏置 | 127 | 7 | 15 | 63 |
| 最大正常值 | ±3.389e38 | ±240 (sat.) | ±57344 | ±2^63 |
| Inf/NaN | 支持 | 不支持（saturation） | 支持（exp=31） | 不支持 |
| 次正规值 | 支持 | 支持 | 支持 | 不适用 |
| Host 解码方式 | 左移 16 位 | 逐字段解析 | 逐字段解析 | 逐字段解析 |

#### 9.2.2 与 BF16 的核心差异

**为什么不能像 BF16 那样直接移位转换？**

BF16 到 FP32 的转换只需左移 16 位（`DecodeBf16`），因为 BF16 与 FP32 共享相同的指数偏置（127），BF16 就是 FP32 的高 16 位，直接移位后指数和尾数字段自然对齐。

三种 FP8 类型与 FP32 的指数偏置不同（7/15/63 vs 127），且尾数位宽差异大（3/2/0 位 vs 23 位），必须逐字段解析转换。尤其：

- **次正规值**（exp=0）没有隐含的 1 位，不能直接放入 FP32 尾数字段，必须先左移尾数直到隐含位出现（归一化），再移除隐含位、调整指数——这与 `DecodeF16` 的处理方式一致。
- **E8M0 无尾数位**，不存在次正规值问题，但需要特殊处理：所有值都是 2 的整数幂（或零）。

### 9.3 需求

1. 在 `NodeTy` 枚举中新增 6 个枚举值：`FP8E4M3`、`FP8E5M2`、`FP8E8M0` 及其 INDEXED 变体
2. 在 Host 侧新增三个解码函数：`DecodeFloat8E4M3()`、`DecodeFloat8E5M2()`、`DecodeFloat8E8M0()`
3. 在 AICore 侧 `LogContext` 中新增三个 Print 回调，并在 `__AiCorePrint` 中增加对应的类型分支
4. 在 `Encode(uint8_t)` 溢出跳过逻辑中增加新 INDEXED 类型的处理
5. 在 `__AiCorePrintTensorImpl`（非 Named 和 Named 路径）中增加三种类型的支持

### 9.4 方案设计

#### 9.4.1 新增枚举类型

```cpp
enum NodeTy { END, NORMAL, FP32, INT, CHAR, STRING, POINTER, BF16, FP16,
              TENSOR_HEADER,
              INDEXED_FP32, INDEXED_INT64, INDEXED_BF16, INDEXED_FP16,
              FP8E4M3, INDEXED_FP8E4M3,
              FP8E5M2, INDEXED_FP8E5M2,
              FP8E8M0, INDEXED_FP8E8M0,
              OVERFLOW_WARNING };
```

新增 6 个枚举值：

| 枚举值 | 用途 |
|--------|------|
| `FP8E4M3` | AiCoreLogF 通用 printf 路径中的 `float8_e4m3_t` 值编码 |
| `INDEXED_FP8E4M3` | 紧凑编码路径中的 `float8_e4m3_t` 值编码 |
| `FP8E5M2` | AiCoreLogF 通用 printf 路径中的 `float8_e5m2_t` 值编码 |
| `INDEXED_FP8E5M2` | 紧凑编码路径中的 `float8_e5m2_t` 值编码 |
| `FP8E8M0` | AiCoreLogF 通用 printf 路径中的 `float8_e8m0_t` 值编码 |
| `INDEXED_FP8E8M0` | 紧凑编码路径中的 `float8_e8m0_t` 值编码 |

> **命名说明**：枚举值使用 `FP8E4M3`/`FP8E5M2`/`FP8E8M0`，与源码 `data_type.h` 中的 `DT_FP8E4M3`/`DT_FP8E5M2`/`DT_FP8E8M0` 对应，类型可区分，避免与 `DT_FP8`（`float8_t`）混淆。

#### 9.4.2 解码函数（Host 侧）

三种类型需要三个独立的解码函数，均将 1 字节原始位解码为 FP32 值。

##### DecodeFloat8E4M3 — E4M3FN 格式

根据 `fp8_convert.cpp` 中 `Fp8E4M3ToFloat32()` 的实现：

```
位布局: [S][EEEE][MMM]     bias=7
次正规值(exp==0)：(-1)^sign × 2^(-6) × (mant/8)
正常值(exp==1..14)：(-1)^sign × 2^(exp-7) × (1 + mant/8)
exp==15：saturation max = (-1)^sign × 240.0
```

> **注意**：`fp8_convert.cpp:64-70` 中 `exp==15` 区域处理为 `sign * 240.0`（max_val），无论 mant 值如何。这意味着 `0x7E`（exp=15,mant=6）和 `0x7F`（exp=15,mant=7）都映射到 240.0，而非标准 OCP 规范中 `0x7F` 映射到 NaN。Host 侧解码须与此行为保持一致。

实现示例（位操作版本，无 `std::pow` 依赖）：

```cpp
#define FP8E4M3_SIGN_MASK       0x80u
#define FP8E4M3_EXP_MASK        0x78u
#define FP8E4M3_MANT_MASK       0x07u
#define FP8E4M3_SIGN_SHIFT      7
#define FP8E4M3_EXP_SHIFT       3
#define FP8E4M3_HIDDEN_BIT      0x08u
#define FP8E4M3_EXP_BIAS        7
#define FP8E4M3_EXP_SATURATE    0xFu
#define FP8E4M3_TO_FP32_MANT_SHIFT 20
#define FP8E4M3_SUBNORMAL_FP32_EXP_BASE (FP32_EXP_BIAS - (FP8E4M3_EXP_BIAS - 1))

INLINE float DecodeFloat8E4M3(uint8_t bits)
{
    uint32_t sign = (bits & FP8E4M3_SIGN_MASK) >> FP8E4M3_SIGN_SHIFT;
    uint32_t exp  = (bits & FP8E4M3_EXP_MASK) >> FP8E4M3_EXP_SHIFT;
    uint32_t mant = bits & FP8E4M3_MANT_MASK;

    uint32_t sign32 = sign << FP32_SIGN_SHIFT;

    // 正负零
    if (exp == 0 && mant == 0) {
        return SafeBitCast<float>(sign32);
    }

    // exp==15 → saturation max 240.0
    // 240.0 = (-1)^sign × 2^7 × 1.875 = 1.875 × 128
    // FP32: sign | (134 << 23) | 0x700000 = sign | 0x43070000
    // 验证: 0x43070000 = 0 10000110 00001110000000000000000
    //        exp=134, mant=0x700000=7<<20, value = 2^7 × (1+7/8) = 128×1.875 = 240
    if (exp == FP8E4M3_EXP_SATURATE) {
        uint32_t satBits = sign32 | ((FP32_EXP_BIAS + 7) << FP32_EXP_SHIFT)
                                  | (0x7u << FP8E4M3_TO_FP32_MANT_SHIFT);
        return SafeBitCast<float>(satBits);
    }

    uint32_t exp32;
    uint32_t mant32;
    if (exp == 0) {
        // 次正规值：归一化（与 DecodeF16 相同的模式）
        exp32 = FP8E4M3_SUBNORMAL_FP32_EXP_BASE;
        while ((mant & FP8E4M3_HIDDEN_BIT) == 0) {
            mant <<= 1;
            --exp32;
        }
        mant &= FP8E4M3_MANT_MASK;  // 移除隐含位
        mant32 = mant << FP8E4M3_TO_FP32_MANT_SHIFT;
    } else {
        // 正常值：调整指数偏置，左移尾数对齐 FP32 mant 字段
        exp32 = exp - FP8E4M3_EXP_BIAS + FP32_EXP_BIAS;
        mant32 = mant << FP8E4M3_TO_FP32_MANT_SHIFT;
    }

    return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT) | mant32);
}
```

##### DecodeFloat8E5M2 — E5M2 格式

根据 `fp8_convert.cpp` 中 `Fp8E5M2ToFloat32()` 的实现：

```
位布局: [S][EEEEE][MM]     bias=15
次正规值(exp==0)：(-1)^sign × 2^(-14) × (mant/4)
正常值(exp==1..30)：(-1)^sign × 2^(exp-15) × (1 + mant/4)
exp==31, mant==0：Inf
exp==31, mant!=0：NaN
```

E5M2 是三种格式中**唯一支持 Inf/NaN 的**。

宏定义与实现：

```cpp
#define FP8E5M2_SIGN_MASK       0x80u
#define FP8E5M2_EXP_MASK        0x7Cu
#define FP8E5M2_MANT_MASK       0x03u
#define FP8E5M2_SIGN_SHIFT      7
#define FP8E5M2_EXP_SHIFT       2
#define FP8E5M2_HIDDEN_BIT      0x04u
#define FP8E5M2_EXP_BIAS        15
#define FP8E5M2_EXP_INF_NAN     0x1Fu
#define FP8E5M2_TO_FP32_MANT_SHIFT 21
#define FP8E5M2_SUBNORMAL_FP32_EXP_BASE (FP32_EXP_BIAS - (FP8E5M2_EXP_BIAS - 1))

INLINE float DecodeFloat8E5M2(uint8_t bits)
{
    uint32_t sign = (bits & FP8E5M2_SIGN_MASK) >> FP8E5M2_SIGN_SHIFT;
    uint32_t exp  = (bits & FP8E5M2_EXP_MASK) >> FP8E5M2_EXP_SHIFT;
    uint32_t mant = bits & FP8E5M2_MANT_MASK;

    uint32_t sign32 = sign << FP32_SIGN_SHIFT;

    // 正负零
    if (exp == 0 && mant == 0) {
        return SafeBitCast<float>(sign32);
    }

    // Inf / NaN：exp=31, mant=0 → Inf; exp=31, mant!=0 → NaN
    if (exp == FP8E5M2_EXP_INF_NAN) {
        uint32_t exp32 = FP32_EXP_INF_NAN;   // 0xFF
        uint32_t mant32 = mant << FP8E5M2_TO_FP32_MANT_SHIFT;
        // mant==0 → Inf, mant!=0 → NaN
        return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT) | mant32);
    }

    uint32_t exp32;
    uint32_t mant32;
    if (exp == 0) {
        // 次正规值：归一化
        exp32 = FP8E5M2_SUBNORMAL_FP32_EXP_BASE;
        while ((mant & FP8E5M2_HIDDEN_BIT) == 0) {
            mant <<= 1;
            --exp32;
        }
        mant &= FP8E5M2_MANT_MASK;  // 移除隐含位
        mant32 = mant << FP8E5M2_TO_FP32_MANT_SHIFT;
    } else {
        // 正常值
        exp32 = exp - FP8E5M2_EXP_BIAS + FP32_EXP_BIAS;
        mant32 = mant << FP8E5M2_TO_FP32_MANT_SHIFT;
    }

    return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT) | mant32);
}
```

##### DecodeFloat8E8M0 — E8M0 纯指数格式

根据 `fp8_convert.cpp` 中 `Fp8E8M0ToFloat32()` 的实现：

```
位布局: [S][EEEEEEE]     bias=63, 无尾数位
值 = (-1)^sign × 2^(exp-63)
所有值都是 2 的整数幂（或零）
```

> **E8M0 的特殊性**：无尾数位意味着没有隐含的 1，也没有次正规值。值域为 [-2^63, ..., -2^-63, -0, +0, 2^-63, ..., 2^63]。不存在 Inf/NaN 的标准定义。这是三种格式中解码逻辑最简单的——无需处理次正规值归一化、隐含位等，直接映射指数偏置即可。

宏定义与实现：

```cpp
#define FP8E8M0_SIGN_MASK       0x80u
#define FP8E8M0_EXP_MASK        0x7Fu
#define FP8E8M0_SIGN_SHIFT      7
#define FP8E8M0_EXP_BIAS        63

INLINE float DecodeFloat8E8M0(uint8_t bits)
{
    uint32_t sign = (bits & FP8E8M0_SIGN_MASK) >> FP8E8M0_SIGN_SHIFT;
    uint32_t exp  = bits & FP8E8M0_EXP_MASK;

    // 正负零：exp=0 → ±0
    if (exp == 0) {
        uint32_t sign32 = sign << FP32_SIGN_SHIFT;
        return SafeBitCast<float>(sign32);
    }

    // 正常值：(-1)^sign × 2^(exp-63)
    // 直接构造 FP32 位模式：sign + biased_exp + mant=0
    uint32_t sign32 = sign << FP32_SIGN_SHIFT;
    uint32_t exp32 = exp - FP8E8M0_EXP_BIAS + FP32_EXP_BIAS;
    // 没有尾数位，mant32 = 0
    return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT));
}
```

> **注意**：当 `exp - 63 + 127` 的结果超出 FP32 的合法指数范围（0~254）时，FP32 位模式可能产生 Inf（exp32 >= 255）或次正规值（exp32 < 1）。当前 E8M0 的 exp 范围为 0~127，`exp32` 范围为 64~190，均在 FP32 正常值范围内，不会出现溢出。

#### 9.4.3 编码格式

三种类型均为 1 字节，编码格式相同。

##### 通用 printf 路径（FP8E4M3 / FP8E5M2 / FP8E8M0）

与 BF16/FP16 走相同编码模板，valLen=1（1 字节）：

```
[type:1B][valLen:2B][val:1B][fmtLen:2B][fmt字符串+'\0']

FP8E4M3/FP8E5M2/FP8E8M0: 1 + 2 + 1 + 2 + fmtLen = 6 + fmtLen 字节
```

##### 紧凑编码路径（INDEXED_FP8E4M3 / INDEXED_FP8E5M2 / INDEXED_FP8E8M0）

与 INDEXED_BF16/INDEXED_FP16 结构一致，value 为 1 字节：

```
┌──────────┬───────────────┬────────────────┐
│ type: 1B │ index: 8B     │ value: 1B      │
│ (uint8)  │ (int64 LE)    │ (原始字节 LE)   │
└──────────┴───────────────┴────────────────┘

INDEXED_FP8E4M3 / INDEXED_FP8E5M2 / INDEXED_FP8E8M0: 1 + 8 + 1 = 10 字节
```

| 类型 | 编码开销 + END | 与 INDEXED_BF16 对比 |
|------|---------------|---------------------|
| INDEXED_FP8E4M3 | 10 + 1 = **11** | 12 + 1 = 13（节省 2B） |
| INDEXED_FP8E5M2 | 10 + 1 = **11** | 12 + 1 = 13（节省 2B） |
| INDEXED_FP8E8M0 | 10 + 1 = **11** | 12 + 1 = 13（节省 2B） |

三种 FP8 类型比 BF16/FP16 节省 2 字节/值（1 字节 vs 2 字节原始值）。

### 9.5 AICore 侧实现

#### 9.5.1 平台守卫

在文件顶部追加：

```cpp
#ifdef __DAV_V310
#define __FP8_EXPLICIT_TYPES_AVAILABLE__ 1
#endif
```

#### 9.5.2 LogContext 新增三个 Print 回调

```cpp
struct LogContext {
    void (*PrintInt)(LogContext* ctx, __gm__ const char** fmt, int64_t val);
    void (*PrintFp32)(LogContext* ctx, __gm__ const char** fmt, float val);
    void (*PrintBf16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
    void (*PrintFp16)(LogContext* ctx, __gm__ const char** fmt, uint16_t rawBits);
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    void (*PrintFp8e4m3)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8e5m2)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
    void (*PrintFp8e8m0)(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits);
#endif
    void (*Print)(LogContext* ctx, __gm__ const char* fmt);
};
```

> **与 BF16 的对照**：BF16 的回调参数为 `uint16_t rawBits`（2 字节）。三种 FP8 类型均为 `uint8_t rawBits`（1 字节），因为三种类型虽然编码格式不同，但存储大小相同（1 字节）。

#### 9.5.3 AicoreLogger 新增静态方法与实例方法

三种类型的实现模式完全一致，仅 NodeTy 枚举值不同：

```cpp
// ===== FP8E4M3 =====
static __aicore__ void __PrintFp8e4m3(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
{
    auto self = reinterpret_cast<AicoreLogger*>(ctx);
    if (self) {
        self->PrintFp8e4m3(fmt, rawBits);
    }
}

__aicore__ void PrintFp8e4m3(__gm__ const char** fmt, uint8_t rawBits)
{
    EncodeFloatType(fmt, FP8E4M3, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
}

// ===== FP8E5M2 =====
static __aicore__ void __PrintFp8e5m2(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
{
    auto self = reinterpret_cast<AicoreLogger*>(ctx);
    if (self) {
        self->PrintFp8e5m2(fmt, rawBits);
    }
}

__aicore__ void PrintFp8e5m2(__gm__ const char** fmt, uint8_t rawBits)
{
    EncodeFloatType(fmt, FP8E5M2, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
}

// ===== FP8E8M0 =====
static __aicore__ void __PrintFp8e8m0(LogContext* ctx, __gm__ const char** fmt, uint8_t rawBits)
{
    auto self = reinterpret_cast<AicoreLogger*>(ctx);
    if (self) {
        self->PrintFp8e8m0(fmt, rawBits);
    }
}

__aicore__ void PrintFp8e8m0(__gm__ const char** fmt, uint8_t rawBits)
{
    EncodeFloatType(fmt, FP8E8M0, reinterpret_cast<uint8_t*>(&rawBits), sizeof(rawBits));
}
```

`Init()` 中注册回调：

```cpp
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    ctx.PrintFp8e4m3 = __PrintFp8e4m3;
    ctx.PrintFp8e5m2 = __PrintFp8e5m2;
    ctx.PrintFp8e8m0 = __PrintFp8e8m0;
#endif
```

#### 9.5.4 __AiCorePrint 增加三种类型分支

`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t` 在 `data_type.h` 中通过 `DTYPE_DESC` 映射为 AICore 编译器提供的内置类型，在 `#if IS_AICORE` 守卫内可直接使用：

```cpp
#if IS_AICORE
    // ... 现有 bfloat16_t / half 分支 ...
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
        ctx->PrintFp8e4m3(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e5m2_t>) {
        ctx->PrintFp8e5m2(ctx, fmt, SafeBitCast<uint8_t>(val));
    } else if constexpr (std::is_same_v<T, float8_e8m0_t>) {
        ctx->PrintFp8e8m0(ctx, fmt, SafeBitCast<uint8_t>(val));
#endif
#endif
```

#### 9.5.5 __AiCorePrintTensorImpl（非 Named 路径）增加三种类型

```cpp
#if IS_AICORE
    // ... 现有 bfloat16_t / half 分支 ...
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
        float v = static_cast<float>(data[i]);  // 先转换为 float
        AiCoreLogF(ctx, "%f\n", v);
    } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
        float v = static_cast<float>(data[i]);  // 先转换为 float
        AiCoreLogF(ctx, "%f\n", v);
    } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
        float v = static_cast<float>(data[i]);  // 先转换为 float
        AiCoreLogF(ctx, "%f\n", v);
#endif
#endif
```

**关键修复**：FP8 类型不能直接传递给 `%f` 格式，必须先转换为 float 类型。这是因为：
1. FP8 类型的位布局与 FP32 不同，直接传递会导致格式化错误
2. 编译器无法自动将 FP8 类型转换为 float，需要显式转换
3. 转换后的 float 值会被 Host 侧正确解码和显示

#### 9.5.6 __AiCorePrintTensorImpl（Named 紧凑路径）增加三种类型

**perElement 计算部分**：

```cpp
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
        perElement = 10 + 1;   // INDEXED_FP8E4M3 (10B) + END (1B) = 11
    } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
        perElement = 10 + 1;   // INDEXED_FP8E5M2 (10B) + END (1B) = 11
    } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
        perElement = 10 + 1;   // INDEXED_FP8E8M0 (10B) + END (1B) = 11
#endif
```

**逐元素编码部分**：

```cpp
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
        } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E4M3, i, &rawBits, sizeof(uint8_t));
        } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E5M2, i, &rawBits, sizeof(uint8_t));
        } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E8M0, i, &rawBits, sizeof(uint8_t));
#endif
```

#### 9.5.7 Encode(uint8_t) 溢出跳过逻辑扩展

```cpp
case INDEXED_FP8E4M3:
case INDEXED_FP8E5M2:
case INDEXED_FP8E8M0:
    tail_ += 8 + 1;   // index(8B) + value(1B)
    break;
```

> **注意**：非 INDEXED 的 `FP8E4M3`/`FP8E5M2`/`FP8E8M0` 类型走 `default` 分支的 `valLen + fmtLen` 变长跳过逻辑，无需新增 case。这是因为通用 printf 路径使用标准编码格式 `[type][valLen][val][fmtLen][fmt]`，与 BF16/FP16 的非 INDEXED 处理方式一致。

### 9.6 Host 侧 Read 解码扩展

#### 9.6.1 通用 printf 路径

在 `Read()` 的内层 switch（`default` 分支内）中新增三个 case：

```cpp
case FP8E4M3: {
    uint8_t bits = Read<uint8_t>(valOff);
    float fv = DecodeFloat8E4M3(bits);
    n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
    break;
}
case FP8E5M2: {
    uint8_t bits = Read<uint8_t>(valOff);
    float fv = DecodeFloat8E5M2(bits);
    n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
    break;
}
case FP8E8M0: {
    uint8_t bits = Read<uint8_t>(valOff);
    float fv = DecodeFloat8E8M0(bits);
    n = snprintf_s(buf, maxSize, maxSize - 1, fmt.c_str(), fv);
    break;
}
```

#### 9.6.2 紧凑编码路径

在 `Read()` 的外层 switch 中新增三个 case：

```cpp
case INDEXED_FP8E4M3: {
    int64_t idx = Read<int64_t>(tail_);
    tail_ += 8;
    uint8_t bits = Read<uint8_t>(tail_);
    tail_ += 1;
    float fv = DecodeFloat8E4M3(bits);
    if (maxSize > 0) {
        n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                       lastTensorName_.c_str(), idx, fv);
    }
    break;
}
case INDEXED_FP8E5M2: {
    int64_t idx = Read<int64_t>(tail_);
    tail_ += 8;
    uint8_t bits = Read<uint8_t>(tail_);
    tail_ += 1;
    float fv = DecodeFloat8E5M2(bits);
    if (maxSize > 0) {
        n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                       lastTensorName_.c_str(), idx, fv);
    }
    break;
}
case INDEXED_FP8E8M0: {
    int64_t idx = Read<int64_t>(tail_);
    tail_ += 8;
    uint8_t bits = Read<uint8_t>(tail_);
    tail_ += 1;
    float fv = DecodeFloat8E8M0(bits);
    if (maxSize > 0) {
        n = snprintf_s(buf, maxSize, maxSize - 1, "%s[%ld] %f\n",
                       lastTensorName_.c_str(), idx, fv);
    }
    break;
}
```

### 9.7 容量评估

#### 打印 1024 个 FP8 值的 buffer 需求

以 name="weight"（6 字符）为例，TENSOR_HEADER = 26 字节。三种 FP8 类型编码开销相同。

| 方案 | 每值开销 | 1024 值 + header | 所需最小 buffer |
|------|---------|-----------------|----------------|
| 通用 printf（非 Named） | 10 + 1 = 11 | ~11.3 KB | 16KB |
| **紧凑编码（Named，含 name+index）** | **10 + 1 = 11** | **~11.3 KB** | **16KB** |

三种 FP8 类型均为 1 字节类型，比 FP16/BF16（2B）和 FP32（4B）更节省空间，紧凑编码下 16KB buffer 可容纳 ⌊(16360-26)/11⌋ = **1485 个 FP8 值**。

### 9.8 输出示例

#### 通用 printf 路径

```cpp
AiCorePrintGmTensor<float8_e4m3_t>(ctx, data, 1024, 0);
```

```
tensor data, range=[0, 1024)
1.500000
0.250000
...
240.000000
```

```cpp
AiCorePrintGmTensor<float8_e5m2_t>(ctx, data, 1024, 0);
```

```
tensor data, range=[0, 1024)
1.500000
0.250000
...
57344.000000
inf
nan
```

```cpp
AiCorePrintGmTensor<float8_e8m0_t>(ctx, data, 1024, 0);
```

```
tensor data, range=[0, 1024)
0.000000
0.000000
...
1.000000
2.000000
4.000000
```

> **E8M0 的输出特点**：所有非零值都是 2 的整数幂，以 `%f` 格式输出时看起来是整数（如 1.000000、2.000000、0.500000 等）。

#### Named 紧凑编码路径

```cpp
AiCorePrintGmTensorNamed<float8_e4m3_t>(ctx, data, 1024, 0, "activation_e4m3");
AiCorePrintGmTensorNamed<float8_e5m2_t>(ctx, data, 1024, 0, "scale_e5m2");
AiCorePrintGmTensorNamed<float8_e8m0_t>(ctx, data, 8, 0, "block_scale");
```

```
tensor 'activation_e4m3', range=[0, 1024)
activation_e4m3[0] 1.500000
activation_e4m3[1] 0.250000
...
activation_e4m3[1023] 240.000000
tensor 'scale_e5m2', range=[0, 1024)
scale_e5m2[0] 1.500000
scale_e5m2[1] 0.250000
...
scale_e5m2[1022] 57344.000000
scale_e5m2[1023] inf
tensor 'block_scale', range=[0, 8)
block_scale[0] 0.000000
block_scale[1] 0.000000
block_scale[2] 1.000000
block_scale[3] 2.000000
block_scale[4] 4.000000
block_scale[5] 0.500000
block_scale[6] 0.250000
block_scale[7] 8.000000
```

### 9.9 代码变更清单

| 文件 | 变更项 | 位置 |
|------|--------|------|
| `aicore_print.h` | 新增 `__FP8_EXPLICIT_TYPES_AVAILABLE__` 宏守卫 | 文件顶部 |
| `aicore_print.h` | 新增 FP8E4M3 / FP8E5M2 / FP8E8M0 宏定义常量 | 文件顶部（F16 宏之后） |
| `aicore_print.h` | 新增 `DecodeFloat8E4M3()` 解码函数 | 文件顶部（DecodeF16 之后） |
| `aicore_print.h` | 新增 `DecodeFloat8E5M2()` 解码函数 | 文件顶部（DecodeFloat8E4M3 之后） |
| `aicore_print.h` | 新增 `DecodeFloat8E8M0()` 解码函数 | 文件顶部（DecodeFloat8E5M2 之后） |
| `aicore_print.h` | NodeTy 枚举新增 6 个值 | `OVERFLOW_WARNING` 之前 |
| `aicore_print.h` | LogContext 新增 3 个 Print 回调 | `Print` 之前 |
| `aicore_print.h` | AicoreLogger 新增 3 个静态方法 + 3 个实例方法 | 现有 Print 方法之后 |
| `aicore_print.h` | AicoreLogger::Init() 注册 3 个回调 | 现有回调注册之后 |
| `aicore_print.h` | AicoreLogger::Encode(uint8_t) 溢出跳过新增 3 个 INDEXED 分支 | 现有 INDEXED 分支之后 |
| `aicore_print.h` | AicoreLogger::Read() switch 新增 6 个 case | 现有类型分支之后 |
| `aicore_print.h` | __AiCorePrint 新增 3 个类型分支 | 现有 half 分支之后 |
| `aicore_print.h` | __AiCorePrintTensorImpl (非 Named) 新增 3 个类型分支 | 现有 half 分支之后 |
| `aicore_print.h` | __AiCorePrintTensorImpl (Named) 新增 3 个类型分支 (perElement + 编码) | 现有 half 分支之后 |

### 9.10 已知问题与修复

#### 9.10.1 类型转换问题导致 aicore error

**问题描述**：
在非 Named 路径中，FP8 类型直接传递给 `AiCoreLogF` 的 `%f` 格式，导致 aicore error：

```cpp
// 错误的实现
AiCoreLogF(ctx, "%f\n", data[i]);  // data[i] 是 float8_e8m0_t
```

**根本原因**：
1. FP8 类型的位布局与 FP32 完全不同（8位 vs 32位）
2. 编译器无法自动将 FP8 类型转换为 float
3. 直接传递给 `%f` 格式会导致格式化错误和内存访问异常

**修复方案**：
显式转换为 float 类型后再传递：

```cpp
// 正确的实现
float v = static_cast<float>(data[i]);  // 先转换为 float
AiCoreLogF(ctx, "%f\n", v);
```

**影响范围**：
- `__AiCorePrintTensorImpl` 非 Named 路径中的三种 FP8 类型
- 仅影响 V310 平台上的 FP8 打印功能

#### 9.10.2 平台守卫不完整

**问题描述**：
FP8 类型定义在 `data_type.h` 中是全局的，但使用代码都有平台守卫，导致不一致：

```cpp
// data_type.h:50-52 - 缺少平台守卫
DTYPE_DESC(DT_FP8E4M3, 1, 8, true, float8_e4m3_t, 36)
DTYPE_DESC(DT_FP8E5M2, 1, 8, true, float8_e5m2_t, 35)
DTYPE_DESC(DT_FP8E8M0, 1, 8, true, float8_e8m0_t, 37)
```

**修复方案**：
在 `data_type.h` 中添加平台守卫和类型定义：

```cpp
// FP8 显式格式类型仅在 V310 平台可用
#ifdef __DAV_V310
typedef struct { uint8_t val; } float8_e4m3_t;
typedef struct { uint8_t val; } float8_e5m2_t;
typedef struct { uint8_t val; } float8_e8m0_t;
#endif
```

#### 9.10.3 解码函数边界检查不足

**问题描述**：
`DecodeFloat8E8M0` 函数缺少指数溢出检查，可能导致异常值：

```cpp
uint32_t exp32 = exp - FP8E8M0_EXP_BIAS + FP32_EXP_BIAS;
return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT));
```

**修复方案**：
添加边界检查，处理指数溢出情况：

```cpp
uint32_t exp32 = exp - FP8E8M0_EXP_BIAS + FP32_EXP_BIAS;

// 边界检查：确保 exp32 在 FP32 合法范围内
if (exp32 >= FP32_EXP_INF_NAN) {
    exp32 = FP32_EXP_INF_NAN;  // 返回 Inf
}

return SafeBitCast<float>(sign32 | (exp32 << FP32_EXP_SHIFT));
```

#### 9.10.4 编译期检查缺失

**问题描述**：
缺少编译期检查来确保 FP8 类型只在支持的平台使用。

**修复方案**：
添加 `static_assert` 检查：

```cpp
static_assert(__FP8_EXPLICIT_TYPES_AVAILABLE__,
           "float8_e8m0_t is only available on DAV_V310 platform");
```

### 9.11 设计决策与注意事项

#### 9.10.1 为什么只支持三种显式格式类型而不支持 `float8_t`（`DT_FP8`）

`float8_t`（`DT_FP8`）与 `float8_e4m3_t`（`DT_FP8E4M3`）在 `data_type.h` 中拥有**不同的枚举值**（`DT_FP8` vs `DT_FP8E4M3`）、**不同的 CANN type code**（28 vs 36），在框架层面是独立的类型。

本方案只支持三种显式格式类型的原因：
1. 三种显式格式类型覆盖了 E4M3、E5M2、E8M0 三种编码格式，功能上完全覆盖 `DT_FP8`（`DT_FP8` 内部就是 E4M3 格式）
2. 显式格式类型精度语义更清晰，打印时用户能准确知道数据的编码格式
3. 避免在 AICore 底层混淆 `DT_FP8` 和 `DT_FP8E4M3` 两个独立类型

#### 9.10.2 为什么为三种类型分别设置独立的 NodeTy 枚举值

虽然 E4M3 与 `DT_FP8` 共享相同的 E4M3FN 解码逻辑，但三种显式格式类型在框架层是独立的 DataType（`DT_FP8E4M3`、`DT_FP8E5M2`、`DT_FP8E8M0`），使用独立的枚举值有以下好处：

1. **类型可区分**：Host 侧解码时能准确知道原始值的格式，而非依赖猜测
2. **向前兼容**：未来如果不同格式的解码逻辑分化，不需要修改编码格式
3. **调试友好**：调试 buffer 内容时可直接从 type 字节看出原始数据格式

#### 9.10.3 AICore 侧类型可用性

`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t` 在 `data_type.h` 中通过 `DTYPE_DESC` 映射为 AICore 编译器提供的内置类型。但它们仅在 `__DAV_V310` 平台上可用，因此所有使用这些类型的代码必须包裹在 `#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__` 中。

实现时需确认：
1. 三种类型在 `__DAV_V310` AICore 编译模式下是否支持 `SafeBitCast` 到 `uint8_t`
2. 三种类型是否满足 `std::is_same_v` 模板匹配

#### 9.10.4 解码函数与 `fp8_convert.cpp` 的一致性

三个解码函数严格遵循 `fp8_convert.cpp` 中对应函数的解码逻辑：

| 类型 | `fp8_convert.cpp` 函数 | 本方案解码函数 | 关键行为 |
|------|------------------------|---------------|---------|
| E4M3 | `Fp8E4M3ToFloat32()` | `DecodeFloat8E4M3()` | exp=15 → saturation 240.0（非标准 OCP） |
| E5M2 | `Fp8E5M2ToFloat32()` | `DecodeFloat8E5M2()` | exp=31, mant=0 → Inf；exp=31, mant≠0 → NaN |
| E8M0 | `Fp8E8M0ToFloat32()` | `DecodeFloat8E8M0()` | 无尾数，纯 2 的幂次 |

#### 9.10.5 E4M3 的 saturation 行为

`fp8_convert.cpp` 中 `Fp8E4M3ToFloat32()` 对 `exp==15` 区域统一处理为 `sign * 240.0`（max_val），无论 mant 值如何。这意味着：
- `0x7E`（exp=15, mant=6）→ 240.0
- `0x7F`（exp=15, mant=7）→ 240.0（而非标准 OCP 中的 NaN）

这与标准 OCP E4M3FN 规范不同（标准规范中 exp=15,mant=0..6 为正常值，最大正常值 ±448，exp=15,mant=7 为 NaN）。本方案遵循 `fp8_convert.cpp` 的实际行为，确保 Host 侧解码与框架一致。

#### 9.10.6 E5M2 的 Inf/NaN 处理

E5M2 是三种格式中**唯一支持 Inf/NaN 的**：

```
exp=31, mant=0 → Inf（正/负取决于符号位）
exp=31, mant≠0 → NaN
```

Host 侧 `%f` 格式化时，Inf 输出为 `inf`/`-inf`，NaN 输出为 `nan`，用户可直观判断。

#### 9.10.7 E8M0 的特殊性

E8M0 格式**无尾数位**，所有非零值都是 2 的整数幂。这使得它：

1. **解码最简单**：无需处理次正规值归一化、隐含位等，直接映射指数偏置即可
2. **显示特殊**：以 `%f` 输出时，所有值看起来是整数或 2 的幂次（如 1.0、2.0、0.5、0.25 等）
3. **用途特定**：通常用作缩放因子（如 Block-FP8 中的 block scale），而非独立数据值

#### 9.10.8 精度显示

三种 FP8 类型精度有限（E4M3 有 3 位尾数，E5M2 有 2 位尾数，E8M0 无尾数），默认 `%f` 输出 6 位小数会显示大量无意义数字。可考虑：
- 使用 `%.4f` 或 `%g` 格式，减少显示噪音
- 或在 Host 侧解码时根据 NodeTy 类型自动选择合适的显示精度

此为显示层面的优化，不影响编码方案。

#### 9.10.9 常量命名风格一致性

所有解码函数的常量均使用 `#define` 命名常量，按类型前缀分组：

| 类型 | 前缀 | 示例 |
|------|------|------|
| FP8E4M3 | `FP8E4M3_` | `FP8E4M3_SIGN_MASK`、`FP8E4M3_EXP_BIAS` |
| FP8E5M2 | `FP8E5M2_` | `FP8E5M2_SIGN_MASK`、`FP8E5M2_EXP_BIAS` |
| FP8E8M0 | `FP8E8M0_` | `FP8E8M0_SIGN_MASK`、`FP8E8M0_EXP_BIAS` |
| 通用 FP32 | `FP32_` | `FP32_SIGN_SHIFT`、`FP32_EXP_BIAS`（已有） |

字段提取统一使用 `(bits & MASK) >> SHIFT` 模式，指数/尾数组装使用 `<< FP32_EXP_SHIFT`，避免硬编码魔法数字。与源码中 `DecodeF16`（`F16_SIGN_MASK` 等）的风格保持一致。

#### 9.10.10 三种 FP8 类型与 BF16 实现路径的完整对照

| 对比项 | BF16 | FP8E4M3 | FP8E5M2 | FP8E8M0 |
|--------|------|---------|---------|---------|
| `__AiCorePrint` 检测 | `is_same_v<T, bfloat16_t>` | `is_same_v<T, float8_e4m3_t>` | `is_same_v<T, float8_e5m2_t>` | `is_same_v<T, float8_e8m0_t>` |
| 原始位提取 | `SafeBitCast<uint16_t>` | `SafeBitCast<uint8_t>` | `SafeBitCast<uint8_t>` | `SafeBitCast<uint8_t>` |
| LogContext 回调参数 | `uint16_t rawBits` | `uint8_t rawBits` | `uint8_t rawBits` | `uint8_t rawBits` |
| 编码方式 | `EncodeFloatType(BF16, ...)` | `EncodeFloatType(FP8E4M3, ...)` | `EncodeFloatType(FP8E5M2, ...)` | `EncodeFloatType(FP8E8M0, ...)` |
| Host 解码复杂度 | 直接左移 16 位 | 逐字段解析 | 逐字段解析（含 Inf/NaN） | 直接映射指数（最简单） |
| Named 紧凑编码 | `INDEXED_BF16` (1+8+2=11B) | `INDEXED_FP8E4M3` (1+8+1=10B) | `INDEXED_FP8E5M2` (1+8+1=10B) | `INDEXED_FP8E8M0` (1+8+1=10B) |
| Encode 溢出跳过 (非 INDEXED) | `default` 分支 | `default` 分支 | `default` 分支 | `default` 分支 |
| Encode 溢出跳过 (INDEXED) | 专用 case `8 + 2` | 专用 case `8 + 1` | 专用 case `8 + 1` | 专用 case `8 + 1` |
| 平台守卫 | `#if IS_AICORE` | `__FP8_EXPLICIT_TYPES_AVAILABLE__` | `__FP8_EXPLICIT_TYPES_AVAILABLE__` | `__FP8_EXPLICIT_TYPES_AVAILABLE__` |

本方案严格遵循 BF16/FP16 的既有模式：AICore 侧通过 `__AiCorePrint` 的类型分支 + `SafeBitCast` 提取原始位，通用 printf 路径走 `EncodeFloatType` + `NodeTy` 编码，紧凑路径走 `EncodeIndexed`，Host 侧根据 `NodeTy` 选择对应解码函数。

---

## 10 扩展：Ring Buffer 溢出 Warning 机制

### 10.1 现状问题

当打印的 tensor 数据量超过 ring buffer 容量时，`Encode(uint8_t)` 会静默地跳过旧记录（`tail_` 前进），导致数据丢失。用户**无任何感知**，只能看到不完整的输出，却不知道：

1. 数据是否发生了丢失
2. 需要多大的 buffer 才能容纳全部数据
3. 在哪里修改 buffer 大小

#### 多 Tensor 累积溢出问题

Ring buffer 是**全局共享**的，多个 tensor 依次打印时会累积占用 buffer。即使单个 tensor 的编码量不超过 buffer 容量，多个 tensor 累积仍可能溢出。

**示例场景**（16KB buffer）：
- tensor1: 100 个 float32（编码量 ~1400 字节）
- tensor2: 100 个 float32（编码量 ~1400 字节）
- tensor3: 1000 个 float32（编码量 ~14000 字节）

单个 tensor 都不溢出（1000 < 1168），但累积 1200 个数据超过 buffer 容量，导致 tensor1 和 tensor2 的数据被覆盖。

### 10.2 需求

1. **检测累积溢出**：通过比较编码前后的 `tail_` 位置，准确检测是否发生了数据覆盖
2. **正常编码 + 追加 Warning**：无论是否溢出都正常编码数据（覆盖就覆盖），编码完成后若检测到覆盖则追加 warning 记录
3. **输出 Warning**：Host 侧解码 warning 记录时输出明确信息，包括 tensor 名称、编码量、buffer 状态变化、配置指引
4. **配置指引**：告诉用户在哪里修改 buffer 大小，修改后需重新编译安装

### 10.3 方案设计

#### 10.3.1 核心思路：通过 tail_ 变化检测覆盖

Ring buffer 的 `tail_` 指针在 `Encode(uint8_t)` 溢出时会前进（跳过旧记录）。通过记录编码前后的 `tail_` 值，可以准确判断是否发生了覆盖：

```
编码前：tail_ = 227  (tensor1 + tensor2 已编码)
编码后：tail_ = 14254 (tensor3 编码完成，tail 被推进)
覆盖检测：tailAfter > tailBefore → 发生了覆盖
```

**优势**：
- 准确检测多 tensor 累积溢出
- 无需维护全局计数器，利用现有 `head_` 和 `tail_` 状态
- 语义清晰：明确告知"此 tensor 的编码导致早期数据被覆盖"

#### 10.3.2 新增编码类型

```cpp
enum NodeTy { END, NORMAL, FP32, INT, CHAR, STRING, POINTER, BF16, FP16,
              TENSOR_HEADER,
              INDEXED_FP32, INDEXED_INT64, INDEXED_BF16, INDEXED_FP16,
              FP8E4M3, INDEXED_FP8E4M3,
              FP8E5M2, INDEXED_FP8E5M2,
              FP8E8M0, INDEXED_FP8E8M0,
              OVERFLOW_WARNING };
```

#### 10.3.3 OVERFLOW_WARNING 编码格式

```
┌──────────┬───────────────┬─────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
│ type: 1B │ nameLen: 2B   │ name: N字节+'\0' │ totalBytes: 8B │ bufferSize: 8B │ tailBefore: 8B │ tailAfter: 8B │
│ (uint8)  │ (short LE)    │ (字符串)         │ (int64 LE)     │ (int64 LE)     │ (int64 LE)     │ (int64 LE)     │
└──────────┴───────────────┴─────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘

示例 name="tensor3": 1 + 2 + 8 + 8 + 8 + 8 + 8 = 43 字节
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `nameLen` + `name` | short + char[] | tensor 名称 |
| `totalBytes` | int64 | 该 tensor 完整编码所需的总字节数（含 header + 所有数据行 + END 标记） |
| `bufferSize` | int64 | 当前 ring buffer 的有效数据区大小（`size_`，即 `PRINT_BUFFER_SIZE - sizeof(Remote)`） |
| `tailBefore` | int64 | 编码前的 tail_ 值（记录 buffer 状态） |
| `tailAfter` | int64 | 编码后的 tail_ 值（记录 buffer 状态） |

#### 10.3.4 AICore 侧实现

##### AicoreLogger 新增辅助方法

```cpp
// 获取 ring buffer 有效数据区大小
__aicore__ int64_t GetBufferSize() const { return size_; }

// 获取当前 tail_ 位置（用于检测覆盖）
__aicore__ int64_t GetTail() const { return tail_; }

// 编码溢出警告信息
template <typename NamePtrT>
__aicore__ void EncodeOverflowWarning(NamePtrT name, int64_t totalBytes, int64_t tailBefore, int64_t tailAfter)
{
    Encode(static_cast<uint8_t>(OVERFLOW_WARNING));
    short nameLen = 0;
    while (name[nameLen]) ++nameLen;
    nameLen += 1;
    auto nlBytes = reinterpret_cast<uint8_t*>(&nameLen);
    Encode(nlBytes[0]);
    Encode(nlBytes[1]);
    for (short i = 0; name[i]; ++i) {
        Encode(static_cast<uint8_t>(name[i]));
    }
    Encode('\0');
    auto tbBytes = reinterpret_cast<uint8_t*>(&totalBytes);
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
        Encode(tbBytes[i]);
    }
    auto bsBytes = reinterpret_cast<uint8_t*>(&size_);
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
        Encode(bsBytes[i]);
    }
    auto tbBeforeBytes = reinterpret_cast<uint8_t*>(&tailBefore);
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
        Encode(tbBeforeBytes[i]);
    }
    auto tbAfterBytes = reinterpret_cast<uint8_t*>(&tailAfter);
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
        Encode(tbAfterBytes[i]);
    }
}
```

##### __AiCorePrintTensorImpl（Named 路径）增加覆盖检测与末尾 Warning

```cpp
template <typename T, typename PtrT, typename NamePtrT>
INLINE void __AiCorePrintTensorImpl(LogContext* ctx, PtrT data, int64_t end,
                                    int64_t begin, NamePtrT name)
{
    using ElemT = std::remove_cv_t<T>;
    auto* logger = reinterpret_cast<AicoreLogger*>(ctx);

    // ===== 记录编码前的 buffer 状态 =====
    int64_t tailBefore = logger->GetTail();

    // ===== 预计算当前 tensor 的编码量 =====
    int64_t count = end - begin;

    // 1. 计算每元素编码开销（INDEXED_type + END）
    int64_t perElement = 0;
    if constexpr (std::is_floating_point_v<ElemT>) {
        perElement = 13 + 1;   // INDEXED_FP32 (13B) + END (1B) = 14

    } else if constexpr (std::is_integral_v<ElemT>) {
        perElement = 17 + 1;   // INDEXED_INT64 (17B) + END (1B) = 18
#if IS_AICORE
    } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
        perElement = 11 + 1;   // INDEXED_BF16 (11B) + END (1B) = 12
    } else if constexpr (std::is_same_v<ElemT, half>) {
        perElement = 11 + 1;   // INDEXED_FP16 (11B) + END (1B) = 12
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
    } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
        perElement = 10 + 1;   // INDEXED_FP8E4M3 (10B) + END (1B) = 11
    } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
        perElement = 10 + 1;   // INDEXED_FP8E5M2 (10B) + END (1B) = 11
    } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
        perElement = 10 + 1;   // INDEXED_FP8E8M0 (10B) + END (1B) = 11
#endif
#endif
    }

    // 2. 计算 nameLen
    int64_t nameLen = 0;
    while (name[nameLen]) ++nameLen;
    nameLen += 1;  // 含 '\0'

    // 3. 计算总字节数
    //    TENSOR_HEADER = type(1) + nameLen_field(2) + name(N) + begin(8) + end(8)
    //    完整编码 = TENSOR_HEADER + END + count × (INDEXED + END)
    int64_t headerSize = 1 + 2 + nameLen + 8 + 8;
    int64_t totalBytes = headerSize + 1 + (perElement * count);

    // ===== 正常编码路径（无论是否溢出都执行） =====

    // 1. 写入 TENSOR_HEADER + END + Sync
    logger->EncodeTensorHeader(name, begin, end);
    logger->EncodeEnd();
    logger->Sync();

    // 2. 逐元素写入 INDEXED_* + END + Sync
    for (int64_t i = begin; i < end; ++i) {
        if constexpr (std::is_floating_point_v<ElemT>) {
            float v = static_cast<float>(data[i]);
            logger->EncodeIndexed(INDEXED_FP32, i, reinterpret_cast<uint8_t*>(&v), sizeof(float));
        } else if constexpr (std::is_integral_v<ElemT>) {
            int64_t v = static_cast<int64_t>(data[i]);
            logger->EncodeIndexed(INDEXED_INT64, i, reinterpret_cast<uint8_t*>(&v), sizeof(int64_t));
        } else if constexpr (std::is_pointer_v<ElemT>) {
            int64_t v = reinterpret_cast<int64_t>(data[i]);
            logger->EncodeIndexed(INDEXED_INT64, i, reinterpret_cast<uint8_t*>(&v), sizeof(int64_t));
#if IS_AICORE
        } else if constexpr (std::is_same_v<ElemT, bfloat16_t>) {
            uint16_t rawBits = SafeBitCast<uint16_t>(data[i]);
            logger->EncodeIndexed(INDEXED_BF16, i, reinterpret_cast<uint8_t*>(&rawBits), sizeof(uint16_t));
        } else if constexpr (std::is_same_v<ElemT, half>) {
            uint16_t rawBits = SafeBitCast<uint16_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP16, i, reinterpret_cast<uint8_t*>(&rawBits), sizeof(uint16_t));
#ifdef __FP8_EXPLICIT_TYPES_AVAILABLE__
        } else if constexpr (std::is_same_v<ElemT, float8_e4m3_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E4M3, i, &rawBits, sizeof(uint8_t));
        } else if constexpr (std::is_same_v<ElemT, float8_e5m2_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E5M2, i, &rawBits, sizeof(uint8_t));
        } else if constexpr (std::is_same_v<ElemT, float8_e8m0_t>) {
            uint8_t rawBits = SafeBitCast<uint8_t>(data[i]);
            logger->EncodeIndexed(INDEXED_FP8E8M0, i, &rawBits, sizeof(uint8_t));
#endif
#endif
        }
        logger->EncodeEnd();
        logger->Sync();
    }

    // ===== 检查是否发生了覆盖 =====
    int64_t tailAfter = logger->GetTail();
    bool overflowOccurred = (tailAfter > tailBefore);

    // ===== 如果发生了覆盖，追加 Warning =====
    if (overflowOccurred) {
        logger->EncodeOverflowWarning(name, totalBytes, tailBefore, tailAfter);
        logger->EncodeEnd();
        logger->Sync();
    }
}
```

##### Encode(uint8_t) 溢出跳过逻辑扩展

```cpp
case OVERFLOW_WARNING: {
    auto nl = Read<short>(tail_);
    tail_ += sizeof(short) + nl;
    tail_ += 8 + 8 + 8 + 8;  // totalBytes + bufferSize + tailBefore + tailAfter
    break;
}
```

##### AicoreLogger 新增辅助方法

```cpp
// 获取 ring buffer 有效数据区大小，供预计算比较
__aicore__ int64_t GetBufferSize() const { return size_; }

// 编码溢出警告信息
template <typename NamePtrT>
__aicore__ void EncodeOverflowWarning(NamePtrT name, int64_t totalBytes)
{
    Encode(static_cast<uint8_t>(OVERFLOW_WARNING));
    short nameLen = 0;
    while (name[nameLen]) ++nameLen;
    nameLen += 1;
    auto nlBytes = reinterpret_cast<uint8_t*>(&nameLen);
    Encode(nlBytes[0]);
    Encode(nlBytes[1]);
    for (short i = 0; name[i]; ++i) {
        Encode(static_cast<uint8_t>(name[i]));
    }
    Encode('\0');
    auto tbBytes = reinterpret_cast<uint8_t*>(&totalBytes);
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
        Encode(tbBytes[i]);
    }
    auto bsBytes = reinterpret_cast<uint8_t*>(&size_);
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
        Encode(bsBytes[i]);
    }
}
```

##### Encode(uint8_t) 溢出跳过逻辑扩展

```cpp
case OVERFLOW_WARNING: {
    auto nl = Read<short>(tail_);
    tail_ += sizeof(short) + nl;
    tail_ += 8 + 8;    // totalBytes + bufferSize
    break;
}
```

#### 10.3.5 Host 侧 Read 解码扩展

```cpp
case OVERFLOW_WARNING: {
    auto nameLen = Read<short>(tail_);
    tail_ += sizeof(short);
    std::string name;
    for (short i = 0; i < nameLen - 1; ++i) {
        name += Read<char>(tail_++);
    }
    tail_++;  // skip '\0'
    auto totalBytes = Read<int64_t>(tail_);
    tail_ += 8;
    auto bufferSize = Read<int64_t>(tail_);
    tail_ += 8;
    auto tailBefore = Read<int64_t>(tail_);
    tail_ += 8;
    auto tailAfter = Read<int64_t>(tail_);
    tail_ += 8;

    // 计算推荐 buffer 大小（含 Remote 开销，向上取整到 KB，留 20% 余量）
    int64_t recommendedBytes = totalBytes + 16;  // sizeof(Remote)
    recommendedBytes = ((recommendedBytes * 12 / 10 + 1023) / 1024) * 1024;

    n = snprintf_s(buf, maxSize, maxSize - 1,
        "[WARNING] Ring buffer overflow detected for tensor '%s'! "
        "This tensor's encoding (%ld bytes) caused earlier data to be overwritten. "
        "Buffer tail advanced from %ld to %ld (delta=%ld bytes), indicating data loss. "
        "Buffer data area size is %ld bytes. "
        "To avoid overflow, set PRINT_BUFFER_SIZE >= %ld (%ld KB) "
        "in framework/src/interface/machine/device/tilefwk/aicpu_common.h:52, "
        "then rebuild and reinstall.\n",
        name.c_str(), totalBytes, tailBefore, tailAfter, (tailAfter - tailBefore),
        bufferSize, recommendedBytes, recommendedBytes / 1024);
    break;
}
```

### 10.4 输出示例

#### 正常情况（无溢出）

```cpp
AiCorePrintGmTensorNamed<float>(ctx, data, 1024, 0, "weight");
```

```
tensor 'weight', range=[0, 1024)
weight[0] 1.000000
weight[1] 2.000000
...
weight[1023] 1024.000000
```

#### 单 Tensor 溢出（数据照常编码，末尾追加 warning）

```cpp
// 打印 1200 个 float32，16KB buffer 放不下
AiCorePrintGmTensorNamed<float>(ctx, data, 1200, 0, "weight");
```

ring buffer 覆盖了早期数据，保留尾部数据，末尾追加 warning：

```
weight[48] 49.000000
weight[49] 50.000000
...
weight[1199] 1200.000000
[WARNING] Ring buffer overflow detected for tensor 'weight'! This tensor's encoding (16828 bytes) caused earlier data to be overwritten. Buffer tail advanced from 0 to 14254 (delta=14254 bytes), indicating data loss. Buffer data area size is 16368 bytes. To avoid overflow, set PRINT_BUFFER_SIZE >= 20480 (20 KB) in framework/src/interface/machine/device/tilefwk/aicpu_common.h:52, then rebuild and reinstall.
```

**特点**：
- 用户能看到尾部数据（带 `name[index]` 前缀，精确定位位置）
- 末尾 warning 告知溢出原因和解决方案
- 如果 warning 自身也被后续数据覆盖了，那说明 buffer 极端紧张，但尾部数据仍然可读

#### 多 Tensor 累积溢出（关键场景）

```cpp
// 三个 tensor 依次打印，累积超过 buffer 容量
AiCorePrintGmTensorNamed<float>(ctx, data1, 100, 0, "tensor1");
AiCorePrintGmTensorNamed<float>(ctx, data2, 100, 0, "tensor2");
AiCorePrintGmTensorNamed<float>(ctx, data3, 1000, 0, "tensor3");
```

输出：
```
tensor1[0] 1.000000
...
tensor1[99] 100.000000
tensor2[0] 1.000000
...
tensor2[99] 100.000000
tensor3[0] 1.000000
...
tensor3[999] 1000.000000
[WARNING] Ring buffer overflow detected for tensor 'tensor3'! This tensor's encoding (14027 bytes) caused earlier data to be overwritten. Buffer tail advanced from 227 to 14254 (delta=14027 bytes), indicating data loss. Buffer data area size is 16368 bytes. To avoid overflow, set PRINT_BUFFER_SIZE >= 20480 (20 KB) in framework/src/interface/machine/device/tilefwk/aicpu_common.h:52, then rebuild and reinstall.
```

**关键点**：
- `tensor1` 和 `tensor2` 的数据被 `tensor3` 覆盖了
- Warning 明确告知是 `tensor3` 导致的覆盖
- `tailBefore=227` 表示编码前 buffer 中已有 227 字节数据（tensor1 + tensor2）
- `tailAfter=14254` 表示编码后 tail 被推进到 14254，说明发生了覆盖

#### 多 Tensor 打印（一个溢出，一个正常）

```cpp
AiCorePrintGmTensorNamed<float>(ctx, bigData, 2000, 0, "weight");
AiCorePrintGmTensorNamed<float>(ctx, smallData, 4, 0, "bias");
```

```
weight[848] 849.000000
...
weight[1999] 2000.000000
[WARNING] Ring buffer overflow detected for tensor 'weight'! This tensor's encoding (28028 bytes) caused earlier data to be overwritten. Buffer tail advanced from 0 to 14254 (delta=14254 bytes), indicating data loss. Buffer data area size is 16368 bytes. To avoid overflow, set PRINT_BUFFER_SIZE >= 33792 (33 KB) in framework/src/interface/machine/device/tilefwk/aicpu_common.h:52, then rebuild and reinstall.
tensor 'bias', range=[0, 4)
bias[0] 0.100000
bias[1] 0.200000
bias[2] 0.300000
bias[3] 0.400000
```

#### 修改 buffer 后恢复正常

将 `PRINT_BUFFER_SIZE` 改为 `32768` 并重新编译安装后：

```
tensor 'weight', range=[0, 2000)
weight[0] 1.000000
weight[1] 2.000000
...
weight[1999] 2000.000000
tensor 'bias', range=[0, 4)
bias[0] 0.100000
bias[1] 0.200000
bias[2] 0.300000
bias[3] 0.400000
```

### 10.5 Buffer 大小配置指引

#### 当前配置位置

| 配置项 | 位置 | 当前值 | 说明 |
|--------|------|--------|------|
| `PRINT_BUFFER_SIZE` | `framework/src/interface/machine/device/tilefwk/aicpu_common.h:52` | `16384` (16 KB) | 每 AICore 的 print ring buffer 大小 |

> **注意**：`PRINT_BUFFER_SIZE` 包含 `sizeof(AicoreLogger::Remote)`（16 字节，含 `head_` + `tail_`），实际可用数据区为 `PRINT_BUFFER_SIZE - 16` 字节。`AicoreLogger::Init()` 中 `size_ = n - sizeof(Remote)` 已处理此偏移。

#### 各数据类型的容量极限（当前 16KB buffer）

以 name="weight"（6 字符）为例，TENSOR_HEADER = 26 字节，可用数据区 = 16368 字节：

| 数据类型 | 每值编码（含 END） | TENSOR_HEADER + END | 16KB 可容纳数量 | 超过即触发 warning |
|----------|-------------------|---------------------|----------------|-------------------|
| float32 (INDEXED_FP32) | 13 + 1 = 14 | 26 + 1 = 27 | ⌊(16368-27)/14⌋ = **1168** | > 1168 |
| int64 (INDEXED_INT64) | 17 + 1 = 18 | 26 + 1 = 27 | ⌊(16368-27)/18⌋ = **908** | > 908 |
| bf16 (INDEXED_BF16) | 11 + 1 = 12 | 26 + 1 = 27 | ⌊(16368-27)/12⌋ = **1362** | > 1362 |
| fp16 (INDEXED_FP16) | 11 + 1 = 12 | 26 + 1 = 27 | ⌊(16368-27)/12⌋ = **1362** | > 1362 |
| fp8e4m3 (INDEXED_FP8E4M3) | 10 + 1 = 11 | 26 + 1 = 27 | ⌊(16368-27)/11⌋ = **1485** | > 1485 |
| fp8e5m2 (INDEXED_FP8E5M2) | 10 + 1 = 11 | 26 + 1 = 27 | ⌊(16368-27)/11⌋ = **1485** | > 1485 |
| fp8e8m0 (INDEXED_FP8E8M0) | 10 + 1 = 11 | 26 + 1 = 27 | ⌊(16368-27)/11⌋ = **1485** | > 1485 |

#### 推荐配置值速查表

| 需打印的元素数量 | float32 所需 buffer | int64 所需 buffer | bf16/fp16 所需 buffer | float8/fp8e4m3/fp8e5m2/fp8e8m0 所需 buffer |
|-----------------|--------------------|--------------------|-----------------------|---------------------------------------------|
| 1024 | ≥ 14.4 KB → **16 KB** | ≥ 18.5 KB → **20 KB** | ≥ 12.3 KB → **16 KB** | ≥ 11.3 KB → **16 KB** |
| 2048 | ≥ 28.6 KB → **32 KB** | ≥ 36.9 KB → **40 KB** | ≥ 24.6 KB → **28 KB** | ≥ 22.6 KB → **24 KB** |
| 4096 | ≥ 57.0 KB → **60 KB** | ≥ 73.7 KB → **76 KB** | ≥ 49.1 KB → **52 KB** | ≥ 45.1 KB → **48 KB** |
| 8192 | ≥ 113.8 KB → **116 KB** | ≥ 147.3 KB → **150 KB** | ≥ 98.1 KB → **100 KB** | ≥ 90.1 KB → **92 KB** |

> **计算公式**：`required = 16 + 1 + 2 + nameLen + 8 + 8 + 1 + count × perElement`
> 
> 其中 `16` = sizeof(Remote)，`1 + 2 + nameLen + 8 + 8` = TENSOR_HEADER，`1` = END for header

#### 修改方法

编辑 `framework/src/interface/machine/device/tilefwk/aicpu_common.h` 第 52 行的 `PRINT_BUFFER_SIZE`，根据上方速查表选择合适的值。修改后需重新编译安装 PyPTO。

### 10.6 代码变更清单

| 文件 | 变更项 | 说明 |
|------|--------|------|
| `aicore_print.h` | NodeTy 枚举新增 `OVERFLOW_WARNING` | 新增溢出警告类型 |
| `aicore_print.h` | AicoreLogger 新增 `GetBufferSize()` 辅助方法 | 返回 `size_` 供预计算比较 |
| `aicore_print.h` | AicoreLogger 新增 `GetTail()` 辅助方法 | 返回 `tail_` 用于检测覆盖 |
| `aicore_print.h` | AicoreLogger 新增 `EncodeOverflowWarning()` 方法 | 编码溢出警告信息（含 tailBefore/tailAfter） |
| `aicore_print.h` | AicoreLogger::Encode(uint8_t) 溢出跳过新增 `OVERFLOW_WARNING` 分支 | 跳过溢出警告记录 |
| `aicore_print.h` | AicoreLogger::Read() 新增 `OVERFLOW_WARNING` 解码 | Host 侧输出 warning 信息 |
| `aicore_print.h` | __AiCorePrintTensorImpl (Named) 增加覆盖检测与末尾 warning | 编码前记录 tail，编码后检测覆盖并追加 warning |

### 10.7 设计决策与注意事项

#### 10.7.1 为什么编码数据的同时也写 Warning

| 对比项 | 只写 Warning 不编码数据 | 编码数据 + 末尾 Warning（本方案） |
|--------|----------------------|--------------------------------|
| 用户可见数据 | 无，只有 warning | 尾部数据可读（带 `name[index]` 前缀） |
| 调试价值 | 低，无法看到任何实际值 | 高，尾部数据仍可辅助定位问题 |
| Warning 丢失风险 | 低（仅 ~26 字节） | 极端情况下 warning 可能被覆盖，但数据仍在 |

核心考量：调试场景下，部分数据比零数据更有价值。紧凑编码的 `name[index]` 前缀让尾部数据依然可精确定位，即使头部被覆盖。

#### 10.7.2 为什么通过 tail_ 变化检测覆盖

| 对比项 | 预计算单个 tensor 编码量 | 比较 tail_ 变化（本方案） |
|--------|------------------------|--------------------------|
| 多 tensor 累积溢出检测 | ❌ 无法检测 | ✅ 准确检测 |
| 实现复杂度 | 低（仅需预计算） | 低（记录 tail 前后值） |
| 状态维护 | 无需额外状态 | 利用现有 `head_` 和 `tail_` |
| 语义清晰度 | 中（预判可能溢出） | 高（明确告知"此 tensor 导致覆盖"） |

**核心优势**：通过比较 `tailBefore` 和 `tailAfter`，可以准确检测多 tensor 累积溢出场景，这是预计算单个 tensor 编码量无法做到的。

#### 10.7.3 tail_ 推进的含义

当 `tailAfter > tailBefore` 时，说明 `Encode(uint8_t)` 在编码过程中检测到 buffer 满，触发了溢出跳过逻辑（`tail_` 前进跳过旧记录）。

```
tailBefore = 227  (buffer 中已有 227 字节数据)
编码 tensor3 (14027 字节)
tailAfter = 14254 (tail 被推进 14027 字节，说明发生了覆盖)
```

#### 10.7.4 Warning 自身被覆盖的极端情况

当 tensor 数据量远超 buffer 容量时，编码完所有数据后写入的 OVERFLOW_WARNING 可能被 ring buffer 中更早的数据覆盖循环时吞掉。但这只发生在数据量极端巨大的场景（例如单个 tensor 编码量超过整个 buffer）。

在常见溢出场景中（如 1200 个 float32 超出 16KB），warning 写入时位于 ring buffer 最新位置，不会被覆盖。

#### 10.7.5 非 Named 路径的处理

非 Named 路径（`AiCorePrintGmTensor` / `AiCorePrintUbTensor`）使用通用 printf 编码（`AiCoreLogF`），每行开销取决于 fmt 字符串内容，无法预计算。建议需要溢出检测的用户迁移到 Named API。

非 Named 路径的溢出仍由 `Encode(uint8_t)` 的原有 ring buffer 机制静默处理（跳过旧记录），行为不变。

#### 10.7.6 多 AICore 并发打印的注意事项

当前每个 AICore 有独立的 ring buffer 实例（`AicoreLogger`），多 AICore 并发打印时各自独立编码，互不影响。Warning 机制在每个 AICore 实例中独立工作。

如果需要跨 AICore 的统一 buffer 管理，需要额外的全局同步机制，这超出了本方案的范围。
