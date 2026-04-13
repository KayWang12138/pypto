# AICore Print 四功能实现设计方案

## 1 现状分析

### 1.1 已实现功能（代码中已存在）

| 功能 | 状态 | 对应代码位置 |
|------|------|-------------|
| FP8 打印支持 | ✅ 已实现 | `DecodeFloat8()`、`FLOAT8`/`INDEXED_FLOAT8` 枚举、`PrintFloat8` 回调、`__AiCorePrint` float8_t 分支 |
| Tensor 名称和索引打印 | ✅ 已实现 | `TENSOR_HEADER`、`INDEXED_*` 枚举、`EncodeTensorHeader()`、`EncodeIndexed()`、Named API |
| L1 数据打印 | ✅ 已实现 | `L1RawCopyToGM()`、`AiCorePrintL1Tensor()` 两个重载 |

### 1.2 本次新增实现

| 功能 | 状态 | 需要的变更 |
|------|------|-----------|
| Ring Buffer 溢出 Warning | ✅ 已实现 | 新增 `OVERFLOW_WARNING` 枚举、编码/解码、预计算逻辑 |

### 1.3 结论

三个功能（FP8、L1 打印、Tensor 名称索引）**已经存在于当前代码中**，本次新增实现了 **Ring Buffer 溢出 Warning** 功能。

---

## 2 Ring Buffer 溢出 Warning 实现方案

### 2.1 修改清单

#### `aicore_print.h` 修改点（共 6 处）

| # | 修改项 | 说明 |
|---|--------|------|
| 1 | NodeTy 枚举新增 `OVERFLOW_WARNING` | 在 `INDEXED_FLOAT8` 之后追加 |
| 2 | AicoreLogger 新增 `GetBufferSize()` | 返回 `size_` 供预计算比较 |
| 3 | AicoreLogger 新增 `EncodeOverflowWarning()` | 编码溢出警告信息（name + totalBytes + bufferSize） |
| 4 | `Encode(uint8_t)` 溢出跳过新增 `OVERFLOW_WARNING` 分支 | ring buffer 满时正确跳过 warning 记录 |
| 5 | `Read()` 新增 `OVERFLOW_WARNING` 解码 | Host 侧输出 warning 信息（含推荐 buffer 大小） |
| 6 | `__AiCorePrintTensorImpl`（Named 路径）增加预计算与末尾 warning | 编码前预判溢出，编码后追加 warning |

### 2.2 详细设计

#### OVERFLOW_WARNING 编码格式

```
┌──────────┬───────────────┬─────────────────┬────────────────┬────────────────┐
│ type: 1B │ nameLen: 2B   │ name: N字节+'\0' │ totalBytes: 8B │ bufferSize: 8B │
│ (uint8)  │ (short LE)    │ (字符串)         │ (int64 LE)     │ (int64 LE)     │
└──────────┴───────────────┴─────────────────┴────────────────┴────────────────┘
```

#### 预计算逻辑

在 `__AiCorePrintTensorImpl`（Named 路径）中：
1. 在编码前根据类型计算每元素固定开销
2. 计算 TENSOR_HEADER + 所有数据行的总字节数
3. 与 `logger->GetBufferSize()` 比较，判定是否溢出
4. 编码完成后（无论是否溢出都正常编码数据），若预判溢出则追加 OVERFLOW_WARNING

### 2.3 测试文件修改

在 `kernel_aicore/TENSOR_s0_Unroll1_PATH0_hiddenfunc0_8_4223363206862604209_0_aic.cpp` 中添加了 4 个测试调用。

### 2.4 测试项与预期结果

| # | 测试项 | 测试代码 | 预期输出 |
|---|--------|---------|---------|
| 1 | 原有 GM Tensor 打印（无名称） | `AiCorePrintGmTensor(ctx, gm_data, 10, 0)` | `tensor data, range=[0, 10)\n` 后跟 10 行浮点数 |
| 2 | Named GM Tensor 打印（带名称索引） | `AiCorePrintGmTensorNamed<float>(ctx, gm_data, 10, 0, "gm_data")` | `tensor 'gm_data', range=[0, 10)\n` 后跟 `gm_data[0] val .. gm_data[9] val` |
| 3 | L1 Tensor 打印（Named） | `AiCorePrintL1Tensor(ctx, l1_data, 10, 0, staging, "l1Tensor_0")` | `tensor 'l1Tensor_0', range=[0, 10)\n` 后跟 `l1Tensor_0[0] val .. l1Tensor_0[9] val` |
| 4 | Ring Buffer 溢出 Warning | `AiCorePrintGmTensorNamed<float>(ctx, gm_data, 1200, 0, "big_data")` | 尾部数据 + `[WARNING] Ring buffer overflow for tensor 'big_data'! Required N bytes ...` |

#### 测试 1 预期输出（原有功能验证）

```
tensor data, range=[0, 10)
<val_0>
<val_1>
...
<val_9>
```

#### 测试 2 预期输出（Tensor 名称索引）

```
tensor 'gm_data', range=[0, 10)
gm_data[0] <val>
gm_data[1] <val>
...
gm_data[9] <val>
```

#### 测试 3 预期输出（L1 数据打印）

```
tensor 'l1Tensor_0', range=[0, 10)
l1Tensor_0[0] <val>
l1Tensor_0[1] <val>
...
l1Tensor_0[9] <val>
```

注意：L1 数据经 ND2NZ 加载后为分形格式，打印的索引为 L1 物理线性偏移，值可能与原始 GM 数据不同。

#### 测试 4 预期输出（溢出 Warning）

```
big_data[<N>] <val>
...
big_data[1199] <val>
[WARNING] Ring buffer overflow for tensor 'big_data'! Required 16829 bytes but buffer data area is only 16368 bytes, earlier data was overwritten. To avoid overflow, set PRINT_BUFFER_SIZE >= 20480 (20 KB) in framework/src/interface/machine/device/tilefwk/aicpu_common.h:52, then rebuild and reinstall.
```

---

## 3 编译与验证结果

### 3.1 PyPTO 编译安装

```bash
rm -rf build_out/ && python build_ci.py && pip install build_out/pypto-*.whl --force-reinstall
```

**结果**：✅ 编译安装成功，pypto 0.2.1 已安装。

### 3.2 Kernel 编译

使用 bisheng 编译器编译测试 kernel：

```bash
bisheng -c -O3 -g -x cce -std=c++17 ... -o kernel_aicore/..._aic.o kernel_aicore/..._aic.cpp
```

**结果**：✅ 编译成功，无错误、无警告。

### 3.3 注意事项

1. **全局 `__gm__` 字符串**：AICore 编译器不允许在函数内部声明 `__gm__` 数组，字符串名称必须定义为全局变量。
2. **L1 打印 staging buffer**：测试中复用 `gmTensor_11`（输出 GM buffer）作为 staging，这在 TStore 写入之前是安全的。
3. **EVENT_ID 冲突**：L1 打印内部使用 `EVENT_ID0`，当前测试代码在 `wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)` 之后执行，EVENT_ID0 已释放，可安全复用。
4. **溢出 Warning 的自身覆盖风险**：当数据量极端巨大时，warning 本身可能被后续编码数据覆盖，但常见溢出场景中 warning 位于 ring buffer 最新位置，不会被覆盖。
