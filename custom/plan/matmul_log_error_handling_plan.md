# Matmul Operation 日志与错误处理整改方案

## 一、整改范围

- **文件**：`framework/src/interface/operation/cube_operation_impl.cpp`
- **新增文件**：无
- **影响范围**：仅 matmul 相关代码，不影响其他 operation

---

## 二、文件位置

**所有定义均在**：`framework/src/interface/operation/cube_operation_impl.cpp`

| 内容 | 位置 |
|------|------|
| 错误码枚举 | 第403行之前，`namespace Matrix` 内 |
| 宏定义 | 错误码枚举之后 |
| 辅助函数 | 宏定义之后 |

**需要添加的 include**：`#include <sstream>`

---

## 三、错误码枚举（9个）

```cpp
enum class MatmulErrorCode : int32_t {
    SUCCESS = 0,
    
    // FC3xxx: 参数错误
    ERR_PARAM_INVALID       = 3000,  // 参数无效（shape、dtype、format等）
    ERR_PARAM_MISMATCH      = 3001,  // 参数不匹配（维度、类型、K轴等）
    ERR_PARAM_UNSUPPORTED   = 3002,  // 不支持的参数
    
    // FC4xxx: 配置错误
    ERR_CONFIG_TILE         = 4000,  // Tile 配置错误
    ERR_CONFIG_ALIGNMENT    = 4001,  // 对齐错误（16B/32B/64元素）
    ERR_CONFIG_UNSUPPORTED  = 4002,  // 不支持的配置组合
    
    // FC5xxx: 运行时错误
    ERR_RUNTIME_NULLPTR     = 5000,  // 空指针错误
    ERR_RUNTIME_STATE       = 5001,  // 内部状态错误
    ERR_RUNTIME_LOGIC       = 5002,  // 逻辑不变量错误
};
```

---

## 四、宏定义

### 4.1 日志宏

```cpp
#define MATMUL_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, "MATMUL", __VA_ARGS__)
#define MATMUL_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO,  "MATMUL", __VA_ARGS__)
#define MATMUL_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN,  "MATMUL", __VA_ARGS__)
#define MATMUL_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, "MATMUL", __VA_ARGS__)
```

**模块名**：`MATMUL`

### 4.2 检查宏（可恢复错误）

```cpp
// 检查条件，失败时记录日志并返回 FAILED
#define MATMUL_CHECK(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MATMUL_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            return FAILED; \
        } \
    } while (0)
```

**使用场景**：参数校验、配置检查、运行时条件

### 4.3 断言宏（致命错误）

```cpp
// 检查条件，失败时记录日志并抛出异常
#define MATMUL_ASSERT(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MATMUL_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            ASSERT(false) << "[ERR-FC" << static_cast<int>(error_code) << "] " << fmt; \
        } \
    } while (0)
```

**使用场景**：内部不变量、不应该发生的错误

---

## 五、辅助函数

```cpp
inline std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}
```

---

## 六、错误处理机制

| 宏 | 行为 | 使用场景 |
|-----|------|----------|
| `MATMUL_CHECK` | 记录日志 + 返回 `FAILED` | 参数校验、配置检查（可恢复） |
| `MATMUL_ASSERT` | 记录日志 + 抛出异常 | 内部不变量（致命错误） |

**双层防御机制**：
- **Layer 1（校验层）**：使用 `MATMUL_CHECK` 返回 `Status`，调用方可处理
- **Layer 2（断言层）**：使用 `MATMUL_ASSERT` 抛出异常，无法恢复

---

## 七、错误码映射表

| 错误码 | 编号 | 含义 | 原检查点合并 |
|--------|------|------|------------|
| **ERR_PARAM_INVALID** | 3000 | 参数无效 | Shape维度无效、Shape值≤0、Bias/Scale Shape无效 |
| **ERR_PARAM_MISMATCH** | 3001 | 参数不匹配 | 维度不匹配、Dtype不匹配、K轴不匹配、M/N轴不匹配 |
| **ERR_PARAM_UNSUPPORTED** | 3002 | 不支持的参数 | 不支持的Dtype、不支持的Format |
| **ERR_CONFIG_TILE** | 4000 | Tile配置错误 | Tile尺寸≤0、L1/L0关系无效 |
| **ERR_CONFIG_ALIGNMENT** | 4001 | 对齐错误 | 16/32字节对齐、64元素对齐、Shape bound对齐 |
| **ERR_CONFIG_UNSUPPORTED** | 4002 | 不支持的配置组合 | GM ACC不支持NZ、配置冲突 |
| **ERR_RUNTIME_NULLPTR** | 5000 | 空指针错误 | tensorPtr==nullptr、GetStorage()==nullptr |
| **ERR_RUNTIME_STATE** | 5001 | 内部状态错误 | Storage访问失败、ValidShape错误 |
| **ERR_RUNTIME_LOGIC** | 5002 | 逻辑不变量错误 | AggregationMap.empty()等内部不变量违反 |

---

## 八、使用示例

### 8.1 参数错误

```cpp
// Shape 维度不匹配
MATMUL_CHECK(ERR_PARAM_MISMATCH,
    operand1.GetShape().size() == operand2.GetShape().size(),
    "Shape dimension mismatch: operand1=%zu, operand2=%zu",
    operand1.GetShape().size(), operand2.GetShape().size());

// 数据类型不匹配
MATMUL_CHECK(ERR_PARAM_MISMATCH,
    operand1.GetDataType() == operand2.GetDataType(),
    "DataType mismatch: %s vs %s",
    DataType2String(operand1.GetDataType()).c_str(),
    DataType2String(operand2.GetDataType()).c_str());

// 不支持的数据类型
MATMUL_CHECK(ERR_PARAM_UNSUPPORTED,
    outType == DataType::DT_FP32 || outType == DataType::DT_FP16,
    "Unsupported output type: %s", DataType2String(outType).c_str());
```

### 8.2 配置错误

```cpp
// Tile 配置错误
MATMUL_CHECK(ERR_CONFIG_TILE,
    kL0 > 0 && kL1a > 0 && mL0 > 0 && nL0 > 0,
    "Invalid tile config: kL0=%ld, kL1a=%ld, mL0=%ld, nL0=%ld",
    kL0, kL1a, mL0, nL0);

// L1/L0 关系错误
MATMUL_CHECK(ERR_CONFIG_TILE,
    kL0 <= kL1a && kL1a % kL0 == 0,
    "Invalid kL0/kL1a relation: kL0=%ld, kL1a=%ld", kL0, kL1a);

// 对齐错误
MATMUL_CHECK(ERR_CONFIG_ALIGNMENT,
    kL0 % 16 == 0 && nL0 % 16 == 0,
    "kL0=%ld and nL0=%ld must be 16-element aligned", kL0, nL0);

// 不支持的配置组合
MATMUL_CHECK(ERR_CONFIG_UNSUPPORTED,
    !(attrParam.isCMatrixNZ && cubeTile.enableSplitK),
    "GM accumulation does not support NZ output format");
```

### 8.3 运行时错误

```cpp
// 空指针检查
MATMUL_ASSERT(ERR_RUNTIME_NULLPTR,
    tensorGraphNodes.aTensorPtr != nullptr,
    "aTensorPtr is null");

// 内部状态错误
MATMUL_ASSERT(ERR_RUNTIME_STATE,
    aMatrix.GetStorage() != nullptr,
    "Cannot get storage for aMatrix");

// 逻辑不变量
MATMUL_ASSERT(ERR_RUNTIME_LOGIC,
    !aggregation.empty(),
    "AggregationMap is empty - internal logic error");
```

### 8.4 日志记录

```cpp
// 函数入口日志
MATMUL_LOGI("Matmul: outType=%s, aShape=%s, bShape=%s, transA=%d, transB=%d",
            DataType2String(outType).c_str(),
            ShapeToString(aMatrix.GetShape()).c_str(),
            ShapeToString(bMatrix.GetShape()).c_str(),
            isATrans, isBTrans);

// 调试日志
MATMUL_LOGD("Tile config: kL0=%ld, kL1a=%ld, nL0=%ld", kL0, kL1a, nL0);

// 警告日志
MATMUL_LOGW("Using deprecated path for matmul operation");
```

---

## 九、日志级别使用

| 级别 | 使用场景 |
|------|----------|
| **DEBUG** | 函数内部流程、中间结果、详细调试 |
| **INFO** | 函数入口/出口、关键参数、重要决策 |
| **WARN** | 非预期但可接受的情况、降级处理 |
| **ERROR** | 错误条件、校验失败（自动由 CHECK/ASSERT 宏记录） |

---

## 十、整改原则

1. **不新增文件**：所有定义在 `cube_operation_impl.cpp` 中
2. **仅用于 Matmul**：不扩展到其他 operation 组件
3. **复用现有基础设施**：基于 `PYPTO_HOST_LOG` 和 `ASSERT`
4. **最小化改动**：保留原有 `OP_CHECK`，逐步替换
5. **函数签名变更**：`void CheckXXX()` → `Status CheckXXX()`
6. **错误码精简**：控制在10个以内，合并相似错误

---

## 十一、方案总结

| 项目 | 内容 |
|------|------|
| **文件** | `cube_operation_impl.cpp` 第403行之前 |
| **模块名** | `MATMUL` |
| **错误码** | 9个（FC3000-FC5002） |
| **日志宏** | `MATMUL_LOGD/I/W/E` |
| **检查宏** | `MATMUL_CHECK` |
| **断言宏** | `MATMUL_ASSERT` |

---

## 十二、验证方法

### 12.1 编译验证

```bash
cd /mnt/workspace/gitCode/cann/pypto
python3 build_ci.py -f python3 --disable_auto_execute
```

### 12.2 日志级别控制

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0      # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

### 12.3 日志输出示例

```
[INFO ] PYPTO(12345):... [cube_operation_impl.cpp:1558][MATMUL]: Matmul: outType=DT_FP32, aShape=[128, 256], bShape=[256, 512], transA=0, transB=0
[DEBUG] PYPTO(12345):... [cube_operation_impl.cpp:612][MATMUL]: CheckOperandShape: dim1=2, dim2=2
[ERROR] PYPTO(12345):... [cube_operation_impl.cpp:615][MATMUL]: [ERR-FC3001] CHECK FAILED: Shape dimension mismatch: 2 vs 3
```
