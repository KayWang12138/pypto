# Matmul 日志与错误处理详细实施方案

## 一、文件修改概述

### 1.1 修改文件
- **文件**：`framework/src/interface/operation/cube_operation_impl.cpp`
- **总行数**：1786 行
- **修改范围**：第16行（include）、第403行之前（新增定义）、多个函数改造

### 1.2 修改类型统计

| 修改类型 | 数量 | 说明 |
|---------|------|------|
| 新增 include | 1 | `<sstream>` |
| 新增错误码枚举 | 1 | 9个错误码 |
| 新增日志宏 | 4 | `MATMUL_LOGD/I/W/E` |
| 新增检查宏 | 2 | `MATMUL_CHECK`、`MATMUL_ASSERT` |
| 新增辅助函数 | 1 | `ShapeToString` |
| 改造检查函数 | 12 | 返回类型从 `void` 改为 `Status` |
| 改造入口函数 | 6 | 添加日志 |

---

## 二、详细修改步骤

### 2.1 步骤1：添加 include（第16行之后）

**位置**：第16行 `#include "interface/configs/config_manager.h"` 之后

**添加内容**：
```cpp
#include <sstream>
```

**修改后的 include 块**（第16-28行）：
```cpp
#include "interface/configs/config_manager.h"
#include <sstream>  // 新增：用于 ShapeToString 辅助函数
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/operator_tracer.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
```

---

### 2.2 步骤2：添加错误码、宏定义和辅助函数（第403行之前）

**位置**：第402行 `} // namespace Deprecate` 之后，第403行 `const int32_t MATRIX_SHAPE_DIM = 2;` 之前

**添加内容**：

```cpp
} // namespace Deprecate

// ============================================================================
// Matmul 专用日志和错误处理机制
// ============================================================================

// 错误码枚举（9个错误码）
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

// 日志宏
#define MATMUL_LOGD(...) PYPTO_HOST_LOG(DLOG_DEBUG, "MATMUL", __VA_ARGS__)
#define MATMUL_LOGI(...) PYPTO_HOST_LOG(DLOG_INFO,  "MATMUL", __VA_ARGS__)
#define MATMUL_LOGW(...) PYPTO_HOST_LOG(DLOG_WARN,  "MATMUL", __VA_ARGS__)
#define MATMUL_LOGE(...) PYPTO_HOST_LOG(DLOG_ERROR, "MATMUL", __VA_ARGS__)

// 检查宏（返回 FAILED）
#define MATMUL_CHECK(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MATMUL_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            return FAILED; \
        } \
    } while (0)

// 断言宏（抛出异常）
#define MATMUL_ASSERT(error_code, cond, fmt, ...) \
    do { \
        if (!(cond)) { \
            MATMUL_LOGE("[ERR-FC%d] " fmt, static_cast<int>(error_code), ##__VA_ARGS__); \
            ASSERT(false) << "[ERR-FC" << static_cast<int>(error_code) << "] " << fmt; \
        } \
    } while (0)

// 辅助函数：将 shape 转换为字符串
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

// ============================================================================
// End of Matmul 日志和错误处理机制
// ============================================================================

const int32_t MATRIX_SHAPE_DIM = 2;
```

---

## 三、函数改造详细方案

### 3.1 CheckOperandShape 函数改造

**位置**：第611-659行

**改造前**：
```cpp
void CheckOperandShape(const Tensor &operand1, const Tensor &operand2)
{
    OP_CHECK(true, {
            ASSERT(operand1.GetShape().size() == operand2.GetShape().size())
        << "Shape dimension mismatch between operand1 and operand2. "
        << "operand1 shape size: " << operand1.GetShape().size()
        << ", operand2 shape size: " << operand2.GetShape().size() << std::endl;
    });

    OP_CHECK(true, {
        ASSERT(operand1.GetShape().size() == operand1.GetStorage()->offset.size())
        << "Shape dimension mismatch with offset size for operand1. "
        << "shape size: " << operand1.GetShape().size() << ", offset size: " << operand1.GetStorage()->offset.size()
        << std::endl;
    });

    OP_CHECK(true, {
            ASSERT(operand2.GetShape().size() == operand2.GetStorage()->offset.size())
        << "Shape dimension mismatch with offset size for operand2. "
        << "shape size: " << operand2.GetShape().size() << ", offset size: " << operand2.GetStorage()->offset.size()
        << std::endl;
    });

    OP_CHECK(true, {
    ASSERT(operand1.GetShape().size() >= SHAPE_DIM2)
        << "The dimension of operand1 must be larger than 2! The dimensin of operand1:" << operand1.GetShape().size()
        << std::endl;
    });

    OP_CHECK(true, {
            ASSERT(operand2.GetShape().size() >= SHAPE_DIM2)
        << "The dimension of operand2 must be larger than 2! The dimensin of operand2:" << operand2.GetShape().size()
        << std::endl;
    });

    for (size_t i = 0; i < operand1.GetShape().size(); ++i) {
        OP_CHECK(true, {
            ASSERT(operand1.GetShape()[i] > 0)
            << "The value of the " << i << "-th dimension of operand1 must be larger than 0" << std::endl;
        });
    }

    for (size_t i = 0; i < operand2.GetShape().size(); ++i) {
        OP_CHECK(true, {
             ASSERT(operand2.GetShape()[i] > 0)
            << "The value of the " << i << "-th dimension of operand2 must be larger than 0" << std::endl;
        });
    }
}
```

**改造后**：
```cpp
Status CheckOperandShape(const Tensor &operand1, const Tensor &operand2)
{
    MATMUL_LOGD("CheckOperandShape: operand1 dim=%zu, operand2 dim=%zu",
                operand1.GetShape().size(), operand2.GetShape().size());
    
    // 检查 shape 维度一致性
    MATMUL_CHECK(ERR_PARAM_MISMATCH,
                 operand1.GetShape().size() == operand2.GetShape().size(),
                 "Shape dimension mismatch: operand1=%zu, operand2=%zu",
                 operand1.GetShape().size(), operand2.GetShape().size());
    
    // 检查 shape 与 offset 的一致性
    MATMUL_CHECK(ERR_PARAM_MISMATCH,
                 operand1.GetShape().size() == operand1.GetStorage()->offset.size(),
                 "operand1 shape size(%zu) != offset size(%zu)",
                 operand1.GetShape().size(), operand1.GetStorage()->offset.size());
    
    MATMUL_CHECK(ERR_PARAM_MISMATCH,
                 operand2.GetShape().size() == operand2.GetStorage()->offset.size(),
                 "operand2 shape size(%zu) != offset size(%zu)",
                 operand2.GetShape().size(), operand2.GetStorage()->offset.size());
    
    // 检查最小维度
    MATMUL_CHECK(ERR_PARAM_INVALID,
                 operand1.GetShape().size() >= SHAPE_DIM2,
                 "operand1 dimension(%zu) must be >= 2",
                 operand1.GetShape().size());
    
    MATMUL_CHECK(ERR_PARAM_INVALID,
                 operand2.GetShape().size() >= SHAPE_DIM2,
                 "operand2 dimension(%zu) must be >= 2",
                 operand2.GetShape().size());
    
    // 检查每个维度的值
    for (size_t i = 0; i < operand1.GetShape().size(); ++i) {
        MATMUL_CHECK(ERR_PARAM_INVALID,
                     operand1.GetShape()[i] > 0,
                     "operand1 dim[%zu] = %ld, must be > 0",
                     i, operand1.GetShape()[i]);
    }
    
    for (size_t i = 0; i < operand2.GetShape().size(); ++i) {
        MATMUL_CHECK(ERR_PARAM_INVALID,
                     operand2.GetShape()[i] > 0,
                     "operand2 dim[%zu] = %ld, must be > 0",
                     i, operand2.GetShape()[i]);
    }
    
    MATMUL_LOGD("CheckOperandShape: PASS");
    return SUCCESS;
}
```

---

### 3.2 CheckL1L0Tile 函数改造

**位置**：第661-674行

**改造前**：
```cpp
void CheckL1L0Tile(const int64_t L0Tile, const int64_t L1Tile, const std::string L0TileName, const std::string L1TileName)
{
    OP_CHECK(true, {
        ASSERT(L0Tile != 0)
            << "Current " << L0TileName << ": " << L0Tile
            << ", Requirement: " << L0TileName << " cannot be zero." << std::endl;
    });
    OP_CHECK(true, {
        ASSERT(L0Tile <= L1Tile && L1Tile % L0Tile == 0)
            << "Current " << L0TileName << ": " << L0Tile << ", " << L1TileName << ": " << L1Tile
            << ", Requirement: " << L0TileName << " <= " << L1TileName << " && "
            << L1TileName << " % " << L0TileName << " == 0" << std::endl;
    });
}
```

**改造后**：
```cpp
Status CheckL1L0Tile(const int64_t L0Tile, const int64_t L1Tile, 
                     const std::string& L0TileName, const std::string& L1TileName)
{
    MATMUL_CHECK(ERR_CONFIG_TILE,
                 L0Tile != 0,
                 "%s cannot be zero, got %ld",
                 L0TileName.c_str(), L0Tile);
    
    MATMUL_CHECK(ERR_CONFIG_TILE,
                 L0Tile <= L1Tile && L1Tile % L0Tile == 0,
                 "Invalid L1/L0 relation: %s=%ld, %s=%ld, require %s <= %s && %s %% %s == 0",
                 L0TileName.c_str(), L0Tile, L1TileName.c_str(), L1Tile,
                 L0TileName.c_str(), L1TileName.c_str(),
                 L1TileName.c_str(), L0TileName.c_str());
    
    return SUCCESS;
}
```

---

### 3.3 CheckCubeTiling 函数改造

**位置**：第676-719行

**改造前**（部分）：
```cpp
void CheckCubeTiling(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int32_t kBL1Idx = 2;
    const int64_t kL0 = cubeTile.k[0];
    const int64_t kL1a = cubeTile.k[1];
    const int64_t kL1b = cubeTile.k[kBL1Idx];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t mL1 = cubeTile.m[1];
    const int64_t nL0 = cubeTile.n[0];
    const int64_t nL1 = cubeTile.n[1];
    OP_CHECK(true, {
        ASSERT(kL0 > 0 && kL1a > 0 && kL1b > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0)
            << "Current kL0: " << kL0 << ", kL1a: " << kL1a << ", kL1b: " << kL1b << ", mL0: " << mL0
            << ", mL1: " << mL1 << ", nL0: " << nL0 << ", nL1: " << nL1
            << ", Requirement: all must be larger than 0." << std::endl;
    });
    // ... 更多检查
}
```

**改造后**（完整）：
```cpp
Status CheckCubeTiling(const Tensor &operand1, const Tensor &operand2, const MatmulAttrParam &attrParam) {
    auto cubeTile = TileShape::Current().GetCubeTile();
    const int32_t kBL1Idx = 2;
    const int64_t kL0 = cubeTile.k[0];
    const int64_t kL1a = cubeTile.k[1];
    const int64_t kL1b = cubeTile.k[kBL1Idx];
    const int64_t mL0 = cubeTile.m[0];
    const int64_t mL1 = cubeTile.m[1];
    const int64_t nL0 = cubeTile.n[0];
    const int64_t nL1 = cubeTile.n[1];
    
    MATMUL_LOGD("CheckCubeTiling: kL0=%ld, kL1a=%ld, kL1b=%ld, mL0=%ld, mL1=%ld, nL0=%ld, nL1=%ld",
                kL0, kL1a, kL1b, mL0, mL1, nL0, nL1);
    
    // 检查所有 tile 值为正
    MATMUL_CHECK(ERR_CONFIG_TILE,
                 kL0 > 0 && kL1a > 0 && kL1b > 0 && mL0 > 0 && mL1 > 0 && nL0 > 0 && nL1 > 0,
                 "Invalid tile values: kL0=%ld, kL1a=%ld, kL1b=%ld, mL0=%ld, mL1=%ld, nL0=%ld, nL1=%ld",
                 kL0, kL1a, kL1b, mL0, mL1, nL0, nL1);
    
    // 检查对齐要求
    MATMUL_CHECK(ERR_CONFIG_ALIGNMENT,
                 kL0 % ALIGN_SIZE_16 == 0 && nL0 % ALIGN_SIZE_16 == 0,
                 "kL0(%ld) and nL0(%ld) must be aligned to 16 elements",
                 kL0, nL0);
    
    // 检查 L1/L0 tile 关系
    if (CheckL1L0Tile(kL0, kL1a, "kL0", "kL1a") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(kL0, kL1b, "kL0", "kL1b") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(nL0, nL1, "nL0", "nL1") != SUCCESS) {
        return FAILED;
    }
    if (CheckL1L0Tile(mL0, mL1, "mL0", "mL1") != SUCCESS) {
        return FAILED;
    }
    
    // 检查字节对齐
    MATMUL_CHECK(ERR_CONFIG_ALIGNMENT,
                 kL0 * BytesOf(operand1.GetDataType()) % ALIGN_SIZE_32 == 0,
                 "kL0 * sizeof(dtype) = %ld bytes, must be 32-byte aligned",
                 kL0 * BytesOf(operand1.GetDataType()));
    
    MATMUL_CHECK(ERR_CONFIG_ALIGNMENT,
                 nL0 * BytesOf(operand2.GetDataType()) % ALIGN_SIZE_32 == 0,
                 "nL0 * sizeof(dtype) = %ld bytes, must be 32-byte aligned",
                 nL0 * BytesOf(operand2.GetDataType()));
    
    MATMUL_LOGD("CheckCubeTiling: PASS");
    return SUCCESS;
}
```

---

### 3.4 Matmul 入口函数改造

**位置**：第1557-1567行

**改造前**：
```cpp
Tensor Matmul(
    DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans, bool isBTrans, bool isCMatrixNZ) {
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), bMatrix.GetStorage());
    auto &cubeTile = TileShape::Current().GetCubeTile();
    if (cubeTile.enableSplitK) {
        return ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam);
    }
    return ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
}
```

**改造后**：
```cpp
Tensor Matmul(
    DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans, bool isBTrans, bool isCMatrixNZ) {
    MATMUL_LOGI("Matmul: outType=%s, aShape=%s, bShape=%s, transA=%d, transB=%d, isCMatrixNZ=%d",
                DataType2String(outType).c_str(),
                ShapeToString(aMatrix.GetShape()).c_str(),
                ShapeToString(bMatrix.GetShape()).c_str(),
                isATrans, isBTrans, isCMatrixNZ);
    
    MatmulAttrParam attrParam(isATrans, isBTrans, isCMatrixNZ);
    
    // 参数校验
    Status checkStatus = CheckMatmulOperands(outType, aMatrix, bMatrix, attrParam);
    MATMUL_ASSERT(ERR_RUNTIME_LOGIC, checkStatus == SUCCESS, "Matmul operands check failed");
    
    MatmulGraphNodes tensorGraphNodes(aMatrix.GetStorage(), bMatrix.GetStorage());
    auto &cubeTile = TileShape::Current().GetCubeTile();
    
    Tensor result;
    if (cubeTile.enableSplitK) {
        MATMUL_LOGD("Matmul: using GM accumulation path");
        result = ConstructGmAccumulationTensorGraph(outType, aMatrix, bMatrix, attrParam);
    } else {
        MATMUL_LOGD("Matmul: using normal path");
        result = ConstructTensorGraph(outType, tensorGraphNodes, attrParam);
    }
    
    MATMUL_LOGI("Matmul: result shape=%s", ShapeToString(result.GetShape()).c_str());
    return result;
}
```

---

## 四、需要改造的函数清单

### 4.1 检查函数（void → Status）

| 函数名 | 行号 | 返回类型变更 | 主要错误码 |
|--------|------|-------------|-----------|
| `CheckOperandShape` | 611-659 | `void` → `Status` | ERR_PARAM_MISMATCH, ERR_PARAM_INVALID |
| `CheckL1L0Tile` | 661-674 | `void` → `Status` | ERR_CONFIG_TILE |
| `CheckCubeTiling` | 676-719 | `void` → `Status` | ERR_CONFIG_TILE, ERR_CONFIG_ALIGNMENT |
| `CheckOperandShapeBound` | 721-747 | `void` → `Status` | ERR_CONFIG_ALIGNMENT |
| `CheckNZFormatAligned` | 749-798 | `void` → `Status` | ERR_CONFIG_ALIGNMENT |
| `CheckCMatrixNZFormatAligned` | 800-833 | `void` → `Status` | ERR_CONFIG_ALIGNMENT |
| `CheckBiasParam` | 835-870 | `void` → `Status` | ERR_PARAM_INVALID, ERR_PARAM_MISMATCH |
| `CheckFixpipeParam` | 872-913 | `void` → `Status` | ERR_PARAM_INVALID, ERR_PARAM_UNSUPPORTED |
| `CheckGmAccumulationParam` | 915-953 | `void` → `Status` | ERR_CONFIG_UNSUPPORTED |
| `CheckMatmulOperands` | 955-984 | `void` → `Status` | ERR_PARAM_UNSUPPORTED, ERR_PARAM_MISMATCH |
| `CheckMXMatmulShape` | 986-1033 | `void` → `Status` | ERR_PARAM_INVALID, ERR_CONFIG_ALIGNMENT |
| `CheckMXMatmulOperands` | 1035-1057 | `void` → `Status` | ERR_PARAM_MISMATCH |

### 4.2 入口函数（添加日志）

| 函数名 | 行号 | 修改内容 |
|--------|------|---------|
| `Matmul` (基础版) | 1557-1567 | 添加入口/出口日志，校验失败使用 MATMUL_ASSERT |
| `Matmul` (扩展版) | 1569-1579 | 添加入口/出口日志，校验失败使用 MATMUL_ASSERT |
| `MatmulMX` (基础版) | 1581-1589 | 添加入口/出口日志，校验失败使用 MATMUL_ASSERT |
| `MatmulMX` (扩展版) | 1591-1600 | 添加入口/出口日志，校验失败使用 MATMUL_ASSERT |
| `BatchMatmul` | 1721-1742 | 添加入口日志 |
| `TransposedBatchMatmul` | 1746-1783 | 添加入口日志 |

---

## 五、改造优先级

### 5.1 P0 优先级（必须完成）

1. **添加错误码和宏定义**（第403行之前）
2. **改造核心检查函数**：
   - `CheckOperandShape` (611-659)
   - `CheckCubeTiling` (676-719)
   - `CheckMatmulOperands` (955-984)
3. **改造主入口函数**：
   - `Matmul` (1557-1567)
   - `Matmul` (1569-1579)

### 5.2 P1 优先级（推荐完成）

1. **改造其他检查函数**：
   - `CheckL1L0Tile` (661-674)
   - `CheckOperandShapeBound` (721-747)
   - `CheckNZFormatAligned` (749-798)
   - `CheckBiasParam` (835-870)
   - `CheckFixpipeParam` (872-913)
   - `CheckGmAccumulationParam` (915-953)

2. **改造 MX 入口函数**：
   - `MatmulMX` (1581-1589)
   - `MatmulMX` (1591-1600)
   - `CheckMXMatmulShape` (986-1033)
   - `CheckMXMatmulOperands` (1035-1057)

### 5.3 P2 优先级（可选完成）

1. **改造 Batch 相关函数**：
   - `BatchMatmul` (1721-1742)
   - `TransposedBatchMatmul` (1746-1783)

---

## 六、错误码使用映射表

### 6.1 参数错误（FC3xxx）

| 函数 | 检查点 | 错误码 |
|------|--------|--------|
| CheckOperandShape | Shape 维度不匹配 | ERR_PARAM_MISMATCH |
| CheckOperandShape | Shape 与 offset 不匹配 | ERR_PARAM_MISMATCH |
| CheckOperandShape | Shape 维度 < 2 | ERR_PARAM_INVALID |
| CheckOperandShape | Shape 值 <= 0 | ERR_PARAM_INVALID |
| CheckMatmulOperands | 输出类型不支持 | ERR_PARAM_UNSUPPORTED |
| CheckMatmulOperands | 输入类型不匹配 | ERR_PARAM_MISMATCH |
| CheckBiasParam | Bias Shape 无效 | ERR_PARAM_INVALID |
| CheckBiasParam | Bias 类型不匹配 | ERR_PARAM_MISMATCH |
| CheckFixpipeParam | Quantize 配置无效 | ERR_PARAM_UNSUPPORTED |
| CheckMXMatmulOperands | Scale Shape 无效 | ERR_PARAM_INVALID |

### 6.2 配置错误（FC4xxx）

| 函数 | 检查点 | 错误码 |
|------|--------|--------|
| CheckL1L0Tile | L0/L1 关系无效 | ERR_CONFIG_TILE |
| CheckCubeTiling | Tile 值 <= 0 | ERR_CONFIG_TILE |
| CheckCubeTiling | 16 元素对齐失败 | ERR_CONFIG_ALIGNMENT |
| CheckCubeTiling | 32 字节对齐失败 | ERR_CONFIG_ALIGNMENT |
| CheckOperandShapeBound | Shape bound 对齐失败 | ERR_CONFIG_ALIGNMENT |
| CheckNZFormatAligned | NZ 对齐失败 | ERR_CONFIG_ALIGNMENT |
| CheckGmAccumulationParam | GM ACC 不支持 NZ | ERR_CONFIG_UNSUPPORTED |
| CheckGmAccumulationParam | GM ACC 配置冲突 | ERR_CONFIG_UNSUPPORTED |

### 6.3 运行时错误（FC5xxx）

| 函数 | 检查点 | 错误码 |
|------|--------|--------|
| Matmul 入口 | 校验函数返回 FAILED | ERR_RUNTIME_LOGIC |
| 内部函数 | 空指针检查 | ERR_RUNTIME_NULLPTR |
| 内部函数 | Storage 访问失败 | ERR_RUNTIME_STATE |
| 内部函数 | 内部不变量违反 | ERR_RUNTIME_LOGIC |

---

## 七、验证检查清单

### 7.1 编译验证

- [ ] 添加 `#include <sstream>` 后编译通过
- [ ] 添加错误码和宏定义后编译通过
- [ ] 改造所有检查函数后编译通过
- [ ] 改造所有入口函数后编译通过
- [ ] 无编译警告

### 7.2 功能验证

- [ ] 正常 matmul 调用成功
- [ ] Shape 不匹配时返回错误码
- [ ] Dtype 不匹配时返回错误码
- [ ] Tile 配置错误时返回错误码
- [ ] 对齐错误时返回错误码
- [ ] 日志正确输出（DEBUG/INFO/ERROR）

### 7.3 日志验证

- [ ] 入口日志显示关键参数
- [ ] 错误日志包含错误码前缀 `[ERR-FCxxxx]`
- [ ] 错误日志包含详细上下文
- [ ] 日志级别控制生效

---

## 八、实施建议

### 8.1 分批实施

1. **第一批**（P0）：核心定义 + 3个核心函数 + 2个入口函数
2. **第二批**（P1）：其他检查函数 + MX 相关函数
3. **第三批**（P2）：Batch 相关函数

### 8.2 向后兼容

- 保留原有 `OP_CHECK` 宏定义（第35-40行）
- 逐步替换，不强制一次性全部改造
- 保持函数签名向后兼容（除返回类型外）

### 8.3 测试建议

1. **单元测试**：为每个错误场景编写测试用例
2. **集成测试**：运行完整的 matmul 测试套件
3. **性能测试**：确认日志和错误处理不影响性能
4. **日志分析**：检查日志格式和内容的正确性
