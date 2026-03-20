# Quantize 算子代码检视报告 (更新版)

**检视日期:** 2026-03-21  
**检视人:** Claw1178  
**涉及 Commit:**
- `b9702f7d` - "add quantize" (新增量化功能)
- `05ebe96f` - "update: 更新文件 quantize.h" (简化重构)

---

## 一、改动概述

### Commit 1: 新增量化功能 (`b9702f7d`)

**新增 25 个文件，+2908 行代码**

这是一个完整的量化算子实现，包含：

| 模块 | 文件 | 说明 |
|------|------|------|
| **文档** | `docs/api/operation/pypto-quantize.md`, `pypto-dequantize.md` | API 文档 |
| **TileOp 层** | `framework/src/interface/tileop/vector/quantize.h` | 核心量化算子实现 |
| **Operation 层** | `framework/src/interface/operation/vector/quantization.cpp` | 张量级量化操作 |
| **Codegen** | `framework/src/codegen/cloudnpu/codegen_vector.cpp` | 代码生成 |
| **解释器** | `framework/src/interface/interpreter/calculator/calc_torch.cpp` | Torch 参考实现 |
| **Python API** | `python/pypto/op/math.py` | 用户接口 `quantize()` |
| **测试** | `tests/st/operation/test_vector_operation_quantize.py` 等 | ST 测试用例 |

### Commit 2: 简化重构 (`05ebe96f`)

**修改 3 个文件，+96/-217 行**

- 大幅精简 `quantize.h`，移除了逐列量化 (axis=-2) 的实现
- 逐列量化改为在 Operation 层通过 Transpose 实现

---

## 二、功能分析

### 2.1 支持的量化类型

| 类型 | 输出范围 | 公式 | 用途 |
|------|---------|------|------|
| **INT8_SYM** (对称量化) | [-128, 127] | `int8 = round(fp32 * scale)` | 权重量化，零点为 0 |
| **INT8_ASYM** (非对称量化) | [0, 255] (UINT8) | `uint8 = round(fp32 * scale + offset)` | 激活量化，支持偏斜分布 |

### 2.2 量化轴 (axis)

- **axis = -1** (逐行量化): 每行一个 scale，scale 形状为 `[..., H, 1]`
- **axis = -2** (逐列量化): 通过 Transpose 转换后在 axis=-1 上执行

### 2.3 架构层次

```
Python API (quantize)
    ↓
Operation 层 (quantization.cpp)
    ↓ (axis=-2 时插入 Transpose)
TileOp 层 (quantize.h) ← 只处理 axis=-1
    ↓
Codegen (codegen_vector.cpp)
    ↓
底层 TileOp 调用 (TQUANT)
```

---

## 三、代码质量评估

### ✅ 优点

#### 1. 架构设计清晰
- 分层合理：Python API → Operation → TileOp → Codegen
- axis=-2 通过 Transpose 实现的设计避免了代码重复

#### 2. 类型安全
- 使用 `static_assert` 在编译期检查参数匹配
- 使用 `constexpr` 函数替代宏，类型安全

#### 3. 边界检查完善
- 对空张量做了防护
- 输入参数有详细校验和友好的错误信息

#### 4. 测试覆盖
- 提供了完整的 ST 测试用例
- 包含 Python 和 C++ 两层测试

---

## 四、已修复的问题 ✅

根据初次检视报告的建议，以下问题已修复：

### 1. 宏定义改为 constexpr 函数 ✅

**Before:**
```cpp
#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))
```

**After:**
```cpp
template<typename T>
constexpr T PtoCeil(T value, T alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}
```

### 2. 魔法数字改为常量 ✅

**Before:**
```cpp
constexpr int paddedCol_dst = PTO_CEIL(dstTileW, 32 / sizeof(int8_t));
```

**After:**
```cpp
constexpr size_t TILE_ALIGNMENT_BYTES = 32;
constexpr int paddedCol_dst = PtoCeil(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));
```

### 3. 错误信息更友好 ✅

**Before:**
```cpp
ASSERT(input.GetShape().size() >= SHAPE_DIM2 && input.GetShape().size() <= SHAPE_DIM5)
    << "The shape.size() only support 2~5";
```

**After:**
```cpp
size_t inputRank = input.GetShape().size();
ASSERT(inputRank >= SHAPE_DIM2 && inputRank <= SHAPE_DIM5)
    << "Quantize input rank must be 2~5, but got rank=" << inputRank;
```

### 4. Python API 参数命名优化 ✅

**Before:** `otype` (容易混淆)  
**After:** `dtype` (更直观)

### 5. Codegen 硬编码改为映射表 ✅

**Before:**
```cpp
if (opCode == Opcode::OP_QUANTIZE_SYM) {
    quantType = "pto::QuantType::INT8_SYM";
} else {
    quantType = "pto::QuantType::INT8_ASYM";
}
```

**After:**
```cpp
static const std::unordered_map<Opcode, std::string> kQuantTypeMap = {
    {Opcode::OP_QUANTIZE_SYM, "pto::QuantType::INT8_SYM"},
    {Opcode::OP_QUANTIZE_ASYM, "pto::QuantType::INT8_ASYM"},
};
auto it = kQuantTypeMap.find(opCode);
ASSERT(it != kQuantTypeMap.end()) << "unknown opcode";
const std::string& quantType = it->second;
```

### 6. 类型校验修正 ✅

**Before:** 声明支持 FP32/FP16/BF16，实际只支持 FP32  
**After:** 明确只支持 FP32，校验和文档一致

### 7. 注释精简 ✅

**Before:** 364 行，注释过于详细  
**After:** 253 行，精简约 130 行冗余注释

---

## 五、剩余建议 (可选优化)

| 优先级 | 问题 | 建议 | 状态 |
|--------|------|------|------|
| 🟡 P1 | Sym/Asym 代码重复 | 抽取公共逻辑到辅助函数 | 未处理 (低优先级) |
| 🟡 P1 | 补充边界测试用例 | 空张量、单元素、极端值 | 未处理 |
| 🟢 P2 | axis=-2 性能 | 三次 Transpose 可能有开销 | 评估中 |

---

## 六、修改统计

| 文件 | 修改 |
|------|------|
| `framework/src/interface/tileop/vector/quantize.h` | +117/-247 (精简注释、constexpr 替代宏) |
| `framework/src/interface/operation/vector/quantization.cpp` | 改进错误信息、修正类型校验 |
| `framework/src/codegen/cloudnpu/codegen_vector.cpp` | 映射表替代硬编码 |
| `python/pypto/op/math.py` | `otype` → `dtype`、文档更新 |

**总计:** 4 个文件修改

---

## 七、总结评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **代码质量** | ⭐⭐⭐⭐⭐ | 已修复所有主要问题 |
| **可维护性** | ⭐⭐⭐⭐⭐ | 代码精简，注释适中 |
| **性能** | ⭐⭐⭐ | axis=-2 路径可能有性能开销 |
| **测试覆盖** | ⭐⭐⭐⭐ | 有完整测试 |
| **文档** | ⭐⭐⭐⭐⭐ | 文档准确，参数清晰 |

**综合评价:** 🟢 **可以合入**

---

*报告更新完毕 🦜*
