# Permute Operation 交付件缺失清单

## 一、已完成的交付件

| 交付件类型 | 文件路径 | 状态 |
|-----------|---------|------|
| Operation 定义 | `interface/operation/vector/permute.h` | ✅ 已完成 |
| Operation 实现 | `interface/operation/vector/permute.cpp` | ✅ 已完成 |
| Opcode 定义 | `interface/operation/opcode.h` | ✅ 已完成 |
| Opcode 注册 | `interface/operation/opcode.cpp` | ✅ 已完成 |
| TileOp 定义 | `interface/tileop/vector/permute.h` | ✅ 已完成 |
| Codegen 映射 | `codegen/cloudnpu/codegen_op_cloudnpu.cpp` | ✅ 已完成 |
| Codegen 头文件 | `codegen/cloudnpu/codegen_op_cloudnpu.h` | ✅ 已完成 |
| Codegen 实现 | `codegen/cloudnpu/codegen_vector_unary_with_tmp.cpp` | ✅ 已完成 |
| C++ ST 测试 | `tests/st/operation/src/test_permute_operation.cpp` | ✅ 已存在 |
| 测试用例 CSV | `tests/st/operation/test_case/Permute_st_test_cases.csv` | ✅ 已存在 |
| Golden 函数 | `tests/st/operation/python/vector_operator_golden.py` | ✅ 已存在 |

---

## 二、缺失的交付件

### 2.1 测试用例 JSON 文件 ❌

**文件路径**: `tests/st/operation/test_case/Permute_st_test_cases.json`

**说明**: 当前只有 CSV 格式的测试用例文件，缺少 JSON 格式文件。JSON 文件通常包含更详细的测试配置信息。

**参考示例** (其他算子的 JSON 格式):
```json
[
    {
        "case_name": "Permute_test_0",
        "input_shape": [128, 128],
        "input_dtype": "fp16",
        "input_format": "ND",
        "input_datarange": [0, 1],
        "output_shape": [128, 128],
        "output_dtype": "fp16",
        "output_format": "ND",
        "view_shape": [128, 128],
        "tile_shape": [128, 128],
        "perm": [1, 0]
    }
]
```

---

### 2.2 算子文档 ❌

**文件路径**: `docs/operators/permute.md`

**说明**: 缺少算子使用文档，应包含：
- 功能描述
- 接口说明
- 参数说明
- 支持的数据类型
- 使用示例
- 性能说明

---

### 2.4 CMakeLists.txt 更新 ⚠️

**文件路径**: `tests/st/operation/CMakeLists.txt`

**说明**: 需要确认 `test_permute_operation.cpp` 是否已添加到编译目标中。

**需要添加的内容**:
```cmake
# 在 tests/st/operation/CMakeLists.txt 中
st_add_op_test(test_permute_operation)
```

---

## 三、交付件完整性检查清单

### 3.1 核心代码文件 (必须)

| 文件 | 路径 | 状态 |
|-----|------|------|
| permute.h | interface/operation/vector/ | ✅ |
| permute.cpp | interface/operation/vector/ | ✅ |
| permute.h | interface/tileop/vector/ | ✅ |
| codegen_op_cloudnpu.cpp | codegen/cloudnpu/ | ✅ |
| codegen_op_cloudnpu.h | codegen/cloudnpu/ | ✅ |
| codegen_vector_unary_with_tmp.cpp | codegen/cloudnpu/ | ✅ |
| opcode.h | interface/operation/ | ✅ |
| opcode.cpp | interface/operation/ | ✅ |

### 3.2 测试文件 (必须)

| 文件 | 路径 | 状态 |
|-----|------|------|
| test_permute_operation.cpp | tests/st/operation/src/ | ✅ |
| Permute_st_test_cases.csv | tests/st/operation/test_case/ | ✅ |
| Permute_st_test_cases.json | tests/st/operation/test_case/ | ❌ 缺失 |

### 3.3 Golden 文件 (必须)

| 文件 | 路径 | 状态 |
|-----|------|------|
| vector_operator_golden.py | tests/st/operation/python/ | ✅ (已包含 Permute) |

### 3.4 文档文件 (推荐)

| 文件 | 路径 | 状态 |
|-----|------|------|
| permute.md | docs/operators/ | ❌ 缺失 |

---

## 四、优先级建议

| 优先级 | 交付件 | 说明 |
|-------|-------|------|
| **P0 (必须)** | Permute_st_test_cases.json | ST 测试必需 |
| **P0 (必须)** | CMakeLists.txt 更新 | 编译必需 |
| **P2 (推荐)** | 算子文档 | 用户参考 |

---

## 五、JSON 测试用例生成建议

可以从现有的 CSV 文件转换生成 JSON 文件，CSV 内容如下：

```csv
case_name,input_shape,input_dtype,input_format,input_datarange,output_shape,output_dtype,output_format,view_shape,tile_shape,perm
Permute_test_0,"[128, 128]",fp16,ND,"[0, 1]","[128, 128]",fp16,ND,"[128, 128]","[128, 128]","[1,0]"
...
```

建议使用 Python 脚本或手动转换为 JSON 格式。

---

## 六、总结

### 已完成: 11 项
- Operation 层代码 (4 项)
- TileOP 层代码 (1 项)
- Codegen 层代码 (3 项)
- ST 测试代码 (2 项)
- Golden 函数 (1 项)

### 缺失: 2 项
1. **Permute_st_test_cases.json** - 测试用例 JSON 文件
2. **permute.md** - 算子文档

### 需确认: 1 项
- CMakeLists.txt 是否已包含 test_permute_operation.cpp
