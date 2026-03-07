# Framework 目录错误码映射文档

本文档记录了 `framework` 目录下所有包含 `FUNCTION_LOG` 宏的文件中典型错误信息的错误码映射关系。

## 错误码定义

所有错误码均定义在 `framework/src/interface/inner/error_code.h` 中，格式为 `F1XXXX`（最多6位字符）。

## 错误码分类

### 一、类型错误 (F10000 - F1000D)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10000 | INVALID_GRAPH_TYPE | 图类型不匹配 | function.cpp | "Current graph type" |
| F10001 | NULL_POINTER | 空指针异常 | function.cpp | "must have a CallOpAttribute" |
| F10002 | INVALID_FUNCTION_TYPE | 函数类型无效 | function.cpp | "Not support connecting other type of function" |
| F10003 | INVALID_TYPE | 无效类型 | config_manager_ng.cpp | "invalid type" |
| F10004 | TYPE_MISMATCH_IMMEDIATE | 类型不匹配：期望立即数 | symbolic_scalar.cpp | "Mismatch immediate type" |
| F10005 | TYPE_MISMATCH_SYMBOL | 类型不匹配：期望符号 | symbolic_scalar.cpp | "Mismatch symbol type" |
| F10006 | TYPE_MISMATCH_EXPRESSION | 类型不匹配：期望表达式 | symbolic_scalar.cpp | "Mismatch expression type" |
| F10007 | TYPE_MISMATCH | 类型不匹配 | tensor.cpp | "Mismatch dimension" |
| F10008 | DATATYPE_MISMATCH | 数据类型不匹配 | tensor.cpp | "GetDataType() == DT_INT32" |
| F10009 | INVALID_SHAPE | 无效形状 | tensor.cpp | "Invalid shape" |
| F1000A | NZ_ALIGNMENT_ERROR | NZ格式对齐错误 | tensor.cpp | "inner axis shape must be 32-byte aligned" |
| F1000B | SHAPE_MISMATCH | 形状不匹配 | function.cpp, logical_tensor.cpp | "Shape mismatch" |
| F1000C | GRAPH_TYPE_MISMATCH | 图类型不匹配 | program.cpp, recorder.cpp | "caller graphType, callee graphType" |
| F1000D | FUNCTION_TYPE_MISMATCH | 函数类型不匹配 | recorder.cpp | "funcType" |

### 二、索引/边界错误 (F10100 - F10105)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10100 | INDEX_OUT_OF_BOUNDS | 索引越界 | function.cpp | "The param address is not stored" |
| F10101 | AXIS_OUT_OF_RANGE | 轴索引越界 | tensor.cpp | "Axis index is out of range" |
| F10102 | SHAPE_OUT_OF_BOUNDS.bounds | 形状越界 | logical_tensor.cpp | "Their size actually are" |
| F10103 | FILE_SIZE_OUT_OF_RANGE | 文件大小超出范围 | file_utils.cpp | "File size is not within range" |
| F10104 | VIEW_DIMENSION_MISMATCH | 视图维度不匹配 | logical_tensor.cpp | "Tensor.view, shape must be same dimension" |
| F10105 | VIEW_OFFSET_MISMATCH | 视图偏移不匹配 | logical_tensor.cpp | "Tensor.view, offset must be same dimension" |

### 三、重复/存在性错误 (F10200 - F1020A)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10200 | DUPLICATE_OPERATION | 重复操作 | function.cpp | "Duplicate OpMagic found" |
| F10201 | DUPLICATE_NAME | 重复名称 | recorder.cpp | "Forbid duplicate name of loop idx" |
| F10202 | RESOURCE_ALREADY_EXISTS | 资源已存在 | tensor_slot.cpp | "already exists in inputSlotDict/outputSlotDict" |
| F10203 | RESOURCE_NOT_FOUND | 资源未找到 | program.cpp | "Failed to find main function" |
| F10204 | OPERATION_NOT_FOUND | 操作未找到 | function.cpp | "Producer not found in opToIndex" |
| F10205 | FUNCTION_NOT_FOUND | 函数未找到 | program.cpp | "Cannot find function iter by magic" |
| F10206 | MAGIC_NOT_FOUND | 魔法数未找到 | logical_tensor.cpp | "rawTensorDict doesn't have magic" |
| F10207 | MAGIC_NAME_EXISTS | 魔法名已存在 | program.cpp | "funcMagicName is already in function map" |
| F10208 | EXPRESSION_NOT_FOUND | 表达式未找到 | symbolic_scalar.h | "has not been found in expressionIndexTable" |
| F10209 | SYMBOL_NOT_FOUND | 符号未找到 | symbolic_scalar.h | "has not been found in symbolValueDict" |
| F1020A | TENSOR_NOT_FOUND | 张量未找到 | tensor_slot.cpp | "TensorSlot not found in slotIndexDict" |

### 四、循环/依赖错误 (F10300 - F10306)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10300 | CYCLE_DETECTED | 循环依赖检测 | function.cpp | "Cycle detected" |
| F10301 | LOOP_BEGIN_END_NESTED | LoopBegin/LoopEnd嵌套错误 | symbolic_scalar.h | LoopBegin或LoopEnd嵌套在表达式中 |
| F10302 | LOOP_INDEX_NAME_DUPLICATE | 循环索引名称重复 | recorder.cpp | "Forbid duplicate name of loop idx" |
| F10303 | UNROLL_TIMES_MUST_BE_POSITIVE | 展开次数必须大于0 | recorder.cpp | "unrollTimes must larger than zero" |
| F10304 | UNROLL_TIMES_EMPTY | 展开次数列表为空 | recorder.cpp | "unrollTimes_ is empty" |
| F10305 | MUST_HAVE_UNROLL_ONE | 必须包含展开次数1 | recorder.cpp | "Must have unroll 1 if user defined custom unroll times" |
| F10306 | UNROLL_TIMES_ALREADY_EXISTS | 展开次数已存在 | recorder.cpp | "unrollTimes already exists in visited" |

### 五、操作数/参数错误 (F10400 - F10409)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10400 | INVALID_OPERATION_OPERAND | 操作数无效 | function.cpp | "index out of bounds for oOperand size" |
| F10401 | INVALID_OPERATION | 无效操作 | function.cpp | "unexpected behavior" |
| F10402 | OPERAND_COUNT_MISMATCH | 操作数数量不匹配 | symbolic_scalar.h | "Lvalue/Rvalue size mismatch" |
| F10403 | OPERAND_SIZE_INVALID | 操作数大小无效 | symbolic_scalar.h | "immediateList.size() invalid" |
| F10404 | PARAM_NOT_STORED | 参数未存储 | function.cpp | "The param address is not stored" |
| F10405 | NO_ACTIVE_FUNCTION | 无活动函数 | program.cpp | "No active function to add operation" |
| F10406 | NO_PARENT | 无父函数 | program.cpp, tensor_slot.cpp | "doesn't have a parent function" |
| F10407 | STACK_NULL | 栈为空 | program.cpp | "The stack of functionMagicName is null" |
| F10408 | STACK_UNDERFLOW | 栈下溢 | config_manager_ng.cpp | "No scope to pop" |
| F10409 | POINTER_LOST | 指针丢失 | program.cpp | "loss of pointer" |

### 六、状态/一致性错误 (F10500 - F1050E)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10500 | ILLEGAL_STATE | 非法状态 | function.cpp | "Not support connecting other type of function" |
| F10501 | INCONSISTENCY_ERROR | 一致性错误 | - | - |
| F10502 | TENSOR_CONSISTENCY_ERROR | 张量一致性错误 | function.cpp | "Output operand is found in incasts" |
| F10503 | PRODUCER_CONSUMER_ERROR | 生产者-消费者关系错误 | function.cpp | "Tensor is not a consumer/producer of the operation" |
| F10504 | GROUP_ID_MISMATCH | 组ID不匹配 | function.cpp | "Operation GroupID mismatch" |
| F10505 | ELEMENT_KEY_MISMATCH | 元素键不匹配 | symbolic_scalar.cpp | "elementKey_ != key" |
| F10506 | TITLE_MISMATCH | 标题不匹配 | symbolic_scalar.cpp | "title_ != title" |
| F10507 | SIZE_MISMATCH | 大小不匹配 | symbolic_scalar.h | "symTable.size() != symExprTable.size()" |
| F10508 | UNINITIALIZED_ASSIGNMENT | 未初始化赋值 | tensor_slot.cpp | "Assigning uninitialized Tensor variable is forbidden" |
| F10509 | SELF_ASSIGNMENT | 自赋值禁止 | tensor.cpp | "Prohibit self-assignment" |
| F1050A | NO_DIMENSIONS | 无维度 | tensor.cpp | "Tensor has no dimensions" |
| F1050B | STORAGE_NULL | 存储为空 | tensor.cpp | "storage_->tensor != nullptr" |
| F1050C | NOT_IN_INPUT_LIST | 不在输入列表中 | tensor.cpp | "Tensor is not in input tensor list" |
| F1050D | NOT_UNDER_DYNAMIC_FUNCTION | 不在动态函数中 | tensor.cpp | "Not under dynamic function" |
| F1050E | SET_CURRENT_FAILED | 设置当前失败 | program.cpp | "Failed to set current function" |

### 七、配置错误 (F10600 - F1060A)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10600 | CONFIG_FILE_READ_FAILED | 配置文件读取失败 | config_manager.cpp, ini_parser.cpp | "ReadJsonFile failed", "ReadINIFile failed" |
| F10601 | CONFIG_FILE_OPEN_FAILED | 配置文件打开失败 | config_manager_ng.cpp, ini_parser.cpp | "Failed to open ini file" |
| F10602 | CONFIG_ATTR_NOT_FOUND | 配置属性未找到 | ini_parser.cpp | "Cannot find attribute 'version' from ini file" |
| F10603 | CONFIG_INVALID_TYPE | 配置类型无效 | config_manager_ng.cpp | "invalid type" |
| F10604 | CONFIG_FIELD_MISSING | 配置字段缺失 | config_manager_ng.cpp | "field['type', 'properties'] not found" |
| F10605 | CONFIG_SET_OPTION_FAILED | 配置选项设置失败 | config_manager_ng.cpp | "Failed to set option" |
| F10606 | CONFIG_KEY_NOT_LOADED | 配置键未加载 | config_manager_ng.cpp | "key has been not loaded" |
| F10607 | CONFIG_VALUE_CONVERT_FAILED | 配置值转换失败 | ini_parser.cpp | "Cannot convert string to size_t"” |
| F10608 | CONFIG_VALUE_OVERFLOW | 配置值溢出 | ini_parser.cpp | "Overflow data" |
| F10609 | CONFIG_SCOPE_NULL | 配置作用域为空 | config_manager_ng.cpp | "Cannot push a null scope" |
| F1060A | CONFIG_SCOPE_STACK_UNDERFLOW | 配置作用域栈下溢 | config_manager_ng.cpp | "No scope to pop" |

### 八、文件I/O错误 (F10700 - F10712)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型倒误消息 |
|--------|--------|------|----------|--------------|
| F10700 | FILE_PATH_TOO_LONG | 文件路径过长 | file_utils.cpp | "file path is too long" |
| F10701 | FILE_PATH_NOT_EXIST | 路径不存在 | file_utils.cpp | "path is not exist" |
| F10702 | FILE_PATH_INVALID | 路径无效 | file_utils.cpp | "Path is too long" |
| F10703 | FILE_NAME_EMPTY | 文件名为空 | file_utils.cpp | "File name is empty" |
| F10704 | FILE_BIN_PATH_INVALID | 二进制文件路径无效 | file_utils.cpp | "Bin file path is not valid" |
| F10705 | DIR_CREATE_FAILED | 创建目录失败 | file_utils.cpp, config_manager.cpp | "Create dir failed" |
| F10706 | DIR_DELETE_FAILED | 删除目录失败 | file_utils.cpp | "Delete dir failed" |
| F10707 | DIR_OPEN_FAILED | 打开目录失败 | file_utils.cpp | "Open directory failed" |
| F10708 | DIR_GETCWD_FAILED | 获取当前工作目录失败 | file_utils.cpp | "failed to call getcwd()" |
| F10709 | FILE_STAT_FAILED | 文件状态获取失败 | file_utils.cpp | "Stat file failed" |
| F1070A | FILE_NOT_FILE | 不是文件 | file_utils.cpp | "is not a file" |
| F1070B | FILE_DELETE_FAILED | 删除文件失败 | file_utils.cpp | "Delete file failed" |
| F1070C | FILE_OPEN_FAILED | 打开文件失败 | file_utils.cpp | "Open file failed" |
| F1070D | FILE_ALREADY_OPEN | 文件已打开 | file_utils.cpp | "file is already open" |
| F1070E | FILE_CONVERT_JSON_FAILED | 转换为JSON失败 | file_utils.cpp | "Fail to convert file to Json" |
| F1070F | FILE_READ_FAILED | 读取文件失败 | file_utils.cpp | "read file failed" |
| F10710 | FILE_READ_EXCEPTION | 文件读取异常 | file_utils.cpp | "Fail to read file" |
| F10711 | FILE_WRITE_FAILED | 文件写入失败 | file_utils.cpp | "Failed open file" |
| F10712 | FILE_LOCK_FAILED | 文件锁定失败 | file_utils.cpp | "Fail to lock file" |

### 九、编译错误 (F10800 - F10804)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10800 | COMPILE_BINARY_OPEN_FAILED | 二进制文件打开失败 | symbolic_scalar.cpp | "open binary file name failed" |
| F10801 | COMPILE_GCC_FAILED | GCC编译失败 | symbolic_scalar.cpp | GCC命令执行失败 |
| F10802 | COMPILE_ASSEMBLE_FAILED | 汇编失败 | symbolic_scalar.cpp | 汇编命令执行失败 |
| F10803 | COMPILE_OBJCOPY_FAILED | Objcopy失败 | symbolic_scalar.cpp | Objcopy命令执行失败 |
| F10804 | COMPILE_BINARY_READ_FAILED | 二进制文件读取失败 | symbolic_scalar.cpp | 读取二进制文件大小不匹配 |

### 十、系统错误 (F10900 - F10904)

| 错误码 | 枚举值 | 描述 | 相关文件 | 典型错误消息 |
|--------|--------|------|----------|--------------|
| F10900 | EXCEPTION_CAUGHT | 捕获异常 | error.h | "Caught exception" |
| F10901 | SEGMENT_FAULT | 段错误 | error.h | "segment fault" |
| F10902 | UNEXPECTED_BEHAVIOR | 未定义行为 | logical_tensor.cpp, symbolic_scalar.cpp | "unexpected behavior" |
| F10903 | UNDEFINED_BEHAVIOR | 未定义行为 | symbolic_scalar.h, symbolic_scalar.cpp | "undefined behavior" |
| F10904 | KIND_UNDEFINED | 类型未定义 | symbolic_scalar.h | "undefined kind" |

## 错误码使用示例

### 类型错误示例
```cpp
// 空指针异常
ASSERT(callAttr != nullptr) << "Operation at index " << i << " must have a CallOpAttribute";
// 错误码: F10001 (NULL_POINTER)

// 类型不匹配：期望立即数
ASSERT(IsImmediate()) << "Mismatch immediate type: " << SymbolicScalarKind2Name(Kind());
// 错误码: F10004 (TYPE_MISMATCH_IMMEDIATE)

// 数据类型不匹配
CHECK(t.GetDataType() == DT_INT32) << "DataType mismatch";
// 错误码: F10008 (DATATYPE_MISMATCH)
```

### 索引/边界错误示例
```cpp
// 索引越界
ASSERT(explicitArgAddrs_.size() > static_cast<uint64_t>(index)) << "The param address is not stored.";
// 错误码: F10100 (INDEX_OUT_OF_BOUNDS)

// 轴索引越界
ASSERT(axis >= 0 && axis < Dim()) << "Axis index " << axis << " is out of range";
// 错误码: F10101 (AXIS_OUT_OF_RANGE)
```

### 重复/存在性错误示例
```cpp
// 重复操作
ASSERT(opToSubgraph.find(op.GetOpMagic()) == opToSubgraph.end())
    << "Same op magic shall only appear once.";
// 错误码: F10200 (DUPLICATE_OPERATION)

// 资源未找到
FUNCTION_LOGE("Failed to find main function.");
// 错误码: F10203 (RESOURCE_NOT_FOUND)
```

### 循环/依赖错误示例
```cpp
// 循环依赖检测
ASSERT(!cycleDetection(op.get(), cycleDetection)) << errorMsg;
// 错误码: F10300 (CYCLE_DETECTED)

// 展开次数必须大于0
CHECK(unrollTimes > 0) << "unrollTimes must larger than zero!";
// 错误码: F10303 (UNROLL_TIMES_MUST_BE_POSITIVE)
```

### 操作数/参数错误示例
```cpp
// 操作数无效
ASSERT(index < callop->oOperand.size())
    << "Index " << index << " out of bounds for oOperand size";
// 错误码: F10400 (INVALID_OPERATION_OPERAND)

// 栈为空
ASSERT(!funcMagicNameStack_.empty()) << "The stack of functionMagicName is null.";
// 错误码: F10407 (STACK_NULL)
```

### 状态/一致性错误示例
```cpp
// 组ID不匹配
ASSERT(operation->GroupID() == i)
    << "Operation GroupID mismatch: Expected: " << i << ", Actual: " << operation->GroupID();
// 错误码: F10504 (GROUP_ID_MISMATCH)

// 自赋值禁止
CHECK(!selfAssign) << "Prohibit self-assignment.";
// 错误码: F10509 (SELF_ASSIGNMENT)
```

### 配置错误示例
```cpp
// 配置文件读取失败
FUNCTION_LOGE("ReadJsonFile failed.");
// 错误码: F10600 (CONFIG_FILE_READ_FAILED)

// 配置属性未找到
FUNCTION_LOGE("Cannot find attribute 'version' from ini file.");
// 错误码: F10602 (CONFIG_ATTR_NOT_FOUND)
```

### 文件I/O错误示例
```cpp
// 文件路径过长
FUNCTION_LOGI("file path %s is too long.", path.c_str());
// 错误码: F10700 (FILE_PATH_TOO_LONG)

// 打开文件失败
FUNCTION_LOGW("Open file [%s] failed.", filePath.c_str());
// 错误码: F1070C (FILE_OPEN_FAILED)
```

### 编译错误示例
```cpp
// 二进制文件打开失败
if (fbin == nullptr) {
    FUNCTION_LOGE("open binary file name failed");
    return {};
}
// 错误码: F10800 (COMPILE_BINARY_OPEN_FAILED)

// GCC编译失败
ASSERT(system(cmdGcc.c_str()) == 0);
// 错误码: F10801 (COMPILE_GCC_FAILED)
```

### 系统错误示例
```cpp
// 捕获异常
FUNCTION_LOGE("Caught exception: %s", e.what());
// 错误码: F10900 (EXCEPTION_CAUGHT)

// 段错误
FUNCTION_LOGE("segment fault!!!\n%s", info);
// 错误码: F10901 (SEGMENT_FAULT)
```

## 错误码设计原则

1. **格式统一**: 所有错误码格式为 `F1XXXX`（最多6位字符）
2. **按类型分类**: 错误码按错误类型而非文件分类，便于理解和维护
3. **预留空间**: 每个类别预留足够的错误码空间（FF个），便于后续扩展
4. **语义清晰**: 错误码枚举名称应清晰表达错误含义
5. **唯一性**: 每个错误码对应一种特定的错误类型，不应混用
6. **可追溯**: 错误码应能追溯到具体的错误源和上下文

## 错误码分配规则

| 错误类别 | 错误码范围 | 说明 |
|----------|------------|------|
| 类型错误 | F10000 - F100FF | 类型不匹配、无效类型等 |
| 索引/边界错误 | F10100 - F101FF | 索引越界、边界检查失败 |
| 重复/存在性错误 | F10200 - F102FF | 重复、资源未找到等 |
| 循环/依赖错误 | F10300 - F103FF | 循环检测、展开次数错误 |
| 操作数/参数错误 | F10400 - F104FF | 操作数无效、参数错误 |
| 状态/一致性错误 | F10500 - F105FF | 状态不一致、逻辑错误 |
| 配置错误 | F10600 - F106FF | 配置文件读写、解析错误 |
| 文件I/O错误 | F10700 - F107FF | 文件路径、目录操作、文件读写错误 |
| 编译错误 | F10800 - F108FF | 编译、汇编、链接错误 |
| 系统错误 | F10900 - F109FF | 异常捕获、段错误、未定义行为 |

## 维护说明

1. **添加新错误码**:
   - 在 `error_code.h` 中定义新的枚举值
   - 确保错误码在预留范围内
   - 更新本文档的映射表

2. **错误码格式**:
   - 使用十六进制表示
   - 格式为 `F1XXXX`（最多6位字符）
   - 示例: F10000, F10101, F1020A

3. **文档同步**:
   - 定期检查文档与代码的一致性
   - 确保错误码映射准确
   - 更新使用示例

4. **错误处理建议**:
   - 生产环境中应使用异常或其他错误码机制替代 ASSERT
   - 错误信息应包含足够的上下文信息，便于调试
   - 考虑错误恢复策略，避免程序崩溃

## 相关文件清单

### 包含 FUNCTION_LOG 宏的文件
- `framework/include/tilefwk/pypto_fwk_log.h` - 日志宏定义
- `framework/src/interface/configs/config_manager.cpp` - 配置管理
- `framework/src/interface/configs/config_manager_ng.cpp` - 新配置管理
- `framework/src/interface/configs/ini_parser.cpp` - INI解析器
- `framework/src/interface/configs/platform.cpp` - 平台配置
- `framework/src/interface/function/function.cpp` - 函数实现
- `framework/src/interface/program/program.cpp` - 程序管理
- `framework/src/interface/program/recorder.cpp` - 记录器
- `framework/src/interface/tensor/logical_tensor.cpp` - 逻辑张量
- `framework/src/interface/tensor/raw_tensor.cpp` - 原始张量
- `framework/src/interface/tensor/raw_tensor.h` - 原始张量头文件
- `framework/src/interface/tensor/symbolic_scalar.cpp` - 符号标量实现
- `framework/src/interface/tensor/symbolic_scalar.h` - 符号标量定义
- `framework/src/interface/tensor/tensor.cpp` - 张量
- `framework/src/interface/tensor/tensormap.cpp` - 张量映射
- `framework/src/interface/tensor/tensor_slot.cpp` - 张量槽
- `framework/src/interface/utils/error.h` - 错误处理
- `framework/src/interface/utils/file_utils.cpp` - 文件工具

### 错误码定义文件
- `framework/src/interface/inner/error_code.h` - 错误码定义
