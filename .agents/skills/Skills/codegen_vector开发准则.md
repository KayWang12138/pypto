# Codegen Vector 开发准则

## 一、概述

Codegen（代码生成）层是 PyPTO 框架的核心组件，负责将 Operation 序列转换为可执行的 CCE 代码。Codegen Vector 专门处理向量运算的代码生成，生成调用 TileOP 的代码。

### 架构位置

```
┌─────────────────────────────────────────────────────────────────┐
│                    Function管理层                                │
│  interface/function/function.h                                  │
│  └─> operations_.push_back(op)                                  │
│  └─> CodeGen(function) 编译                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Codegen代码生成层                             │
│  codegen/cloudnpu/codegen_vector_binary.cpp                     │
│  └─> CodeGenOpCloudNPU::GenOpCode()                             │
│      └─> PrintBinaryStatic(param)                               │
│          生成: Binary_<float,...>(dst, src0, src1);             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TileOp执行层                                  │
│  interface/tileop/vector/binary.h                               │
│  └─> BinaryComputeImpl<ADD>(dst, src0, src1)                    │
│      └─> pto::TADD(dst, src0, src1)                             │
└─────────────────────────────────────────────────────────────────┘
```

## 二、目录结构

```
codegen/
├── cloudnpu/                      # CloudNPU 代码生成
│   ├── codegen_cloudnpu.cpp/.h   # CloudNPU 主代码生成
│   ├── codegen_op_cloudnpu.cpp/.h # Operation 代码生成基类
│   ├── codegen_vector.cpp        # 向量运算代码生成
│   ├── codegen_vector_binary.cpp # 二元运算代码生成
│   ├── codegen_vector_unary.cpp  # 一元运算代码生成
│   ├── codegen_vector_sort.cpp   # 排序运算代码生成
│   ├── codegen_scalar.cpp        # 标量运算代码生成
│   ├── codegen_mte.cpp           # 内存传输代码生成
│   ├── codegen_cube.cpp          # Cube 运算代码生成
│   ├── codegen_distributed.cpp   # 分布式代码生成
│   └── op_print_param_def.h      # 参数打印定义
├── stmt_mgr/                      # 语句管理
│   ├── codegen_for_block.cpp/.h  # For 循环代码生成
├── symbol_mgr/                    # 符号管理
│   ├── codegen_symbol.cpp/.h     # 符号表管理
├── utils/                         # 工具函数
│   ├── codegen_utils.cpp/.h      # 代码生成工具
│   ├── codegen_error.h           # 错误处理
│   └── parallel_execute.cpp/.h   # 并行执行
├── codegen.cpp/.h                # 代码生成基类
├── codegen_cce.cpp/.h            # CCE 代码生成
├── codegen_factory.h             # 工厂模式
└── codegen_op.cpp/.h             # Operation 代码生成基类
```

## 三、核心类设计

### 3.1 CodeGenOp 基类

```cpp
class CodeGenOp {
public:
    explicit CodeGenOp(const CodeGenOpCtx &ctx);
    virtual ~CodeGenOp() = default;
    
    virtual void Init(const Operation &ops);
    virtual std::string GenOpCode() const = 0;  // 纯虚函数，子类实现
    
protected:
    // Operation 信息
    const Operation &originalOp;
    Opcode opCode;
    std::string tileOpName;
    
    // 操作数信息
    int operand[MAX_OPERANDS];           // buffer id
    int operandWithMagic[MAX_OPERANDS];  // 带 magic 的操作数
    OperandType operandType[MAX_OPERANDS];
    DataType operandDtype[MAX_OPERANDS];
    
    // Shape 信息
    std::vector<int64_t> offset[MAX_OPERANDS];
    std::vector<int64_t> shape[MAX_OPERANDS];
    std::vector<int64_t> rawShape[MAX_OPERANDS];
    std::vector<int64_t> originShape[MAX_OPERANDS];
    std::vector<SymbolicScalar> dynamicOffset[MAX_OPERANDS];
    std::vector<SymbolicScalar> dynamicValidShape[MAX_OPERANDS];
    
    // 属性
    std::map<std::string, Any> opAttrs;
    
    // 符号管理器
    std::shared_ptr<SymbolManager> sm;
    
    // 标志位
    bool isSupportLayout;
    bool isDynamicFunction;
    bool isSupportDynamicAligned;
};
```

### 3.2 CodeGenOpCtx 上下文

```cpp
struct CodeGenOpCtx {
    std::shared_ptr<SymbolManager> symbolManager;
    Function &topFunc;
    Function &subFunc;
    const Operation &operation;
    const std::map<int, int> &locToOffset;
    bool isMainBlock;
    bool isDynamicAligned;
};
```

### 3.3 CodeGenOpCloudNPU 派生类

```cpp
class CodeGenOpCloudNPU : public CodeGenOp {
public:
    // 二元运算代码生成
    std::string GenBinaryOp() const;
    std::string GenBinaryOpWithTmp() const;
    
    // 一元运算代码生成
    std::string GenCastOp() const;
    std::string GenDupOp() const;
    
    // 打印函数
    std::string PrintBinaryStatic(const PrintBinaryParam &param) const;
    std::string PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const;
    std::string PrintBinaryTileTensor() const;
    
    // TileTensor 支持
    std::string QueryTileTensorNameByIdx(int idx) const;
    std::string GetLastUse() const;
};
```

## 四、开发规范

### 4.1 命名规范

| 类型 | 命名规则 | 示例 |
|------|----------|------|
| 代码生成函数 | Gen + 操作名 + Op | `GenBinaryOp`, `GenCastOp` |
| 打印函数 | Print + 操作名 + 场景 | `PrintBinaryStatic`, `PrintBinaryTileTensor` |
| 参数结构体 | Print + 操作名 + Param | `PrintBinaryParam`, `PrintUnaryParam` |
| 常量 | 大写 + 下划线 | `SHAPE_DIM4`, `MAX_OPERANDS` |

### 4.2 必须包含的头文件

```cpp
#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen_op_cloudnpu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
```

### 4.3 操作数索引定义

```cpp
// MISO: Multi-Input Single-Output
enum class MISOIdx {
    DST_IDX = 0,   // 输出
    SRC0_IDX = 1,  // 第一个输入
    SRC1_IDX = 2   // 第二个输入
};

// MIMO: Multi-Input Multi-Output
enum class MIMOIdx {
    DST_IDX = 0,   // 输出
    TMP_IDX = 1,   // 临时空间
    SRC0_IDX = 2,  // 第一个输入
    SRC1_IDX = 3   // 第二个输入
};
```

## 五、开发步骤

### 5.1 定义参数结构体

```cpp
struct PrintBinaryParam {
    std::string s0Var;       // 源操作数0变量名
    std::string s1Var;       // 源操作数1变量名
    std::string dVar;        // 目标变量名
    std::string src0DtypeStr; // 源操作数0数据类型字符串
    std::string src1DtypeStr; // 源操作数1数据类型字符串
    std::string dstDtypeStr;  // 目标数据类型字符串
};
```

### 5.2 实现代码生成函数

#### 5.2.1 主入口函数

```cpp
std::string CodeGenOpCloudNPU::GenBinaryOp() const {
    // 1. 获取变量名
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    
    // 2. 获取数据类型字符串
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    
    // 3. 处理偏移量
    auto offset0 = GetOperandStartOffset(ID0);
    auto offset1 = GetOperandStartOffset(ID1);
    auto offset2 = GetOperandStartOffset(ID2);
    
    if (!offset0.ConcreteValid() || offset0.Concrete() != 0) {
        dVar += "+" + GetOperandStartOffset(ID0).Dump();
    }
    if (!offset1.ConcreteValid() || offset1.Concrete() != 0) {
        s0Var += "+" + GetOperandStartOffset(ID1).Dump();
    }
    if (!offset2.ConcreteValid() || offset2.Concrete() != 0) {
        s1Var += "+" + GetOperandStartOffset(ID2).Dump();
    }
    
    // 4. 调用打印函数
    return PrintBinary({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
}
```

#### 5.2.2 静态 Shape 打印函数

```cpp
std::string CodeGenOpCloudNPU::PrintBinaryStatic(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    
    // 1. 归一化 Shape 到 4 维
    std::vector<int64_t> os0 = NormalizeShape(originShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> os1 = NormalizeShape(originShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    
    // 2. 构建模板参数列表
    std::ostringstream os;
    std::vector<std::string> paramList;
    
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*OS0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os0[i]));
    }
    paramList.emplace_back("/*OS1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(os1[i]));
    }
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    paramList.emplace_back("/*S0*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    paramList.emplace_back("/*S1*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    
    // 3. 添加广播参数
    int64_t brcOperandIdx = 0;
    if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        paramList.emplace_back(GetBrcOprandIdxStr(brcOperandIdx));
    }
    
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    
    // 4. 构建函数调用参数
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    
    // 5. 生成最终代码
    os << tileOpName.c_str() << "_<" << templateParam << ">" 
       << "(" << tiloOpCallParam << ");\n";
    
    return os.str();
}
```

#### 5.2.3 动态 Shape 打印函数

```cpp
std::string CodeGenOpCloudNPU::PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const {
    const std::string &dstDtypeStr = param.dstDtypeStr;
    const std::string &src0DtypeStr = param.src0DtypeStr;
    const std::string &src1DtypeStr = param.src1DtypeStr;
    const std::string &dVar = param.dVar;
    const std::string &s0Var = param.s0Var;
    const std::string &s1Var = param.s1Var;
    
    // 1. 归一化 Shape
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    
    // 2. 获取动态 Shape
    std::vector<SymbolicScalar> dynSrcShape0 = dynamicValidShape[ID1];
    std::vector<SymbolicScalar> dynSrcShape1 = dynamicValidShape[ID2];
    
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape0, SHAPE_DIM4 - dynamicValidShape[ID1].size(), 1);
    FillIntVecWithDummyInHead<SymbolicScalar>(dynSrcShape1, SHAPE_DIM4 - dynamicValidShape[ID2].size(), 1);
    
    // 3. 构建模板参数
    std::ostringstream os;
    std::vector<std::string> paramList;
    
    paramList.emplace_back(dstDtypeStr);
    paramList.emplace_back("/*DS*/");
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    // ... 添加其他参数
    
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    
    // 4. 构建函数调用参数（包含动态 Shape 表达式）
    paramList.clear();
    std::string dst = "(__ubuf__ " + dstDtypeStr + "*)" + dVar;
    std::string src0 = "(__ubuf__ " + src0DtypeStr + "*)" + s0Var;
    std::string src1 = "(__ubuf__ " + src1DtypeStr + "*)" + s1Var;
    paramList.emplace_back(dst);
    paramList.emplace_back(src0);
    paramList.emplace_back(src1);
    
    // 添加动态 Shape 表达式
    for (auto dynShape : dynSrcShape0) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    for (auto dynShape : dynSrcShape1) {
        paramList.emplace_back(SymbolicExpressionTable::BuildExpression(dynShape));
    }
    
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">" 
       << "(" << tiloOpCallParam << ");\n";
    
    return os.str();
}
```

#### 5.2.4 TileTensor 打印函数

```cpp
std::string CodeGenOpCloudNPU::PrintBinaryTileTensor() const {
    // 1. 查询 TileTensor 名称
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    
    // 2. 构建参数列表
    std::vector<std::string> tileOpCallParamList = {dstTensor, src0Tensor, src1Tensor};
    
    // 3. 构建模板参数列表
    std::vector<std::string> templateParamList;
    
    // 添加 LastUse
    std::string lastUse = GetLastUse();
    if (!lastUse.empty()) {
        templateParamList.emplace_back(lastUse);
    }
    
    // 添加广播参数
    int64_t brcOperandIdx = 0;
    if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
        templateParamList.emplace_back(GetBrcOprandIdxStr(brcOperandIdx));
    }
    
    // 4. 生成代码
    std::ostringstream oss;
    oss << tileOpName;
    if (!templateParamList.empty()) {
        oss << WrapParamByAngleBrackets(templateParamList);
    }
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    
    return oss.str();
}
```

#### 5.2.5 分发函数

```cpp
std::string CodeGenOpCloudNPU::PrintBinary(const PrintBinaryParam &param) const {
    // 根据不同场景选择不同的打印方式
    if (isSupportLayout) {
        return PrintBinaryTileTensor();  // TileTensor 模式
    }
    if (isDynamicFunction) {
        return PrintBinaryDynamicUnaligned(param);  // 动态 Shape 模式
    }
    return PrintBinaryStatic(param);  // 静态 Shape 模式
}
```

### 5.3 注册代码生成器

在 `codegen_op_cloudnpu.cpp` 中注册：

```cpp
std::string CodeGenOpCloudNPU::GenOpCode() const {
    switch (opCode) {
        case Opcode::OP_ADD:
        case Opcode::OP_SUB:
        case Opcode::OP_MUL:
        case Opcode::OP_DIV:
            return GenBinaryOp();
        case Opcode::OP_CAST:
            return GenCastOp();
        case Opcode::OP_DUPLICATE:
            return GenDupOp();
        // ... 添加其他操作
        default:
            ASSERT(GenCodeErr::UNSUPPORTED_OP, false) << "Unsupported opcode: " << static_cast<int>(opCode);
            return "";
    }
}
```

## 六、关键工具函数

### 6.1 Shape 处理

```cpp
// 归一化 Shape 到指定维度
std::vector<int64_t> NormalizeShape(const std::vector<int64_t> &shape, size_t targetDim);

// 填充虚拟维度
template <typename T>
void FillIntVecWithDummyInHead(std::vector<T> &vec, size_t dummyCount, const T &dummyValue);

// 连接字符串
std::string JoinString(const std::vector<std::string> &strs, const std::string &connector);
```

### 6.2 数据类型转换

```cpp
// 数据类型转 CCE 字符串
std::string DataType2CCEStr(DataType dtype);

// 示例：
// DT_FP32 -> "float"
// DT_FP16 -> "half"
// DT_INT32 -> "int32_t"
```

### 6.3 参数封装

```cpp
// 用尖括号封装模板参数
std::string WrapParamByAngleBrackets(const std::vector<std::string> &params);

// 用圆括号封装函数参数
std::string WrapParamByParentheses(const std::vector<std::string> &params);

// 示例：
// WrapParamByAngleBrackets({"float", "half"}) -> "<float, half>"
// WrapParamByParentheses({"dst", "src"}) -> "(dst, src)"
```

### 6.4 符号表达式

```cpp
// 构建符号表达式
std::string SymbolicExpressionTable::BuildExpression(const SymbolicScalar &scalar);

// 获取操作数起始偏移
Element GetOperandStartOffset(int operandIdx) const;
```

### 6.5 属性获取

```cpp
// 获取属性值
template <typename T>
bool GetAttr(const std::string &key, T &value) const;

// 示例：
int64_t brcOperandIdx = 0;
if (GetAttr(OpAttributeKey::brcbIdx, brcOperandIdx)) {
    // 使用 brcOperandIdx
}
```

## 七、代码生成模式

### 7.1 三种生成模式

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| Static | 静态 Shape | 编译期确定所有 Shape 参数 |
| DynamicUnaligned | 动态 Shape | 运行时传入 Shape 参数 |
| TileTensor | TileTensor 模式 | 使用 TileTensor 对象，更简洁 |

### 7.2 模式选择逻辑

```cpp
std::string PrintBinary(const PrintBinaryParam &param) const {
    if (isSupportLayout) {
        // 优先使用 TileTensor 模式
        return PrintBinaryTileTensor();
    }
    if (isDynamicFunction) {
        // 动态 Shape 使用 DynamicUnaligned 模式
        return PrintBinaryDynamicUnaligned(param);
    }
    // 默认使用 Static 模式
    return PrintBinaryStatic(param);
}
```

## 八、特殊场景处理

### 8.1 需要临时空间的操作

```cpp
std::string CodeGenOpCloudNPU::GenBinaryOpWithTmp() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC1_IDX));
    
    std::vector<std::string> tileOpCallParamList = {dstTensor, src0Tensor, src1Tensor, tmpTensor};
    
    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}
```

### 8.2 标量操作

```cpp
std::string CodeGenOpCloudNPU::GenVectorScalarOpWithTmp() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::DST_IDX));
    std::string tmpTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::TMP_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MIMOIdx::SRC0_IDX));
    
    // 处理标量值
    std::string srcScalar;
    if (extOperandVal.IsFloat()) {
        srcScalar = FormatFloat(extOperandVal.Cast<float>());
    } else if (extOperandVal.IsUnsigned() || extOperandVal.IsSigned()) {
        srcScalar = std::visit(
            [](const auto &val) -> std::string { return std::to_string(val); }, 
            extOperandVal.GetVariantData());
    }
    
    std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, srcScalar, tmpTensor};
    
    std::ostringstream oss;
    oss << tileOpName << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}
```

### 8.3 广播操作

```cpp
std::string GetBrcOprandIdxStr(int64_t brcbOperandIdx) {
    std::string ret = "TileOp::";
    switch (brcbOperandIdx) {
        case ToUnderlying(BroadcastOperand::NONE): 
            ret.append("BroadcastOperand::NONE"); 
            break;
        case ToUnderlying(BroadcastOperand::LEFT_OPERAND): 
            ret.append("BroadcastOperand::LEFT_OPERAND"); 
            break;
        case ToUnderlying(BroadcastOperand::RIGHT_OPERAND): 
            ret.append("BroadcastOperand::RIGHT_OPERAND"); 
            break;
        default: 
            ret.append("BroadcastOperand::NONE");
    }
    return ret;
}
```

### 8.4 数据传输操作

```cpp
std::string CodeGenOpCloudNPU::GenTransposeDataMove() const {
    bool isCopyLocalToGM = opCode == Opcode::OP_TRANSPOSE_MOVEOUT;
    unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
    unsigned localIdx = isCopyLocalToGM ? 1 : 0;
    
    std::string localVar = sm->QueryVarNameByTensorMagic(operandWithMagic[localIdx]);
    std::string gmVar = GenGmParamVar(gmIdx);
    
    // 处理偏移
    AppendLocalBufferVarOffset({
        {gmIdx, std::ref(gmVar)},
        {localIdx, std::ref(localVar)}
    });
    
    std::string localDtypeStr = DataType2CCEStr(operandDtype[localIdx]);
    std::string gmDtypeStr = DataType2CCEStr(operandDtype[gmIdx]);
    
    return PrintTransposeDataMove({gmIdx, localIdx, localVar, gmShape, localDtypeStr, gmDtypeStr});
}
```

## 九、错误处理

### 9.1 断言宏

```cpp
// 使用 ASSERT 宏进行错误检查
ASSERT(GenCodeErr::PRINT_FAILED, condition) << "error message";
ASSERT(GenCodeErr::UNSUPPORTED_OP, condition) << "Unsupported opcode";

// 错误码定义
namespace GenCodeErr {
    constexpr int PRINT_FAILED = 1;
    constexpr int UNSUPPORTED_OP = 2;
    constexpr int INVALID_PARAM = 3;
}
```

### 9.2 日志记录

```cpp
// 使用 CODEGEN_LOGI 记录信息日志
CODEGEN_LOGI("genBinaryOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(src0RawShape).c_str());
CODEGEN_LOGI("dynamic gmShape param: %s", IntVecToStr(gmShapeExpr).c_str());
```

## 十、最佳实践

### 10.1 代码组织

1. **分离生成逻辑**：每个操作类型有独立的生成函数
2. **使用参数结构体**：封装复杂参数，提高可读性
3. **统一命名规范**：遵循命名规范，便于维护

### 10.2 性能优化

1. **减少字符串拼接**：使用 `ostringstream` 提高效率
2. **预计算 Shape**：在循环外计算 Shape 相关信息
3. **避免重复查询**：缓存符号表查询结果

### 10.3 可维护性

1. **添加详细注释**：说明生成逻辑和参数含义
2. **使用工具函数**：封装常用操作，提高代码复用
3. **保持函数简洁**：每个函数只做一件事

## 十一、完整示例

```cpp
// codegen_vector_myop.cpp

#include "interface/utils/log.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen_op_cloudnpu.h"
#include "securec.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {

// 参数结构体
struct PrintMyOpParam {
    std::string s0Var;
    std::string s1Var;
    std::string dVar;
    std::string src0DtypeStr;
    std::string src1DtypeStr;
    std::string dstDtypeStr;
};

// 静态 Shape 打印
std::string CodeGenOpCloudNPU::PrintMyOpStatic(const PrintMyOpParam &param) const {
    std::vector<int64_t> s0 = NormalizeShape(rawShape[ID1], SHAPE_DIM4);
    std::vector<int64_t> s1 = NormalizeShape(rawShape[ID2], SHAPE_DIM4);
    std::vector<int64_t> ds = NormalizeShape(rawShape[ID0], SHAPE_DIM4);
    
    std::ostringstream os;
    std::vector<std::string> paramList;
    
    // 构建模板参数
    paramList.emplace_back(param.dstDtypeStr);
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(ds[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s0[i]));
    }
    for (int i = 0; i < SHAPE_DIM4; ++i) {
        paramList.emplace_back(std::to_string(s1[i]));
    }
    
    std::string templateParam = JoinString(paramList, CONN_COMMA);
    
    // 构建函数参数
    paramList.clear();
    std::string dst = "(__ubuf__ " + param.dstDtypeStr + "*)" + param.dVar;
    std::string src0 = "(__ubuf__ " + param.src0DtypeStr + "*)" + param.s0Var;
    std::string src1 = "(__ubuf__ " + param.src1DtypeStr + "*)" + param.s1Var;
    paramList.insert(paramList.end(), {dst, src0, src1});
    
    std::string tiloOpCallParam = JoinString(paramList, CONN_COMMA);
    os << tileOpName.c_str() << "_<" << templateParam << ">" 
       << "(" << tiloOpCallParam << ");\n";
    
    return os.str();
}

// TileTensor 打印
std::string CodeGenOpCloudNPU::PrintMyOpTileTensor() const {
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
    
    std::vector<std::string> tileOpCallParamList = {dstTensor, src0Tensor, src1Tensor};
    
    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(tileOpCallParamList) << STMT_END;
    return oss.str();
}

// 分发函数
std::string CodeGenOpCloudNPU::PrintMyOp(const PrintMyOpParam &param) const {
    if (isSupportLayout) {
        return PrintMyOpTileTensor();
    }
    return PrintMyOpStatic(param);
}

// 主入口函数
std::string CodeGenOpCloudNPU::GenMyOp() const {
    std::string s0Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID1]);
    std::string dVar = sm->QueryVarNameByTensorMagic(operandWithMagic[ID0]);
    std::string s1Var = sm->QueryVarNameByTensorMagic(operandWithMagic[ID2]);
    
    CODEGEN_LOGI("genMyOp %s, src0RawShape is %s", tileOpName.c_str(), IntVecToStr(rawShape[ID1]).c_str());
    
    std::string dstDtypeStr = DataType2CCEStr(operandDtype[ID0]);
    std::string src0DtypeStr = DataType2CCEStr(operandDtype[ID1]);
    std::string src1DtypeStr = DataType2CCEStr(operandDtype[ID2]);
    
    // 处理偏移
    auto offset0 = GetOperandStartOffset(ID0);
    auto offset1 = GetOperandStartOffset(ID1);
    auto offset2 = GetOperandStartOffset(ID2);
    
    if (!offset0.ConcreteValid() || offset0.Concrete() != 0) {
        dVar += "+" + offset0.Dump();
    }
    if (!offset1.ConcreteValid() || offset1.Concrete() != 0) {
        s0Var += "+" + offset1.Dump();
    }
    if (!offset2.ConcreteValid() || offset2.Concrete() != 0) {
        s1Var += "+" + offset2.Dump();
    }
    
    return PrintMyOp({s0Var, s1Var, dVar, src0DtypeStr, src1DtypeStr, dstDtypeStr});
}

} // namespace npu::tile_fwk
```

## 十二、生成的代码示例

### 12.1 静态 Shape 代码

```cpp
Binary_<float, /*OS0*/1,1,128,128, /*OS1*/1,1,128,128, /*DS*/1,1,128,128, /*S0*/1,1,128,128, /*S1*/1,1,128,128>(
    (__ubuf__ float *)dst, 
    (__ubuf__ float *)src0, 
    (__ubuf__ float *)src1
);
```

### 12.2 动态 Shape 代码

```cpp
Binary_<float, /*DS*/1,1,128,128, /*S0*/1,1,128,128, /*S1*/1,1,128,128>(
    (__ubuf__ float *)dst, 
    (__ubuf__ float *)src0, 
    (__ubuf__ float *)src1,
    RUNTIME_COA_GET_PARAM_VALID_SHAPE(1, 1, 0),
    RUNTIME_COA_GET_PARAM_VALID_SHAPE(1, 1, 1),
    RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0),
    RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)
);
```

### 12.3 TileTensor 代码

```cpp
TAdd<LastUse3Dim<0, 0, 0>, TileOp::BroadcastOperand::NONE>(ubTensor_0, ubTensor_1, ubTensor_2);
```

## 十三、参考文件

- [codegen_vector_binary.cpp](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/codegen/cloudnpu/codegen_vector_binary.cpp) - 二元运算代码生成
- [codegen_vector_unary.cpp](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/codegen/cloudnpu/codegen_vector_unary.cpp) - 一元运算代码生成
- [codegen_vector.cpp](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/codegen/cloudnpu/codegen_vector.cpp) - 向量运算代码生成
- [codegen_op.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/codegen/codegen_op.h) - CodeGenOp 基类
- [codegen_utils.h](file:///N:/Users/w00933451/GitCode/Permute/pypto_per/framework/src/codegen/utils/codegen_utils.h) - 工具函数
