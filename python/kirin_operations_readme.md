[toc]

## 1. 比较计算

### 1.1. pypto.eq

#### 1.1.1. 算子计算原理
逐元素等于比较运算。
参考：pypto/docs/api/operation/pypto-eq.md
#### 1.1.2. pypto前段接口（python）以及支持范围
函数原型
```python
eq(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```
参数说明
| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor\float\Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

约束说明

1.  input 和 other 类型须保持一致。
2.  **支持一维广播。**
#### 1.1.3. c++ tensor graph接口
```c++
// 根据OpType区分具体的操作类型
// 比较两个Tensor
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Tensor和Scalar
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Scalar和Tensor
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}
```
<br>

```c++
enum class OpType {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
};
```
#### 1.1.4. c++ tile graph接口
```c++
// 两个Tensor间逐元素比较  对应的pto instruction是TCMP
void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

// Tensor和Scalar逐元素比较 对应的pto instruction是TCMPS
void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}
```
#### 1.1.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TCMP:比较两个 Tile 并写入一个打包的谓词掩码。
TCMPS:将 Tile 与标量比较并写入逐元素比较结果。

TStore：ub to gm搬运
#### 1.1.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素比较x和y，将结果保存到z
// 此处<0, 0> 指明OpType为EQ
// ubTensor_5为临时buffer
TCompare<0, 0>(ubTensor_z, ubTensor_x, ubTensor_y, ubTensor_5);

// 搬运z从ub到gm
TStore(gmTensor_z, ubTensor_z, Coor2Dim(4, 4));
```
#### 1.1.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.1.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 输入组合：input + other 严格满足一维广播规则


| 用例名称  | input维度 | other维度 | 切分        | 输入dtype | 说明                     |
|----------|-----------|-----------|-------------|-----------|--------------------------|
| fp16_001 | (112)     | (112)     | (50)        | fp16      | 1D 尾轴对齐，w切分        |
| fp16_002 | (64)      | (1)       | (32)        | fp16      | 1D 一维广播，w切分        |
| fp16_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp16      | 2D 无广播，h+w切分        |
| fp16_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp16      | 3D 无广播，c+h+w切分      |
| fp16_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp16      | 4D 无广播，n+c+h+w切分    |
| fp16_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp16      | 2D 大尺寸，h+w切分        |
| fp16_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_014 | (8, 1)      | (1,)       | (4, 1)      | fp16      | 跨维度广播，h+w切分       |
| fp16_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp16      | 4D 标准尺寸，n+c+h+w切分  |
| fp32_001 | (112)     | (112)     | (50)        | fp32      | 1D 尾轴对齐，w切分        |
| fp32_002 | (64)      | (1)       | (32)        | fp32      | 1D 一维广播，w切分        |
| fp32_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp32      | 2D 无广播，h+w切分        |
| fp32_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp32      | 3D 无广播，c+h+w切分      |
| fp32_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp32      | 4D 无广播，n+c+h+w切分    |
| fp32_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp32      | 2D 大尺寸，h+w切分        |
| fp32_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_014 | (8, 1)      | (1,)       | (4, 1)      | fp32      | 跨维度广播，h+w切分       |
| fp32_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp32      | 4D 标准尺寸，n+c+h+w切分  |
### 1.2. pypto.ge

#### 1.2.1. 算子计算原理
逐元素大于等于比较运算。
#### 1.2.2. pypto前段接口（python）以及支持范围
函数原型
```python
ge(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```
参数说明
| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor\float\Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

约束说明

1.  input 和 other 类型须保持一致。
2.  **支持一维广播。**
#### 1.2.3. c++ tensor graph接口
```c++
// 根据OpType区分具体的操作类型
// 比较两个Tensor
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Tensor和Scalar
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Scalar和Tensor
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}
```
#### 1.2.4. c++ tile graph接口
```c++
// 两个Tensor间逐元素比较  对应的pto instruction是TCMP
void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

// Tensor和Scalar逐元素比较 对应的pto instruction是TCMPS
void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}
```
#### 1.2.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TCMP:比较两个 Tile 并写入一个打包的谓词掩码。
TCMPS:将 Tile 与标量比较并写入逐元素比较结果。

TStore：ub to gm搬运
#### 1.2.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素比较x和y，将结果保存到z
// 此处<5, 0> 指明OpType为GE
// ubTensor_5为临时buffer
TCompare<5, 0>(ubTensor_z, ubTensor_x, ubTensor_y, ubTensor_5);

// 搬运z从ub到gm
TStore(gmTensor_z, ubTensor_z, Coor2Dim(4, 4));
```
#### 1.2.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.2.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 输入组合：input + other 严格满足一维广播规则


| 用例名称  | input维度 | other维度 | 切分        | 输入dtype | 说明                     |
|----------|-----------|-----------|-------------|-----------|--------------------------|
| fp16_001 | (112)     | (112)     | (50)        | fp16      | 1D 尾轴对齐，w切分        |
| fp16_002 | (64)      | (1)       | (32)        | fp16      | 1D 一维广播，w切分        |
| fp16_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp16      | 2D 无广播，h+w切分        |
| fp16_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp16      | 3D 无广播，c+h+w切分      |
| fp16_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp16      | 4D 无广播，n+c+h+w切分    |
| fp16_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp16      | 2D 大尺寸，h+w切分        |
| fp16_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_014 | (8, 1)      | (1,)       | (4, 1)      | fp16      | 跨维度广播，h+w切分       |
| fp16_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp16      | 4D 标准尺寸，n+c+h+w切分  |
| fp32_001 | (112)     | (112)     | (50)        | fp32      | 1D 尾轴对齐，w切分        |
| fp32_002 | (64)      | (1)       | (32)        | fp32      | 1D 一维广播，w切分        |
| fp32_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp32      | 2D 无广播，h+w切分        |
| fp32_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp32      | 3D 无广播，c+h+w切分      |
| fp32_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp32      | 4D 无广播，n+c+h+w切分    |
| fp32_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp32      | 2D 大尺寸，h+w切分        |
| fp32_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_014 | (8, 1)      | (1,)       | (4, 1)      | fp32      | 跨维度广播，h+w切分       |
| fp32_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp32      | 4D 标准尺寸，n+c+h+w切分  |
### 1.3. pypto.gt

#### 1.3.1. 算子计算原理
逐元素大于比较运算。
#### 1.3.2. pypto前段接口（python）以及支持范围
函数原型
```python
gt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```
参数说明
| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor\float\Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

约束说明

1.  input 和 other 类型须保持一致。
2.  **支持一维广播。**
#### 1.3.3. c++ tensor graph接口
```c++
// 根据OpType区分具体的操作类型
// 比较两个Tensor
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Tensor和Scalar
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Scalar和Tensor
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}
```
#### 1.3.4. c++ tile graph接口
```c++
// 两个Tensor间逐元素比较  对应的pto instruction是TCMP
void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

// Tensor和Scalar逐元素比较 对应的pto instruction是TCMPS
void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}
```
#### 1.3.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TCMP:比较两个 Tile 并写入一个打包的谓词掩码。
TCMPS:将 Tile 与标量比较并写入逐元素比较结果。

TStore：ub to gm搬运
#### 1.3.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素比较x和y，将结果保存到z
// 此处<4, 0> 指明OpType为GT
// ubTensor_5为临时buffer
TCompare<4, 0>(ubTensor_z, ubTensor_x, ubTensor_y, ubTensor_5);

// 搬运z从ub到gm
TStore(gmTensor_z, ubTensor_z, Coor2Dim(4, 4));
```
#### 1.3.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.3.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 输入组合：input + other 严格满足一维广播规则


| 用例名称  | input维度 | other维度 | 切分        | 输入dtype | 说明                     |
|----------|-----------|-----------|-------------|-----------|--------------------------|
| fp16_001 | (112)     | (112)     | (50)        | fp16      | 1D 尾轴对齐，w切分        |
| fp16_002 | (64)      | (1)       | (32)        | fp16      | 1D 一维广播，w切分        |
| fp16_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp16      | 2D 无广播，h+w切分        |
| fp16_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp16      | 3D 无广播，c+h+w切分      |
| fp16_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp16      | 4D 无广播，n+c+h+w切分    |
| fp16_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp16      | 2D 大尺寸，h+w切分        |
| fp16_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_014 | (8, 1)      | (1,)       | (4, 1)      | fp16      | 跨维度广播，h+w切分       |
| fp16_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp16      | 4D 标准尺寸，n+c+h+w切分  |
| fp32_001 | (112)     | (112)     | (50)        | fp32      | 1D 尾轴对齐，w切分        |
| fp32_002 | (64)      | (1)       | (32)        | fp32      | 1D 一维广播，w切分        |
| fp32_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp32      | 2D 无广播，h+w切分        |
| fp32_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp32      | 3D 无广播，c+h+w切分      |
| fp32_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp32      | 4D 无广播，n+c+h+w切分    |
| fp32_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp32      | 2D 大尺寸，h+w切分        |
| fp32_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_014 | (8, 1)      | (1,)       | (4, 1)      | fp32      | 跨维度广播，h+w切分       |
| fp32_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp32      | 4D 标准尺寸，n+c+h+w切分  |
### 1.4. pypto.le

#### 1.4.1. 算子计算原理
逐元素小于等于比较运算。
#### 1.4.2. pypto前段接口（python）以及支持范围
函数原型
```python
le(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```
参数说明
| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor\float\Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

约束说明

1.  input 和 other 类型须保持一致。
2.  **支持一维广播。**
#### 1.4.3. c++ tensor graph接口
```c++
// 根据OpType区分具体的操作类型
// 比较两个Tensor
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Tensor和Scalar
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Scalar和Tensor
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}
```
#### 1.4.4. c++ tile graph接口
```c++
// 两个Tensor间逐元素比较  对应的pto instruction是TCMP
void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

// Tensor和Scalar逐元素比较 对应的pto instruction是TCMPS
void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}
```
#### 1.4.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TCMP:比较两个 Tile 并写入一个打包的谓词掩码。
TCMPS:将 Tile 与标量比较并写入逐元素比较结果。

TStore：ub to gm搬运
#### 1.4.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素比较x和y，将结果保存到z
// 此处<3, 0> 指明OpType为LE
// ubTensor_5为临时buffer
TCompare<3, 0>(ubTensor_z, ubTensor_x, ubTensor_y, ubTensor_5);

// 搬运z从ub到gm
TStore(gmTensor_z, ubTensor_z, Coor2Dim(4, 4));
```
#### 1.4.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.4.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 输入组合：input + other 严格满足一维广播规则
数据类型：FP16/FP32


| 用例名称  | input维度 | other维度 | 切分        | 输入dtype | 说明                     |
|----------|-----------|-----------|-------------|-----------|--------------------------|
| fp16_001 | (112)     | (112)     | (50)        | fp16      | 1D 尾轴对齐，w切分        |
| fp16_002 | (64)      | (1)       | (32)        | fp16      | 1D 一维广播，w切分        |
| fp16_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp16      | 2D 无广播，h+w切分        |
| fp16_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp16      | 3D 无广播，c+h+w切分      |
| fp16_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp16      | 4D 无广播，n+c+h+w切分    |
| fp16_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp16      | 2D 大尺寸，h+w切分        |
| fp16_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_014 | (8, 1)      | (1,)       | (4, 1)      | fp16      | 跨维度广播，h+w切分       |
| fp16_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp16      | 4D 标准尺寸，n+c+h+w切分  |
| fp32_001 | (112)     | (112)     | (50)        | fp32      | 1D 尾轴对齐，w切分        |
| fp32_002 | (64)      | (1)       | (32)        | fp32      | 1D 一维广播，w切分        |
| fp32_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp32      | 2D 无广播，h+w切分        |
| fp32_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp32      | 3D 无广播，c+h+w切分      |
| fp32_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp32      | 4D 无广播，n+c+h+w切分    |
| fp32_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp32      | 2D 大尺寸，h+w切分        |
| fp32_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_014 | (8, 1)      | (1,)       | (4, 1)      | fp32      | 跨维度广播，h+w切分       |
| fp32_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp32      | 4D 标准尺寸，n+c+h+w切分  |
### 1.5. pypto.lt

#### 1.5.1. 算子计算原理
逐元素小于比较运算。
#### 1.5.2. pypto前段接口（python）以及支持范围
函数原型
```python
lt(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```
参数说明
| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor\float\Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

约束说明

1.  input 和 other 类型须保持一致。
2.  **支持一维广播。**
#### 1.5.3. c++ tensor graph接口
```c++
// 根据OpType区分具体的操作类型
// 比较两个Tensor
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Tensor和Scalar
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Scalar和Tensor
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}
```
#### 1.5.4. c++ tile graph接口
```c++
// 两个Tensor间逐元素比较  对应的pto instruction是TCMP
void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

// Tensor和Scalar逐元素比较 对应的pto instruction是TCMPS
void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}
```
#### 1.5.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TCMP:比较两个 Tile 并写入一个打包的谓词掩码。
TCMPS:将 Tile 与标量比较并写入逐元素比较结果。

TStore：ub to gm搬运
#### 1.5.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素比较x和y，将结果保存到z
// 此处<2, 0> 指明OpType为LT
// ubTensor_5为临时buffer
TCompare<2, 0>(ubTensor_z, ubTensor_x, ubTensor_y, ubTensor_5);

// 搬运z从ub到gm
TStore(gmTensor_z, ubTensor_z, Coor2Dim(4, 4));
```
#### 1.5.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.5.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 输入组合：input + other 严格满足一维广播规则


| 用例名称  | input维度 | other维度 | 切分        | 输入dtype | 说明                     |
|----------|-----------|-----------|-------------|-----------|--------------------------|
| fp16_001 | (112)     | (112)     | (50)        | fp16      | 1D 尾轴对齐，w切分        |
| fp16_002 | (64)      | (1)       | (32)        | fp16      | 1D 一维广播，w切分        |
| fp16_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp16      | 2D 无广播，h+w切分        |
| fp16_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp16      | 3D 无广播，c+h+w切分      |
| fp16_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp16      | 4D 无广播，n+c+h+w切分    |
| fp16_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp16      | 2D 大尺寸，h+w切分        |
| fp16_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_014 | (8, 1)      | (1,)       | (4, 1)      | fp16      | 跨维度广播，h+w切分       |
| fp16_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp16      | 4D 标准尺寸，n+c+h+w切分  |
| fp32_001 | (112)     | (112)     | (50)        | fp32      | 1D 尾轴对齐，w切分        |
| fp32_002 | (64)      | (1)       | (32)        | fp32      | 1D 一维广播，w切分        |
| fp32_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp32      | 2D 无广播，h+w切分        |
| fp32_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp32      | 3D 无广播，c+h+w切分      |
| fp32_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp32      | 4D 无广播，n+c+h+w切分    |
| fp32_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp32      | 2D 大尺寸，h+w切分        |
| fp32_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_014 | (8, 1)      | (1,)       | (4, 1)      | fp32      | 跨维度广播，h+w切分       |
| fp32_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp32      | 4D 标准尺寸，n+c+h+w切分  |
### 1.6. pypto.ne

#### 1.6.1. 算子计算原理
逐元素不等于比较运算。
#### 1.6.2. pypto前段接口（python）以及支持范围
函数原型
```python
ne(input: Tensor, other: Union[Tensor, float, Element]) -> Tensor
```
参数说明
| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。 <br> 支持的类型为：Tensor\float\Element。 <br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32，两个源操作数的数据类型必须保持一致。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

返回Shape与输入Tensor一致、数据类型为DT\_BOOL的Tensor。若input对应位置的元素值等于other对应位置的元素值，则该位置的返回值为True，其余位置的返回值为False。

约束说明

1.  input 和 other 类型须保持一致。
2.  **支持一维广播。**
#### 1.6.3. c++ tensor graph接口
```c++
// 根据OpType区分具体的操作类型
// 比较两个Tensor
Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Tensor和Scalar
Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

// 比较Scalar和Tensor
Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}
```
#### 1.6.4. c++ tile graph接口
```c++
// 两个Tensor间逐元素比较  对应的pto instruction是TCMP
void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

// Tensor和Scalar逐元素比较 对应的pto instruction是TCMPS
void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}
```
#### 1.6.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TCMP:比较两个 Tile 并写入一个打包的谓词掩码。
TCMPS:将 Tile 与标量比较并写入逐元素比较结果。

TStore：ub to gm搬运
#### 1.6.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素比较x和y，将结果保存到z
// 此处<1, 0> 指明OpType为NE
// ubTensor_5为临时buffer
TCompare<1, 0>(ubTensor_z, ubTensor_x, ubTensor_y, ubTensor_5);

// 搬运z从ub到gm
TStore(gmTensor_z, ubTensor_z, Coor2Dim(4, 4));
```
#### 1.6.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.6.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 输入组合：input + other 严格满足一维广播规则

| 用例名称  | input维度 | other维度 | 切分        | 输入dtype | 说明                     |
|----------|-----------|-----------|-------------|-----------|--------------------------|
| fp16_001 | (112)     | (112)     | (50)        | fp16      | 1D 尾轴对齐，w切分        |
| fp16_002 | (64)      | (1)       | (32)        | fp16      | 1D 一维广播，w切分        |
| fp16_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp16      | 2D 无广播，h+w切分        |
| fp16_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp16      | 2D 一维广播，h+w切分      |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp16      | 3D 无广播，c+h+w切分      |
| fp16_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp16      | 3D 一维广播，c+h+w切分    |
| fp16_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp16      | 4D 无广播，n+c+h+w切分    |
| fp16_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp16      | 2D 大尺寸，h+w切分        |
| fp16_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp16      | 4D 一维广播，n+c+h+w切分  |
| fp16_014 | (8, 1)      | (1,)       | (4, 1)      | fp16      | 跨维度广播，h+w切分       |
| fp16_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp16      | 4D 标准尺寸，n+c+h+w切分  |
| fp32_001 | (112)     | (112)     | (50)        | fp32      | 1D 尾轴对齐，w切分        |
| fp32_002 | (64)      | (1)       | (32)        | fp32      | 1D 一维广播，w切分        |
| fp32_003 | (3, 128)  | (3, 128)  | (2, 64)     | fp32      | 2D 无广播，h+w切分        |
| fp32_004 | (2, 96)   | (1, 96)   | (1, 48)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_005 | (5, 1)    | (5, 32)   | (5, 16)     | fp32      | 2D 一维广播，h+w切分      |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 32) | fp32      | 3D 无广播，c+h+w切分      |
| fp32_007 | (2, 1, 48) | (2, 3, 48) | (2, 1, 24) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_008 | (3, 64, 1) | (3, 64, 48) | (3, 32, 1) | fp32      | 3D 一维广播，c+h+w切分    |
| fp32_009 | (1, 1, 32, 32) | (1, 1, 32, 32) | (1, 1, 16, 16) | fp32      | 4D 无广播，n+c+h+w切分    |
| fp32_010 | (1, 4, 16, 16) | (1, 1, 16, 16) | (1, 2, 8, 8) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_011 | (2, 1, 24, 24) | (2, 3, 24, 24) | (2, 1, 12, 12) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_012 | (4, 80)     | (4, 80)     | (2, 40)     | fp32      | 2D 大尺寸，h+w切分        |
| fp32_013 | (1, 2, 40, 40) | (1, 2, 1, 40) | (1, 1, 20, 20) | fp32      | 4D 一维广播，n+c+h+w切分  |
| fp32_014 | (8, 1)      | (1,)       | (4, 1)      | fp32      | 跨维度广播，h+w切分       |
| fp32_015 | (2, 3, 16, 16) | (2, 3, 16, 16) | (1, 3, 8, 8) | fp32      | 4D 标准尺寸，n+c+h+w切分  |
### 1.7. pypto.maximum

#### 1.7.1. 算子计算原理
计算输入与另一输入的逐元素最大值
#### 1.7.2. pypto前段接口（python）以及支持范围
函数原型

```python
maximum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为 float, Element, Tensor类型。 <br> 当为float 类型时会自动转换为 Element 类型，其中 float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为 float, Element, Tensor类型。 <br> 当为 float 类型时会自动转换为 Element 类型，其中 float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 <br> 类型和数据类型必须与源操作数一保持一致。 |

源操作数一与源操作数二之间至少一者为Tensor。

返回值说明

当两个源操作数均为Tensor时，两个Tensor必须满足广播关系。该接口返回一个与源操作数一和源操作数二广播后形状相同的Tensor，数据类型与源操作数相同，其元素为源操作数一和源操作数二的逐元素最大值。
**且源操作数为Tensor时，源操作数一和源操作数二均仅支持单轴广播。**

当两个源操作数之中存在一个Tensor时，返回与输入Tensor相同形状的Tensor，其元素为源操作数一和源操作数二的逐元素最大值。
#### 1.7.3. c++ tensor graph接口
```c++
// 计算两个Tensor的最大值
Tensor Maximum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(
        BinaryOperation<BinaryOpType::MAXIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

// 计算Tensor和Scalar的最大值
Tensor Maximum(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    ASSERT(operand1.GetDataType() == operand2.GetDataType()) << "The datatype of the two input must be equal";
    std::vector<DataType> MAXS_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_BF16};
    ASSERT(std::find(MAXS_SUPPORT_DATATYPES.begin(), MAXS_SUPPORT_DATATYPES.end(), operand1.GetDataType()) !=
           MAXS_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}
```
#### 1.7.4. c++ tile graph接口
```c++
// 计算两个Tensor的最大值
template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationScalar<T>(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

// 计算Tensor和Scalar的最大值
template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationScalar<T>(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}
```
#### 1.7.5. 算子npu计算使用的pto instruction
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// LastUse2Dim<0, 0>表示该张量在此操作后立即释放
// 将x扩展至2维，并存入ubTensor_x1
TExpand<LastUse2Dim<0, 0>, 2>(ubTensor_x1, ubTensor_x);

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素取x和y的最大值，结果存储到y中
TMax<LastUse3Dim<0, 0, 0>>(ubTensor_y, ubTensor_y, ubTensor_x1);

// 将结果从ub搬运到gm
TStore(gmTensor_z, ubTensor_y, Coor2Dim(4, 4));
```
#### 1.7.6. 算子kernel的计算过程（搬运+计算）
TLoad：搬运buffer从gm到ub

TExpand：将标量广播到目标Tile中

TMax:两个Tile的逐元素求最大值

TStore：ub to gm搬运
#### 1.7.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.7.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 广播约束：Tensor + Tensor 时仅支持单轴广播

| 用例名称  | input维度   | other维度   | 切分         | 输入dtype | 说明 |
|----------|-------------|-------------|--------------|-----------|------|
| fp16_001 | (112)       | (112)       | (56)         | fp16      | 1D，W单轴切分 |
| fp16_002 | (64)        | (1)         | (32)         | fp16      | 1D，单轴广播，W单轴切分 |
| fp16_003 | (3, 128)    | (3, 128)    | (3, 64)      | fp16      | 2D，W单轴切分 |
| fp16_004 | (2, 96)     | (1, 96)     | (2, 96)      | fp16      | 2D，单轴广播，N单轴切分 |
| fp16_005 | (5, 1)      | (5, 32)     | (5, 1)       | fp16      | 2D，单轴广播，H单轴切分 |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 64)  | fp16      | 3D，C单轴切分 |
| fp16_007 | (2, 1, 48)  | (2, 3, 48)  | (2, 1, 24)   | fp16      | 3D，单轴广播，W单轴切分 |
| fp16_008 | (3, 64, 1)  | (3, 64, 48) | (3, 32, 1)   | fp16      | 3D，单轴广播，H单轴切分 |
| fp16_009 | (1,1,32,32) | (1,1,32,32) | (1,1,16,32)  | fp16      | 4D，H单轴切分 |
| fp16_010 | (1,4,16,16) | (1,1,16,16) | (1,2,16,16)  | fp16      | 4D，单轴广播，C单轴切分 |
| fp16_011 | (2,1,24,24) | (2,3,24,24) | (1,1,24,24)  | fp16      | 4D，单轴广播，N单轴切分 |
| fp16_012 | (4, 80)     | 标量        | (2, 40)      | fp16      | 2D，Tensor+标量，H+W切分 |
| fp16_013 | (1,2,40,40) | 标量        | (1,1,40,20)  | fp16      | 4D，Tensor+标量，W单轴切分 |
| fp16_014 | (8, 1)      | 标量        | (4, 1)       | fp16      | 2D，Tensor+标量，N单轴切分 |
| fp16_015 | (2,3,16,16) | (2,3,16,16) | (1,3,8,8)    | fp16      | 4D，N+H+W多轴切分 |
| fp32_001 | (112)       | (112)       | (56)         | fp32      | 1D，W单轴切分 |
| fp32_002 | (64)        | (1)         | (32)         | fp32      | 1D，单轴广播，W单轴切分 |
| fp32_003 | (3, 128)    | (3, 128)    | (3, 64)      | fp32      | 2D，W单轴切分 |
| fp32_004 | (2, 96)     | (1, 96)     | (2, 96)      | fp32      | 2D，单轴广播，N单轴切分 |
| fp32_005 | (5, 1)      | (5, 32)     | (5, 1)       | fp32      | 2D，单轴广播，H单轴切分 |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 64)  | fp32      | 3D，C单轴切分 |
| fp32_007 | (2, 1, 48)  | (2, 3, 48)  | (2, 1, 24)   | fp32      | 3D，单轴广播，W单轴切分 |
| fp32_008 | (3, 64, 1)  | (3, 64, 48) | (3, 32, 1)   | fp32      | 3D，单轴广播，H单轴切分 |
| fp32_009 | (1,1,32,32) | (1,1,32,32) | (1,1,16,32)  | fp32      | 4D，H单轴切分 |
| fp32_010 | (1,4,16,16) | (1,1,16,16) | (1,2,16,16)  | fp32      | 4D，单轴广播，C单轴切分 |
| fp32_011 | (2,1,24,24) | (2,3,24,24) | (1,1,24,24)  | fp32      | 4D，单轴广播，N单轴切分 |
| fp32_012 | (4, 80)     | 标量        | (2, 40)      | fp32      | 2D，Tensor+标量，H+W切分 |
| fp32_013 | (1,2,40,40) | 标量        | (1,1,40,20)  | fp32      | 4D，Tensor+标量，W单轴切分 |
| fp32_014 | (8, 1)      | 标量        | (4, 1)       | fp32      | 2D，Tensor+标量，N单轴切分 |
| fp32_015 | (2,3,16,16) | (2,3,16,16) | (1,3,8,8)    | fp32      | 4D，N+H+W多轴切分 |
### 1.8. pypto.minimum

#### 1.8.1. 算子计算原理
计算输入与另一输入的最小值。
#### 1.8.2. pypto前段接口（python）以及支持范围
函数原型

```python
maximum(
    input: Union[Tensor, Element, int, float], other: Union[Tensor, Element, int, float]
) -> Tensor
```

参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为 int, float, Element, Tensor类型。 <br> 当为 int 或者 float 类型时会自动转换为 Element 类型，其中 int 对应 DT_INT_32，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other   | 输入      | 源操作数。 <br> 支持的类型为 int, float, Element, Tensor类型。 <br> 当为 int 或者 float 类型时会自动转换为 Element 类型，其中 int 对应 DT_INT_32，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。 <br> Tensor和Element支持的数据类型为：DT_FP16，DT_FP32 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 <br> 类型和数据类型必须与源操作数一保持一致。 |

源操作数一与源操作数二之间至少一者为Tensor。

返回值说明

当两个源操作数均为Tensor时，两个Tensor必须满足广播关系。该接口返回一个与源操作数一和源操作数二广播后形状相同的Tensor，数据类型与源操作数相同，其元素为源操作数一和源操作数二的逐元素最大值。
**且源操作数为Tensor时，源操作数一和源操作数二均仅支持单轴广播。**

当两个源操作数之中存在一个Tensor时，返回与输入Tensor相同形状的Tensor，其元素为源操作数一和源操作数二的逐元素最大值。
#### 1.8.3. c++ tensor graph接口
```c++
// 计算两个Tensor的最小值
Tensor Minimum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(
        BinaryOperation<BinaryOpType::MINIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

// 计算Tensor和Scalar的最小值
Tensor Minimum(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    ASSERT(operand1.GetDataType() == operand2.GetDataType()) << "The datatype of the two input must be equal";
    std::vector<DataType> MINS_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_BF16};
    ASSERT(std::find(MINS_SUPPORT_DATATYPES.begin(), MINS_SUPPORT_DATATYPES.end(), operand1.GetDataType()) !=
           MINS_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MIN>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}
```
#### 1.8.4. c++ tile graph接口
```c++
// 计算两个Tensor的最小值
template <BinaryOpType T>
void BinaryOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBinaryOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

// 计算Tensor和Scalar的最小值
template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationScalar<T>(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}
```
#### 1.8.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TExpand：将标量广播到目标Tile中

TMin:两个Tile的逐元素最小值

TStore：ub to gm搬运

#### 1.8.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(4, 4));

// LastUse2Dim<0, 0>表示该张量在此操作后立即释放
// 将x扩展至2维，并存入ubTensor_x1
TExpand<LastUse2Dim<0, 0>, 2>(ubTensor_x1, ubTensor_x);

// 搬运源操作数y从gm到ub
TLoad(ubTensor_y, gmTensor_y, Coor2Dim(4, 4));

// 逐元素取x和y的最小值，结果存储到y中
TMin<LastUse3Dim<0, 0, 0>>(ubTensor_y, ubTensor_y, ubTensor_x1);

// 将结果从ub搬运到gm
TStore(gmTensor_z, ubTensor_y, Coor2Dim(4, 4));
```
#### 1.8.7. tile shape设置约束
TileShape维度应和输出一致。
#### 1.8.8. 用例设计
测试因子:
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* 广播约束：Tensor + Tensor 时仅支持单轴广播

| 用例名称  | input维度   | other维度   | 切分         | 输入dtype | 说明 |
|----------|-------------|-------------|--------------|-----------|------|
| fp16_001 | (112)       | (112)       | (56)         | fp16      | 1D，W单轴切分 |
| fp16_002 | (64)        | (1)         | (32)         | fp16      | 1D，单轴广播，W单轴切分 |
| fp16_003 | (3, 128)    | (3, 128)    | (3, 64)      | fp16      | 2D，W单轴切分 |
| fp16_004 | (2, 96)     | (1, 96)     | (2, 96)      | fp16      | 2D，单轴广播，N单轴切分 |
| fp16_005 | (5, 1)      | (5, 32)     | (5, 1)       | fp16      | 2D，单轴广播，H单轴切分 |
| fp16_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 64)  | fp16      | 3D，C单轴切分 |
| fp16_007 | (2, 1, 48)  | (2, 3, 48)  | (2, 1, 24)   | fp16      | 3D，单轴广播，W单轴切分 |
| fp16_008 | (3, 64, 1)  | (3, 64, 48) | (3, 32, 1)   | fp16      | 3D，单轴广播，H单轴切分 |
| fp16_009 | (1,1,32,32) | (1,1,32,32) | (1,1,16,32)  | fp16      | 4D，H单轴切分 |
| fp16_010 | (1,4,16,16) | (1,1,16,16) | (1,2,16,16)  | fp16      | 4D，单轴广播，C单轴切分 |
| fp16_011 | (2,1,24,24) | (2,3,24,24) | (1,1,24,24)  | fp16      | 4D，单轴广播，N单轴切分 |
| fp16_012 | (4, 80)     | 标量        | (2, 40)      | fp16      | 2D，Tensor+标量，H+W切分 |
| fp16_013 | (1,2,40,40) | 标量        | (1,1,40,20)  | fp16      | 4D，Tensor+标量，W单轴切分 |
| fp16_014 | (8, 1)      | 标量        | (4, 1)       | fp16      | 2D，Tensor+标量，N单轴切分 |
| fp16_015 | (2,3,16,16) | (2,3,16,16) | (1,3,8,8)    | fp16      | 4D，N+H+W多轴切分 |
| fp32_001 | (112)       | (112)       | (56)         | fp32      | 1D，W单轴切分 |
| fp32_002 | (64)        | (1)         | (32)         | fp32      | 1D，单轴广播，W单轴切分 |
| fp32_003 | (3, 128)    | (3, 128)    | (3, 64)      | fp32      | 2D，W单轴切分 |
| fp32_004 | (2, 96)     | (1, 96)     | (2, 96)      | fp32      | 2D，单轴广播，N单轴切分 |
| fp32_005 | (5, 1)      | (5, 32)     | (5, 1)       | fp32      | 2D，单轴广播，H单轴切分 |
| fp32_006 | (1, 64, 64) | (1, 64, 64) | (1, 32, 64)  | fp32      | 3D，C单轴切分 |
| fp32_007 | (2, 1, 48)  | (2, 3, 48)  | (2, 1, 24)   | fp32      | 3D，单轴广播，W单轴切分 |
| fp32_008 | (3, 64, 1)  | (3, 64, 48) | (3, 32, 1)   | fp32      | 3D，单轴广播，H单轴切分 |
| fp32_009 | (1,1,32,32) | (1,1,32,32) | (1,1,16,32)  | fp32      | 4D，H单轴切分 |
| fp32_010 | (1,4,16,16) | (1,1,16,16) | (1,2,16,16)  | fp32      | 4D，单轴广播，C单轴切分 |
| fp32_011 | (2,1,24,24) | (2,3,24,24) | (1,1,24,24)  | fp32      | 4D，单轴广播，N单轴切分 |
| fp32_012 | (4, 80)     | 标量        | (2, 40)      | fp32      | 2D，Tensor+标量，H+W切分 |
| fp32_013 | (1,2,40,40) | 标量        | (1,1,40,20)  | fp32      | 4D，Tensor+标量，W单轴切分 |
| fp32_014 | (8, 1)      | 标量        | (4, 1)       | fp32      | 2D，Tensor+标量，N单轴切分 |
| fp32_015 | (2,3,16,16) | (2,3,16,16) | (1,3,8,8)    | fp32      | 4D，N+H+W多轴切分 |
## 2. 二元数学运算

### 2.1. pypto.mul

#### 2.1.1. 算子计算原理

#### 2.1.2. pypto前段接口（python）以及支持范围

#### 2.1.3. c++ tensor graph接口

#### 2.1.4. c++ tile graph接口

#### 2.1.5. 算子npu计算使用的pto instruction

#### 2.1.6. 算子kernel的计算过程（搬运+计算）

#### 2.1.7. tile shape设置约束

#### 2.1.8. 用例设计

### 2.2. pypto.add

#### 2.2.1. 算子计算原理

#### 2.2.2. pypto前段接口（python）以及支持范围

#### 2.2.3. c++ tensor graph接口

#### 2.2.4. c++ tile graph接口

#### 2.2.5. 算子npu计算使用的pto instruction

#### 2.2.6. 算子kernel的计算过程（搬运+计算）

#### 2.2.7. tile shape设置约束

#### 2.2.8. 用例设计

### 2.3. pypto.div

#### 2.3.1. 算子计算原理

#### 2.3.2. pypto前段接口（python）以及支持范围

#### 2.3.3. c++ tensor graph接口

#### 2.3.4. c++ tile graph接口

#### 2.3.5. 算子npu计算使用的pto instruction

#### 2.3.6. 算子kernel的计算过程（搬运+计算）

#### 2.3.7. tile shape设置约束

#### 2.3.8. 用例设计

### 2.4. pypto.sub

#### 2.4.1. 算子计算原理

#### 2.4.2. pypto前段接口（python）以及支持范围

#### 2.4.3. c++ tensor graph接口

#### 2.4.4. c++ tile graph接口

#### 2.4.5. 算子npu计算使用的pto instruction

#### 2.4.6. 算子kernel的计算过程（搬运+计算）

#### 2.4.7. tile shape设置约束

#### 2.4.8. 用例设计

## 3. 规约运算

### 3.1. pypto.amax

#### 3.1.1. 算子计算原理
参考pypto\docs\api\operation\pypto-amax.md  
对一个多维向量在指定的维度求最大值。
定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对Shape为\(2, 3\)的二维矩阵进行运算，指定在第一维求最大值，输出结果为\[4, 5, 6\]；指定在第二维求最大值，输出结果为\[3, 6\]。

**图 1**  amax按第一个维度计算示例  
![](../docs/api/figures/pypto.amax_1.png)
**图 2**  amax按最后一个维度计算示例  
![](../docs/api/figures/pypto.amax_2.png)
#### 3.1.2. pypto前段接口（python）以及支持范围
```python
amax(input: Tensor, dim: int, keepdim: bool = False) -> Tensor: 
```
##### 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 源操作数。 <br> 支持任意单轴。                                       |
| keepdim | 输入      | 源操作数 <br> 控制在进行归约后，是否保持被压缩的维度。 <br> 默认值为False。 |

##### 返回值说明

返回输出Tensor，输出Tensor的Shape与keepdim参数相关。

若keepdim参数为 True，则在执行归约操作后保留被归约的维度。输出Tensor在除dim指定的维度外，其他维度的Shape与输入Tensor的Shape一致，而在dim指定的维度上的大小为 1。

若keepdim参数为False（默认），则被归约的维度会从输出Tensor中移除，而tileshape中对应的维度不变, 所以建议在调其他operation前重设tileshape。
#### 3.1.3. c++ tensor graph接口
```cpp
Tensor Amax(const Tensor &self, int axis = -1, bool keepDim=false);
```
#### 3.1.4. c++ tile graph接口
`amax` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级归约操作，使用 "MAX" 或 "MAX_COMBINE_AXIS" 操作符，根据张量形状和归约维度选择合适的实现方式。
```cpp
void RowMaxSingleOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "MAX", iOperand[0], oOperand[0], axis);
}

void RowMaxCombineOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "MAX_COMBINE_AXIS", iOperand[0], oOperand[0], axis);
}
```
#### 3.1.5. 算子npu计算使用的pto instruction
**PTO 指令**:TLOAD、TPairMax、TROWMaxSingle、TStore
#### 3.1.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 将输入张量从全局内存加载到 NPU 的局部内存
   - 根据 Tile 大小进行分块处理
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 对每个 Tile 执行局部最大值计算
   - 利用 NPU 的向量处理能力，并行计算多个元素
   - 对于大张量，采用多级归约策略

3. **结果存储**：
   - 将计算结果写回全局内存
   - 根据 `keepdim` 参数调整输出形状
#### 3.1.7. tile shape设置约束
1. TileShape大小不超过 64KB；

2. 尾轴要 32bytes 对齐；

3. TileShape次尾轴要小于等于255，即 TileShape\[-2\]<=255.
#### 3.1.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| amax_fp16_001 | (112) | (50) | fp16 | 1d尾轴对齐，w切分 |
| amax_fp16_002 | (100) | (100) | fp16 | 1d尾轴不对齐，无切分 |
| amax_fp16_003 | (4, 128) | (2, 32) | fp16 | 2d尾轴对齐，h,w切分 |
| amax_fp16_004 | (4, 130) | (1, 130) | fp16 | 2d尾轴不对齐，h切分 |
| amax_fp16_005 | (2, 4, 160) | (1, 2, 32) | fp16 | 3d尾轴对齐，c,h,w切分 |
| amax_fp16_006 | (2, 4, 140) | (1, 2, 140) | fp16 | 3d尾轴不对齐，c,h切分 |
| amax_fp16_007 | (2, 5, 152) | (1, 5, 32) | fp16 | 3d尾轴不对齐，c,w切分 |
| amax_fp16_008 | (2, 3, 170) | (1, 3, 170) | fp16 | 3d尾轴不对齐，c切分 |
| amax_fp16_009 | (5, 2, 4, 176) | (2, 1, 2, 16) | fp16 | 4d尾轴对齐，n,c,h,w切分 |
| amax_fp16_010 | (5, 2, 4, 130) | (1, 1, 1, 130) | fp16 | 4d尾轴不对齐，n,c,h切分 |
| amax_fp16_011 | (2, 3, 5, 134) | (1, 1, 5, 32) | fp16 | 4d尾轴不对齐，n,c,w切分 |
| amax_fp16_012 | (4, 2, 6, 135) | (2, 2, 3, 32) | fp16 | 4d尾轴不对齐，n,h,w切分 |
| amax_fp16_013 | (6, 2, 4, 130) | (1, 1, 4, 130) | fp16 | 4d尾轴不对齐，n,c切分 |
| amax_fp16_014 | (3, 2, 3, 139) | (1, 2, 1, 139) | fp16 | 4d尾轴不对齐，n,h切分 |
| amax_fp16_015 | (6, 3, 5, 141) | (3, 3, 5, 32) | fp16 | 4d尾轴不对齐，n,w切分 |
| amax_fp32_001 | (112) | (50) | fp32 | 1d尾轴对齐，w切分 |
| amax_fp32_002 | (100) | (100) | fp32 | 1d尾轴不对齐，无切分 |
| amax_fp32_003 | (4, 128) | (2, 32) | fp32 | 2d尾轴对齐，h,w切分 |
| amax_fp32_004 | (4, 130) | (1, 130) | fp32 | 2d尾轴不对齐，h切分 |
| amax_fp32_005 | (2, 4, 160) | (1, 2, 32) | fp32 | 3d尾轴对齐，c,h,w切分 |
| amax_fp32_006 | (2, 4, 140) | (1, 2, 140) | fp32 | 3d尾轴不对齐，c,h切分 |
| amax_fp32_007 | (2, 5, 152) | (1, 5, 32) | fp32 | 3d尾轴不对齐，c,w切分 |
| amax_fp32_008 | (2, 3, 170) | (1, 3, 170) | fp32 | 3d尾轴不对齐，c切分 |
| amax_fp32_009 | (5, 2, 4, 176) | (2, 1, 2, 16) | fp32 | 4d尾轴对齐，n,c,h,w切分 |
| amax_fp32_010 | (5, 2, 4, 130) | (1, 1, 1, 130) | fp32 | 4d尾轴不对齐，n,c,h切分 |
| amax_fp32_011 | (2, 3, 5, 134) | (1, 1, 5, 32) | fp32 | 4d尾轴不对齐，n,c,w切分 |
| amax_fp32_012 | (4, 2, 6, 135) | (2, 2, 3, 32) | fp32 | 4d尾轴不对齐，n,h,w切分 |
| amax_fp32_013 | (6, 2, 4, 130) | (1, 1, 4, 130) | fp32 | 4d尾轴不对齐，n,c切分 |
| amax_fp32_014 | (3, 2, 3, 139) | (1, 2, 1, 139) | fp32 | 4d尾轴不对齐，n,h切分 |
| amax_fp32_015 | (6, 3, 5, 141) | (3, 3, 5, 32) | fp32 | 4d尾轴不对齐，n,w切分 |
### 3.2. pypto.amin
```python
amin(input: Tensor, dim: int, keepdim: bool = False) -> Tensor: 
```
#### 3.2.1. 算子计算原理
参考pypto\docs\api\operation\pypto-amin.md  
对一个多维向量在指定的维度求最小值。

定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对Shape为\(2, 3\)的二维矩阵进行运算，指定在第一维求最小值，输出结果为\[1, 2, 3\]；指定在第二维求最小值，输出结果为\[1, 4\]。

**图 1**  amin按第一个维度计算示例  
![](../docs/api/figures/pypto.amin_1.png)
**图 2**  amin按最后一个维度计算示例  
![](../docs/api/figures/pypto.amin_2.png)
#### 3.2.2. pypto前段接口（python）以及支持范围
```python
amin(input: Tensor, dim: int, keepdim: bool = False) -> Tensor: 
```
##### 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 源操作数。 <br> 支持任意单轴。                                       |
| keepdim | 输入      | 源操作数 <br> 控制在进行归约后，是否保持被压缩的维度。 <br> 默认值为False。 |

##### 返回值说明

返回输出Tensor，输出Tensor的Shape与keepdim参数相关。

若keepdim参数为 True，则在执行归约操作后保留被归约的维度。输出Tensor在除dim指定的维度外，其他维度的Shape与输入Tensor的Shape一致，而在dim指定的维度上的大小为 1。

若keepdim参数为 False（默认），则被归约的维度会从输出Tensor中移除，而tileshape中对应的维度不变, 所以建议在调其他operation前重设tileshape。
#### 3.2.3. c++ tensor graph接口
```cpp
Tensor Amin(const Tensor &self, int axis = -1, bool keepDim=false);
```
#### 3.2.4. c++ tile graph接口
`amin` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级归约操作，使用 "MIN" 或 "MIN_COMBINE_AXIS" 操作符，根据张量形状和归约维度选择合适的实现方式。
```cpp
void RowMinSingleOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "MIN", iOperand[0], oOperand[0], axis);
}
```
#### 3.2.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLOAD、TPairMin、TROWMinSingle、TStore
#### 3.2.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 将输入张量从全局内存加载到 NPU 的局部内存
   - 根据 Tile 大小进行分块处理
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 对每个 Tile 执行局部最小值计算
   - 利用 NPU 的向量处理能力，并行计算多个元素
   - 对于大张量，采用多级归约策略

3. **结果存储**：
   - 将计算结果写回全局内存
   - 根据 `keepdim` 参数调整输出形状
#### 3.2.7. tile shape设置约束
1. TileShape大小不超过 64KB；

2. 尾轴要 32bytes 对齐；

3. TileShape 次尾轴要小于等于255，即 TileShape\[-2\]<=255.
#### 3.2.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| amin_fp16_001 | (112) | (50) | fp16 | 1d尾轴对齐，w切分 |
| amin_fp16_002 | (100) | (100) | fp16 | 1d尾轴不对齐，无切分 |
| amin_fp16_003 | (4, 128) | (2, 32) | fp16 | 2d尾轴对齐，h,w切分 |
| amin_fp16_004 | (4, 130) | (1, 130) | fp16 | 2d尾轴不对齐，h切分 |
| amin_fp16_005 | (2, 4, 160) | (1, 2, 32) | fp16 | 3d尾轴对齐，c,h,w切分 |
| amin_fp16_006 | (2, 4, 140) | (1, 2, 140) | fp16 | 3d尾轴不对齐，c,h切分 |
| amin_fp16_007 | (2, 5, 152) | (1, 5, 32) | fp16 | 3d尾轴不对齐，c,w切分 |
| amin_fp16_008 | (2, 3, 170) | (1, 3, 170) | fp16 | 3d尾轴不对齐，c切分 |
| amin_fp16_009 | (5, 2, 4, 176) | (2, 1, 2, 16) | fp16 | 4d尾轴对齐，n,c,h,w切分 |
| amin_fp16_010 | (5, 2, 4, 130) | (1, 1, 1, 130) | fp16 | 4d尾轴不对齐，n,c,h切分 |
| amin_fp16_011 | (2, 3, 5, 134) | (1, 1, 5, 32) | fp16 | 4d尾轴不对齐，n,c,w切分 |
| amin_fp16_012 | (4, 2, 6, 135) | (2, 2, 3, 32) | fp16 | 4d尾轴不对齐，n,h,w切分 |
| amin_fp16_013 | (6, 2, 4, 130) | (1, 1, 4, 130) | fp16 | 4d尾轴不对齐，n,c切分 |
| amin_fp16_014 | (3, 2, 3, 139) | (1, 2, 1, 139) | fp16 | 4d尾轴不对齐，n,h切分 |
| amin_fp16_015 | (6, 3, 5, 141) | (3, 3, 5, 32) | fp16 | 4d尾轴不对齐，n,w切分 |
| amin_fp32_001 | (112) | (50) | fp32 | 1d尾轴对齐，w切分 |
| amin_fp32_002 | (100) | (100) | fp32 | 1d尾轴不对齐，无切分 |
| amin_fp32_003 | (4, 128) | (2, 32) | fp32 | 2d尾轴对齐，h,w切分 |
| amin_fp32_004 | (4, 130) | (1, 130) | fp32 | 2d尾轴不对齐，h切分 |
| amin_fp32_005 | (2, 4, 160) | (1, 2, 32) | fp32 | 3d尾轴对齐，c,h,w切分 |
| amin_fp32_006 | (2, 4, 140) | (1, 2, 140) | fp32 | 3d尾轴不对齐，c,h切分 |
| amin_fp32_007 | (2, 5, 152) | (1, 5, 32) | fp32 | 3d尾轴不对齐，c,w切分 |
| amin_fp32_008 | (2, 3, 170) | (1, 3, 170) | fp32 | 3d尾轴不对齐，c切分 |
| amin_fp32_009 | (5, 2, 4, 176) | (2, 1, 2, 16) | fp32 | 4d尾轴对齐，n,c,h,w切分 |
| amin_fp32_010 | (5, 2, 4, 130) | (1, 1, 1, 130) | fp32 | 4d尾轴不对齐，n,c,h切分 |
| amin_fp32_011 | (2, 3, 5, 134) | (1, 1, 5, 32) | fp32 | 4d尾轴不对齐，n,c,w切分 |
| amin_fp32_012 | (4, 2, 6, 135) | (2, 2, 3, 32) | fp32 | 4d尾轴不对齐，n,h,w切分 |
| amin_fp32_013 | (6, 2, 4, 130) | (1, 1, 4, 130) | fp32 | 4d尾轴不对齐，n,c切分 |
| amin_fp32_014 | (3, 2, 3, 139) | (1, 2, 1, 139) | fp32 | 4d尾轴不对齐，n,h切分 |
| amin_fp32_015 | (6, 3, 5, 141) | (3, 3, 5, 32) | fp32 | 4d尾轴不对齐，n,w切分 |
### 3.3. pypto.sum

#### 3.3.1. 算子计算原理
参考pypto\docs\api\operation\pypto-sum.md  
对一个多维向量按照指定的维度进行数据累加。

定义指定计算的维度（Reduce轴）为R轴，非指定维度（Normal轴）为A轴。如下图所示，对Shape为\(2, 3\)的二维矩阵进行运算，指定在第一维计算数据的累加，输出结果为\[5, 7, 9\]；指定在第二维计算数据的累加，输出结果为\[6, 15\]。

**图 1**  sum按第一个维度计算示例  
![](../docs/api/figures/pypto.sum_1.png)
**图 2**  sum按最后一个维度计算示例  
![](../docs/api/figures/pypto.sum_2.png)
#### 3.3.2. pypto前段接口（python）以及支持范围
```python
sum(input: Tensor,  dim: int, keepdim: bool = False) -> Tensor: 
```
##### 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32。 <br> 不支持空Tensor；Shape仅支持2-4维，Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 源操作数。 <br> 支持任意单轴。 |
| keepdim | 输入      | 源操作数。 <br> 控制在进行归约后，是否保持被压缩的维度。 <br> 默认值为False。 |

##### 返回值说明

返回输出Tensor，输出Tensor的Shape与keepdim参数相关。

若keepdim参数为 True，则在执行归约操作后保留被归约的维度。输出Tensor在除dim指定的维度外，其他维度的Shape与输入Tensor的Shape一致，而在dim指定的维度上的大小为 1。

若keepdim参数为 False（默认），则被归约的维度会从输出Tensor中移除，而tileshape中对应的维度不变, 所以建议在调其他operation前重设tileshape。
#### 3.3.3. c++ tensor graph接口
```cpp
Tensor Sum(const Tensor &self, int axis = -1, bool keepDim=false);
```
#### 3.3.4. c++ tile graph接口
`sum` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级归约操作，使用 "SUM" 或 "SUM_COMBINE_AXIS" 操作符，根据张量形状和归约维度选择合适的实现方式。
```cpp
void RowSumSingleOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "SUM", iOperand[0], oOperand[0], axis);
}

void RowSumCombineOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto axis = op.GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    TiledReduceSingle(function, tileShape, "SUM_COMBINE_AXIS", iOperand[0], oOperand[0], axis);
}
```
#### 3.3.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLOAD、TPairSum、TROWSumSingle、TStore
#### 3.3.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 将输入张量从全局内存加载到 NPU 的局部内存
   - 根据 Tile 大小进行分块处理
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 对每个 Tile 执行局部求和计算
   - 利用 NPU 的向量处理能力，并行计算多个元素
   - 对于大张量，采用多级归约策略

3. **结果存储**：
   - 将计算结果写回全局内存
   - 根据 `keepdim` 参数调整输出形状
#### 3.3.7. tile shape设置约束
1. TileShape大小不超过 64KB；

2. 尾轴要 32bytes 对齐；

3. TileShape次尾轴要小于等于255，即 TileShape\[-2\]<=255.
#### 3.3.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| sum_fp32_001 | (112) | (50) | fp32 | 1d尾轴对齐，w切分 |
| sum_fp32_002 | (100) | (100) | fp32 | 1d尾轴不对齐，无切分 |
| sum_fp32_003 | (4, 128) | (2, 32) | fp32 | 2d尾轴对齐，h,w切分 |
| sum_fp32_004 | (4, 130) | (1, 130) | fp32 | 2d尾轴不对齐，h切分 |
| sum_fp32_005 | (2, 4, 160) | (1, 2, 32) | fp32 | 3d尾轴对齐，c,h,w切分 |
| sum_fp32_006 | (2, 4, 140) | (1, 2, 140) | fp32 | 3d尾轴不对齐，c,h切分 |
| sum_fp32_007 | (2, 5, 152) | (1, 5, 32) | fp32 | 3d尾轴不对齐，c,w切分 |
| sum_fp32_008 | (2, 3, 170) | (1, 3, 170) | fp32 | 3d尾轴不对齐，c切分 |
| sum_fp32_009 | (5, 2, 4, 176) | (2, 1, 2, 16) | fp32 | 4d尾轴对齐，n,c,h,w切分 |
| sum_fp32_010 | (5, 2, 4, 130) | (1, 1, 1, 130) | fp32 | 4d尾轴不对齐，n,c,h切分 |
| sum_fp32_011 | (2, 3, 5, 134) | (1, 1, 5, 32) | fp32 | 4d尾轴不对齐，n,c,w切分 |
| sum_fp32_012 | (4, 2, 6, 135) | (2, 2, 3, 32) | fp32 | 4d尾轴不对齐，n,h,w切分 |
| sum_fp32_013 | (6, 2, 4, 130) | (1, 1, 4, 130) | fp32 | 4d尾轴不对齐，n,c切分 |
| sum_fp32_014 | (3, 2, 3, 139) | (1, 2, 1, 139) | fp32 | 4d尾轴不对齐，n,h切分 |
| sum_fp32_015 | (6, 3, 5, 141) | (3, 3, 5, 32) | fp32 | 4d尾轴不对齐，n,w切分 |
## 4. 基础数学运算

### 4.1. pypto.exp、pypto.neg、pypto.abs、pypto.sqrt、pypto.reciprocal

#### 4.1.1. 算子计算原理
__pypto.exp__：计算输入Tensor中每个元素的 e 的指数，逐元素运算，返回与输入形状相同的Tensor。

__pypto.neg__：计算输入Tensor中每个元素的负数，逐元素运算，返回与输入形状相同的Tensor。

__pypto.abs__：计算输入Tensor中每个元素的绝对值，逐元素运算。

__pypto.sqrt__：计算输入Tensor中每个元素的平方根，逐元素运算。输入为负数时返回 NaN。

__pypto.reciprocal__：计算输入Tensor中每个元素的倒数，逐元素运算。

#### 4.1.2. pypto前段接口（python）以及支持范围
| exp(input: Tensor) -> Tensor | neg(input: Tensor) -> Tensor | abs(input: Tensor) -> Tensor | sqrt(input: Tensor) -> Tensor | reciprocal(input: Tensor) -> Tensor |

__参数说明__

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16，neg额外支持DT_INT32，DT_INT16。 <br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

#### 4.1.3. c++ tensor graph接口
统一调用TensorUnaryOperation接口

```
template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function &function, LogicalTensorPtr operand, std::optional<DataType> datatype = std::nullopt) {
    auto opName = GetUnaryOpName<T>();
    CheckTensorShape(operand, opName);
    datatype = datatype.value_or(operand->tensor->datatype);
    auto result = std::make_shared<LogicalTensor>(
        function, *datatype, operand->shape, operand->GetDynValidShape(), operand->Format());
    function.AddOperation(GetUnaryOpNameCode<T>(), {operand}, {result});
    return result;
}
```

#### 4.1.4. c++ tile graph接口
统一调用TiledUnaryOperation接口
```
template <UnaryOpType T>
void TiledUnaryOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, const LogicalTensorPtr &result, uint32_t workspaceSize = 0) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        if (workspaceSize == 0) {
            function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_UINT8, std::vector<int64_t>{workspaceSize});
            function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile, workspace});
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledUnaryOperation<T>(function, tileShape, cur + 1, input, result, workspaceSize);
    }
}

```

#### 4.1.5. 算子npu计算使用的pto instruction
__Tload__：gm to ub

__TUnary__：unary基础运算，包含：__exp__、__neg__(调用TMul接口，与-1或-1.相乘)、__abs__、__sqrt__、__reciprocal__

__TStore__：ub to gm

#### 4.1.6. 算子kernel的计算过程（搬运+计算）
```
// load x from gm to ub
TLoad(ub_x, gm_x)

// Template Unary(input)
Tunary(ub_x)

// store x from ub to gm
TStore(gm_y, ub_x)
```
```
// load x from gm to ub
TLoad(ub_x, gm_x)

// Reciprocal(input)
TReciprocal(ub_y, ub_x)

// store x from ub to gm
TStore(gm_y, ub_y)
```

#### 4.1.7. tile shape设置约束
无
#### 4.1.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32

| 用例名称  | 输入维度 | 切分 | 输入dtype | 说明 |
|----------|---------|-----------|------|------|
| Unary_fp16_001  | (112) | (50) | fp16 | 1d尾轴对齐，w切分 |
| Unary_fp16_002  | (100) | (100) | fp16 | 1d尾轴不对齐，无切分 |
| Unary_fp16_003  | (4,128) | (2,32) | fp16 | 2d尾轴对齐，h,w切分 |
| Unary_fp16_004  | (4,130) | (1,130) | fp16 | 2d尾轴不对齐，h切分 |
| Unary_fp16_005  | (2,4,160) | (1,2,32) | fp16 | 3d尾轴对齐，c,h,w切分 |
| Unary_fp16_006  | (2,4,140) | (1,2,140) | fp16 | 3d尾轴不对齐，c,h切分 |
| Unary_fp16_007  | (2,5,152) | (1,5,32) | fp16 | 3d尾轴不对齐，c,w切分 |
| Unary_fp16_008  | (2,3,170) | (1,3,170) | fp16 | 3d尾轴不对齐，c切分 |
| Unary_fp16_009  | (5,2,4,176) | (2,1,2,16) | fp16 | 4d尾轴对齐，n,c,h,w切分 |
| Unary_fp16_010  | (5,2,4,130) | (1,1,1,130) | fp16 | 4d尾轴不对齐，n,c,h切分 |
| Unary_fp16_011  | (2,3,5,134) | (1,1,5,32) | fp16 | 4d尾轴不对齐，n,c,w切分 |
| Unary_fp16_012  | (4,2,6,135) | (2,2,3,32) | fp16 | 4d尾轴不对齐，n,h,w切分 |
| Unary_fp16_013  | (6,2,4,130) | (1,1,4,130) | fp16 | 4d尾轴不对齐，n,c切分 |
| Unary_fp16_014  | (3,2,3,139) | (1,2,1,139) | fp16 | 4d尾轴不对齐，n,h切分 |
| Unary_fp16_015  | (6,3,5,141) | (3,3,5,32) | fp16 | 4d尾轴不对齐，n,w切分 |
| Unary_fp32_001  | (112) | (50) | fp32 | 1d尾轴对齐，w切分 |
| Unary_fp32_002  | (100) | (100) | fp32 | 1d尾轴不对齐，无切分 |
| Unary_fp32_003  | (4,128) | (2,32) | fp32 | 2d尾轴对齐，h,w切分 |
| Unary_fp32_004  | (4,130) | (1,130) | fp32 | 2d尾轴不对齐，h切分 |
| Unary_fp32_005  | (2,4,160) | (1,2,32) | fp32 | 3d尾轴对齐，c,h,w切分 |
| Unary_fp32_006  | (2,4,140) | (1,2,140) | fp32 | 3d尾轴不对齐，c,h切分 |
| Unary_fp32_007  | (2,5,152) | (1,5,32) | fp32 | 3d尾轴不对齐，c,w切分 |
| Unary_fp32_008  | (2,3,170) | (1,3,170) | fp32 | 3d尾轴不对齐，c切分 |
| Unary_fp32_009  | (5,2,4,176) | (2,1,2,16) | fp32 | 4d尾轴对齐，n,c,h,w切分 |
| Unary_fp32_010  | (5,2,4,130) | (1,1,1,130) | fp32 | 4d尾轴不对齐，n,c,h切分 |
| Unary_fp32_011  | (2,3,5,134) | (1,1,5,32) | fp32 | 4d尾轴不对齐，n,c,w切分 |
| Unary_fp32_012  | (4,2,6,135) | (2,2,3,32) | fp32 | 4d尾轴不对齐，n,h,w切分 |
| Unary_fp32_013  | (6,2,4,130) | (1,1,4,130) | fp32 | 4d尾轴不对齐，n,c切分 |
| Unary_fp32_014  | (3,2,3,139) | (1,2,1,139) | fp32 | 4d尾轴不对齐，n,h切分 |
| Unary_fp32_015  | (6,3,5,141) | (3,3,5,32) | fp32 | 4d尾轴不对齐，n,w切分 |

### 4.2. pypto.rsqrt

#### 4.2.1. 算子计算原理

计算输入Tensor中每个元素的平方根倒数，逐元素运算。当输入为负数时返回 NaN，输入为零时返回 Inf。

#### 4.2.2. pypto前段接口（python）以及支持范围

rsqrt(input: Tensor) -> Tensor

__参数说明__

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32, DT_FP16。 <br> 不支持空Tensor；Shape仅支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

#### 4.2.3. c++ tensor graph接口
```
Tensor Rsqrt(const Tensor &self) {
    DECLARE_TRACER();

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto sqrtSelf = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    auto ones = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
        SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto result = CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), ones, sqrtSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return result;
}
```
#### 4.2.4. c++ tile graph接口

调用cast、sqrt、Full、Div算子的tile graph接口

#### 4.2.5. 算子npu计算使用的pto instruction

__TVecDup__：按给定值填充

__TLoad__：gm to ub

__TSqrt__：计算平方根

__pipe_barrier__: 同步

__TDiv__：除法运算

__TStore__： ub to gm

#### 4.2.6. 算子kernel的计算过程（搬运+计算）

```
// filling with 1
TVecDup(ub_y, 1)

// load x from gm to ub
TLoad(ub_x, gm_x)

//√input
TSqrt(ub_X)

//sync
pipe_barrier(PIPE_V)

//division
TDiv(ub_y, ub_x)

//store x from ub to gm
TStore(gm_y, ub_x)
```

#### 4.2.7. tile shape设置约束
无
#### 4.2.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32

| 用例名称  | 输入维度 | 切分 | 输入dtype | 说明 |
|----------|---------|-----------|------|------|
| rsqrt_fp16_001  | (112) | (50) | fp16 | 1d尾轴对齐，w切分 |
| rsqrt_fp16_002  | (100) | (100) | fp16 | 1d尾轴不对齐，无切分 |
| rsqrt_fp16_003  | (4,128) | (2,32) | fp16 | 2d尾轴对齐，h,w切分 |
| rsqrt_fp16_004  | (4,130) | (1,130) | fp16 | 2d尾轴不对齐，h切分 |
| rsqrt_fp16_005  | (2,4,160) | (1,2,32) | fp16 | 3d尾轴对齐，c,h,w切分 |
| rsqrt_fp16_006  | (2,4,140) | (1,2,140) | fp16 | 3d尾轴不对齐，c,h切分 |
| rsqrt_fp16_007  | (2,5,152) | (1,5,32) | fp16 | 3d尾轴不对齐，c,w切分 |
| rsqrt_fp16_008  | (2,3,170) | (1,3,170) | fp16 | 3d尾轴不对齐，c切分 |
| rsqrt_fp16_009  | (5,2,4,176) | (2,1,2,16) | fp16 | 4d尾轴对齐，n,c,h,w切分 |
| rsqrt_fp16_010  | (5,2,4,130) | (1,1,1,130) | fp16 | 4d尾轴不对齐，n,c,h切分 |
| rsqrt_fp16_011  | (2,3,5,134) | (1,1,5,32) | fp16 | 4d尾轴不对齐，n,c,w切分 |
| rsqrt_fp16_012  | (4,2,6,135) | (2,2,3,32) | fp16 | 4d尾轴不对齐，n,h,w切分 |
| rsqrt_fp16_013  | (6,2,4,130) | (1,1,4,130) | fp16 | 4d尾轴不对齐，n,c切分 |
| rsqrt_fp16_014  | (3,2,3,139) | (1,2,1,139) | fp16 | 4d尾轴不对齐，n,h切分 |
| rsqrt_fp16_015  | (6,3,5,141) | (3,3,5,32) | fp16 | 4d尾轴不对齐，n,w切分 |
| rsqrt_fp32_001  | (112) | (50) | fp32 | 1d尾轴对齐，w切分 |
| rsqrt_fp32_002  | (100) | (100) | fp32 | 1d尾轴不对齐，无切分 |
| rsqrt_fp32_003  | (4,128) | (2,32) | fp32 | 2d尾轴对齐，h,w切分 |
| rsqrt_fp32_004  | (4,130) | (1,130) | fp32 | 2d尾轴不对齐，h切分 |
| rsqrt_fp32_005  | (2,4,160) | (1,2,32) | fp32 | 3d尾轴对齐，c,h,w切分 |
| rsqrt_fp32_006  | (2,4,140) | (1,2,140) | fp32 | 3d尾轴不对齐，c,h切分 |
| rsqrt_fp32_007  | (2,5,152) | (1,5,32) | fp32 | 3d尾轴不对齐，c,w切分 |
| rsqrt_fp32_008  | (2,3,170) | (1,3,170) | fp32 | 3d尾轴不对齐，c切分 |
| rsqrt_fp32_009  | (5,2,4,176) | (2,1,2,16) | fp32 | 4d尾轴对齐，n,c,h,w切分 |
| rsqrt_fp32_010  | (5,2,4,130) | (1,1,1,130) | fp32 | 4d尾轴不对齐，n,c,h切分 |
| rsqrt_fp32_011  | (2,3,5,134) | (1,1,5,32) | fp32 | 4d尾轴不对齐，n,c,w切分 |
| rsqrt_fp32_012  | (4,2,6,135) | (2,2,3,32) | fp32 | 4d尾轴不对齐，n,h,w切分 |
| rsqrt_fp32_013  | (6,2,4,130) | (1,1,4,130) | fp32 | 4d尾轴不对齐，n,c切分 |
| rsqrt_fp32_014  | (3,2,3,139) | (1,2,1,139) | fp32 | 4d尾轴不对齐，n,h切分 |
| rsqrt_fp32_015  | (6,3,5,141) | (3,3,5,32) | fp32 | 4d尾轴不对齐，n,w切分 |
## 5. 数据转换

### 5.1. pypto.cast

#### 5.1.1. 算子计算原理

根据源操作数和目的操作数Tensor的数据类型进行精度转换，如果目的操作数是整型且源操作数的数值超过整型的数据表示范围进行精度转换结果为目的操作数的最大值或者最小值。

#### 5.1.2. pypto前段接口（python）以及支持范围

cast(input: Tensor, dtype: DataType, mode: CastMode = CastMode.CAST_NONE) -> Tensor

__参数说明__

| 参数名     | 输入/输出 | 说明                                                                 |
|------------|-----------|----------------------------------------------------------------------|
| input      | 输入      | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_INT8，DT_UINT8，DT_INT16，DT_INT32。 <br> 不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| dtype      | 输入      | 精度转换后的数据类型。 <br> 支持的数据类型为：DT_FP32，DT_FP16，DT_INT8，DT_UINT8，DT_INT16，DT_INT32。 |
| CastMode   | 输入      | 源操作数枚举类型，用以控制精度转换处理模式，具体定义为：[CastMode](../docs\api\datatype\CastMode.md) 。<br> 默认为 CAST_NONE，常见类型之间的转换，框架会自动转换，与torch对齐，详见约束说明。 |

__约束说明__

1.  目的操作数是整型且源操作数的数值超过整型的数据表示范围进行精度转换结果为目的操作数的最大值或者最小值。例如DT\_FP16转DT\_INT8时，若输入是130.0，将会输出127（DT\_INT8的上界）
2.  支持以下转化：
    1.  DT\_FP16 到 DT\_FP32\\DT\_INT32\\DT\_INT16\\DT\_INT8\\DT\_UINT8 转化
    2.  DT\_FP32 到 DT\_FP16\\DT\_INT16\\DT\_INT32  转化
    3.  DT\_INT32 到 DT\_FP32 转化
    4.  DT\_UINT8 到 DT\_FP16 转化
    5.  DT\_INT8 到 DT\_FP16 转化
    6.  DT\_INT16 到 DT\_FP32\\DT\_FP16 转化

3.  支持精度转换处理模式CastMode，默认处理模式如下：
    1.  DT\_FP32 -\> DT\_FP16 : CAST\_RINT，与 Torch 对齐。
    2.  DT\_FP16\\DT\_INT32 -\> DT\_FP32 : 与 Torch 对齐。
    3.  DT\_FP16 -\> DT\_TNT8:  CAST\_TRUNC，见约束1。
    4.  DT\_INT32-\> DT\_FP16 : 与 Torch 对齐。

4.  当 cast 前后类型相同的时候，某些场景下会产生空操作，不保证精度。

#### 5.1.3. c++ tensor graph接口

```
Tensor Cast(const Tensor &self, DataType dstDataType, CastMode mode) {
    DECLARE_TRACER();
    ASSERT(self.GetShape().size() == self.GetStorage()->offset.size()) << "The shape size of self and offset should be equal";
    // Cast to same dType with no mode will do nothing
    if (self.GetStorage()->tensor->datatype == dstDataType && (mode == CAST_NONE || mode == CAST_RINT)) {
        return self;
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
        dstDataType, mode);
}
```
```
template <CastOpType T>
LogicalTensorPtr TensorCastOperation(
    Function &function, LogicalTensorPtr self, const DataType &dstDataType, const CastMode &mode = CAST_NONE) {
    auto result = std::make_shared<LogicalTensor>(function, dstDataType, self->shape, self->dynValidShape_);
    auto &op = function.AddOperation(GetCastOpName<T>(), {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
    return result;
}
```

#### 5.1.4. c++ tile graph接口
```
template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const CastMode &mode) {
    if (cur == static_cast<int>(input.tensor.GetShape().size())) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto &op = function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledCastOperation<T>(function, tileShape, cur + 1, input, result, mode);
    }
}
```
```
template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const CastMode &mode) {
    ASSERT(operand->shape.size() == operand->offset.size()) << "The shape size of operand and offset should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCastOperation<T>(function, tileShape, 0, input, result, mode);
}
```

#### 5.1.5. 算子npu计算使用的pto instruction
Tload: gm to ub

TCast：类型转换

TStore：ub to gm

#### 5.1.6. 算子kernel的计算过程（搬运+计算）
```
// load x from gm to ub
TLoad(ub_x, gm_x)

//type casting
TCast(ub_y, ub_x)

// store x from ub to gm
TStore(gm_y, ub_y)
```

#### 5.1.7. tile shape设置约束
无
#### 5.1.8. 用例设计
测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 输入/输出数据类型：FP16/FP32->(FP32/FP16/INT8/Uint8/INT16/INT32)、(FP32/FP16/INT8/Uint8/INT16/INT32)->FP16/FP32
* 转换类型：CAST_NONE、CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC、CAST_ODD

| 用例名称  | 输入维度 | 切分 | 输入dtype | 输出dtype | 处理模式 | 说明 |
|----------|---------|-----------|------|------|------|------|
| cast_001  | (112) | (50) | FP16 | FP32 | CAST_NONE | 1d尾轴对齐，w切分 |
| cast_002  | (100) | (100) | INT32 | FP32 | CAST_RINT | 1d尾轴不对齐，无切分 |
| cast_003  | (4,128) | (2,32) | INT16 | FP32 | CAST_ROUND | 2d尾轴对齐，h,w切分 |
| cast_004  | (4,130) | (1,130) | FP32 | FP16 | CAST_FLOOR | 2d尾轴不对齐，h切分 |
| cast_005  | (2,4,160) | (1,2,32) | INT8 | FP16 | CAST_CEIL | 3d尾轴对齐，c,h,w切分 |
| cast_006  | (2,4,140) | (1,2,140) | Uint8 | FP16 | CAST_TRUNC | 3d尾轴不对齐，c,h切分 |
| cast_007  | (2,5,152) | (1,5,32) | INT16 | FP16| CAST_ODD | 3d尾轴不对齐，c,w切分 |
| cast_008  | (2,3,170) | (1,3,170) | FP32 | INT16 | CAST_NONE | 3d尾轴不对齐，c切分 |
| cast_009  | (5,2,4,176) | (2,1,2,16) | FP32 | INT32 | CAST_NONE | 4d尾轴对齐，n,c,h,w切分 |
| cast_010  | (5,2,4,130) | (1,1,1,130) | FP16 | INT8 | CAST_NONE | 4d尾轴不对齐，n,c,h切分 |
| cast_011  | (2,3,5,134) | (1,1,5,32) | FP16 | Uint8 | CAST_NONE | 4d尾轴不对齐，n,c,w切分 |
| cast_012  | (4,2,6,135) | (2,2,3,32) | FP16 | INT16 | CAST_NONE | 4d尾轴不对齐，n,h,w切分 |
| cast_013  | (6,2,4,130) | (1,1,4,130) | FP16 | INT32 | CAST_NONE | 4d尾轴不对齐，n,c切分 |
| cast_014  | (3,2,3,139) | (1,2,1,139) | FP16 | FP32 | CAST_NONE | 4d尾轴不对齐，n,h切分 |
| cast_015  | (6,3,5,141) | (3,3,5,32) | FP16 | FP32 | CAST_NONE | 4d尾轴不对齐，n,w切分 |

## 6. 视图操作

### 6.1. pypto.transpose

#### 6.1.1. 算子计算原理
返回一个Tensor，该Tensor是输入Tensor的转置版本。指定的维度 dim0 和 dim1 将被交换。
#### 6.1.2. pypto前段接口（python）以及支持范围
```python
transpose(input: Tensor, dim0: int, dim1: int) -> Tensor
```
##### 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br> 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_FP16，DT_INT16，DT_UINT16，DT_FP32，DT_INT32，DT_UINT32。<br> 不支持空Tensor；Shape仅支持2-5维；Shape Size不大于2147483647（即INT32_MAX）。<br> 算子对不同 Shape 支持不同，详见约束说明。 |
| dim0    | 输入      | 源操作数，要交换的第一个维度的索引，从0开始计数。 |
| dim1    | 输入      | 源操作数，要交换的第二个维度的索引，从0开始计数。 |

##### 返回值说明
返回一个与输入数据类型一致的Tensor，其中 dim0 与 dim1 的维度位置被对调。
#### 6.1.3. c++ tensor graph接口
```CPP
Tensor Transpose(const Tensor &self, std::vector<int> perm);
```
#### 6.1.4. c++ tile graph接口
`transpose` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级转置操作，根据输入张量的形状和指定的维度交换生成相应的 Tile 操作。
```CPP
void TiledInnerTranspose(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<int> &shape) {
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape);
}
```
#### 6.1.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLoad、TTransMoveOut
#### 6.1.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 将输入张量从全局内存加载到 NPU 的局部内存
   - 根据 Tile 大小进行分块处理
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 对每个 Tile 执行局部转置操作
   - 利用 NPU 的向量处理能力，并行处理多个元素
   - 对于大张量，采用分块转置策略

3. **结果存储**：
   - 将转置后的结果写回全局内存
   - 确保输出张量的形状正确
#### 6.1.7. tile shape设置约束
1. TileShape和输入input维度一致，用于切分input。

2.输入维度dim0，dim1 必须大于0，小于input维度。

3.当前Transpose实现存在约束，只能支持以下场景转置：

-   2维：任意轴
-   3维：任意轴
-   4维：支持：0轴和2轴，1轴和 3轴，2轴和3轴, 1轴和 2轴,  不支持：0轴和3轴,  0轴和1轴
-   5维：支持：3轴和 4轴，其他不支持

4.涉及尾轴转置的场景，需要预留一块临时空间，用来搬运。

示例：

input : \[a, b, c, d\]  TileShape为\[t0, t1, t2, t3\] 数据类型为DT\_FP32

dim0: 2

dim1: 3

预留的临时空间为：t0 \* t1 \* align\(t2, 16\) \* align\(t3, 32 / sizeof\(DT\_FP32\)\)
#### 6.1.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| transpose_fp16_001 | (2, 3) | (1, 3) | fp16 | 2d转置，交换0和1轴 |
| transpose_fp16_002 | (3, 4) | (1, 4) | fp16 | 2d转置，交换0和1轴 |
| transpose_fp16_003 | (3, 4, 5) | (1, 2, 5) | fp16 | 3d转置，交换1和2轴 |
| transpose_fp16_004 | (2, 3, 4) | (1, 3, 2) | fp16 | 3d转置，交换0和2轴 |
| transpose_fp16_005 | (2, 3, 4, 5) | (1, 1, 2, 5) | fp16 | 4d转置，交换2和3轴 |
| transpose_fp16_006 | (2, 3, 4, 5) | (1, 3, 1, 5) | fp16 | 4d转置，交换1和2轴 |
| transpose_fp16_007 | (2, 3, 4, 5, 6) | (1, 1, 1, 2, 6) | fp16 | 5d转置，交换3和4轴 |
| transpose_fp16_008 | (4, 5) | (2, 5) | fp16 | 2d转置，交换0和1轴 |
| transpose_fp16_009 | (2, 5, 6) | (1, 5, 3) | fp16 | 3d转置，交换1和2轴 |
| transpose_fp16_010 | (3, 2, 4, 5) | (1, 2, 2, 5) | fp16 | 4d转置，交换2和3轴 |
| transpose_fp32_001 | (2, 3) | (1, 3) | fp32 | 2d转置，交换0和1轴 |
| transpose_fp32_002 | (3, 4) | (1, 4) | fp32 | 2d转置，交换0和1轴 |
| transpose_fp32_003 | (3, 4, 5) | (1, 2, 5) | fp32 | 3d转置，交换1和2轴 |
| transpose_fp32_004 | (2, 3, 4) | (1, 3, 2) | fp32 | 3d转置，交换0和2轴 |
| transpose_fp32_005 | (2, 3, 4, 5) | (1, 1, 2, 5) | fp32 | 4d转置，交换2和3轴 |
| transpose_fp32_006 | (2, 3, 4, 5) | (1, 3, 1, 5) | fp32 | 4d转置，交换1和2轴 |
| transpose_fp32_007 | (2, 3, 4, 5, 6) | (1, 1, 1, 2, 6) | fp32 | 5d转置，交换3和4轴 |
| transpose_fp32_008 | (4, 5) | (2, 5) | fp32 | 2d转置，交换0和1轴 |
| transpose_fp32_009 | (2, 5, 6) | (1, 5, 3) | fp32 | 3d转置，交换1和2轴 |
| transpose_fp32_010 | (3, 2, 4, 5) | (1, 2, 2, 5) | fp32 | 4d转置，交换2和3轴 |
| transpose_int32_001 | (2, 3) | (1, 3) | int32 | 2d转置，交换0和1轴 |
| transpose_int32_002 | (3, 4) | (1, 4) | int32 | 2d转置，交换0和1轴 |
| transpose_int32_003 | (3, 4, 5) | (1, 2, 5) | int32 | 3d转置，交换1和2轴 |
| transpose_int32_004 | (2, 3, 4, 5) | (1, 1, 2, 5) | int32 | 4d转置，交换2和3轴 |
| transpose_int32_005 | (4, 5) | (2, 5) | int32 | 2d转置，交换0和1轴 |
### 6.2. pypto.reshape
#### 6.2.1. 算子计算原理
改变Tensor形状，改变valid\_shape部分的形状\(Shape\)
#### 6.2.2. pypto前段接口（python）以及支持范围
```python
reshape(input: Tensor,shape: List[int],*,valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, inplace: bool = False) -> Tensor
```
##### 参数说明


| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。 <br> 支持的数据类型为：PyPTO支持的数据类型 <br> 不支持空Tensor，Shape Size不大于INT32_MAX。 |
| shape       | 输入      | 目标Shape。 <br> Shape Size不大于INT32_MAX；某个维度为-1时，支持自动推导。 |
| valid_shape | 输入      | 输出Tensor的有效数据的Shape，且valid_shape Size不大于INT32_MAX。 |
| inplace     | 输入      | 是否为inplace；参数为True时，不会为输出申请新地址； |

##### 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，形状\(Shape\)为输入参数指定的shape。

##### 约束说明

inplace为True时，需要保证输入输出分别是当前loop的输入输出；输出不可作为整个Function的输出
#### 6.2.3. c++ tensor graph接口
```cpp
Tensor Reshape(const Tensor &operand, const std::vector<int64_t> &dstshape, const std::vector<SymbolicScalar> &validShape={}, const bool inplace=false); 
Tensor Reshape(const Tensor &operand, const std::initializer_list<int64_t> &dstshape, const std::initializer_list<SymbolicScalar> &validShape={}, const bool inplace=false); 
Tensor Reshape(const Tensor &operand, const std::vector<SymbolicScalar> &dstShape, const bool inplace);

void Reshape(const Tensor &operand, Tensor &dst);
```
#### 6.2.4. c++ tile graph接口
`reshape` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级形状调整操作，根据输入张量的形状和目标形状生成相应的 Tile 操作。
```CPP
void TiledInnerReshape(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result, const bool isInplace = false) {
    auto &op = function.AddOperation("TILE_RESHAPE", {operand}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "isInplace", isInplace);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", result->GetDynValidShape());
    op.oOperand.front()->SetIsDummy();
}
```
#### 6.2.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLoad、TStore
#### 6.2.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 如果是原地操作，直接使用原内存
   - 如果不是原地操作，将数据从输入张量复制到新张量
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 调整内存访问的映射关系
   - 不需要实际的计算操作，主要是内存布局的调整

3. **结果存储**：
   - 如果是原地操作，直接修改原张量的形状信息
   - 如果不是原地操作，将数据写入新张量
#### 6.2.7. tile shape设置约束
inplace为True时，需要保证输入输出分别是当前loop的输入输出；输出不可作为整个Function的输出
#### 6.2.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| reshape_fp16_001 | (2, 2) | (1, 2) | fp16 | 2d转4d，保持数据 |
| reshape_fp16_002 | (4, 4) | (2, 2) | fp16 | 2d转1d，保持数据 |
| reshape_fp16_003 | (2, 3, 4) | (1, 1, 2) | fp16 | 3d转2d，保持数据 |
| reshape_fp16_004 | (2, 2, 2, 2) | (1, 1, 1, 1) | fp16 | 4d转2d，保持数据 |
| reshape_fp16_005 | (8) | (4) | fp16 | 1d转2d，保持数据 |
| reshape_fp16_006 | (3, 9) | (1, 3) | fp16 | 2d转3d，保持数据 |
| reshape_fp16_007 | (2, 2, 6) | (1, 1, 2) | fp16 | 3d转4d，保持数据 |
| reshape_fp16_008 | (4, 4, 4, 4) | (2, 2, 2, 2) | fp16 | 4d转1d，保持数据 |
| reshape_fp16_009 | (12) | (6) | fp16 | 1d转3d，保持数据 |
| reshape_fp16_010 | (2, 6) | (1, 3) | fp16 | 2d转4d，保持数据 |
| reshape_fp32_001 | (2, 2) | (1, 2) | fp32 | 2d转4d，保持数据 |
| reshape_fp32_002 | (4, 4) | (2, 2) | fp32 | 2d转1d，保持数据 |
| reshape_fp32_003 | (2, 3, 4) | (1, 1, 2) | fp32 | 3d转2d，保持数据 |
| reshape_fp32_004 | (2, 2, 2, 2) | (1, 1, 1, 1) | fp32 | 4d转2d，保持数据 |
| reshape_fp32_005 | (8) | (4) | fp32 | 1d转2d，保持数据 |
| reshape_fp32_006 | (3, 9) | (1, 3) | fp32 | 2d转3d，保持数据 |
| reshape_fp32_007 | (2, 2, 6) | (1, 1, 2) | fp32 | 3d转4d，保持数据 |
| reshape_fp32_008 | (4, 4, 4, 4) | (2, 2, 2, 2) | fp32 | 4d转1d，保持数据 |
| reshape_fp32_009 | (12) | (6) | fp32 | 1d转3d，保持数据 |
| reshape_fp32_010 | (2, 6) | (1, 3) | fp32 | 2d转4d，保持数据 |
| reshape_inplace_fp32_001 | (2, 2) | (1, 2) | fp32 | 2d转4d，inplace操作 |
| reshape_inplace_fp32_002 | (4, 4) | (2, 2) | fp32 | 2d转1d，inplace操作 |
| reshape_inplace_fp32_003 | (2, 3, 4) | (1, 1, 2) | fp32 | 3d转2d，inplace操作 |
| reshape_inplace_fp32_004 | (2, 2, 2, 2) | (1, 1, 1, 1) | fp32 | 4d转2d，inplace操作 |
| reshape_inplace_fp32_005 | (8) | (4) | fp32 | 1d转2d，inplace操作 |
### 6.3. pypto.view

#### 6.3.1. 算子计算原理
从输入Tensor中取出部分视图，用于后续计算。
#### 6.3.2. pypto前段接口（python）以及支持范围
```python
view(input: Tensor, shape: List[int] = None, offsets: List[Union[int, SymbolicScalar]] = None, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None, dtype: DataType = None,
) -> Tensor:
```
##### 参数说明


| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| input       | 输入      | 源操作数。<br> 支持的数据类型为：PyPto支持的数据类型<br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| shape       | 输入      | 获取出视图的大小。<br> Shape Size不大于2147483647（即INT32_MAX） |
| offsets     | 输入      | 获取视图时每个维度相对于input的偏移。<br> 需要保证offsets小于input的Shape |
| valid_shape | 输入      | 取出示意图块的有效数据大小。<br> 需要保证valid_shape小于input的Shape |
| dtype       | 输入      | 返回值的数据类型，允许将输入数据解读为不同数据类型 |

##### 返回值说明

返回输出Tensor，Tensor的数据类型和input相同，Shape为参数shape指定大小，若指定了valid\_shape，则真实大小为valid\_shape。若指定dtype，则会将输入按照dtype进行读取。

#### 6.3.3. c++ tensor graph接口
```cpp
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<int64_t> &offsets);
Tensor View(const Tensor &operand, const DataType dstDataType);
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::vector<SymbolicScalar> &newOffsets);
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes, const std::initializer_list<SymbolicScalar> &newOffsets);
Tensor View(const Tensor &operand, const std::vector<int64_t> &shapes,
    const std::vector<SymbolicScalar> &newValidShapes, const std::vector<SymbolicScalar> &newOffsets);
```
#### 6.3.4. c++ tile graph接口
`view` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级视图操作，根据输入张量的形状、偏移和目标形状生成相应的 Tile 操作。
```CPP
void TiledViewTypeOperation(Function &function, const TileShape &tileShape,
    const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    assert(operand->shape.size() == operand->offset.size());

    TileInfo operandTileInfo(operand->shape.size(), operand->offset.size());
    auto input = Input{operand, operandTileInfo};

    float factor = (float)BytesOf(operand->tensor->datatype) / (float)BytesOf(result->tensor->datatype);
    // 检查TileShape是否符合要求
    if(factor < 1){
        auto vecTile = tileShape.GetVecTile();
        auto lastDim = vecTile[vecTile.size() - 1];
        ASSERT(isInteger(lastDim * factor)) << "TileShape lastDim * factor must be int";
    }
    TiledViewTypeOperation(function, tileShape, 0, input, factor, result);
}
```
#### 6.3.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLoad、TStore
#### 6.3.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 不需要实际的数据搬运，只是创建一个视图
   - 建立从视图到原始张量的内存映射关系
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 不需要实际的计算操作，主要是内存访问的重映射
   - 建立视图的内存访问逻辑

3. **结果存储**：
   - 不需要实际的存储操作，视图共享原始张量的内存
   - 只是创建一个新的张量对象，指向原始张量的特定区域
#### 6.3.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| view_fp16_001 | (4, 8) | (2, 4) | fp16 | 2d视图，从(0,4)开始，形状(4,4) |
| view_fp16_002 | (4, 8) | (2, 4) | fp16 | 2d视图，带valid_shape(2,4) |
| view_fp16_003 | (2, 4, 8) | (1, 2, 4) | fp16 | 3d视图，从(0,0,4)开始，形状(2,4,4) |
| view_fp16_004 | (2, 4, 8) | (1, 2, 4) | fp16 | 3d视图，带valid_shape(2,2,4) |
| view_fp16_005 | (8) | (4) | fp16 | 1d视图，从(4)开始，形状(4) |
| view_fp16_006 | (3, 6) | (1, 3) | fp16 | 2d视图，从(0,3)开始，形状(3,3) |
| view_fp16_007 | (2, 3, 6) | (1, 1, 3) | fp16 | 3d视图，从(0,0,3)开始，形状(2,3,3) |
| view_fp16_008 | (2, 2, 2, 4) | (1, 1, 1, 2) | fp16 | 4d视图，从(0,0,0,2)开始，形状(2,2,2,2) |
| view_fp16_009 | (6) | (3) | fp16 | 1d视图，从(2)开始，形状(4) |
| view_fp16_010 | (2, 8) | (1, 4) | fp16 | 2d视图，从(0,4)开始，形状(2,4) |
| view_fp32_001 | (4, 8) | (2, 4) | fp32 | 2d视图，从(0,4)开始，形状(4,4) |
| view_fp32_002 | (4, 8) | (2, 4) | fp32 | 2d视图，带valid_shape(2,4) |
| view_fp32_003 | (2, 4, 8) | (1, 2, 4) | fp32 | 3d视图，从(0,0,4)开始，形状(2,4,4) |
| view_fp32_004 | (2, 4, 8) | (1, 2, 4) | fp32 | 3d视图，带valid_shape(2,2,4) |
| view_fp32_005 | (8) | (4) | fp32 | 1d视图，从(4)开始，形状(4) |
| view_fp32_006 | (3, 6) | (1, 3) | fp32 | 2d视图，从(0,3)开始，形状(3,3) |
| view_fp32_007 | (2, 3, 6) | (1, 1, 3) | fp32 | 3d视图，从(0,0,3)开始，形状(2,3,3) |
| view_fp32_008 | (2, 2, 2, 4) | (1, 1, 1, 2) | fp32 | 4d视图，从(0,0,0,2)开始，形状(2,2,2,2) |
| view_fp32_009 | (6) | (3) | fp32 | 1d视图，从(2)开始，形状(4) |
| view_fp32_010 | (2, 8) | (1, 4) | fp32 | 2d视图，从(0,4)开始，形状(2,4) |
| view_dtype_fp32_001 | (2, 2) | (1, 2) | fp32 | 2d视图，转换为int8 |
| view_dtype_fp32_002 | (2, 2) | (1, 2) | fp32 | 2d视图，转换为fp16 |
| view_dtype_fp32_003 | (4) | (2) | fp32 | 1d视图，转换为int8 |
| view_dtype_fp32_004 | (2, 4) | (1, 2) | fp32 | 2d视图，转换为fp16 |
| view_dtype_fp32_005 | (3, 3) | (1, 3) | fp32 | 2d视图，转换为int8 |
### 6.4. pypto.assemble

#### 6.4.1. 算子计算原理
以offsets指定的out索引位置为基准，将输入Tensor input赋值到输出Tensor out的对应区域。
#### 6.4.2. pypto前段接口（python）以及支持范围
```python
assemble(input: Tensor, offsets: List[Union[int, SymbolicScalar]], out: Tensor) -> None

assemble(inputs: List[Tuple[Tensor, List[Union[int, SymbolicScalar]]]], out: Tensor, parallel: bool = False) -> None
```
##### 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| inputs   | 输入      | 源操和输出偏移组成的Tuple列表。 <br> 单个支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| offsets | 输入      | 相对于目标输出的偏移。 <br> 需要保证offsets小于out的Shape。          |
| out     | 输入      | 目的操作数。 <br> 支持的数据类型为：PyPto支持的数据类型。 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| parallel | 输入      | 是否并行执行。 <br> 默认值为False |

##### 返回值说明

无返回值，会直接对out进行修改。
#### 6.4.3. c++ tensor graph接口
`assemble` 算子的 C++ 接口会调用相应的底层实现，将小张量组装到大多张量的指定位置。
```CPP
Tensor Assemble(const std::vector<std::pair<Tensor, std::vector<int64_t>>> &tensors); // 00 Tuple列表输入，Python前端直接调用这个接口
struct AssembleItem {
    Tensor tensor;
    std::vector<SymbolicScalar> offsets;
};
void Assemble(const std::vector<AssembleItem> &items, Tensor &src, bool parallelInAssemble = false); //01 Tuple列表输入，最终底层调用这个接口

void Assemble(const Tensor &tensor, const std::vector<SymbolicScalar> &dynOffset, Tensor &dest); //1 正常调用此接口
```
#### 6.4.4. c++ tile graph接口
`assemble` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级组装操作，根据源张量、偏移和目标张量生成相应的 Tile 操作。
```CPP
void TiledAssemble(Function &function, const TileShape &tileShape,
    const std::shared_ptr<LogicalTensor> &operand, const std::shared_ptr<LogicalTensor> &result,
    std::shared_ptr<AssembleOpAttribute> attr) {
    assert(operand->shape.size() == operand->offset.size());

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledAssemble(function, tileShape, 0, input, result, attr);
}
```
#### 6.4.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLoad、TStore
#### 6.4.6. 算子kernel的计算过程（搬运+计算）
1. **数据搬运**：
   - 将源张量的数据从全局内存加载到 NPU 的局部内存
   - 将数据从局部内存写入目标张量的指定位置
   - 优化内存访问模式，提高缓存命中率

2. **计算**：
   - 不需要实际的计算操作，主要是数据复制
   - 对于多个源张量，可以并行执行组装操作

3. **结果存储**：
   - 将组装后的数据写回目标张量的全局内存
   - 确保数据正确写入指定位置
#### 6.4.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| assemble_fp16_001 | (2, 2) | (1, 2) | fp16 | 单张量组装，偏移(0,0)到(4,4)张量 |
| assemble_fp16_002 | (2, 2) | (1, 2) | fp16 | 多张量组装，并行执行 |
| assemble_fp16_003 | (3, 3) | (1, 3) | fp16 | 单张量组装，偏移(1,1)到(5,5)张量 |
| assemble_fp16_004 | (2, 2, 2) | (1, 1, 2) | fp16 | 3d张量组装，偏移(0,0,0)到(3,3,3)张量 |
| assemble_fp16_005 | (4) | (2) | fp16 | 1d张量组装，偏移(1)到(6)张量 |
| assemble_fp16_006 | (2, 4) | (1, 2) | fp16 | 2d张量组装，偏移(0,2)到(3,6)张量 |
| assemble_fp16_007 | (1, 2, 2) | (1, 1, 2) | fp16 | 3d张量组装，偏移(0,1,1)到(2,3,3)张量 |
| assemble_fp16_008 | (2, 2, 2, 2) | (1, 1, 1, 2) | fp16 | 4d张量组装，偏移(0,0,0,0)到(3,3,3,3)张量 |
| assemble_fp16_009 | (3) | (3) | fp16 | 1d张量组装，偏移(0)到(5)张量 |
| assemble_fp16_010 | (1, 3) | (1, 3) | fp16 | 2d张量组装，偏移(0,0)到(2,5)张量 |
| assemble_fp32_001 | (2, 2) | (1, 2) | fp32 | 单张量组装，偏移(0,0)到(4,4)张量 |
| assemble_fp32_002 | (2, 2) | (1, 2) | fp32 | 多张量组装，并行执行 |
| assemble_fp32_003 | (3, 3) | (1, 3) | fp32 | 单张量组装，偏移(1,1)到(5,5)张量 |
| assemble_fp32_004 | (2, 2, 2) | (1, 1, 2) | fp32 | 3d张量组装，偏移(0,0,0)到(3,3,3)张量 |
| assemble_fp32_005 | (4) | (2) | fp32 | 1d张量组装，偏移(1)到(6)张量 |
| assemble_fp32_006 | (2, 4) | (1, 2) | fp32 | 2d张量组装，偏移(0,2)到(3,6)张量 |
| assemble_fp32_007 | (1, 2, 2) | (1, 1, 2) | fp32 | 3d张量组装，偏移(0,1,1)到(2,3,3)张量 |
| assemble_fp32_008 | (2, 2, 2, 2) | (1, 1, 1, 2) | fp32 | 4d张量组装，偏移(0,0,0,0)到(3,3,3,3)张量 |
| assemble_fp32_009 | (3) | (3) | fp32 | 1d张量组装，偏移(0)到(5)张量 |
| assemble_fp32_010 | (1, 3) | (1, 3) | fp32 | 2d张量组装，偏移(0,0)到(2,5)张量 |
| assemble_int32_001 | (2, 2) | (1, 2) | int32 | 单张量组装，偏移(0,0)到(4,4)张量 |
| assemble_int32_002 | (2, 2) | (1, 2) | int32 | 多张量组装，并行执行 |
| assemble_int32_003 | (3, 3) | (1, 3) | int32 | 单张量组装，偏移(1,1)到(5,5)张量 |
| assemble_int32_004 | (4) | (2) | int32 | 1d张量组装，偏移(1)到(6)张量 |
| assemble_int32_005 | (2, 4) | (1, 2) | int32 | 2d张量组装，偏移(0,2)到(3,6)张量 |
### 6.5. pypto.unsqueeze

#### 6.5.1. 算子计算原理
为输入Tensor增加维度。
#### 6.5.2. pypto前段接口（python）以及支持范围
```python
unsqueeze(input: Tensor, dim: int) -> Tensor
```
##### 参数说明


| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br> 支持的数据类型为：PyPto支持的数据类型<br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 指定插入新维度的位置（索引）。<br> 支持负索引。<br> 需在 [-input.dim - 1, input.dim] 范围内。 |

##### 返回值说明

返回在指定维度dim处新增大小为1的维度的输出Tensor，与输入Tensor共享数据且属性一致。
#### 6.5.3. c++ tensor graph接口
```cpp
Tensor Unsqueeze(const Tensor &old, int unsqueezeDimNum);
```
#### 6.5.4. c++ tile graph接口
`unsqueeze` 算子在 Tile Graph 层面会生成针对 NPU 硬件的 Tile 级维度扩展操作，根据输入张量和指定的维度位置生成相应的 Tile 操作。
同reshape
```CPP
Tensor Unsqueeze(const Tensor &old, int unsqueezeDimNum) {
    DECLARE_TRACER();

    ASSERT(unsqueezeDimNum < static_cast<int>(old.GetShape().size()) + 1 && unsqueezeDimNum >= -static_cast<int>(old.GetShape().size()) - 1);
    size_t unsqueezeDim = unsqueezeDimNum;
    if (unsqueezeDimNum < 0) {
        unsqueezeDim = unsqueezeDimNum + old.GetShape().size() + 1;
    }
    std::vector<int64_t> newShape(old.GetStorage()->shape);
    newShape.insert(newShape.begin() + unsqueezeDim, 1);
    auto validShape = old.GetStorage()->GetDynValidShape();
    ASSERT(!validShape.empty());
    validShape.insert(validShape.begin() + unsqueezeDim, 1);
    return Reshape(old, newShape, validShape);
}
```
#### 6.5.5. 算子npu计算使用的pto instruction
- **PTO 指令**：TLoad、TStore
#### 6.5.6. 算子kernel的计算过程（搬运+计算）

1. **数据搬运**：
   - 不需要实际的数据搬运，只是调整张量的形状信息
   - 建立从新张量到原始张量的内存映射关系

2. **计算**：
   - 不需要实际的计算操作，主要是形状信息的调整
   - 建立新维度的内存访问逻辑

3. **结果存储**：
   - 不需要实际的存储操作，新张量共享原始张量的内存
   - 只是创建一个新的张量对象，具有扩展后的形状
#### 6.5.8. 用例设计
| 用例名称 | 输入维度 | tileshape | 输入dtype | 说明 |
|----------|---------|-----------|-----------|------|
| unsqueeze_fp16_001 | (2) | (2) | fp16 | 1d张量在0维添加维度 |
| unsqueeze_fp16_002 | (2, 3) | (1, 3) | fp16 | 2d张量在0维添加维度 |
| unsqueeze_fp16_003 | (2, 3) | (1, 3) | fp16 | 2d张量在1维添加维度 |
| unsqueeze_fp16_004 | (2, 3, 4) | (1, 1, 4) | fp16 | 3d张量在2维添加维度 |
| unsqueeze_fp16_005 | (2, 3, 4, 5) | (1, 1, 1, 5) | fp16 | 4d张量在3维添加维度 |
| unsqueeze_fp16_006 | (5) | (5) | fp16 | 1d张量在0维添加维度 |
| unsqueeze_fp16_007 | (3, 4) | (1, 4) | fp16 | 2d张量在0维添加维度 |
| unsqueeze_fp16_008 | (3, 4, 5) | (1, 2, 5) | fp16 | 3d张量在1维添加维度 |
| unsqueeze_fp16_009 | (2, 3, 4, 5, 6) | (1, 1, 1, 1, 6) | fp16 | 5d张量在4维添加维度 |
| unsqueeze_fp16_010 | (4) | (4) | fp16 | 1d张量在0维添加维度 |
| unsqueeze_fp32_001 | (2) | (2) | fp32 | 1d张量在0维添加维度 |
| unsqueeze_fp32_002 | (2, 3) | (1, 3) | fp32 | 2d张量在0维添加维度 |
| unsqueeze_fp32_003 | (2, 3) | (1, 3) | fp32 | 2d张量在1维添加维度 |
| unsqueeze_fp32_004 | (2, 3, 4) | (1, 1, 4) | fp32 | 3d张量在2维添加维度 |
| unsqueeze_fp32_005 | (2, 3, 4, 5) | (1, 1, 1, 5) | fp32 | 4d张量在3维添加维度 |
| unsqueeze_fp32_006 | (5) | (5) | fp32 | 1d张量在0维添加维度 |
| unsqueeze_fp32_007 | (3, 4) | (1, 4) | fp32 | 2d张量在0维添加维度 |
| unsqueeze_fp32_008 | (3, 4, 5) | (1, 2, 5) | fp32 | 3d张量在1维添加维度 |
| unsqueeze_fp32_009 | (2, 3, 4, 5, 6) | (1, 1, 1, 1, 6) | fp32 | 5d张量在4维添加维度 |
| unsqueeze_fp32_010 | (4) | (4) | fp32 | 1d张量在0维添加维度 |
| unsqueeze_int32_001 | (2) | (2) | int32 | 1d张量在0维添加维度 |
| unsqueeze_int32_002 | (2, 3) | (1, 3) | int32 | 2d张量在0维添加维度 |
| unsqueeze_int32_003 | (2, 3, 4) | (1, 1, 4) | int32 | 3d张量在2维添加维度 |
| unsqueeze_int32_004 | (5) | (5) | int32 | 1d张量在0维添加维度 |
| unsqueeze_int32_005 | (3, 4) | (1, 4) | int32 | 2d张量在0维添加维度 |
## 7. 张量操作

### 7.1. pypto.full

#### 7.1.1. 算子计算原理

#### 7.1.2. pypto前段接口（python）以及支持范围

#### 7.1.3. c++ tensor graph接口

#### 7.1.4. c++ tile graph接口

#### 7.1.5. 算子npu计算使用的pto instruction

#### 7.1.6. 算子kernel的计算过程（搬运+计算）

#### 7.1.7. tile shape设置约束

#### 7.1.8. 用例设计

### 7.2. pypto.concat

#### 7.2.1. 算子计算原理

#### 7.2.2. pypto前段接口（python）以及支持范围

#### 7.2.3. c++ tensor graph接口

#### 7.2.4. c++ tile graph接口

#### 7.2.5. 算子npu计算使用的pto instruction

#### 7.2.6. 算子kernel的计算过程（搬运+计算）

#### 7.2.7. tile shape设置约束

#### 7.2.8. 用例设计

### 7.3. pypto.where

#### 7.3.1. 算子计算原理
condition 为一个布尔类型的掩码张量（mask tensor）。对于张量中任意位置的元素，该操作基于布尔掩码张量 condition 进行逐元素选择。其计算行为可形式化表示为如下表达式。

$$
result_{i}=
\begin{cases}
input_{i} & \text{if } condition_{i}==True \\
other_{i} & \text{if } condition_{i}==False
\end{cases}
$$

condition 须为Tensor，input 和 other 可以为 Tensor、 float  以及 Element，广播规则如下（**只支持单轴广播**）：

1.  input, other, condition 均为Tensor，result 的 Shape 由三者广播得到。

    例：input:[1,20,20], other:[20,1,20], condition:[20,20,1], result:[20,20,20]

2.  只有 input, condition 为 Tensor时，result 的 Shape 由两者广播得到。

    例：input:[1,20,20], condition:[20,20,1], result:[20,20,20]

3.  只有 other, condition 为 Tensor时，result 的 Shape 由两者广播得到。

    例：other:[20,1,20], condition:[20,20,1], result:[20,20,20]

4.  只有 condition 为 Tensor时，result 的 Shape 与 condition 一致。
#### 7.3.2. pypto前段接口（python）以及支持范围
函数原型

```python
where(
    condition: Tensor,
    input: Union[Tensor, float, Element],
    other: Union[Tensor, float, Element]
) -> Tensor
```

参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| condition   | 输入      | 支持的类型为：Tensor。<br> Tensor支持的数据类型为：DT_BOOL。<br> 不支持空Tensor；Shape支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。<br> 作为条件选择input或者other的元素。 |
| input       | 输入      | 支持的类型为 float\Element\Tensor类型。<br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。<br> Tensor和Element支持的数据类型为：DT_FP32，DT_FP16。<br> 不支持空Tensor；Shape支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| other       | 输入      | 支持的类型为 float\Element\Tensor类型。<br> 当为float类型时会自动转换为 Element 类型，float 对应 DT_FP32。当需要使用其他数据类型时，可以通过 Element 构建。<br> Tensor和Element支持的数据类型为：DT_FP32，DT_FP16。<br> 不支持空Tensor；Shape支持2-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

返回值说明

result ：Tensor，Shape由输入的广播得到，详细广播场景可看上文。数据类型和input、other保持一致。

约束说明

1. 建议优先使用 Element，传入 float 标量对于 fp16 场景，不保证正确性。
#### 7.3.3. c++ tensor graph接口
```c++
// 根据条件从两个Tensor中选取元素
Tensor Where(const Tensor &condition, const Tensor &input, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, input, other);
}

// 根据条件从Tensor和Scalar中选取元素
Tensor Where(const Tensor &condition, const Tensor &input, const Element &otherValue) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, input, otherValue);
}

// 根据条件从Scalar和Tensor中选取元素
Tensor Where(const Tensor &condition, const Element &inputValue, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, inputValue, other);
}

// 根据条件从两个Scalar中选取元素
Tensor Where(const Tensor &condition, const Element &inputValue, const Element &otherValue) {
    DECLARE_TRACER();
    RETURN_CALL(WhereOperation, *Program::GetInstance().GetCurrentFunction(), condition, inputValue, otherValue);
}
```
#### 7.3.4. c++ tile graph接口
```c++
// 根据条件从两个Tensor中选取元素
void WhereOperationTileFuncTT(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(function, tileShape, iOperand[0], iOperand[1], iOperand[2], oOperand[0]);
}

// 根据条件从Tensor和Scalar中选取元素
void WhereOperationTileFuncTS(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(
        function, tileShape, iOperand[0], iOperand[1], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

// 根据条件从Scalar和Tensor中选取元素
void WhereOperationTileFuncST(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), iOperand[1], oOperand[0]);
}

// 根据条件从Tensor和Scalar中选取元素
void WhereOperationTileFuncSS(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledWhereOperation(function, tileShape, iOperand[0], op.GetVectorElementAttribute(OpAttributeKey::vectorScalar)[0],
        op.GetVectorElementAttribute(OpAttributeKey::vectorScalar)[1], oOperand[0]);
}
```
#### 7.3.5. 算子npu计算使用的pto instruction
TLoad：搬运buffer从gm到ub

TWhere: 通过TASSIGN、TSEL封装实现
TASSIGN:将Tile对象绑定到实现定义的片上地址（手动选择）
TSEL：使用掩码Tile在两个Tile之间进行逐元素选择

TStore：ub to gm搬运
#### 7.3.6. 算子kernel的计算过程（搬运+计算）
```c++
// 搬运源操作数condition从gm到ub
TLoad(ubTensor_c, gmTensor_c, Coor2Dim(0, 1));

// 搬运源操作数x从gm到ub
TLoad(ubTensor_x, gmTensor_x, Coor2Dim(0, 8));

// TT表示从两个Tensor中按条件选择元素
TWhereTT(ubTensor_x, ubTensor_1, ubTensor_c, ubTensor_x, ubTensor_y);
// TS表示从Tensor和Scalar中按条件选择元素
TWhereTS(ubTensor_x, ubTensor_1, ubTensor_c, ubTensor_x, ubTensor_y);
// ST表示从Scalar和Tensor中按条件选择元素
TWhereST(ubTensor_x, ubTensor_1, ubTensor_c, ubTensor_x, ubTensor_y);
// SS表示从两个Scalar中按条件选择元素
TWhereSS(ubTensor_x, ubTensor_1, ubTensor_c, ubTensor_x, ubTensor_y);

// 将结果从ub搬运到gm
TStore(gmTensor_2, ubTensor_x, Coor2Dim(0, 8));
```
#### 7.3.7. tile shape设置约束
TileShape维度应和输出一致。
#### 7.3.8. 用例设计
测试因子
* 输入维度：2~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32
* condition的最后一维 = input和other广播后shape的最后一维 ÷ 8

| 用例名称  | input维度   | other维度   | condition维度 | 切分         | 输入dtype | 说明 |
| --------- | ----------- | ----------- | ------------ | ------------ | --------- | ---- |
| fp16_001  | (1, 112)    | (1, 112)    | (1, 14)      | (1, 56)      | fp16      | 2D 三Tensor同shape，W单轴切分 |
| fp16_002  | (1, 64)     | (1, 1)      | (1, 8)       | (1, 32)      | fp16      | 2D 单轴广播，W单轴切分 |
| fp16_003  | (3, 128)    | (3, 128)    | (3, 16)      | (3, 64)      | fp16      | 2D 三Tensor同shape，W单轴切分 |
| fp16_004  | (2, 96)     | (1, 96)     | (2, 12)      | (2, 96)      | fp16      | 2D 单轴广播，N单轴切分 |
| fp16_005  | (5, 1)      | (5, 32)     | (5, 4)       | (5, 1)       | fp16      | 2D 单轴广播，H单轴切分 |
| fp16_006  | (1, 64, 64) | (1, 64, 64) | (1, 8, 64)   | (1, 32, 64)  | fp16      | 3D 三Tensor同shape，C单轴切分 |
| fp16_007  | (2, 1, 48)  | (2, 3, 48)  | (2, 3, 6)    | (2, 1, 24)   | fp16      | 3D 单轴广播，W单轴切分 |
| fp16_008  | (3, 64, 1)  | (3, 64, 48) | (3, 8, 6)    | (3, 32, 1)   | fp16      | 3D 单轴广播，H单轴切分 |
| fp16_009  | (1,1,32,32) | (1,1,32,32) | (1,1,4,32)   | (1,1,16,32)  | fp16      | 4D 三Tensor同shape，H单轴切分 |
| fp16_010  | (1,4,16,16) | (1,1,16,16) | (1,4,2,16)   | (1,2,16,16)  | fp16      | 4D 单轴广播，C单轴切分 |
| fp16_011  | (2,1,24,24) | (2,3,24,24) | (1,3,3,24)   | (1,1,24,24)  | fp16      | 4D 单轴广播，N单轴切分 |
| fp16_012  | (4, 80)     | 标量        | (4, 10)      | (2, 40)      | fp16      | 2D Tensor+标量，H+W双轴切分 |
| fp16_013  | (1,2,40,40) | 标量        | (1,2,5,40)   | (1,1,40,20)  | fp16      | 4D Tensor+标量，W单轴切分 |
| fp16_014  | (8, 1)      | 标量        | (8, 1)       | (4, 1)       | fp16      | 2D Tensor+标量，N单轴切分 |
| fp16_015  | (2,3,16,16) | (2,3,16,16) | (2,3,2,2)    | (1,3,8,8)    | fp16      | 4D 三Tensor同shape，N+H+W多轴切分 |
| fp32_001  | (1, 112)    | (1, 112)    | (1, 14)      | (1, 56)      | fp32      | 2D 三Tensor同shape，W单轴切分 |
| fp32_002  | (1, 64)     | (1, 1)      | (1, 8)       | (1, 32)      | fp32      | 2D 单轴广播，W单轴切分 |
| fp32_003  | (3, 128)    | (3, 128)    | (3, 16)      | (3, 64)      | fp32      | 2D 三Tensor同shape，W单轴切分 |
| fp32_004  | (2, 96)     | (1, 96)     | (2, 12)      | (2, 96)      | fp32      | 2D 单轴广播，N单轴切分 |
| fp32_005  | (5, 1)      | (5, 32)     | (5, 4)       | (5, 1)       | fp32      | 2D 单轴广播，H单轴切分 |
| fp32_006  | (1, 64, 64) | (1, 64, 64) | (1, 8, 64)   | (1, 32, 64)  | fp32      | 3D 三Tensor同shape，C单轴切分 |
| fp32_007  | (2, 1, 48)  | (2, 3, 48)  | (2, 3, 6)    | (2, 1, 24)   | fp32      | 3D 单轴广播，W单轴切分 |
| fp32_008  | (3, 64, 1)  | (3, 64, 48) | (3, 8, 6)    | (3, 32, 1)   | fp32      | 3D 单轴广播，H单轴切分 |
| fp32_009  | (1,1,32,32) | (1,1,32,32) | (1,1,4,32)   | (1,1,16,32)  | fp32      | 4D 三Tensor同shape，H单轴切分 |
| fp32_010  | (1,4,16,16) | (1,1,16,16) | (1,4,2,16)   | (1,2,16,16)  | fp32      | 4D 单轴广播，C单轴切分 |
| fp32_011  | (2,1,24,24) | (2,3,24,24) | (1,3,3,24)   | (1,1,24,24)  | fp32      | 4D 单轴广播，N单轴切分 |
| fp32_012  | (4, 80)     | 标量        | (4, 10)      | (2, 40)      | fp32      | 2D Tensor+标量，H+W双轴切分 |
| fp32_013  | (1,2,40,40) | 标量        | (1,2,5,40)   | (1,1,40,20)  | fp32      | 4D Tensor+标量，W单轴切分 |
| fp32_014  | (8, 1)      | 标量        | (8, 1)       | (4, 1)       | fp32      | 2D Tensor+标量，N单轴切分 |
| fp32_015  | (2,3,16,16) | (2,3,16,16) | (2,3,2,2)    | (1,3,8,8)    | fp32      | 4D 三Tensor同shape，N+H+W多轴切分 |

## 8. 矩阵运算

### 8.1. pypto.matmul

参考：pypto/docs/api/operation/pypto-matmul.md

#### 8.1.1. 算子计算原理

实现input 、mat2矩阵的矩阵乘运算，计算公式为：out = input @ mat2

-   input 、mat2为源操作数，input 为左矩阵；mat2为右矩阵
-   out 为目的操作数，存放矩阵乘结果的矩阵

#### 8.1.2. pypto前段接口（python）以及支持范围

## 函数原型

```python
matmul(input, mat2, out_dtype, *, a_trans = False, b_trans = False, c_matrix_nz = False, extend_params=None) -> Tensor
```

参数说明


| 参数名            | 输入/输出 | 说明                                                                 |
|-------------------|-----------|----------------------------------------------------------------------|
| input             | 输入      | 表示输入左矩阵。不支持输入空Tensor。 <br> 支持的数据类型为：DT_INT8, DT_FP16。需保证左右矩阵一致。 <br> 支持的矩阵维度：2维、3维、4维，且左右矩阵维度需保持一致。 <br> 输入矩阵支持的Format为：TILEOP_ND, TILEOP_NZ。 <br> 当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。 <br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。 <br> 内轴外轴：当输入矩阵input非转置时，对应数据排布为[M, K]，此时外轴为M，内轴为K；当输入矩阵input转置时，对应数据排布为[K, M]，此时外轴为K，内轴为M； <br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐 <br> 当矩阵维度为3维或者4维时，不支持pypto.view场景。 |
| mat2              | 输入      | 表示输入右矩阵。不支持输入空Tensor。 <br> 支持的数据类型为：DT_INT8, DT_FP16。 <br> 支持的矩阵维度：2维、3维、4维，且左右矩阵维度需保持一致。 <br> 输入矩阵支持的Format为：TILEOP_ND, TILEOP_NZ。 <br> 当Format为TILEOP_ND（ND格式）时，外轴范围为[1, 2^31 - 1]，内轴范围为[1, 65535]。 <br> 当Format为TILEOP_NZ（NZ格式）时，其Shape维度需满足内轴32字节对齐（当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。 <br> 内轴外轴：当输入矩阵mat2非转置时，对应数据排布为[K, N]，此时外轴为K，内轴为N；当输入矩阵mat2转置时，对应数据排布为[N, K]，此时外轴为N，内轴为K； <br> 在使用pypto.view接口的场景，应保证传入View的Shape维度也满足内轴32字节对齐（其中当输出矩阵数据类型为DT_INT32时，内轴为16元素对齐），外轴16元素对齐。 <br> 当矩阵维度为3维或者4维时，不支持pypto.view场景。 |
| out_dtype         | 输出      | 表示输出矩阵数据类型，支持DT_FP16，DT_INT8。 |
| a_trans           | 输入      | 参数a_trans表示输入左矩阵是否转置，**端侧不支持，必须是False。** |
| b_trans           | 输入      | 参数b_trans表示输入右矩阵是否转置，默认为False。 |
| c_matrix_nz       | 输入      | 参数c_matrix_nz表示输出矩阵的Format是否采用NZ格式，默认为False，当前仅支持设置False，即输出矩阵仅支持ND格式。 |
| extend_params     | 输入      | 支持bias、fixpipe反量化及TF32舍入模式TransMode功能，数据类型为字典格式。 <br> 参数：bias_tensor <br> 类型：Tensor <br> 功能说明： <br> - 输入左右矩阵数据类型为DT_FP16时，Bias矩阵数据类型可选DT_FP16和DT_FP32。 <br> - 输入左右矩阵数据类型为DT_INT8时，Bias矩阵数据类型只能为DT_INT32。<br> - bias_tensor只支持ND格式。 <br> - bias_tensor的第一维度应置1，且N维度需要与mat2矩阵的N维度相等。 <br> - Bias不支持多核切K功能。 <br> - 仅支持矩阵维度为2维场景。 <br> 参数：scale <br> 类型：float <br> 功能说明： <br> - 输入为float类型，取1为符号位 + 8位指数位 + 10位尾数位参与运算。 <br> 参数：scale_tensor <br> 类型：Tensor <br> 功能说明： <br> - scale_tensor输入固定为uint64_t 的Tensor。计算时会转换uint64_t为float类型的低32位bit后，取1为符号位 + 8位指数位 + 10位尾数位参与运算。 <br> - scale_tensor的第一维度必须置1，且N维度需要与mat2矩阵的N维度相等。 <br> - scale_tensor只支持ND格式。 <br> - 仅支持矩阵维度为2维场景。 <br> 参数：relu_type <br> 类型：[ReLuType](../datatype/ReLuType.md) <br> 功能说明： <br> - 支持RELU和NO_RELU两种模式。 <br> - 仅支持矩阵维度为2维场景。 <br> 参数：trans_mode <br> 类型：[TransMode](../datatype/TransMode.md) <br> 端侧场景不支持trans_mode |

返回值说明

返回值为out 矩阵（Tensor）。

约束说明

-   调用matmul接口前需要通过pypto.set\_cube\_tile\_shapes设置M、N、K轴上的切分大小
-   当矩阵维度为3维或者4维时，需要调用pypto.set\_vec\_tile\_shapes接口设置vector的TileShape切分，如未设置，接口内部会设置2维的vec\_tile\_shape，其值为128，128。
-   调用matmul接口的输入为调用pypto.reshape后的NZ格式时，需要调用pypto.set\_matrix\_size接口设置pypto.reshape前的输入到matmul的原始Shape的m,k,n值。
-   调用matmul接口的输入矩阵维度为3维/4维并且数据格式为NZ格式时，需要调用pypto.set\_matrix\_size接口设置输入到matmul的原始Shape的m,k,n值。

#### 8.1.3. c++ tensor graph接口

pypto/framework/src/interface/operation/cube_operation_impl.cpp

```
Tensor Matmul(
    DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, bool isATrans, bool isBTrans, bool isCMatrixNZ);

Tensor Matmul(DataType outType, const Tensor &aMatrix, const Tensor &bMatrix, const MatmulExtendParam &param,
    bool isATrans, bool isBTrans, bool isCMatrixNZ);（新增MatmulExtendParam参数，包括量化、bias等信息）

Tensor BatchMatmul(DataType dataType, const Tensor &aMatrix, const Tensor &bMatrix, const bool isTransA,
    const bool isTransB, const bool isCMatrixNZ);
```

#### 8.1.4. c++ tile graph接口

pypto/framework/src/interface/operation/cube_operation_impl.cpp

```
void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op)
```

#### 8.1.5. 算子npu计算使用的pto instruction

TLoad<ND2NZ>：gm到l1的搬运，做nd2nz转换（n,d -> d1,n1,n0,d0；其中n0、d0为32B对齐单位）

TExtract：l1到l0a/l0b的搬运

TMatmul_：矩阵乘计算

TStore<NZ2ND>：l1到gm的搬运，做nz2nd转换（d1,n1,n0,d0 -> n,d；其中n0、d0为32B对齐单位）

#### 8.1.6. 算子kernel的计算过程（搬运+计算）

（云侧计算流程）

```
// 搬运a矩阵从gm到l1，使用nd2nz模式，转换为硬件对齐格式
TLoad<ND2NZ>(a_l1, a_gm)

// 搬运a矩阵从l1到l0a
TExtract(a_l0a, a_l1)

// 搬运b矩阵从gm到l1，使用nd2nz模式，转换为硬件对齐格式
TLoad<ND2NZ>(b_l1, b_gm)

// 搬运b矩阵从l1到l0b
TExtract(b_l0b, b_l1)

// 计算a_l0a和b_l0b的矩阵乘，将输出放在c_loc l0c buffer上，当前c_l0c格式是nz
TMatmul_(c_l0c, a_l0a, b_l0b)

// 搬运c矩阵从l0c到gm，使用nz2nd模式，转换为原始格式
TStore<NZ2ND>(c_gm, c_loc)
```

（端侧计算流程）

主要差别：TMatmul在云侧实现是 l0a，l0b -> l0c；TMatmul在端侧实现是 l1, l0b -> l1。

```
// 搬运a矩阵从gm到l1，使用nd2nz模式，转换为硬件对齐格式 **注意，这里没有l0a buffer，TMatmul的a输入直接从l1读**
TLoad<ND2NZ>(a_l1, a_gm)

// 搬运b矩阵从gm到l1，使用nd2nz模式，转换为硬件对齐格式
TLoad<ND2NZ>(b_l1, b_gm)

// 搬运b矩阵从l1到l0b
TExtract(b_l0b, b_l1)

// 计算a_l1和b_l0b的矩阵乘，将输出放在c_l1 l1 buffer上，当前c_l1格式是nz **注意，这里输出在l1 buffer，不是l0c buffer**
TMatmul_(c_l1, a_l1, b_l0b)

// 搬运c矩阵从l1到gm，使用nz2nd模式，转换为原始格式
TStore<NZ2ND>(c_gm, c_l1)
```

#### 8.1.7. tile shape设置约束

m1,m0,k1,k0,n1,n0
分别m,k,n代表矩阵乘的m,k@k,n；1切分代表l1 buffer的切分，l0代表l0b buffer的切分（m1=m0，m0没有用）

buffer大小约束：
* m1\*k1\*a_dtype_bytes + k1\*n1\*b_dtype_bytes + m1\*n1\*int32_bytes\*contains_partial_sum + m1\*n1\*c_dtype_bytes + elewise_size\*eltwise_dtype_bytes\*contains_eltwise <= L1_size
* k0\*n0\*b_dtype_bytes <= L0B_size
* n*int32_bytes <= BT_size
* n*uint64_bytes <= fixpipe_quant_pre_size
* n*uint32_bytes <= fixpipe_quant_pre_act_size
* n*uint64_bytes <= fixpipe_quant_post_size
* 1*uint64_bytes <= fixpipe_quant_post_eltwise_antiquant_size（肯定满足）

算子约束：
tile m是16的整数倍，tile k和tile n是32B的整数倍。

#### 8.1.8. 用例设计

测试因子
* a、b矩阵维度（必须相同）：2~4维
* bias: 有/没有
* 切分：m1,m0,k1,k0,n1,n0组合
* 数据类型：FP16FP16（输出可以S8或者FP16）/S8S8（输出可以S8或者FP16）量化

| 用例名称 | A输入维度 | B输入维度 | bias | b_trans | 切分 (m1,m0,k1,k0,n1,n0) | 输入dtype | 输出dtype | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| matmul_001 | (16,16) | (16,16) | 有 | false | (16,16,16,16,16,16) | FP16/FP16 | FP16 | 基础FP16用例 |
| matmul_002 | (64,130) | (130,64) | 没有 | false | (64,64,16,16,16,16) | FP16/FP16 | S8 | K非对齐，验证FP16到S8量化 |
| matmul_003 | (64,128) | (80,128) | 有 | true | (32,32,16,16,16,16) | FP16/FP16 | FP16 | N非对齐，B转置，输出FP16 |
| matmul_004 | (80,256) | (256,32) | 有 | false | (64,64,16,16,32,32) | FP16/FP16 | S8 | M非对齐，验证Bias累加后量化 |
| matmul_005 | (4,32,32) | (4,32,32) | 没有 | false | (16,16,16,16,16,16) | FP16/FP16 | FP16 | 3维带Batch，标准对齐切分 |
| matmul_006 | (2,2,64,64) | (2,2,64,64) | 有 | false | (32,32,32,32,32,32) | FP16/FP16 | FP16 | 4维高维张量，验证多维Offset逻辑 |
| matmul_007 | (128,64) | (64,128) | 没有 | false | (64,64,32,32,64,64) | S8/S8 | S8 | 纯S8整型路径，验证饱和处理 |
| matmul_008 | (64,64) | (64,128) | 有 | true | (32,32,32,32,64,64) | S8/S8 | FP16 | S8转FP16，验证带Bias的反量化路径 |
| matmul_009 | (8,48,48) | (8,48,48) | 有 | false | (16,32,16,32,16,32) | S8/S8 | S8 | 3维S8，非2幂次维度(48)切分验证 |
| matmul_010 | (2,3,32,64) | (2,3,64,32) | 没有 | false | (16,16,32,32,16,16) | S8/S8 | FP16 | 4维S8，验证非对称M/N维度 |
| matmul_011 | (128,33) | (33,128) | 有 | false | (64,64,32,32,64,64) | S8/S8 | S8 | 2d s8量化 |
| matmul_012 | (64,128) | (64,128) | 有 | true | (32,32,64,64,32,32) | FP16/FP16 | S8 | 综合用例：B转置 + Bias + 输出量化 |

## 9. 芯片能力增强

### 9.1. pypto.index_put_

参考：pypto/docs/api/operations/pypto-indexput_.md
主要用于实现onnx scatter nd。

#### 9.1.1. 算子计算原理

根据索引indices将values的多个或多块数据更新到self中。如果accumulate参数为True，则表示在更新时，values和原本存储在相应位置的值进行累加；如果accumulate为False，则会直接覆盖原本的值。

结果示例如下：

```python
输入数据 x:      [[1 1 1],
                 [1 1 1],
                 [0 0 0]]
      indices:   ([1 2], )
      values:    [[0 1 0],
                  [0 2 0]]
原地更新后的 x:   [[1 1 1],
                 [1 2 1],
                 [0 2 0]]               # accumulate is True
                 [[1 1 1],
                 [0 1 0],
                 [0 2 0]]               # accumulate is False
```

计算过程：
```
indices=[1,2] 代表在row 1、2实现替换。

1. 当accumulate=True
Row 1（indices[0] = 1）= x[indices[0]][:] + values[0][:] = x[1][:] + values[0][:] = [1,1,1] + [0,1,0] = [1,2,1]
Row 2（indices[1] = 2）= x[indices[1]][:] + values[1][:] = x[2][:] + values[1][:] = [0,0,0] + [0,2,0] = [0,2,0]
Result = [[1,1,1],[1,2,1],[0,2,0]]

2. 当accumulate=False
Row 1（indices[0] = 1）= values[0][:] = [0,1,0]
Row 2（indices[1] = 2）= values[1][:] = [0,2,0]
Result = [[1,1,1],[0,1,0],[0,2,0]]
```

#### 9.1.2. pypto前段接口（python）以及支持范围

函数原型

```python
index_put_(input: Tensor, indices: tuple, values: Tensor, accumulate: bool = False) -> None
```

参数说明

|   参数名   | 输入/输出 | 说明                                                                  |
|------------|-----------|----------------------------------------------------------------------|
|   input    |    输入   | 源操作数。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32。 <br> 不支持空Tensor，Shape仅支持1-4维，Shape Size不大于2147483647（即INT32_MAX）。 |
|  indices   |   输入    | Tensor类型的元组，每个Tensor表示一个维度的索引。 <br> 支持的类型为：tuple\[Tensor\], 每个Tensor均为一维，且维度相同。 <br> Tensor支持的数据类型为：DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32，DT_INT64，DT_UINT64。 <br> 不支持空Tensor，tuple中Tensor的个数不大于input的维数。 |
|   values   |   输入    | 待更新到input中的值。 <br> 支持的类型为：Tensor。 <br> Tensor支持的数据类型为：DT_FP16，DT_FP32。 <br> 不支持空Tensor，维数不大于input的维数。 |
| accumulate |   输入    | 累加参数，默认为False。 <br> 支持的类型为：bool。 |

返回值说明

对input进行原地操作，无返回值

约束说明

1. indices中的一维Tensor维度相同，不支持broadcast。indices中第i个Tensor中的值须小于input中第i-1维的Shape大小。当indices的选取会对同一个位置进行重复更新时，结果是未确定的。

2. values不支持broadcast，其第0维的shape须和indices中一维Tensor的shape相同。

3. input的维度、indices中Tensor的个数和values的维度之间需满足：（input.shape.size） + 1 = (indices.size) + (values.shape.size)。

4. input和values的数据类型须相同。

5. viewshape为一维，针对indices中的每个一维Tensor和values的第0维进行切分，values的其它维度不做切分。

#### 9.1.3. c++ tensor graph接口

pypto/framework/src/interface/operation/vector/indexing.cpp

```
void IndexPut_(Tensor &self, const std::vector<Tensor> &indices, const Tensor &values, bool accumulate = false)
```

#### 9.1.4. c++ tile graph接口

pypto/framework/src/interface/operation/vector/indexing.cpp

```
void IndexPutOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op)
```

#### 9.1.5. 算子npu计算使用的pto instruction

TLoad：搬运buffer从gm到ub

TIndexPut：搬运ub_input到gm_output，并且在ub_indices，搬运ub_values到对应的gm_output，实现accumulate功能如果开关有打开。

#### 9.1.6. 算子kernel的计算过程（搬运+计算）

```
// 搬运input从gm到ub
TLoad(ub_input, gm_input)

// 搬运indices从gm到ub
TLoad(ub_indices, gm_indices)

// 搬运values从gm到ub
TLoad(ub_values, gm_values)

// 实现IndexPut并且输出到gm
TIndexPut(gm_output, ub_input, ub_indices, ub_values)
```

#### 9.1.7. tile shape设置约束

TileShape为一维，针对indices中的每个一维Tensor和values的第0维进行切分，values的其它维度不做切分。indices和values的TileShape大小总和不能超过UB内存的大小。

#### 9.1.8. 用例设计

测试因子
* 输入维度：1~4维
* 切分：基于indices中的每一个维Tensor和values的第0维切分，1d
* 数据类型：input/indices：FP16/FP32，values：INT8，UINT8，INT16，UINT16，INT32，UINT32，INT64，UINT64

| 用例名称  | input维度 | indices维度 | values维度 | input/values dtype | indices dtype | 切分 | 说明 |
|----------|-----------|------------|------------|--------------------|---------------|------|------|
| index_put_001 | (3,3) | (2,) | (2,3) | FP16 | INT32 | 3 | 2d输入，indices包含2个索引，替换2行数据 |
| index_put_002 | (5,) | (3,) | (3,) | FP32 | INT8 | 3 | 1d输入，基础索引操作，values为INT8 |
| index_put_003 | (2,4,4) | (2,) | (2,4,4) | FP32 | UINT8 | 2 | 3d输入，切分第一维，values为UINT8 |
| index_put_004 | (2,2,3,3) | (2,) | (2,2,3,3) | FP16 | INT16 | 2 | 4d输入，高维张量索引更新，values为INT16 |
| index_put_005 | (8,8) | (4,) | (4,8) | FP32 | UINT16 | 4 | 2d输入，较大范围切分，values为UINT16 |
| index_put_006 | (4,5,6) | (2,) | (2,5,6) | FP16 | INT32 | 2 | 3d输入，验证FP16下INT32 values的转换 |
| index_put_007 | (3,3,3,3) | (3,) | (3,3,3,3) | FP32 | UINT32 | 3| 4d输入，边界维度测试，values为UINT32 |
| index_put_008 | (10,) | (5,) | (5,) | FP16 | INT64 | 5 | 1d输入，长整型values测试 |
| index_put_009 | (2,2) | (2,) | (2,2) | FP32 | INT32 | 2 | 最小维度组合，验证最高精度整数values |
| index_put_010 | (3,4,5) | (4,) | (4,4,5) | FP16 | UINT64 | 3 | 3d输入，验证indices长度与values第0维一致性 |

## 10. 激活类型

### 10.1. pypto.sigmoid

#### 10.1.1. 算子计算原理

对输入Tensor的每个元素应用 sigmoid 激活函数，计算公式为：

$$
sigmoid(input) = \frac{1}{1 + e^{-input}}
$$

参考：pypto/docs/api/operation/pypto-sigmoid.md

#### 10.1.2. pypto前段接口（python）以及支持范围

函数原型
```python
sigmoid(input: Tensor) -> Tensor
```

参数说明
| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。 <br> 支持的数据类型为：DT_FP16、DT_FP32。 <br> shape支持1~4维度 <br> 不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |


返回值说明
返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入元素经sigmoid 函数映射到 \(0, 1\) 区间的结果。

#### 10.1.3. c++ tensor graph接口

Tensor Sigmoid(Tensor &input);

通过小算子：mul、exp、add、div operation实现计算。

pypto/framework/src/operator/activation/sigmoid.cpp

```
Tensor Sigmoid(Tensor &input) {
    // 1/(1+exp(-x))
    auto dtype = input.GetStorage()->Datatype();
    if (dtype != DT_FP32) {
        input = Cast(input, DataType::DT_FP32);
    }
    auto expRes = Exp(Mul(input, Element(DataType::DT_FP32, F_NEGA_1)));
    auto res = Add(expRes, Element(DataType::DT_FP32, F_1));
    Element src(DataType::DT_FP32, 1.0f);
    auto ones = Full(src, DataType::DT_FP32, res.GetShape());
    res = Div(ones, res);
    if (dtype != DT_FP32) {
        res = Cast(res, dtype);
    }
    return res;
}
```

#### 10.1.4. c++ tile graph接口

没有独立的tile graph接口，在ExpandFunction阶段，调用小算子的tile graph接口。

#### 10.1.5. 算子npu计算使用的pto instruction

TLoad：gm to ub搬运

TVecDup：broadcast实现，扩展一个vec

TMulS：vec multiply scalar计算

TExp：vec exp计算

TAddS：vec add scalar计算

TDiv：vec div dev计算

TStore：ub to gm搬运

#### 10.1.6. 算子kernel的计算过程（搬运+计算）

```
// load x from gm to ub
TLoad(ub_x, gm_x)

// input * -1
TMulS(ub_x, ub_x, -1)

// e^(-input)
TExp(ub_x, ub_x)

// 1 + e^(-input)
TaddS(ub_x, ub_x, 1)

// broadcast 1s
TVecDup(ub_ones, 1)

// 1 / (1 + e^(-input))
TDiv(ub_ones, ub_ones, ub_x)

// store x from ub to gm
TStore(gm_y, ub_ones)
```

#### 10.1.7. tile shape设置约束

无。

#### 10.1.8. 用例设计

测试因子
* 输入维度：1~4维
* 切分：n,c,h,w全量组合
* 数据类型：FP16/FP32

| 用例名称  | 输入维度 | 切分 | 输入dtype | 说明 |
|----------|---------|-----------|------|------|
| sigmoid_fp16_001  | (112) | (50) | fp16 | 1d尾轴对齐，w切分 |
| sigmoid_fp16_002  | (100) | (100) | fp16 | 1d尾轴不对齐，无切分 |
| sigmoid_fp16_003  | (4,128) | (2,32) | fp16 | 2d尾轴对齐，h,w切分 |
| sigmoid_fp16_004  | (4,130) | (1,130) | fp16 | 2d尾轴不对齐，h切分 |
| sigmoid_fp16_005  | (2,4,160) | (1,2,32) | fp16 | 3d尾轴对齐，c,h,w切分 |
| sigmoid_fp16_006  | (2,4,140) | (1,2,140) | fp16 | 3d尾轴不对齐，c,h切分 |
| sigmoid_fp16_007  | (2,5,152) | (1,5,32) | fp16 | 3d尾轴不对齐，c,w切分 |
| sigmoid_fp16_008  | (2,3,170) | (1,3,170) | fp16 | 3d尾轴不对齐，c切分 |
| sigmoid_fp16_009  | (5,2,4,176) | (2,1,2,16) | fp16 | 4d尾轴对齐，n,c,h,w切分 |
| sigmoid_fp16_010  | (5,2,4,130) | (1,1,1,130) | fp16 | 4d尾轴不对齐，n,c,h切分 |
| sigmoid_fp16_011  | (2,3,5,134) | (1,1,5,32) | fp16 | 4d尾轴不对齐，n,c,w切分 |
| sigmoid_fp16_012  | (4,2,6,135) | (2,2,3,32) | fp16 | 4d尾轴不对齐，n,h,w切分 |
| sigmoid_fp16_013  | (6,2,4,130) | (1,1,4,130) | fp16 | 4d尾轴不对齐，n,c切分 |
| sigmoid_fp16_014  | (3,2,3,139) | (1,2,1,139) | fp16 | 4d尾轴不对齐，n,h切分 |
| sigmoid_fp16_015  | (6,3,5,141) | (3,3,5,32) | fp16 | 4d尾轴不对齐，n,w切分 |
| sigmoid_fp32_001  | (112) | (50) | fp32 | 1d尾轴对齐，w切分 |
| sigmoid_fp32_002  | (100) | (100) | fp32 | 1d尾轴不对齐，无切分 |
| sigmoid_fp32_003  | (4,128) | (2,32) | fp32 | 2d尾轴对齐，h,w切分 |
| sigmoid_fp32_004  | (4,130) | (1,130) | fp32 | 2d尾轴不对齐，h切分 |
| sigmoid_fp32_005  | (2,4,160) | (1,2,32) | fp32 | 3d尾轴对齐，c,h,w切分 |
| sigmoid_fp32_006  | (2,4,140) | (1,2,140) | fp32 | 3d尾轴不对齐，c,h切分 |
| sigmoid_fp32_007  | (2,5,152) | (1,5,32) | fp32 | 3d尾轴不对齐，c,w切分 |
| sigmoid_fp32_008  | (2,3,170) | (1,3,170) | fp32 | 3d尾轴不对齐，c切分 |
| sigmoid_fp32_009  | (5,2,4,176) | (2,1,2,16) | fp32 | 4d尾轴对齐，n,c,h,w切分 |
| sigmoid_fp32_010  | (5,2,4,130) | (1,1,1,130) | fp32 | 4d尾轴不对齐，n,c,h切分 |
| sigmoid_fp32_011  | (2,3,5,134) | (1,1,5,32) | fp32 | 4d尾轴不对齐，n,c,w切分 |
| sigmoid_fp32_012  | (4,2,6,135) | (2,2,3,32) | fp32 | 4d尾轴不对齐，n,h,w切分 |
| sigmoid_fp32_013  | (6,2,4,130) | (1,1,4,130) | fp32 | 4d尾轴不对齐，n,c切分 |
| sigmoid_fp32_014  | (3,2,3,139) | (1,2,1,139) | fp32 | 4d尾轴不对齐，n,h切分 |
| sigmoid_fp32_015  | (6,3,5,141) | (3,3,5,32) | fp32 | 4d尾轴不对齐，n,w切分 |