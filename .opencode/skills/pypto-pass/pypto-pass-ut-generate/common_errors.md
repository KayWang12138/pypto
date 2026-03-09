# Pass UT开发常见错误记录

本文档记录在开发Pass单元测试（UT）过程中遇到的常见错误和解决方案，避免重复犯错。

---

## 语法错误类

### 错误1：Operations()调用语法错误

**错误代码**：
```cpp
function->Operations().().size()
```

**正确代码**：
```cpp
function->Operations().size()
```

**原因**：Operations()返回的是引用或对象，不需要再调用()

**解决方案**：检查所有Operations()的调用，确保只调用一次()

---

### 错误2：中文字符混入代码

**错误代码**：
```cpp
op.GetOOperands()[0]->Datatype与其他()
```

**正确代码**：
```cpp
op.GetOOperands()[0]->Datatype()
```

**原因**：输入法切换问题导致中文字符混入

**解决方案**：使用英文输入法，仔细检查代码中的函数名

---

### 错误3：重复定义变量

**错误代码**：
```cpp
const int opNum1 = 1;
// ... some code ...
const int opNum1 = 1;  // 重复定义
```

**正确代码**：
```cpp
const int opNum1 = 1;
// ... some code ...
// 不要重复定义，直接使用已定义的变量
```

**原因**：在不同测试用例中使用了相同的变量名，但忘记检查作用域

**解决方案**：每个测试用例使用不同的变量名，或确保变量作用域正确

---

### 错误4：类定义中使用中文冒号

**错误代码**：
```cpp
class AutoCastExtendedTest : public testing::Test {
public:
    static void SetUpTestCase() {}
```

**正确代码**：
```cpp
class AutoCastExtendedTest : public testing::Test {
public:
    static void SetUpTestCase() {}
```

**原因**：输入法切换导致冒号变成中文冒号

**解决方案**：使用英文输入法编写代码

---

## 逻辑错误类

### 错误5：错误的编译阶段配置

**错误代码**：
```cpp
config::SetHostOption(COMPILE_STAGE, CS::EXECUTE_GRAPH);
```

**正确代码**（对于TensorGraph阶段的Pass）：
```cpp
config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
```

**原因**：未根据Pass所在的文件夹目录选择正确的编译阶段

**解决方案**：根据Pass所在目录选择编译策略
- tensor_graph_pass: CS_TENSOR_GRAPH
- tile_graph_pass: CS_TILE_GRAPH
- block_graph_pass: CS_EXECUTE_GRAPH

---

### 错误6：错误的期望操作数

**错误代码**：期望的操作数与实际不符

**原因**：未仔细分析Pass的业务逻辑，对插入或删除的操作数预估错误

**解决方案**：
1. 先运行Pass，打印实际的操作数
2. 根据实际结果调整期望值
3. 确保理解每个功能点会插入/删除多少操作

---

### 错误7：未验证数据类型转换

**错误代码**：只验证操作数，不验证数据类型

**正确代码**：验证Cast操作后的数据类型是否正确

**原因**：AutoCast Pass的核心功能是数据类型转换，必须验证转换结果

**解决方案**：在遍历操作时，检查每个操作输入输出的数据类型

---

### 错误8：未验证Tensor连接关系

**错误代码**：只验证操作数，不验证Tensor的连接关系

**正确代码**：验证操作删除后，Tensor的消费者和生产者关系是否正确

**原因**：删除操作后需要确保图结构正确

**解决方案**：验证关键Tensor的消费者和生产者数量

---

## 测试覆盖类

### 错误9：测试用例覆盖不全

**错误**：只测试了部分场景

**原因**：未全面分析Pass的所有功能点

**解决方案**：
1. 列出Pass的所有功能点（如InsertBF16Cast、RemoveRedundantCastChain等）
2. 为每个功能点设计至少一个测试用例
3. 考虑边界情况（空图、无Cast、冗余链等）

---

### 错误10：未测试架构差异

**错误**：未测试不同NPU架构下的行为差异

**原因**：AutoCast Pass在DAV_3510和其他架构上有不同行为

**解决方案**：针对不同架构设计专门的测试用例

---

### 错误11：未测试边界情况

**错误**：只测试正常场景，未测试边界情况

**原因**：边界情况容易暴露Pass的bug

**解决方案**：测试以下边界情况
- 空图
- 单个操作
- 多个相同操作
- 操作链
- 多个消费者

---

## 环境配置类

### 错误12：未重置Program和Config

**错误代码**：测试用例间未重置环境

**正确方法**：在SetUp()中重置Program和Config

**原因**：测试用例间可能相互影响

**解决方案**：确保每个测试用例开始前环境是干净的

---

### 错误13：NPU架构未恢复

**错误代码**：修改NPU架构后未恢复

**正确代码**：测试结束后恢复默认架构

**原因**：影响后续测试用例

**解决方案**：在测试用例结束前恢复NPUArch::DAV_UNKNOWN

---

## 内存管理类

### 错误14：智能指针使用不当

**错误代码**：
```cpp
auto tensor = new LogicalTensor(...);  // 未使用智能指针
```

**正确代码**：
```cpp
auto tensor = std::make_shared<LogicalTensor>(...);  // 使用智能指针
```

**原因**：未使用智能指针可能导致内存泄漏

**解决方案**：使用std::make_shared创建对象

---

### 错误15：Operation引用获取错误

**错误代码**：
```cpp
Operation *op = function.AddOperation(...);  // 返回的是引用
```

**正确代码**：
```cpp
Operation &op = function.AddOperation(...);  // 使用引用
```

**原因**：AddOperation返回的是引用，不是指针

**解决方案**：使用引用接收AddOperation的返回值

---

## 数据结构类

### 错误16：Shape比较错误

**错误代码**：
```cpp
if (tensor1->shape == tensor2->shape) { ... }  // shape是成员变量
```

**正确代码**：
```cpp
if (tensor1->GetShape() == tensor2->GetShape()) { ... }  // 使用getter方法
```

**原因**：shape是成员变量，应该使用GetShape()方法

**解决方案**：使用GetShape()方法获取shape

---

### 错误17：DataType比较错误

**错误代码**：
```cpp
if (tensor->Datatype() == DataType::DT_FP32) { ... }  // 方法名错误
```

**正确代码**：
```cpp
if (tensor->GetDatatype() == DataType::DT_FP32) { ... }  // 使用GetDatatype()
```

**原因**：DataType的getter方法名是GetDatatype()

**解决方案**：使用GetDatatype()方法

---

## Opcode相关类

### 错误18：Opcode枚举使用错误

**错误代码**：
```cpp
if (op.GetOpcode() == OP_CAST) { ... }  // 缺未命名空间
```

**正确代码**：
```cpp
if (op.GetOpcode() == Opcode::OP_CAST) { ... }  // 使用Opcode命名空间
```

**原因**：Opcode是枚举类，需要使用Opcode::前缀

**解决方案**：使用Opcode::前缀访问枚举值

---

### 错误19：Opcode集合查找错误

**错误代码**：
```cpp
if (opSet.find(op.GetOpcode()) != opSet.end()) { ... }  // 正确
```

**原因**：Opcode是枚举类，可以直接在集合中查找

**解决方案**：确保使用正确的集合查找方法

---

## Attribute相关类

### 错误20：Attribute获取错误

**错误代码**：
```cpp
auto attr = op.GetOpAttribute();  // 返回的是shared_ptr
```

**正确代码**：
```cpp
auto attr = op.GetOpAttribute().get();  // 获取原始指针
```

**原因**：GetOpAttribute()返回的是shared_ptr，需要使用get()获取原始指针

**解决方案**：使用get()方法获取原始指针或直接使用shared_ptr

---

### 错误21：Attribute类型转换错误

**错误代码**：
```cpp
auto attr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute());  // 错误
```

**正确代码**：
```cpp
auto attr = dynamic_cast<ViewOpAttribute*>(op.GetOpAttribute().get());  // 正确
```

**原因**：需要先获取原始指针再进行类型转换

**解决方案**：先使用get()获取原始指针，再进行dynamic_cast

---

## Tensor连接类

### 错误22：消费者/生产者获取错误

**错误代码**：
```cpp
auto consumers = tensor->consumers;  // consumers是成员变量
```

**正确代码**：
```cpp
auto consumers = tensor->GetConsumers();  // 使用getter方法
```

**原因**：应该使用GetConsumers()方法

**解决方案**：使用GetConsumers()和GetProducers()方法

---

### 错误23：Tensor连接关系验证错误

**错误代码**：
```cpp
EXPECT_EQ(tensor->GetConsumers().size(), 1);  // 未验证具体消费者
```

**正确代码**：
```cpp
EXPECT_EQ(tensor->GetConsumers().size(), 1);
EXPECT_EQ(tensor->GetConsumers()[0]->GetOpcode(), Opcode::OP_ADD);  // 验证具体消费者
```

**原因**：需要验证连接关系的正确性

**解决方案**：验证消费者/生产者的具体操作类型

---

## Pass执行类

### 错误24：未调用PreCheck/PostCheck

**错误代码**：
```cpp
pass.RunOnFunction(function);  // 只调用RunOnFunction
```

**正确代码**：
```cpp
EXPECT_EQ(pass.PreCheck(function), SUCCESS);  // 调用PreCheck
EXPECT_EQ(pass.RunOnFunction(function), SUCCESS);  // 调用RunOnFunction
EXPECT_EQ(pass.PostCheck(function), SUCCESS);  // 调用PostCheck
```

**原因**：需要验证Pass的前置和后置检查

**解决方案**：调用PreCheck和PostCheck并验证返回值

---

### 错误25：Pass返回值未验证

**错误代码**：
```cpp
pass.RunOnFunction(function);  // 未验证返回值
```

**正确代码**：
```cpp
EXPECT_EQ(pass.RunOnFunction(function), SUCCESS);  // 验证返回值
```

**原因**：需要验证Pass执行是否成功

**解决方案**：使用EXPECT_EQ验证Pass的返回值

---

## 遍历相关类

### 错误26：遍历时修改容器

**错误代码**：
```cpp
for (auto &op : function->Operations()) {
    if (shouldRemove) {
        function.RemoveOperation(op);  // 遍历时修改容器
    }
}
```

**正确代码**：
```cpp
auto opList = function->Operations().DuplicatedOpList();  // 先复制
for (auto op : opList) {
    if (shouldRemove) {
        function.RemoveOperation(op);  // 安全删除
    }
}
```

**原因**：遍历时修改容器会导致未定义行为

**解决方案**：先复制操作列表，再进行修改

---

### 错误27：遍历顺序依赖

**错误代码**：假设遍历顺序固定

**原因**：容器的遍历顺序可能不固定

**解决方案**：不要依赖遍历顺序，使用操作ID或Magic ID进行验证

---

## 验证相关类

### 错误28：只验证数量不验证内容

**错误代码**：
```cpp
EXPECT_EQ(function->Operations().size(), 3);  // 只验证数量
```

**正确代码**：
```cpp
EXPECT_EQ(function->Operations().size(), 3);
uint32_t cast_num = 0;
for (auto &op : function->Operations()) {
    if (op.GetOpcode() == Opcode::OP_CAST) {
        ++cast_num;
    }
}
EXPECT_EQ(cast_num, 2);  // 验证具体操作数量
```

**原因**：需要验证操作的具体类型

**解决方案**：遍历操作，统计各类型操作的数量

---

### 错误29：未验证操作输入输出

**错误代码**：只验证操作存在，不验证输入输出

**正确代码**：验证操作的输入输出Tensor是否正确

**原因**：需要确保操作的输入输出连接正确

**解决方案**：验证操作的IOperands和OOperands

---

## 编译相关类

### 错误30：头文件包含错误

**错误代码**：缺少必要的头文件

**原因**：未包含必要的头文件导致编译错误

**解决方案**：参考现有测试文件，确保包含所有必要的头文件

---

### 错误31：命名空间使用错误

**错误代码**：
```cpp
using namespace std;
using namespace npu;
// ... some code ...
Operation op;  // 未指定命名空间
```

**正确代码**：
```cpp
using namespace std;
using namespace npu::tile_fwk;
// ... some code ...
Operation op;  // 在正确的命名空间中
```

**原因**：命名空间使用不当导致编译错误

**解决方案**：确保在正确的命名空间中编写代码

---

## 调试相关类

### 错误32：缺少调试信息

**错误代码**：测试失败时没有足够的调试信息

**正确代码**：使用EXPECT_EQ并提供有意义的消息

**原因**：缺少调试信息难以定位问题

**解决方案**：使用EXPECT_EQ、EXPECT_NE等宏并提供有意义的消息

---

### 错误33：未打印中间结果

**错误代码**：不打印中间结果进行调试

**正确代码**：打印关键中间结果

**原因**：难以定位问题所在

**解决方案**：在关键位置打印中间结果

---

## 总结

### 最容易犯的错误（Top 10）

1. **Operations()调用语法错误**：`Operations().()` 多余的括号
2. **变量重复定义**：在同一测试用例中多次定义相同变量名
3. **编译阶段配置错误**：使用了错误的COMPILE_STAGE值
4. **数据类型验证不完整**：只检查操作数，不检查数据类型
5. **中文字符混入**：输入法未切换导致代码中混入中文字符
6. **未重置Program和Config**：测试用例间未重置环境
7. **定义变量未使用**：声明一个变量，但在后续过程中未使用
8. **智能指针使用不当**：未使用智能指针导致内存泄漏
9. **Operation引用获取错误**：AddOperation返回的是引用，不是指针
10. **遍历时修改容器**：遍历时修改容器导致未定义行为


### 避免错误的最佳实践

1. **仔细检查代码语法**：特别是函数调用和括号匹配
2. **使用不同的变量名**：避免重复定义变量
3. **根据Pass所在目录选择正确的编译阶段**
4. **验证数据类型转换的正确性**：检查每个操作的输入输出数据类型
5. **使用英文输入法编写代码**：避免中文字符混入
6. **在SetUp()中重置环境**：确保每个测试用例开始前环境是干净的
7. **恢复NPU架构**：测试用例结束前恢复默认架构
8. **使用智能指针**：避免内存泄漏
9. **使用引用接收AddOperation的返回值**
10. **先复制操作列表再修改**：避免遍历时修改容器

### 调试技巧

1. **打印中间结果**：在关键位置打印中间结果
2. **分步验证**：逐步验证每个功能点
3. **使用断言**：使用EXPECT_EQ、EXPECT_NE等宏
4. **检查图结构**：验证操作和Tensor的连接关系
5. **使用gcov**：统计代码覆盖率，确保测试充分

### 测试设计原则

1. **覆盖所有功能点**：为每个功能点设计测试用例
2. **测试边界情况**：空图、单个操作、操作链等
3. **验证图结构**：不仅验证操作数，还要验证连接关系
4. **测试架构差异**：针对不同架构设计专门的测试用例
5. **逐步验证**：先验证简单场景，再验证复杂场景
