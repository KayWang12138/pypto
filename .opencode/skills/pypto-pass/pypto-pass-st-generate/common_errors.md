# Pass ST开发常见错误记录

本文档记录在开发Pass集成测试（ST）过程中遇到的常见错误和解决方案，避免重复犯错。

---

## 语法错误类

### 错误1：FUNCTION宏使用错误

**错误代码**：
```cpp
FUNCTION("STCase1") {
    exp = Exp(input);
    // 缺少闭合大括号
```

**正确代码**：
```cpp
FUNCTION("STCase1") {
    exp = Exp(input);
}
```

**原因**：FUNCTION宏需要完整的函数体，包括闭合大括号

**解决方案**：确保FUNCTION宏后有完整的函数体

---

### 错误2：Tensor构造函数参数错误

**错误代码**：
```cpp
Tensor input(DT_FP32, shape, "input");  // 缺少必要参数
```

**正确代码**：
```cpp
Tensor input(DT_FP32, shape, "input");
```

**原因**：Tensor构造函数的参数顺序和类型不正确

**解决方案**：参考Tensor类的构造函数定义

---

### 错误3：宏定义中使用中文标点

**错误代码**：
```cpp
FUNCTION("STCase1") {
    exp = Exp(input);  // 中文标点
}
```

**正确代码**：
```cpp
FUNCTION("STCase1") {
    exp = Exp(input);
}
```

**原因**：输入法切换导致标点变成中文标点

**解决方案**：使用英文输入法编写代码

---

## 逻辑错误类

### 错误4：Pass策略注册错误

**错误代码**：
```cpp
passManager.RegisterStrategy("AutoCastTestStrategy", {
    {"AutoCast", PassName::AUTO_CAST},  // 缺少其他必要Pass
});
```

**正确代码**：
```cpp
passRegisterStrategy("AutoCastTestStrategy", {
    {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
    {"AutoCast", PassName::AUTO_CAST},
});
```

**原因**：ST测试需要注册完整的Pass链，不仅仅是当前Pass

**解决方案**：参考现有ST测试文件，注册完整的Pass链

---

### 错误5：编译阶段配置错误

**错误代码**：
```cpp
config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);  // 错误的编译阶段
```

**正确代码**：
```cpp
config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);  // 正确的编译阶段
```

**原因**：AutoCast是TensorGraph阶段的Pass，不是ExecuteGraph阶段

**解决方案**：根据Pass所在目录选择正确的编译阶段

---

### 错误6：Function获取错误

**错误代码**：
```cpp
Function* func = Program::GetInstance().GetFunctionByRawName("STCase1");  // 错误的函数名
```

**正确代码**：
```cpp
Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");  // 正确的函数名
```

**原因**：FUNCTION宏会自动添加"TENSOR_"前缀

**解决方案**：使用正确的函数名格式

---

## 测试覆盖类

### 错误7：测试用例覆盖不全

**错误**：只测试了部分场景

**原因**：未全面分析通过当前Pass的业务场景

**解决方案**：
1. 列出当前Pass的所有业务场景
2. 为每个场景设计至少一个ST用例
3. 考虑边界情况和复杂场景

---

### 错误8：未验证Pass效果

**错误代码**：只运行Pass，不验证结果

**正确代码**：
```cpp
auto ret = passManager.RunPass(Program::GetInstance(), *func, "AutoCastTestStrategy");
EXPECT_EQ(ret, SUCCESS);
// 验证Pass效果
uint32_t cast_num = 0;
for (auto &op : func->Operations()) {
    if (op.GetOpcode() == Opcode::OP_CAST) {
        ++cast_num;
    }
}
EXPECT_EQ(cast_num, expected_cast_num);
```

**原因**：需要验证Pass执行后的效果

**解决方案**：遍历操作，验证Pass的效果

---

## 环境配置类

### 错误9：未重置Program和Config

**错误代码**：测试用例间未重置环境

**正确方法**：在SetUp()中重置Program和Config

**原因**：测试用例间可能相互影响

**解决方案**：确保每个测试用例开始前环境是干净的

---

### 错误10：未设置TileShape

**错误代码**：未设置TileShape

**正确代码**：
```cpp
TileShape::Current().SetVecTile({64, 64});
TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
```

**原因**：某些Pass需要TileShape配置

**解决方案**：在SetUp()中设置TileShape

---

## 数据结构类

### 错误11：Tensor使用错误

**错误代码**：
```cpp
Tensor input;  // 未初始化
input = Tensor(DT_FP32, shape, "input");  // 赋值错误
```

**正确代码**：
```cpp
Tensor input(DT_FP32, shape, "input");  // 直接构造
```

**原因**：Tensor类不支持默认构造和赋值

**解决方案**：使用构造函数直接创建Tensor

---

### 错误12：OP宏使用错误

**错误代码**：
```cpp
exp = Exp(input, "exp");  // OP宏不支持命名参数
```

**正确代码**：
```cpp
exp = Exp(input);  // OP宏只接受操作数
```

**原因**：OP宏不支持命名参数

**解决方案**：使用OP宏的正确语法

---

## Pass执行类

### 错误13：Pass策略名错误

**错误代码**：
```cpp
passManager.RunPass(Program::GetInstance(), *func, "AutoCast");  // 错误的策略名
```

**正确代码**：
```cpp
passManager.RunPass(Program::GetInstance(), *func, "AutoCastTestStrategy");  // 正确的策略名
```

**原因**：需要使用注册的策略名，而不是Pass名

**解决方案**：使用正确的策略名

---

### 错误14：未验证返回值

**错误代码**：
```cpp
passManager.RunPass(Program::GetInstance(), *func, "AutoCastTestStrategy");  // 未验证返回值
```

**正确代码**：
```cpp
auto ret = passManager.RunPass(Program::GetInstance(), *func, "AutoCastTestStrategy");
EXPECT_EQ(ret, SUCCESS);  // 验证返回值
```

**原因**：需要验证Pass执行是否成功

**解决方案**：使用EXPECT_EQ验证返回值

---

## 编译相关类

### 错误15：头文件包含错误

**错误代码**：缺少必要的头文件

**原因**：未包含必要的头文件导致编译错误

**解决方案**：参考现有ST测试文件，确保包含所有必要的`头文件`

---

### 错误16：宏定义错误

**错误代码**：
```cpp
#include "tilefwk/tilefwk.h"  // 缺少必要的宏定义
```

**正确代码**：
```cpp
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"  // 包含内部宏定义
```

**原因**：需要包含内部宏定义

**解决方案**：确保包含所有必要的头文件

---

## 验证相关类

### 错误17：只验证操作数不验证内容

**错误代码**：
```cpp
EXPECT_EQ(func->Operations().size(), 3);  // 只验证数量
```

**正确代码**：
```cpp
EXPECT_EQ(func->Operations().size(), 3);
uint32_t cast_num = 0;
for (auto &op : func->Operations()) {
    if (op.GetOpcode() == Opcode::OP_CAST) {
        ++cast_num;
    }
}
EXPECT_EQ(cast_num, 2);  // 验证具体操作数量
```

**原因**：需要验证操作的具体类型

**解决方案**：遍历操作，统计各类型操作的数量

---

### 错误18：未验证数据类型

**错误代码**：只验证操作存在，不验证数据类型

**正确代码**：
```cpp
for (auto &op : func->Operations()) {
    if (op.GetOpcode() == Opcode::OP_CAST) {
        EXPECT_EQ(op.GetIOperands()[0]->GetDatatype(), DataType::DT_BF16);
        EXPECT_EQ(op.GetOOperands()[0]->GetDatatype(), DataType::DT_FP32);
    }
}
```

**原因**：需要验证数据类型转换是否正确

**解决方案**：验证操作的输入输出数据类型

---

### 错误19：ST测试使用UT测试方式

**错误代码**：使用UT测试的方式编写ST测试

**正确代码**：使用FUNCTION宏和Tensor类编写ST测试

**原因**：ST测试需要使用不同的API和宏定义

**解决方案**：参考现有ST测试文件，使用FUNCTION宏和Tensor类

---

### 错误20：只注册当前Pass

**错误代码**：
```cpp
passManager.RegisterStrategy("AutoCastTestStrategy", {
    {"AutoCast", PassName::AUTO_CAST},  // 只注册当前Pass
});
```

**正确代码**：
```cpp
passManager.RegisterStrategy("AutoCastTestStrategy", {
    {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},  // 注册完整Pass链
    {"AutoCast", PassName::AUTO_CAST},
});
```

**原因**：ST测试需要根据用户描述注册完整的Pass链，不仅仅是当前Pass

**解决方案**：参考现有ST测试文件，并根据用户描述，注册完整的Pass链

---

### 错误21：缺少必要的宏定义

**错误代码**：
```cpp
#include "tilefwk/function.h"  // 缺少必要的宏定义
```

**正确代码**：
```cpp
#include "tilefwk/function.h"
#include "test_suite_stest_ops.h"  // 包含ST测试相关的宏定义
```

**原因**：ST测试需要特定的宏定义

**解决方案**：确保包含所有必要的头文件

---

## 总结

### 最容易犯的错误（Top 10）

1. **FUNCTION宏使用错误**：缺少闭合大括号
2. **Tensor构造函数参数错误**：参数顺序或类型不正确
3. **宏定义中使用中文标点**：输入法切换导致
4. **Pass策略注册错误**：只注册当前Pass，未注册完整Pass链
5. **编译阶段配置错误**：使用了错误的COMPILE_STAGE值
6. **Function获取错误**：函数名格式不正确
7. **未验证Pass效果**：只运行Pass，不验证结果
8. **未重置Program和Config**：测试用例间未重置环境
9. **未设置TileShape**：某些Pass需要TileShape配置
10. **Pass策略名错误**：使用了错误的策略名

### 避免错误的最佳实践

1. **确保FUNCTION宏有完整的函数体**：包括闭合大括号
2. **参考Tensor类的构造函数定义**：使用正确的参数
3. **使用英文输入法编写代码**：避免中文标点
4. **注册完整的Pass链**：不仅仅是当前Pass
5. **根据Pass所在目录选择正确的编译阶段**
6. **使用正确的函数名格式**：FUNCTION宏会自动添加"TENSOR_"前缀
7. **验证Pass执行后的效果**：遍历操作，验证结果
8. **在SetUp()中重置环境**：确保每个测试用例开始前环境是干净的
9. **设置TileShape**：在SetUp()中设置TileShape
10. **使用正确的策略名**：使用注册的策略名，而不是Pass名
11. **验证数据类型转换**：不仅验证操作数，还要验证数据类型转换是否正确
12. **ST测试使用UT测试方式**：使用FUNCTION宏和Tensor类编写ST测试
13. **只注册当前Pass**：ST测试需要注册完整的Pass链
14. **使用正确的测试基类**：ST测试需要继承特定的测试基类
15. **包含必要的宏定义**：确保包含所有必要的头文件和宏定义

### 测试设计原则

1. **覆盖所有业务场景**：为每个业务场景设计ST用例
2. **验证Pass效果**：不仅运行Pass，还要验证结果
3. **注册完整的Pass链**：确保Pass能够正常执行
4. **使用正确的编译阶段**：根据Pass所在目录选择
5. **逐步验证**：先验证简单场景，再验证复杂场景
