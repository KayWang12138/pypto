---
name: Pass-Ut-Generate
description: 根据Pass业务描述，生成单元测试用例（UT）。当用户输入业务情况时，能根据业务，生成对应Pass的Ut用例。
license: 完整条款见 LICENSE.txt
---

# Pass 业务单元测试生成

## 概述

本文档描述如何根据 Pass 业务描述生成对应的单元测试用例（UT）。

## 生成流程一

### 步骤 1：环境配置

    观察当前.cpp文件中，是否初始化环境，若已完成初始化，则跳过该步骤。
    初始化环境详细步骤：
        （1）创建新的类，命名xxx.cpp，文件名为test_xxx.cpp，其中xxx为输入的pass名称
        （2）声明该类继承于gtest框架
        （3）该类中，编写相关函数：
                1.所有测试用例全局初始化函数--static void SetUpTestCase() {}，若未明确指定内容，则为空实现
                2.所有测试用例全局清理函数--static void TearDownTestCase() {}， 若未明确指定内容，则为空实现
                3.每个测试用例执行前的初始化测试环境函数--void SetUp() override {}，若未指定内容，则默认生成以下代码体，其中
```cpp
                    void SetUp() override {
                        Program::GetInstance().Reset();
                        config::Reset();
                        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH); //其中 pypto\framework\src\interface\configs\config_manager_ng.h中的COMPILE_STAGE策略，通过pass所在目录，得到所处的编译策略
                        config::SetHostConfig(KEY_STRATEGY, "XXXTestStrategy"); //其中xxx未pass名字
                        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
                        TileShape::Current().SetVecTile({64, 64}); 
                        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
                    }
```
                4.每个测试用例执行后的清理测试环境函数--void TearDown() override {}，若为明确指定内容，则为空实现

### 步骤 2：搭建测试用例框架

    搭建测试用例框架 TEST_F(XXXX, XXX){ }，其中XXXX为上述新建的测试类，XXX为该测试用例名字，可根据根据业务内容生成。
    另外，在TEST_F(XXXX, XXX){ }上方位置处可以添加该测试用例注释，描述经过该pass前后的变化，例如：
 ```cpp
            /*
        TESTRemoveDummyExpand
        inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                            ->sqrt->outCast2{8,16}
                                            ->reciprocal->outCast3{8,16}
        inCast{8,16}->exp->outCast1
                    ->sqrt->outCast2
                    ->reciprocal->outCast3
        */
        TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
            ...
        }
```

### 步骤 3：构建function

    构建整张计算图function，利用智能指针进行创建，并在创建后判断是否为空。若用户未明确指定参数，则belongTo为新建Program实例，funcMagicName和funcRawName均为TestXXX，XXX为pass名字，parentFunc为空指针。
    
    详细的信息如下：
        Function类常用构造函数：
```cpp
        Function(const Program &belongTo, const std::string &funcMagicName, const std::string &funcRawName,
            Function *parentFunc);
```
        Program常用获取实例：
```cpp
        Program &Program::GetInstance() {
            static Program sProgram;
            return sProgram;
        }
```
        详细代码可以参考：pypto\framework\src\interface\program\program.cpp和pypto\framework\src\interface\function\function.h

### 步骤 4：创建Tensor

    构建计算图中的Tensor，利用智能指针，根据业务需求，创建所需要的Tensor。

    LogicalTensor类常用构造函数：
        LogicalTensor(Function &function, DataType t, Shape tshape, TileOpFormat tformat = TileOpFormat::TILEOP_ND, std::string tname = "",
        NodeType tnodetype = NodeType::LOCAL);
    
    详细LogicalTensor类信息，参考pypto\framework\src\interface\tensor\logical_tensor.h


### 步骤 5：创建Operation及绑定function输入输出

    构建计算图中的Operation，利用智能指针，根据业务需求，创建所需要的Operation。

    常用的创建Operation及绑定function函数：
```cpp
            Operation &Program::AddOperation(const Opcode opCode,
        const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
        const std::vector<std::shared_ptr<LogicalTensor>> &oOperand) {
        // Add the operation to the current function
        if (currentFunctionMagicName_ == PROGRAM_ENTRY_FUNCTION_NAME) {
            FUNCTION_LOGE("Error: No active function to add operation.");
            ASSERT(false) << "No active function to add operation.";
        }
        return currentFunctionPtr_->AddOperation(opCode, iOperand, oOperand);
    }
```
    详细创建Operation及绑定function函数信息，参考pypto\framework\src\interface\program\program.cpp

    Opcode信息，请参考：pypto\framework\src\interface\operation\opcode.h和pypto\framework\src\interface\operation\opcode.cpp

### 步骤 6：对业务功能进行校验

    根据业务功能，校验pass运行后的处理结果是否符合预期。

    例如校验function经过pass后，assemble数量是否符合预期，可以通过遍历function，查找assemble的数量进行比对，具体代码如下：
```cpp
        uint32_t assemble_num = kNumZero;
        for (auto &op : currFunctionPtr->Operations()) {
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                ++assemble_num;
            }
        }
        EXPECT_EQ(assemble_num, kNumZero);
```

### 步骤 7：利用现有的环境，执行生成的UT，并对错误进行修改

    当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确，xxx为该测试用例类的名字，对于错误进行改正。
    例如：python3 build_ci.py -c -u=TestRemoveRedundantOpPass.* -j=24 -f=cpp
    
## 生成流程二

### 步骤 1：环境配置

    观察当前.cpp文件中，是否初始化环境，若已完成初始化，则跳过该步骤。
    初始化环境详细步骤：
        （1）创建新的类，命名xxx.cpp，文件名为test_xxx.cpp，其中xxx为输入的pass名称
        （2）声明该类继承于gtest框架
        （3）该类中，编写相关函数：
                1.所有测试用例全局初始化函数--static void SetUpTestCase() {}，若未明确指定内容，则为空实现
                2.所有测试用例全局清理函数--static void TearDownTestCase() {}， 若未明确指定内容，则为空实现
                3.每个测试用例执行前的初始化测试环境函数--void SetUp() override {}，若未指定内容，则默认生成以下代码体，其中
```cpp
                    void SetUp() override {
                        Program::GetInstance().Reset();
                        config::Reset();
                        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH); //其中 pypto\framework\src\interface\configs\config_manager_ng.h中的COMPILE_STAGE策略，通过pass所在目录，得到所处的编译策略
                        config::SetHostConfig(KEY_STRATEGY, "XXXTestStrategy"); //其中xxx未pass名字
                        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
                        TileShape::Current().SetVecTile({64, 64}); 
                        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
                    }
```
                4.每个测试用例执行后的清理测试环境函数--void TearDown() override {}，若为明确指定内容，则为空实现

### 步骤 2：搭建测试用例框架

    搭建测试用例框架 TEST_F(XXXX, XXX){ }，其中XXXX为上述新建的测试类，XXX为该测试用例名字，可根据根据业务内容生成。
    另外，在TEST_F(XXXX, XXX){ }上方位置处可以添加该测试用例注释，描述经过该pass前后的变化，例如：
```cpp
            /*
        TESTRemoveDummyExpand
        inCast{8,16}->expand->ubTensor{8,16}->exp->outCast1{8,16}
                                            ->sqrt->outCast2{8,16}
                                            ->reciprocal->outCast3{8,16}
        inCast{8,16}->exp->outCast1
                    ->sqrt->outCast2
                    ->reciprocal->outCast3
        */
        TEST_F(TestRemoveRedundantOpPass, RemoveRedundantOpUTest1) {
            ...
        }
```
### 步骤 3：构建function

    利用ComputationalGraphBuilder类来构建function，通过调用AddTensor()和AddTensors()来实现function中Tensor的构建，调用AddOp()和AddOps()来实现function中Op的构建。
    通过调用SetInCast()和SetOutCast()来实现对function的输入输出构建。

    ComputationalGraphBuilder类信息，请参考：pypto\framework\tests\ut\passes\src\computational_graph_builder.h
    Opcode信息，请参考：pypto\framework\src\interface\operation\opcode.h和pypto\framework\src\interface\operation\opcode.cpp
    Operation信息，请参考: pypto\framework/src/interface/operation/operation.cpp
    详细LogicalTensor类信息，参考pypto\framework\src\interface\tensor\logical_tensor.h
    Tensor创建过程中DataType信息，请参考：pypto\framework\include\tilefwk\data_type.h

### 步骤 4：对业务功能进行校验

    根据业务功能，校验pass运行后的处理结果是否符合预期。

    例如校验function经过pass后，Redunce_Acc数量是否符合预期，可以通过遍历function，查找Redunce_Acc的数量进行比对，具体代码如下：
```cpp
        opReduceAccCount = 0;
        for(auto &op : function->Operations()) {
            if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
                opReduceAccCount++;
            }
        }
        EXPECT_EQ(opReduceAccCount, 0);
```
### 步骤 5：利用现有的环境，执行生成的UT，并对错误进行修改

    当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确，xxx为该测试用例类的名字，对于错误进行改正。
    例如：python3 build_ci.py -c -u=TestRemoveRedundantOpPass.* -j=24 -f=cpp

## 注意事项
    要符合代码编程规范，例如：未使用的变量定义、未修改的引用加入const、魔鬼数字等
    生成的用例请真实执行，并对用例进行修改验证。

## 参考资料

| 类别 | 文件路径 |
|------|----------|
| 生成流程一示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |
| 生成流程二示例 | `pypto/framework/tests/ut/passes/src/test_cube_process.cpp` |
| ComputationalGraphBuilder | `pypto/framework/tests/ut/passes/src/computational_graph_builder.h` |
| function信息 | `pypto\framework\src\interface\program\program.cpp和pypto\framework\src\interface\function\function.h`|
| Opcode 定义 | `pypto/framework/src/interface/operation/opcode.h` |
| op属性定义 | `framework/src/interface/operation/attribute.h` |
| DataType 定义 | `pypto/framework/include/tilefwk/data_type.h` |
| LogicalTensor 定义 | `pypto\framework\src\interface\tensor\logical_tensor.h` |
| Operation信息 | `pypto\framework/src/interface/operation/operation.cpp`  |
| COMPILE_STAGE策略 | `pypto\framework\src\interface\configs\config_manager_ng.h` |

---

## 编译阶段参考值

| 阶段 | 配置值 |
|------|--------|
| TENSOR GRAPH 执行 | `CS_TENSOR_GRAPH` |
| TILE GRAPH 执行 | `CS_TILE_GRAPH` |
| BLOCK GRAPH 执行 | `CS_EXECUTE_GRAPH` |

根据 Pass 所在文件夹目录选择对应的编译策略。


