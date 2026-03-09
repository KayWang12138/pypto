---
name: pass-st-generate
description: 根据Pass业务描述，生成集成测试用例（ST）。当用户输入业务情况时，能根据业务，生成对应Pass的ST用例。
license: 完整条款见 LICENSE.txt
---

# Pass 业务单元测试生成

## 概述

本技能用于分析如何根据用户描述生成对应的Pass侧集成测试用例，结合pypto-pass-module-analyzer/SKILL.md技能分析Pass业务，帮助设计相关集成测试用例（ST）。

## 触发机制
 	 
 	当用户输入包含以下关键字或者相关内容时，自动触发此技能：
 	 
 	 （1）设计Pass模块XXX的ST用例**，注册策略为**：设计指定 Pass 模块的功能的ST用例，并说明注册策略
 	 （2）设计Pass模块XXX的XXX功能**，注册策略为**：设计指定 Pass 模块的指定功能的ST用例，并说明注册策略
 	 （3）设计XXX功能的相关Pass模块的ST用例**，注册策略为**：设计与XXX功能相关Pass的ST用例，并说明注册策略
 
 	触发示例：

 	 （1）"设计Pass模块AutoCast的ST用例，注册策略为RemoveRedundantReshape与Autocast"
 	 （2）"设计Pass模块AutoCast的对于不支持BF16 OP插入Cast的功能的ST用例，注册策略为RemoveRedundantReshape与Autocast"
 	 （3）"设计删除冗余Op功能的相关Pass的ST用例，注册策略为RemoveRedundantReshape与Autocast"

## 使用场景
 	 
 	当需要设计 PyPTO pass 中模块的功能或业务集成测试用例时使用此技能。

## ST生成流程

### 步骤 1：分析业务

    根据用户描述的业务情况，分析相关业务：当用户描述为具体Pass的具体业务时，根据@.opencode/skills/pypto-pass/pypto-pass-module-analyzer/SKILL.md，分析业务场景与注册策略中Pass之间的功能与联系，并根据当前业务和用户指定的注册策略，设计相应的ST用例，例如：设计Splitk这个Pass消除RedunceAcc功能的ST，注册策略为RemoveRedundantOp与SplitK。
    
    注：用户需要显示的指定注册策略，否则默认只注册当前设计用例的Pass。

### 步骤 2：环境配置

    在pypto/framework/tests/ut/passes/src/test_xxx.cpp寻找对应的测试文件，观察当前文件中，是否初始化环境，若已完成初始化，则跳过该步骤。其中xxx一般为Pass名称，
    当未找到对应的测试文件时，可以查找test_xxx.cpp中创建类的名字是否与所要求设计Pass名字是否一致。
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
                        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH); //其中 pypto/framework/src/interface/configs/config_manager_ng.h中的COMPILE_STAGE策略，通过pass所在目录，得到所处的编译策略
                        config::SetHostConfig(KEY_STRATEGY, "XXXTestStrategy"); //其中xxx未pass名字
                        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
                        TileShape::Current().SetVecTile({64, 64}); 
                        TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
                    }
```
                4.每个测试用例执行后的清理测试环境函数--void TearDown() override {}，若为明确指定内容，则为空实现

### 步骤 3：搭建测试用例框架

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

### 步骤 4：构建Tensor

    利用Tensor类来构建整张计算图所需要的Tensor。
    Tensor类常用的构造函数：
```cpp
    Tensor(DataType dataType, const Shape &shape, std::string name = "", TileOpFormat format = TileOpFormat::TILEOP_ND);
```
    例如：
```cpp
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, shape, "view");
    Tensor expand(DT_FP32, shape, "expand");
    Tensor output1(DT_FP32, shape, "output1");
    Tensor output2(DT_FP32, shape, "output2");
```
`
    详细Tensor类信息，请参考: `pypto/framework/include/tilefwk/tensor.h`
    Tensor创建过程中DataType信息，请参考：`pypto/framework/include/tilefwk/data_type.h`

### 步骤 5：构建Fuction

    使用宏定义FUNCTION方式，构建计算图。
    在FUNCTION宏定义中，声明需要的OP，并绑定上所连接的TENSOR信息。
    常用的FUNCTION宏定义使用：
    FUNCTION(name) {
        变量 1 = OP(TENSOR1, ...);
        变量 2 = OP(TENSOR1, ...);
    }
    例如:
```cpp
    FUNCTION("STCase1") {
        exp = Exp(input);
        view = View(exp, shape, {kNumZero, kNumZero});
        expand = Expand(view, shape);
        output1 = Exp(expand);
        output2 = Exp(expand);
    }
```
    通过Program实例中的通过计算图名字获取function：
        Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR__xxx"); //其中xxx为上述宏FUNCTION定义的名字

    详细FUNCTION信息、RecordFunc信息，请参考：pypto/framework/include/tilefwk/function.h
    详细OP信息，请参考：pypto/framework/src/interface/interpreter/calc.h
    详细Program信息，请参考：pypto/framework/src/interface/program/program.cpp
    

### 步骤 6：注册执行策略

    使用passManager.RegisterStrategy（）进行执行策略的注册，通过passManager.RunPass（）按照注册顺序执行相关Pass
    例如：
    ```cpp
    passManager.RegisterStrategy("AutoCastTestStrategy", {
        {"AssignMemoryType", PassName::ASSIGN_MEMORY_TYPE},
        {"AutoCast", PassName::AUTO_CAST},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "AutoCastTestStrategy");
    ```
    详细pass_manager信息，请参考：pypto/framework/src/passes/pass_mgr/pass_manager.h
    用户需要显示的指定注册策略，否则默认只注册当前pass。
    
### 步骤 7：对业务功能进行校验

    根据业务功能，校验pass运行后的处理结果是否符合预期。

    例如校验function经过pass后，Cast数量是否符合预期，可以通过遍历function，查找Cast的数量进行比对，具体代码如下：
    ```cpp
        opCastCount = 0;
        for(auto &op : function->Operations()) {
            if(op.GetOpcode() == Opcode::OP_CAST) {
                opCastCount++;
            }
        }
        EXPECT_EQ(opCastCount, expected_cast_num);
    ```

### 步骤 8：利用现有的环境，执行生成的UT，并对错误进行修改(若当前环境正常，需在当前环境中验证)

    当生成UT用例后，执行Python3 build_ci.py -c -u=xxx.* -j=24来验证用例是否正确，xxx为该测试用例类的名字，对于错误进行改正。
    当执行超时时，优先执行Python3 build_ci.py -u=xxx.* -j=24。
    若此时还存在执行超时问题，则执行Python3 build_ci.py -u=xxx.xx -j=24，xx为测试用例
    例如: python3 build_ci.py -c -u=TestAutoCast.* -j=24 -f=cpp
          python3 build_ci.py -u=TestAutoCast.* -j=24 -f=cpp
          python3 build_ci.py -c -u=TestAutoCast.TestInsertBF16CastForExp -j=24 -f=cpp

## 注意事项

    要符合常见的CPP代码编程规范，例如常见的编程错误：未使用的变量定义、未修改的引用加入const、魔鬼数字等；
    生成的用例请真实执行，并对生成的用例进行验证修改；
    设计ST用例时，请参考[common_errors.md](./common_errors.md)中的最容易犯的错误（Top 10）与避免错误的最佳实践内容，避免重复犯错。

## 常见错误记录

    详细的常见错误记录请参考：[common_errors.md](./common_errors.md)

    该文档包含以下错误类别：
    - 语法错误类
    - 逻辑错误类
    - 测试覆盖类
    - 环境配置类
    - 数据结构类
    - Pass执行类
    - 编译相关类
    - 验证相关类

## 参考资料

| 类别 | 文件路径 |
|------|----------|
| 生成流程示例 | `pypto/framework/tests/ut/passes/src/test_removeredundantop.cpp` |
| FUNCTION信息 | `pypto/framework/include/tilefwk/function.h`|
| OP信息 | `pypto/framework/src/interface/interpreter/calc.h` |
| Tensor信息 | `pypto/framework/include/tilefwk/tensor.h` |
| DataType 定义 | `pypto/framework/include/tilefwk/data_type.h` |
| Pass Manager信息 | `pypto/framework/src/passes/pass_mgr/pass_manager.h` |
| COMPILE_STAGE策略 | `pypto/framework/src/interface/configs/config_manager_ng.h` |
| Program信息 | `pypto/framework/src/interface/program/program.cpp` |

## 编译阶段参考值

| 阶段 | 配置值 |
|------|--------|
| TENSOR GRAPH 执行 | `CS_TENSOR_GRAPH` |
| TILE GRAPH 执行 | `CS_TILE_GRAPH` |
| BLOCK GRAPH 执行 | `CS_EXECUTE_GRAPH` |

根据 Pass 所在文件夹目录选择对应的编译策略。


