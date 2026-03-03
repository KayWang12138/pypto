---
name: Pass-function
description: 根据用户描述的Pass名称，阅读相关逻辑代码，输出总结Pass业务和相关的代码和业务意见的.md文档。
license: 完整条款见 LICENSE.txt
---

## 概述

根据用户所提出的Pass，总结该Pass业务，并针对该Pass存在的代码问题及业务问题提出建议。

## 方法

### 步骤 1 ：寻找对应的Pass文件
    
根据用户描述的Pass名称，寻找`pypto/framework/src/passes`下对应的Pass文件。

### 步骤 2：阅读Pass文件代码逻辑

根据对应的Pass文件，阅读Pass文件代码逻辑。主要包含三方面内容：
    1.Precheck：对输入的function进行预校验，满足该Pass需求；
    2.RunOnfunction：该Pass的核心处理逻辑，包含了该Pass的核心功能；
    3.PostCheck：对于处理后function进行自校验，是否满足了预期功能。

### 步骤 3：总结Pass业务及相关建议

通过阅读代码逻辑，总结出该Pass对应业务，并提出以下建议：
    业务建议：针对业务，是否存在可以优化点，例如：考虑性能、功能等方面，提出优化点；
    代码建议：针对代码，是否存在代码质量问题，是否可以改进代码结构，提出改进点。
最终输出文档xxx_Business.md，格式参考 `pass_function/template_format.md`

## Pass相关基础知识

### 1. 总体描述

当前端用例完成对operation及tensor的定义后，框架会针对tensor及相关操作，生成由Tensor合Operation交错连接的图结构，形成计算图，即function结构。计算图经过Pypto编译优化流程，完成从原始计算图到可执行图的编译过程，最终生成可在昇腾硬件环境中可运行的可执行代码，以实现实际的计算任务。
Pass负责对计算图进行一系列的优化编译，生成优化后的、能在硬件上运行的计算图。通过编译Pass将Tensor Graph转换为Tile Graph、Block Graph和Execution Graph，每一步包括一系列Pass优化流程。
Pass中存在以下重要组件：
    Pass基类，定义了Pass的基础功能，支持dump function及 dump graph功能；
    所有Pass继承Pass基类，实现各自的功能；
    支持用户注册功能Pass，Pass Registrar 统一管理功能Pass；
    Pass Manager 负责提供对外调用接口，管理Pass执行顺序及整体的Pass DFX能力。

### 2. Pass基类

Pass基类提供统一对外调用接口，包括执行接口、自校验接口，DFX调试能力。
Pass主流程接口：
    Run(Function): Pass统一对外调用接口
    RunOnFunction(Function &function)：Pass核心处理逻辑接口，子类需要实现该接口
Pass自校验接口：
    PreCheck(Function &function)：校验Pass输入是否符合预期
    PostCheck(Function &function)：校验Pass输出是否符合预期
详细信息，请参考: `pypto/framework/src/passes/pass_interface/pass.h`

### 3. Pass Registrar

Pass Registrar提供Pass注册接口，统一管理Pass。Pass Registrar使用单例模式，保存注册Pass，并且提供支持用户通过Pass名称获取对应的Pass对象。
Pass Registrar提供宏包装的Pass注册接口，简化用户调用接口
详细信息，请参考: `pypto/framework/src/passes/pass_mgr/pass_registry.h`

### 4. Pass Manager

Pass Manager是Pass模块对外提供接口。Pass Manager支持配置Pass执行顺序。Pass Manager按照配置的Pass顺序依次执行Pass，并根据Pass的DFX选项，执行Pass的DFX功能。
Pass DFX功能通过json文件进行配置，Pass Manager负责解析json配置文件，并将每个json的DFX配置设置到对应的Pass，执行Pass的DFX能力。
详细信息，请参考: `pypto/framework/src/passes/pass_mgr/pass_manager.h`

### 5. Pass执行顺序及对应的功能概略

Pass执行顺序参考`pypto/framework/src/passes/pass_mgr/pass_manager.cpp`中的void PassManager::RegDefaultStrategy(){}注册策略。

整体来说计算图进入Pass时为Tensor Graph阶段，到ExpandFunction时，对其中的大Tensor等进行了tile分块操作，转换为Tile Graph阶段。SubgraphToFunction将计算图进行切分子图，此时转换为Block Graph阶段。

Tensor Graph阶段：RemoveRedundantReshape、AutoCast、InferMemoryConflict、RemoveUndrivenView主要为计算逻辑推导、冗余节点删除、cast类型转换、地址冲突等；ExpandFunction根据前端设置的tile大小，对大Tensor进行切分，并将对应的op分解为tileop，并进入Tile Graph阶段。

Tile Graph阶段：MergeViewAssemble、SplitReshape、SplitRawTensor、SplitLargeFanoutTensor、DuplicateOp、AssignMemoryType、InferDiscontinuousInput、RemoveRedundantOp、SplitK等主要对计算图进行化简或者改图，消除冗余节点、拆分数据依赖；GraphPartition、NBufferMerge、L1CopyInReuseMerge、CommonOperationEliminate等主要针对计算图进行切图和合图，标记好每个子图；IntraSubgraphAdapter、GenerateMoveOp、PadLocalBuffer、RemoveUnalignedReshape等对于子图边界的view/assemble以及convert op进行翻译；ReplaceTensor、PreGraphProcess、InferDynShape主要为根据指令约束，进行op约束处理。

Block Graph阶段：SubgraphToFunction完成计算图的切分；InferParamIndex进行vaildshape的参数化推导；GlobalMemoryReuse、SrcDstBufferMerge进行内存复用；InsertSync、OoOSchedule等为自动调度功能。

