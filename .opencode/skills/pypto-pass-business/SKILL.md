---
name: Pass-function
description: 根据用户描述的Pass名称，阅读相关逻辑代码，输出总结Pass业务和相关的代码和业务意见的.md文档。
license: 完整条款见 LICENSE.txt
---

## 概述

根据用户所提出的Pass，总结该Pass业务，并针对该Pass存在的代码问题及业务问题提出建议。

## 方法

### 步骤 1 ：寻找对应的Pass文件
    
    根据用户描述的Pass名称，寻找pypto/framework/src/passes下对应的Pass文件。

### 步骤 2：阅读Pass文件代码逻辑

    根据对应的Pass文件，阅读Pass文件代码逻辑。

### 步骤 3：总结Pass业务及相关建议

    通过阅读代码逻辑，总结出该Pass对应业务，针对业务，提出是否存在优化点；针对代码，提出改进点。
    输出文档xxx_Business.md

## .md格式

    <!-- 请参考pypto/pass_function/geshi.md -->

## Pass相关基础知识

### 1. 总体描述

    Pass基类（pypto/framework/src/passes/pass_interface/pass.h），定义了Pass的基础功能，支持dump function及 dump graph功能；
    所有Pass继承Pass基类，实现各自的功能；
    支持用户注册功能Pass，Pass Registrar （pypto/framework/src/passes/pass_mgr/pass_registry.h）统一管理功能Pass；
    Pass Manager（pypto/framework/src/passes/pass_mgr/pass_manager.h） 负责提供对外调用接口，管理Pass执行顺序及整体的Pass DFX能力。

### 2. Pass基类

    Pass基类提供统一对外调用接口，包括执行接口、自校验接口，DFX调试能力。
    Pass主流程接口：
        Run(Function): Pass统一对外调用接口
        RunOnFunction(Function &function)：Pass核心处理逻辑接口，子类需要实现该接口
    Pass自校验接口：
        PreCheck(Function &function)：校验Pass输入是否符合预期
        PostCheck(Function &function)：校验Pass输出是否符合预期
        
### 3. Pass Registrar

    Pass Registrar提供Pass注册接口，统一管理Pass。Pass Registrar使用单例模式，保存注册Pass，并且提供支持用户通过Pass名称获取对应的Pass对象
    Pass Registrar提供宏包装的Pass注册接口，简化用户调用接口

### 4. Pass Manager

    Pass Manager是Pass模块对外提供接口。Pass Manager支持配置Pass执行顺序。Pass Manager按照配置的Pass顺序依次执行Pass，并根据Pass的DFX选项，执行Pass的DFX功能。
    Pass DFX功能通过json文件进行配置，Pass Manager负责解析json配置文件，并将每个json的DFX配置设置到对应的Pass，执行Pass的DFX能力。

### 5. Pass执行顺序及对应的功能概略

```cpp
void PassManager::RegDefaultStrategy() {
    RegisterStrategy(
        "PVC2_OOO", {
            {   "RemoveRedundantReshape",      PassName::REMOVE_REDUNDANT_RESHAPE},
            {                 "AutoCast",                     PassName::AUTO_CAST},
            {      "InferMemoryConflict",         PassName::INFER_MEMORY_CONFLICT},
            {       "RemoveUndrivenView",          PassName::REMOVE_UNDRIVEN_VIEW},
            {           "ExpandFunction",               PassName::EXPAND_FUNCTION},
            {        "MergeViewAssemble",           PassName::MERGE_VIEW_ASSEMBLE},
            {             "SplitReshape",                 PassName::SPLIT_RESHAPE},
            {           "SplitRawTensor",              PassName::SPLIT_RAW_TENSOR},
            {   "SplitLargeFanoutTensor",     PassName::SPLIT_LARGE_FANOUT_TENSOR},
            {              "DuplicateOp",                  PassName::DUPLICATE_OP},
            {         "AssignMemoryType",            PassName::ASSIGN_MEMORY_TYPE},
            {  "InferDiscontinuousInput",     PassName::INFER_DISCONTINUOUS_INPUT},
            {        "RemoveRedundantOp",           PassName::REMOVE_REDUNDANT_OP},
            {  "InsertOpForViewAssemble",    PassName::INSERT_OP_FOR_VIEWASSEMBLE},
            {                   "SplitK",                       PassName::SPLIT_K},
            {           "GraphPartition",               PassName::GRAPH_PARTITION},
            {          "ReduceCopyMerge",             PassName::REDUCE_COPY_MERGE},
            {             "NBufferMerge",                PassName::N_BUFFER_MERGE},
            {       "L1CopyInReuseMerge",        PassName::L1_COPY_IN_REUSE_MERGE},
            {     "IntraSubgraphAdapter",        PassName::INTRA_SUBGRAPH_ADAPTER},
            {           "GenerateMoveOp",              PassName::GENERATE_MOVE_OP},
            { "CommonOperationEliminate",    PassName::COMMON_OPERATION_ELIMINATE},
            {              "AxisCombine",                  PassName::AXIS_COMBINE},
            {           "PadLocalBuffer",              PassName::PAD_LOCAL_BUFFER},
            {   "RemoveUnalignedReshape",      PassName::REMOVE_UNALIGNED_RESHAPE},
            {            "ReplaceTensor",                PassName::REPLACE_TENSOR},
            {          "PreGraphProcess",             PassName::PRE_GRAPH_PROCESS},
            {            "InferDynShape",               PassName::INFER_DYN_SHAPE},
            {       "SubgraphToFunction",          PassName::SUBGRAPH_TO_FUNCTION},
            {          "InferParamIndex",             PassName::INFER_PARAM_INDEX},
            {        "SrcDstBufferMerge",          PassName::SRC_DST_BUFFER_MERGE},
            {                 "AddAlloc",                     PassName::ADD_ALLOC},
            {              "OoOSchedule",                  PassName::OOO_SCHEDULE},
            {        "GlobalMemoryReuse",           PassName::GLOBAL_MEMORY_REUSE},
            {              "RemoveAlloc",                  PassName::REMOVE_ALLOC},
            {           "CopyOutResolve",              PassName::COPY_OUT_RESOLVE},
            {               "InsertSync",                   PassName::INSERT_SYNC},
            {         "MixSubgraphSplit",            PassName::MIX_SUBGRAPH_SPLIT},
            {             "LoopaxesProc",             PassName::LOOPAXES_PROC},
            {           "CodegenPreproc",               PassName::CODEGEN_PREPROC},
    });
    RegisterStrategy(
        "FunctionUnroll", {
            {               "LoopUnroll",                   PassName::LOOP_UNROLL}
    });
    RegisterStrategy(
        "ExecuteGraph", {
            {          "DynAttrToStatic",            PassName::DYN_ATTR_TO_STATIC},
    });
}
```

当完成对operation及tensor的定义后，框架会针对tensor及相关操作，形成计算图，即function结构。Pass负责对计算图进行一系列的优化编译，生成优化后的、能在硬件上运行的计算图。
整体来说计算图进入Pass时为Tensor Graph阶段，到ExpandFunction时，对其中的大Tensor等进行了tile分块操作，转换为Tile Graph阶段。SubgraphToFunction将计算图进行切分子图，此时转换为Block Graph阶段。
Tensor Graph阶段：RemoveRedundantReshape、AutoCast、InferMemoryConflict、RemoveUndrivenView主要为计算逻辑推导、冗余节点删除、cast类型转换、地址冲突等；ExpandFunction根据前端设置的tile大小，对大Tensor进行切分，并将对应的op分解为tileop，并进入Tile Graph阶段。
Tile Graph阶段：MergeViewAssemble、SplitReshape、SplitRawTensor、SplitLargeFanoutTensor、DuplicateOp、AssignMemoryType、InferDiscontinuousInput、RemoveRedundantOp、SplitK等主要对计算图进行化简或者改图，消除冗余节点、拆分数据依赖；GraphPartition、NBufferMerge、L1CopyInReuseMerge、CommonOperationEliminate等主要针对计算图进行切图和合图，标记好每个子图；IntraSubgraphAdapter、GenerateMoveOp、PadLocalBuffer、RemoveUnalignedReshape等对于子图边界的view/assemble以及convert op进行翻译；ReplaceTensor、PreGraphProcess、InferDynShape主要为根据指令约束，进行op约束处理。
Block Graph阶段：SubgraphToFunction完成计算图的切分；InferParamIndex进行vaildshape的参数化推导；GlobalMemoryReuse、SrcDstBufferMerge进行内存复用；InsertSync、OoOSchedule等为自动调度功能。

