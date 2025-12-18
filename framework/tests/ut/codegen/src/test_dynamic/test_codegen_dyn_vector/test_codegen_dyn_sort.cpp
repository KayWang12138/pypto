/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_sort.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
constexpr const unsigned OP_MAGIC3 = 3;
constexpr const unsigned OP_MAGIC4 = 4;
class TestCodegenDynSort : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

struct TestContext {
    Function *function;
    std::shared_ptr<LogicalTensor> localTensor;
    std::shared_ptr<LogicalTensor> localOutTensor;
    Operation *op;
};

std::string generateCodeForOp(Operation *op) {
    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(*op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    cop.Init(*op);
    return cop.GenOpCode();
}

TestContext prepareSortParamForUT(Opcode opcode) {
    std::vector<int64_t> shape = {64, 64};

    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string funcName = "ADD";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto localTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC3, dynValidShape});
    auto localOutTensor =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, OP_MAGIC4, dynValidShape});

    auto &op = function->AddOperation(opcode, {localTensor}, {localOutTensor});

    function->GetTensorMap().inverseMap_[localTensor->GetMagic()] = localTensor;
    function->GetTensorMap().inverseMap_[localOutTensor->GetMagic()] = localOutTensor;

    TestContext param;
    param.function = function;
    param.localTensor = localTensor;
    param.localOutTensor = localOutTensor;
    param.op = &op;
    return param;
}

TEST_F(TestCodegenDynSort, TestDynBitSort) {
    auto param = prepareSortParamForUT(Opcode::OP_BITSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynBitSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynMrgSort) {
    auto param = prepareSortParamForUT(Opcode::OP_MRGSORT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "axis", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynExtract) {
    auto param = prepareSortParamForUT(Opcode::OP_EXTRACT);
    param.op->SetAttribute(OP_ATTR_PREFIX + "kvalue", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "mode", 1);
    param.op->SetAttribute(OP_ATTR_PREFIX + "order", 1);
    param.op->SetAttribute("GmTensorParamIdxInCallFunc", 0);

    std::string res = generateCodeForOp(param.op);
    std::string expect =
        R"!!!(TileOp::DynExtract<float, float, 1, 64, 64, 1, 1, 1>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

struct TopKParams {
    int32_t shape0;
    int32_t shape1;
    int32_t k;
    bool isLargest;
};

void TopKOnBoardFunc(TopKParams &params) {
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);

    int32_t shape0 = params.shape0;
    int32_t shape1 = params.shape1;
    int32_t k = params.k;
    bool isLargest = params.isLargest;

    std::vector<int64_t> input_shape = {shape0, shape1};
    std::vector<int64_t> output_shape = {shape0, k};
    TileShape::Current().SetVecTile({shape0, shape1});
    Tensor input_a(DataType::DT_FP32, input_shape, "A");
    auto output = std::make_tuple(
        Tensor(DataType::DT_FP32, output_shape, "npu_val"), Tensor(DataType::DT_FP32, output_shape, "resDics"));
    config::SetBuildStatic(true);
    FUNCTION("TOPK_T_TILETENSOR", {input_a, std::get<0>(output), std::get<1>(output)}) {
        output = TopK(input_a, k, -1, isLargest);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TOPK_T_TILETENSOR");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 9622810300601126008

extern "C" [aicore] void TENSOR_TOPK_T_TILETENSOR_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E16384 = (float __ubuf__ *)get_imm(0x0); // size: 0x4000
float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float __ubuf__ *UB_S16384_E81920 = (float __ubuf__ *)get_imm(0x4000); // size: 0x10000
float *UB_S16384_E81920_T = (float *)get_imm(0x4000); // size: 0x10000
int32_t __ubuf__ *UB_S81920_E98304 = (int32_t __ubuf__ *)get_imm(0x14000); // size: 0x4000
int32_t *UB_S81920_E98304_T = (int32_t *)get_imm(0x14000); // size: 0x4000
float __ubuf__ *UB_S98304_E114688 = (float __ubuf__ *)get_imm(0x18000); // size: 0x4000
float *UB_S98304_E114688_T = (float *)get_imm(0x18000); // size: 0x4000
uint64_t sym_13_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 0);
uint64_t sym_13_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 1);
uint64_t sym_7_dim_0 = 128; //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_7_dim_1 = 32; //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
uint64_t sym_8_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_8_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
using UBTileTensorINT32Dim2_5 = TileTensor<int32_t, StaticLayout2Dim<128, 32, 128, 32>, Hardware::UB>;
using GMTileTensorINT32Dim2_6 = TileTensor<__gm__ int32_t, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_3 = TileTensor<float, StaticLayout2Dim<128, 128, 128, 128>, Hardware::UB>;
using UBTileTensorFP32Dim2_4 = TileTensor<float, StaticLayout2Dim<128, 64, 128, 64>, Hardware::UB>;
using GMTileTensorFP32Dim2_2 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_1 = TileTensor<float, StaticLayout2Dim<128, 32, 128, 32>, Hardware::UB>;
GMTileTensorINT32Dim2_6 gmTensor_11((__gm__ int32_t*)((__gm__ GMTensorInfo*)(param) + 2)->Addr, DynLayout2Dim(Shape2Dim(128, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_1 ubTensor_9((uint64_t)UB_S98304_E114688_T);
UBTileTensorINT32Dim2_5 ubTensor_7((uint64_t)UB_S81920_E98304_T);
GMTileTensorFP32Dim2_2 gmTensor_13((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 1)->Addr, DynLayout2Dim(Shape2Dim(128, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_4 ubTensor_5((uint64_t)UB_S16384_E81920_T);
UBTileTensorFP32Dim2_3 ubTensor_3((uint64_t)UB_S16384_E81920_T);
GMTileTensorFP32Dim2_2 gmTensor_2((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout2Dim(Shape2Dim(128, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_1 ubTensor_1((uint64_t)UB_S0_E16384_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_1, gmTensor_2, Coord2Dim(0, 0));
set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
TBitSort<1, 0, 1>(ubTensor_3, ubTensor_1);
pipe_barrier(PIPE_V);
SUBKERNEL_PHASE2
TMrgSort<1, 32, 1>(ubTensor_5, ubTensor_3);
pipe_barrier(PIPE_V);
TExtract<32, 1, 1>(ubTensor_7, ubTensor_5);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TExtract<32, 0, 1>(ubTensor_9, ubTensor_5);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_11, ubTensor_7, Coord2Dim(0, 0));
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
TStore(gmTensor_13, ubTensor_9, Coord2Dim(0, 0));
}
)!!!";

    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynSort, TestDynTopKTileTensor) {
    TopKParams params;
    params.shape0 = 128;
    params.shape1 = 32;
    params.k = 32;
    params.isLargest = true;
    TopKOnBoardFunc(params);
}

TEST_F(TestCodegenDynSort, TestDynTiledMgrSort) {
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    std::vector<int64_t> shape = {64, 64};
    std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    TileShape::Current().SetVecTile(shape);
    Tensor inputA(DT_FP32, shape, "A");
    Tensor inputB(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");
    
    std::string funcName = "TestDynTiledMgrSort";
    FUNCTION(funcName, {inputA, inputB, output}) {
        output = Add(inputA, inputB);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetUnderDynamicFunction(true);
    auto localTensorInput1 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput2 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorInput3 =
        CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorRes = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});
    auto localTensorTmp = CreateLogicalTensor({*function, DataType::DT_FP32, MemoryType::MEM_UB, shape, dynValidShape});

    auto &op = function->AddOperation(Opcode::OP_TILEDMRGSORT,
        {localTensorInput1, localTensorInput2, localTensorInput3, localTensorInput3}, {localTensorRes, localTensorTmp});
    op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
    op.SetAttribute(OP_ATTR_PREFIX + "validBit", 2);
    op.SetAttribute(OP_ATTR_PREFIX + "kvalue", 3);

    std::shared_ptr<SymbolManager> symbolManager = std::make_shared<SymbolManager>();
    CodeGenCtx ctx;
    CodeGenCloudNPU cga(ctx);
    cga.GenAllocForLocalBuffer(op, symbolManager);
    CodeGenOpCloudNPU cop(symbolManager, FunctionType::DYNAMIC_LOOP_PATH, {}, true);
    function->GetTensorMap().inverseMap_[localTensorInput1->GetMagic()] = localTensorInput1;
    function->GetTensorMap().inverseMap_[localTensorInput2->GetMagic()] = localTensorInput2;
    function->GetTensorMap().inverseMap_[localTensorInput3->GetMagic()] = localTensorInput3;
    function->GetTensorMap().inverseMap_[localTensorRes->GetMagic()] = localTensorRes;
    function->GetTensorMap().inverseMap_[localTensorTmp->GetMagic()] = localTensorTmp;

    cop.Init(op);
    std::string res = cop.GenOpCode();
    std::string expect =
        R"!!!(TileOp::DynTiledMrgSort<float, 1, 1, 64, 64, 1, 1, 64, 64, 64, 3, 2>((__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, (__ubuf__ float*)UB_S0_E0, 1, 1, 64, 64, 64, 64, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk