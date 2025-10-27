/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_binary.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynBinary : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

void TestAddDynBody(const std::vector<int64_t> &shape, const std::vector<int64_t> &tile_shape, const std::string &name,
    bool isNeedCalcMinForBinaryOperands = false) {
    TileShape::Current().SetVecTile(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, input_b, output}) {
        output = Add(input_a, input_b);
    }

    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_" + name);

    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
            if (op.GetOpcode() == Opcode::OP_ADD && isNeedCalcMinForBinaryOperands) {
                op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
            }
        }
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar()};
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBinary, TestCodegenAddDim2) {
    TestAddDynBody({64, 64}, {64, 64}, "ADD");
}

TEST_F(TestCodegenDynBinary, TestCodegenAddDim2SrcNotSameShape) {
    TestAddDynBody({64, 64}, {64, 64}, "ADD", true);
}

TEST_F(TestCodegenDynBinary, TestAddsDynamic) {
    std::vector<int64_t> shape = {64, 64};
    TileShape::Current().SetVecTile({64, 64});
    Tensor input_a(DataType::DT_FP32, shape, "A");
    Element value(DataType::DT_FP32, 1.5);
    Tensor output(DataType::DT_FP32, shape, "C");
    ConfigManager::Instance();
    config::SetBuildStatic(true);
    FUNCTION("ADD_S", {input_a, output}) {
        output = Add(input_a, value);
    }

    auto function = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD_S");
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
        DynParamInfo fakeParam = {3, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar()};
        subFunc.second->dynParamTable_.emplace("sym_2_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_2_dim_1", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_0", fakeParam);
        subFunc.second->dynParamTable_.emplace("sym_4_dim_1", fakeParam);
    }

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBinary, TestGatherEle) {
    constexpr const int32_t nRoutedExperts = 256;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({16, 32});
    Tensor inputScores(DT_FP32, outputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    std::string funcName = "GATHER_ELEMET_T";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor}) {
        outputTensor = GatherElements(inputTmpScores, inputScores, 1); // [b*s,8]
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    function->SetUnderDynamicFunction(true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    for (auto &subFunc : function->rootFunc_->programs_) {
        for (auto &op : subFunc.second->Operations()) {
            if (OpcodeManager::Inst().IsCopyIn(op.GetOpcode()) || OpcodeManager::Inst().IsCopyOut(op.GetOpcode())) {
                if (IsCopyIn(op.GetOpcode()))
                    op.SetIOpAttrOffset(0, 0);
                else
                    op.SetOOpAttrOffset(0, 0);
                op.SetAttribute("GmTensorParamIdxInCallFunc", 0);
            }
        }
        DynParamInfo fakeParam = {2, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar()};
        subFunc.second->InsertDynParam("sym_2_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_2_dim_1", fakeParam);
        subFunc.second->InsertDynParam("sym_4_dim_0", fakeParam);
        subFunc.second->InsertDynParam("sym_4_dim_1", fakeParam);
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBinary, AddUnalignTileTensor) {
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(64, 64);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    // NEXTNEXT: delete after tileop adapted layout mode
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> inputShape = {b * sq, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor input1(DT_FP32, inputShape, "intput1");
    Tensor input2(DT_FP32, inputShape, "intput2");
    Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    std::string loopName = "L0";

    FUNCTION("main", {input1, input2, curSeq}, {out}) {
        LOOP(loopName, FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            auto seq = GetTensorData(curSeq, {batchId, 0});
            Tensor intput11 = View(input1, {sq, d}, {seq, d}, {batchId, 0});
            Tensor intput22 = View(input2, {sq, d}, {seq, d}, {batchId, 0});
            auto tmp = Add(intput11, intput22);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    std::vector<int> actSeqsData(b, 100);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input1, 1.0),
        RawTensorData::CreateConstantTensor<float>(input2, 1.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + loopName + SUB_FUNC_SUFFIX);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});

    std::string res = GetResultFromCpp(*function);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 13526864639772037405

extern "C" [aicore] void TENSOR_L0_Unroll1_PATH0_3_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E16384 = (float __ubuf__ *)get_imm(0x0); // size: 0x4000
float *UB_S0_E16384_T = (float *)get_imm(0x0); // size: 0x4000
float __ubuf__ *UB_S16384_E32768 = (float __ubuf__ *)get_imm(0x4000); // size: 0x4000
float *UB_S16384_E32768_T = (float *)get_imm(0x4000); // size: 0x4000
uint64_t sym_18_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 0);
uint64_t sym_18_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 10, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 1, 10, 2, 1);
uint64_t sym_19_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 0);
uint64_t sym_19_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(2, 1, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 0, 1, 2, 1);
uint64_t sym_7_dim_0 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 0);
uint64_t sym_7_dim_1 = GET_PARAM_VALID_SHAPE_BY_IDX(param, 2, 19, 2, 1);
using GMTileTensorFP32Dim2_1 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_0 = TileTensor<float, LocalLayout2Dim<64, 64>, Hardware::UB>;
GMTileTensorFP32Dim2_1 gmTensor_7((__gm__ float*)GET_PARAM_ADDR(param, 2, 19), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 2, 19)), Stride2Dim(GET_PARAM_STRIDE_2(param, 2, 19))));
GMTileTensorFP32Dim2_1 gmTensor_3((__gm__ float*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1))));
UBTileTensorFP32Dim2_0 ubTensor_2((float*)UB_S16384_E32768_T, (Shape2Dim(sym_19_dim_0, sym_19_dim_1)));
GMTileTensorFP32Dim2_1 gmTensor_1((__gm__ float*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 1, 10)), Stride2Dim(GET_PARAM_STRIDE_2(param, 1, 10))));
UBTileTensorFP32Dim2_0 ubTensor_0((float*)UB_S0_E16384_T, (Shape2Dim(sym_18_dim_0, sym_18_dim_1)));
SUBKERNEL_PHASE1
TLoad(ubTensor_0, gmTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 10, 1))));
TLoad(ubTensor_2, gmTensor_3, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 1, 1))));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TAdd(ubTensor_0, ubTensor_0, ubTensor_2);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_7, ubTensor_0, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(2, 19, 1))));
}
)!!!";

    EXPECT_EQ(res, expect);
}
} // namespace npu::tile_fwk