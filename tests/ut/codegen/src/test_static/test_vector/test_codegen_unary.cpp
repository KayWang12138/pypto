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
 * \file test_codegen_unary.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenUnary : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

void TestRowMaxSingleBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Amax(input_a, -1);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowMaxSingleDim2) {
    TestRowMaxSingleBody({8, 128}, {8, 1}, {2, 64}, "ROWMAXSINGLE_DIM2");
}

TEST_F(TestCodegenUnary, RowMaxSingleDim3) {
    TestRowMaxSingleBody({8, 4, 128}, {8, 4, 1}, {2, 1, 64}, "ROWMAXSINGLE_DIM3");
}

TEST_F(TestCodegenUnary, RowMaxSingleDim4) {
    TestRowMaxSingleBody({8, 4, 4, 128}, {8, 4, 4, 1}, {2, 1, 1, 64}, "ROWMAXSINGLE_DIM4");
}

void TestRowSumSingleBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Sum(input_a, -1);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowSumSingleDim2) {
    TestRowSumSingleBody({8, 128}, {8, 1}, {2, 64}, "ROWSUMSINGLE_DIM2");
}

TEST_F(TestCodegenUnary, RowSumSingleDim3) {
    TestRowSumSingleBody({8, 4, 128}, {8, 4, 1}, {2, 1, 64}, "ROWSUMSINGLE_DIM3");
}

TEST_F(TestCodegenUnary, RowSumSingleDim4) {
    TestRowSumSingleBody({8, 4, 4, 128}, {8, 4, 4, 1}, {2, 1, 1, 64}, "ROWSUMSINGLE_DIM4");
}

void TestTransposeVnchwconvBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int> transposeShape,
    std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, outShape, "output");
    config::SetBuildStatic(true);
    FUNCTION(name, {input, output}) {
        output = Transpose(input, transposeShape);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim2) {
    TestTransposeVnchwconvBody({16, 32}, {32, 16}, {0, 1}, {16, 16}, "TRANSPOSE_VNCHWCONV_DIM2");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim4) {
    TestTransposeVnchwconvBody({1, 2, 32, 16}, {1, 2, 16, 32}, {3, 2}, {1, 1, 16, 16}, "TRANSPOSE_VNCHWCONV_DIM4");
}

TEST_F(TestCodegenUnary, TransposeVnchwconvDim5) {
    TestTransposeVnchwconvBody(
        {1, 1, 2, 32, 16}, {1, 1, 2, 16, 32}, {3, 4}, {1, 1, 1, 16, 16}, "TRANSPOSE_VNCHWCONV_DIM5");
}

void TestRowMaxExpandBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = RowMaxExpand(input_a);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowMaxExpandDim2) {
    TestRowMaxExpandBody({128, 64}, {128, 64}, {16, 16}, "ROWMAXEXPAND_DIM2");
}

void TestCastBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_INT32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Cast(input_a, DT_FP32);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, CastDim1) {
    TestCastBody({128}, {128}, {64}, "CAST_DIM2");
}

void TestExpandBody(std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name) {
    TileShape::Current().SetVecTile(tileShape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, outShape, "C");

    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Expand(input_a, outShape);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, ExpandDim2Axis0) {
    TestExpandBody({1, 22}, {22, 22}, {2, 2}, "EXPAND_T");
}

TEST_F(TestCodegenUnary, ExpandDim4Axis0) {
    TestExpandBody({1, 22, 8, 17}, {4, 22, 8, 17}, {2, 16, 4, 8}, "EXPAND_T");
}

TEST_F(TestCodegenUnary, ExpandDim4Axis1) {
    TestExpandBody({4, 1, 8, 17}, {4, 22, 8, 17}, {2, 16, 4, 8}, "EXPAND_T");
}

void TestRowSumBody(
    std::vector<int64_t> shape, std::vector<int64_t> outShape, std::vector<int64_t> tileShape, std::string name, unsigned axis) {
    TileShape::Current().SetVecTile(tileShape);

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor output(DataType::DT_FP32, outShape, "C");

    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Sum(input_a, axis);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, RowSumDim4Axis2) {
    TestRowSumBody({3, 2, 8, 255}, {3, 2, 1, 255}, {2, 8, 8, 255}, "ROWSUMAXIS2", 2);
}

TEST_F(TestCodegenUnary, RowSumDim3Axis1) {
    TestRowSumBody({2, 8, 255}, {2, 1, 255}, {8, 8, 255}, "ROWSUMAXIS1", 1);
}

TEST_F(TestCodegenUnary, RowSumDim2Axis0) {
    TestRowSumBody({8, 255}, {1, 255}, {8, 255}, "ROWSUMAXIS0", 0);
}

TEST_F(TestCodegenUnary, TestVecDup) {
    std::vector<int64_t> shape{32, 1, 32};
    Element src(DataType::DT_INT32, static_cast<int64_t>(2));
    std::string funcName = "VECDUP";
    TileShape::Current().SetVecTile({16, 1, 16});

    Tensor output(DataType::DT_INT32, shape, "C");
    config::SetBuildStatic(true);
    FUNCTION(funcName, {output}) {
        output = npu::tile_fwk::VectorDuplicate(src, DT_INT32, shape);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenUnary, TestVecDupUnaligned) {
    std::vector<int64_t> shape{2, 2, 256, 7};
    Element src(DataType::DT_FP32, 2.0);
    TileShape::Current().SetVecTile({1, 1, 256, 16});

    Tensor output(DataType::DT_FP32, shape, "C");

    std::string funcName = "VECDUP_T";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {output}) {
        output = npu::tile_fwk::VectorDuplicate(src, DT_FP32, shape);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk