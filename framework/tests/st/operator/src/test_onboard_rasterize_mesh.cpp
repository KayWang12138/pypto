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
 * \file test_onboard_abs.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class RasterizeMeshOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

// static int GetTensorSize(Tensor &tensor) {
//     auto &shape = tensor.GetShape();
//     return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
// }

TEST_F(RasterizeMeshOnBoardTest, TestAddTest) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(64, 64);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    int b = 1;
    int sq = 128;
    int d = 64;
    std::vector<int64_t> inputShape = {b * sq, d};
    std::vector<int64_t> outShape = {b * sq, d};

    Tensor input1(DT_FP32, inputShape, "intput1");
    Tensor input2(DT_FP32, inputShape, "intput2");
    Tensor curSeq(DT_INT32, {b, 1}, "curSeq");
    Tensor out(DT_FP32, outShape, "out");

    std::vector<int> actSeqsData(b, 100);
    std::vector<float> golden(b * sq * d, 0.001f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(input1, 1.0),
        RawTensorData::CreateConstantTensor<float>(input2, 1.0),
        RawTensorData::CreateTensor<int32_t>(curSeq, actSeqsData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.001f),
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    for (int i = 0; i < b; i++) {
        int offset = i * sq * d;
        std::fill(golden.begin() + offset, golden.begin() + offset + actSeqsData[i] * d, 2.0);
    }

    FUNCTION("main", {input1, input2, curSeq}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(b)) {
            auto seq = GetTensorData(curSeq, {batchId, 0});
            Tensor intput11 = View(input1, {sq, d}, {seq, d}, {batchId, 0});
            Tensor intput22 = View(input2, {sq, d}, {seq, d}, {batchId, 0});
            auto tmp = Add(intput11, intput22);
            Assemble(tmp, {batchId * sq, 0}, out);
        }
    }

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(RasterizeMeshOnBoardTest, test_operation_greater) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile(32, 32);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    int S0 = 2;
    int S1 = 200;

    std::vector<int64_t> shape = {S0, S1};
    std::vector<int64_t> shape_out = {S0, S1};

    // DataType dtype = DataType::DT_FP32;
    int cap_in = shape[0] * shape[1];
    int cap_out = shape_out[0] * shape_out[1];

    std::vector<float> greater_x(cap_in);
    std::vector<float> greater_y(cap_in);
    std::vector<uint8_t> golden(cap_out);

    readInput<float>(GetGoldenDir() + "/greater_x.bin", greater_x);
    readInput<float>(GetGoldenDir() + "/greater_y.bin", greater_y);

    Tensor input_a(DataType::DT_FP32, shape, "A");
    Tensor input_b(DataType::DT_FP32, shape, "B");
    Tensor output(DataType::DT_BOOL, shape_out, "C");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(input_a, greater_x),
        RawTensorData::CreateTensor<float>(input_b, greater_y),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<uint8_t>(output, golden),
    });

    readInput<uint8_t>(GetGoldenDir() + "/greater_res.bin", golden);

    PROGRAM("GREATER") {
        FUNCTION("GREATER_FUNC", {input_a, input_b}, {output}) {
            LOOP("LOOP_1", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(1)) {
                UNUSED(bIdx);
                output = Compare(input_a, input_b, OpType::GT, OutType::BOOL);
            }
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    for (int i = 0; i < cap_out; i++) {
        printf("golden[%d]: %d,  outs[%d]: %d \n", i, (int)golden[i], i, (int)*((uint8_t *)outs->data() + i));
    }

    bool ret = resultCmp((uint8_t *)golden.data(), (uint8_t *)outs->data(), cap_out, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RasterizeMeshOnBoardTest, test_operation_where) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile({1, 128});
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    int S0 = 2;
    int S1 = 200;

    std::vector<int64_t> shape = {S0, S1};
    std::vector<int64_t> shape_out = {S0, S1};

    // DataType dtype = DataType::DT_FP32;
    int cap_in = shape[0] * shape[1];
    int cap_out = shape_out[0] * shape_out[1];

    std::vector<uint8_t>  index_value(cap_in);
    std::vector<float> where_y_value(cap_in);
    std::vector<float> golden(cap_out);

    readInput<uint8_t>(GetGoldenDir() + "/index.bin", index_value);
    readInput<float>(GetGoldenDir() + "/where_y.bin", where_y_value);

    Tensor index(DataType::DT_BOOL, shape, "A");
    Tensor input(DataType::DT_FP32, shape, "B");
    Tensor output(DataType::DT_FP32, shape_out, "C");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<uint8_t>(index, index_value),
        RawTensorData::CreateTensor<float>(input, where_y_value),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<float>(output, golden),
    });

    readInput<float>(GetGoldenDir() + "/where_res.bin", golden);

    PROGRAM("WHERE_P") {
        FUNCTION("WHERE_FUNC", {index, input}, {output}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                output = Where(index, input, Element(DataType::DT_FP32, INFINITY));
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    for (int i = 0; i < cap_out; i++) {
        printf("golden[%d]: %f,  outs[%d]: %f \n", i, golden[i], i, *((float *)outs->data() + i));
    }

    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
}

TEST_F(RasterizeMeshOnBoardTest, test_operation_and) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile({1, 128});
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    int S0 = 2;
    int S1 = 201;
    std::vector<int64_t> shape = {S0, S1};
    std::vector<int64_t> shape_out = {S0, S1};

    // DataType dtype = DataType::DT_FP32;
    int cap_in = shape[0] * shape[1];
    int cap_out = shape_out[0] * shape_out[1];

    std::vector<uint8_t> index_value1(cap_in);
    std::vector<uint8_t> index_value2(cap_in);
    std::vector<uint8_t> golden(cap_out);

    readInput<uint8_t>(GetGoldenDir() + "/index_1.bin", index_value1);
    readInput<uint8_t>(GetGoldenDir() + "/index_2.bin", index_value2);


    Tensor index1(DataType::DT_BOOL, shape, "A");
    Tensor index2(DataType::DT_BOOL, shape, "B");
    Tensor output(DataType::DT_BOOL, shape_out, "C");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<uint8_t>(index1, index_value1),
        RawTensorData::CreateTensor<uint8_t>(index2, index_value2),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<uint8_t>(output, golden),
    });

    readInput<uint8_t>(GetGoldenDir() + "/and_res.bin", golden);
    PROGRAM("AND_P") {
        TileShape::Current().SetVecTile({32, 32});
        FUNCTION("AND_FUNC", {index1, index2}, {output}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                output = LogicalAnd(index1, index2);
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    for (int i = 0; i < cap_out; i++) {
        printf("golden[%d]: %d,  outs[%d]: %d \n", i, (int)golden[i], i, (int)*((uint8_t *)outs->data() + i));
    }

    std::vector<uint8_t> res(cap_out);
    for (int i = 0; i < cap_out; i++) {
        res[i] = *((uint8_t *)outs->data() + i);
    }
    bool ret = resultCmp((uint8_t *)golden.data(), (uint8_t *)res.data(), cap_out, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RasterizeMeshOnBoardTest, test_operation_gatherelement) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    int S0 = 16;
    int S1 = 10000;

    int D0 = 16;
    int D1 = 10000;

    std::vector<int64_t> shape = {S0, S1};
    std::vector<int64_t> shape_index = {D0, D1};
    std::vector<int64_t> shape_out   = {D0, D1};

    // DataType dtype = DataType::DT_FP32;
    int cap_in = shape[0] * shape[1];
    int cap_in2 = shape_index[0] * shape_index[1];    
    int cap_out = shape_out[0] * shape_out[1];

    std::vector<float> input_value(cap_in);
    std::vector<int32_t> sort_index_value(cap_in2);
    std::vector<float> golden(cap_out);

    readInput<float>(GetGoldenDir() + "/input.bin", input_value);
    readInput<int32_t>(GetGoldenDir() + "/sort_index.bin", sort_index_value);
    readInput<float>(GetGoldenDir() + "/gather_res.bin", golden);

    Tensor input(DataType::DT_FP32, shape, "A");
    Tensor index(DataType::DT_INT32, shape, "B");
    Tensor output(DataType::DT_FP32, shape_out, "C");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(input, input_value),
        RawTensorData::CreateTensor<int32_t>(index, sort_index_value),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<float>(output, golden),
    });

    TileShape::Current().SetVecTile({1, S1});

    PROGRAM("GATHER_P") {
        FUNCTION("GATHER_FUNC", {input, index}, {output}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                output = GatherElements(input, index, -1);
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    for (int i = 0; i < cap_out; i++) {
        printf("golden[%d]: %f,  outs[%d]: %f \n", i, golden[i], i, *((float *)outs->data() + i));
    }

    bool ret = resultCmp(golden, (float *)outs->data(), 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RasterizeMeshOnBoardTest, test_operation_min_max) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    TileShape::Current().SetVecTile({16, 32});
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    // shape, capacity, ptr
    int S0 = 16;
    int S1 = 32;
    int D0 = 16;
    int D1 = 1;

    std::vector<int64_t> shape = {S0, S1};
    std::vector<int64_t> shape_out   = {D0, D1};
    std::vector<int64_t> shape_out0  = {S0, S1};

    int cap_in  = shape[0] * shape[1];
    int cap_out = shape_out[0] * shape_out[1];
    int cap_out0 = shape_out0[0] * shape_out0[1];

    std::vector<float> input_value(cap_in);
    std::vector<int32_t> golden_sort_index(cap_out0);
    std::vector<int32_t> golden_min_index(cap_out);

    readInput<float>(GetGoldenDir()   + "/input.bin", input_value);
    readInput<int32_t>(GetGoldenDir() + "/sort_index.bin", golden_sort_index);
    readInput<int32_t>(GetGoldenDir() + "/min_index.bin",  golden_min_index);

    Tensor input(DataType::DT_FP32, shape, "A");
    Tensor output0(DataType::DT_INT32, shape_out0, "B");
    Tensor output(DataType::DT_INT32, shape_out, "C");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(input, input_value),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(output0, golden_sort_index),
        RawTensorData::CreateTensor<int32_t>(output, golden_min_index),
    });

    PROGRAM("MIN_MAX_P") {
        FUNCTION("MIN_MAX_FUNC", {input}, {output0, output}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                auto input_new = ScalarMulS(input, Element(DataType::DT_FP32, 2.0f));
                auto sort = ArgSort(input_new, -1, false);
                output0 = sort;
                Element ele_one(DataType::DT_INT32, 0.0f);
                Tensor index = Full(ele_one, DT_INT32, shape_out);
                output = GatherElements(sort, index, -1);
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs0 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);

    for (int i = 0; i < cap_out0; i++) {
        printf("golden[%d]: %d,  outs[%d]: %d \n", i, golden_sort_index[i], i, *((int32_t *)outs0->data() + i));
    }

    for (int i = 0; i < cap_out; i++) {
        printf("golden[%d]: %d,  outs[%d]: %d \n", i, golden_min_index[i], i, *((int32_t *)outs1->data() + i));
    }

    // bool ret = resultCmp(golden_sort_index, (int32_t *)outs0->data(), 0);

    bool ret = resultCmp(golden_min_index, (int32_t *)outs1->data(), 0);
    EXPECT_EQ(ret, true);
}

Tensor Squeeze(Tensor a, int axis) {
    std::vector<int64_t> newShape(a.GetStorage()->shape);
    // printf("ori shape: %d, %d, %d \n", newShape[0], newShape[1], newShape[2]);
    if (axis < 0)
        axis = newShape.size() + axis;
    newShape.erase(newShape.begin() + axis);
    // printf("shape: %d, %d \n", newShape[0], newShape[1]);
    auto out = Reshape(a, newShape);
    return out;
}

Tensor Dot(Tensor a, Tensor b, int axis) {
    auto res_mul = Mul(a, b);
    auto sum = Sum(res_mul, axis, true);
    auto out = Squeeze(sum, axis);
    return out;
}

Tensor Dot_DIM2(Tensor a, Tensor b, int axis) {
    auto mul_a_b = Mul(a, b);
    auto mul_a_b_0 = View(mul_a_b, {mul_a_b.GetShape()[0], 1}, {0, 0});// ( #S0, 1)
    auto mul_a_b_1 = View(mul_a_b, {mul_a_b.GetShape()[0], 1}, {0, 1});// ( #S0, 1)

    auto dot = Add(mul_a_b_0, mul_a_b_1);
    auto dot_out = Squeeze(dot, axis);
    return dot_out;
}

Tensor Dot_DIM3(Tensor a, Tensor b, int axis) {
    auto mul_a_b = Mul(a, b);
    auto mul_a_b_0 = View(mul_a_b, {mul_a_b.GetShape()[0], mul_a_b.GetShape()[1], 1}, {0, 0, 0});// ( #S0, S1, 1)
    auto mul_a_b_1 = View(mul_a_b, {mul_a_b.GetShape()[0], mul_a_b.GetShape()[1], 1}, {0, 0, 1});// ( #S0, S1, 1)

    auto dot = Add(mul_a_b_0, mul_a_b_1);
    auto dot_out = Squeeze(dot, axis);
    return dot_out;
}

Tensor PointLineDistanceSquare(Tensor point, Tensor vert_va, Tensor vert_vb, Tensor screen_to_ndc_scale) {
    auto ba = Sub(vert_vb, vert_va);     // (#faces, 2)
    auto p_unsq = Unsqueeze(point, -2);  // (#point, 1, 2)
    auto pa = Sub(p_unsq, vert_va);      // (#point, #faces, 2)
    auto ba_unsq = Unsqueeze(ba, 0);     // (1, #faces, 2)

    // auto t_bot = Dot(ba, ba, -1);
    // auto t_top = Dot(ba_unsq, pa, -1);

    auto t_bot = Dot(ba, ba, -1);      // (#faces,)
    auto t_top = Dot(ba_unsq, pa, -1); // (#point, #faces)

    auto t = Div(t_top, t_bot);          // (#point, #faces )

    auto t_clip = Clip(t, Element(DataType::DT_FP32, 0.0f), Element(DataType::DT_FP32, 1.0f)); // (#point, #faces )

    auto proj = Add(Unsqueeze(vert_va, 0), Mul(Unsqueeze(t_clip, -1), ba_unsq)); // (#point, #faces, 2)

    auto p_unsq_expand = Expand(p_unsq, proj.GetShape());// (#point, #faces, 2)
    auto delta = Sub(proj, p_unsq_expand);
    auto dist  = Mul(delta, screen_to_ndc_scale);

    return Dot(dist, dist, -1);
}

Tensor TriangleSquaredDistance(Tensor point,Tensor tri_v0,Tensor tri_v1,Tensor tri_v2,Tensor scale) {
    auto dist_v01 = PointLineDistanceSquare(point, tri_v0, tri_v1, scale);
    auto dist_v12 = PointLineDistanceSquare(point, tri_v1, tri_v2, scale);
    auto dist_v20 = PointLineDistanceSquare(point, tri_v2, tri_v0, scale);
    auto dist_f = Minimum(dist_v01, dist_v12);
    auto output_dist = Minimum(dist_f, dist_v20);
    return output_dist;// (#point, #faces)
}

Tensor PointLineDistanceSquareSingle(Tensor point_x, Tensor point_y, Tensor vert_va_x, Tensor vert_va_y, Tensor vert_vb_x,Tensor vert_vb_y, Tensor screen_to_ndc_scale) {
    auto ba_x = Sub(vert_vb_x, vert_va_x);   // (#faces)
    auto ba_y = Sub(vert_vb_y, vert_va_y);   // (#faces)

    auto p_x_expand = Expand(Unsqueeze(point_x, -1), {point_x.GetShape()[0],vert_va_x.GetShape()[0]});  // (#point, #faces)
    auto p_y_expand = Expand(Unsqueeze(point_y, -1), {point_x.GetShape()[0],vert_va_x.GetShape()[0]});  // (#point, #faces)
    auto pa_x = Sub(p_x_expand, vert_va_x);    // (#point, #faces)
    auto pa_y = Sub(p_y_expand, vert_va_y);    // (#point, #faces)

    auto ba_x_unsq = Unsqueeze(ba_x, 0);     // (1, #faces)
    auto ba_y_unsq = Unsqueeze(ba_y, 0);     // (1, #faces)

    auto t_bot = Add(Mul(ba_x, ba_x), Mul(ba_y, ba_y));//(#faces)
    auto t_top = Add(Mul(ba_x_unsq, pa_x), Mul(ba_y_unsq, pa_y));//(#p, #faces)

    auto t = Div(t_top, t_bot);          // (#point, #faces )

    auto t_clip = Clip(t, Element(DataType::DT_FP32, 0.0f), Element(DataType::DT_FP32, 1.0f)); // (#point, #faces)

    auto proj_x = Add(Unsqueeze(vert_va_x, 0), Mul(t_clip, ba_x_unsq)); // (#point, #faces)
    auto proj_y = Add(Unsqueeze(vert_va_y, 0), Mul(t_clip, ba_y_unsq)); // (#point, #faces)

    auto delta_x = Sub(proj_x, p_x_expand);// (#point, #faces)
    auto delta_y = Sub(proj_y, p_y_expand);// (#point, #faces)

    auto dist_x  = Mul(delta_x, screen_to_ndc_scale);// (#point, #faces)
    auto dist_y  = Mul(delta_y, screen_to_ndc_scale);// (#point, #faces)

    auto out_sd2 = Add(Mul(dist_x, dist_x), Mul(dist_y, dist_y));// (#point, #faces)

    return  out_sd2;
}

Tensor TriangleSquaredDistanceSingle(Tensor point_x, Tensor point_y, 
        Tensor tri_v0_x,Tensor tri_v0_y,Tensor tri_v1_x,Tensor tri_v1_y,Tensor tri_v2_x,Tensor tri_v2_y,Tensor scale) {
    auto dist_v01 = PointLineDistanceSquareSingle(point_x, point_y, tri_v0_x, tri_v0_y, tri_v1_x, tri_v1_y, scale);// (#point, #faces)
    auto dist_v12 = PointLineDistanceSquareSingle(point_x, point_y, tri_v1_x, tri_v1_y, tri_v2_x, tri_v2_y, scale);// (#point, #faces)
    auto dist_v20 = PointLineDistanceSquareSingle(point_x, point_y, tri_v2_x, tri_v2_y, tri_v0_x, tri_v0_y, scale);// (#point, #faces)

    auto dist_f = Minimum(dist_v01, dist_v12);
    auto output_dist = Minimum(dist_f, dist_v20);
    return output_dist;// (#point, #faces)
}



Tensor PointLineDistanceSquareSingle2(Tensor point_x, Tensor point_y, Tensor vert_va_x, Tensor vert_va_y, Tensor vert_vb_x,Tensor vert_vb_y, Tensor screen_to_ndc_scale) {
    auto ba_x = Sub(vert_vb_x, vert_va_x);              // (1, #faces)
    auto ba_y = Sub(vert_vb_y, vert_va_y);              // (1, #faces)

    auto pa_x = Sub(point_x, vert_va_x);                // (#point, #faces)
    auto pa_y = Sub(point_y, vert_va_y);                // (#point, #faces)

    auto t_bot = Add(Mul(ba_x, ba_x), Mul(ba_y, ba_y)); //(1, #faces)
    auto t_top = Add(Mul(ba_x, pa_x), Mul(ba_y, pa_y)); //(#p, #faces)

    auto t = Div(t_top, t_bot);                         // (#point, #faces)

    auto t_clip = Clip(t, Element(DataType::DT_FP32, 0.0f), Element(DataType::DT_FP32, 1.0f)); // (#point, #faces)

    auto proj_x = Add(vert_va_x, Mul(t_clip, ba_x));    // (#point, #faces)
    auto proj_y = Add(vert_va_y, Mul(t_clip, ba_y));    // (#point, #faces)

    auto delta_x = Sub(proj_x, point_x);// (#point, #faces)
    auto delta_y = Sub(proj_y, point_y);// (#point, #faces)

    auto dist_x  = Mul(delta_x, screen_to_ndc_scale);// (#point, #faces)
    auto dist_y  = Mul(delta_y, screen_to_ndc_scale);// (#point, #faces)

    auto out_sd2 = Add(Mul(dist_x, dist_x), Mul(dist_y, dist_y));// (#point, #faces)

    return  out_sd2;
}

Tensor TriangleSquaredDistanceSingle2(Tensor point_x, Tensor point_y, 
        Tensor tri_v0_x,Tensor tri_v0_y,Tensor tri_v1_x,Tensor tri_v1_y,Tensor tri_v2_x,Tensor tri_v2_y, Tensor scale) {
    auto dist_v01 = PointLineDistanceSquareSingle2(point_x, point_y, tri_v0_x, tri_v0_y, tri_v1_x, tri_v1_y, scale);// (#point, #faces)
    auto dist_v12 = PointLineDistanceSquareSingle2(point_x, point_y, tri_v1_x, tri_v1_y, tri_v2_x, tri_v2_y, scale);// (#point, #faces)
    auto dist_v20 = PointLineDistanceSquareSingle2(point_x, point_y, tri_v2_x, tri_v2_y, tri_v0_x, tri_v0_y, scale);// (#point, #faces)

    auto dist_f = Minimum(dist_v01, dist_v12);
    auto output_dist = Minimum(dist_f, dist_v20);
    return output_dist;// (#point, #faces)
}


TEST_F(RasterizeMeshOnBoardTest, test_point_triangle_distance) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    // std::string bin_path = "_8_11.bin";
    // int face_num = 976;
    // int point_num = 8 * 8;

    // std::string bin_path = "_9_5.bin";
    // int face_num = 272;
    // int point_num = 8 * 8;

    // std::string bin_path = "_9_10.bin";
    // int face_num = 368;
    // int point_num = 8 * 8;

    std::string bin_path = "_11_8.bin";
    int face_num = 96;
    int point_num = 8 * 8;

    // std::string bin_path = "_11_10.bin";
    // int face_num = 192;
    // int point_num = 8 * 8;

    int coord_xy = 2;

    std::vector<int64_t> shape_point      = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts = {face_num, coord_xy};
    std::vector<int64_t> shape_out = {point_num, face_num};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = shape_point[0] * shape_point[1];
    int cap_verts  = shape_face_verts[0] * shape_face_verts[1];
    int cap_out = shape_out[0] * shape_out[1];

    std::vector<float> point_value(cap_point);
    std::vector<float> tri_v0_value(cap_verts);
    std::vector<float> tri_v1_value(cap_verts);
    std::vector<float> tri_v2_value(cap_verts);
    std::vector<float> scale_value(1);
    std::vector<float> golden_dist(cap_out);

    readInput<float>(GetGoldenDir() + "/point" + bin_path, point_value);
    readInput<float>(GetGoldenDir() + "/face_verts_0" + bin_path, tri_v0_value);
    readInput<float>(GetGoldenDir() + "/face_verts_1" + bin_path, tri_v1_value);
    readInput<float>(GetGoldenDir() + "/face_verts_2" + bin_path, tri_v2_value);
    readInput<float>(GetGoldenDir() + "/scale" + bin_path, scale_value);

    Tensor point (DataType::DT_FP32, shape_point, "A");
    Tensor tri_v0(DataType::DT_FP32, shape_face_verts, "B");
    Tensor tri_v1(DataType::DT_FP32, shape_face_verts, "C");
    Tensor tri_v2(DataType::DT_FP32, shape_face_verts, "D");
    Tensor scale(DataType::DT_FP32, {1}, "E");
    Tensor output_dist(DataType::DT_FP32, shape_out, "Z");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, point_value),
        RawTensorData::CreateTensor<float>(tri_v0, tri_v0_value),
        RawTensorData::CreateTensor<float>(tri_v1, tri_v1_value),
        RawTensorData::CreateTensor<float>(tri_v2, tri_v2_value),
        RawTensorData::CreateTensor<float>(scale, scale_value),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<float>(output_dist, golden_dist),
    });
    readInput<float>(GetGoldenDir() + "/distance" + bin_path, golden_dist);

    TileShape::Current().SetVecTile({64, 16, 8});

    PROGRAM("PointTriangleDistance") {
        FUNCTION("PointTriangleDistanceFunc", {point, tri_v0, tri_v1, tri_v2, scale}, {output_dist}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                output_dist = TriangleSquaredDistance(point, tri_v0, tri_v1, tri_v2, scale);
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    // for (int i = 0; i < cap_out; i++) {
    //     printf("golden[%d]: %f,  outs[%d]: %f \n", i, golden_dist[i], i, *((float *)outs->data() + i));
    // }

    bool ret = resultCmp(golden_dist, (float *)outs->data(), 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(RasterizeMeshOnBoardTest, test_point_triangle_singed_distance) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    // std::string bin_path = "_8_11.bin";
    // int face_num = 976;
    // int point_num = 8 * 8;

    // std::string bin_path = "_9_5.bin";
    // int face_num = 272;
    // int point_num = 8 * 8;

    // std::string bin_path = "_9_10.bin";
    // int face_num = 368;
    // int point_num = 8 * 8;

    std::string bin_path = "_11_8.bin";
    int face_num = 96;
    int point_num = 8 * 8;

    // std::string bin_path = "_11_10.bin";
    // int face_num = 192;
    // int point_num = 8 * 8;

    int coord_xy = 2;
    int coord_xyz = 3;

    std::vector<int64_t> shape_point      = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts = {face_num, coord_xy};
    std::vector<int64_t> shape_bary_coords = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_out = {point_num, face_num};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = shape_point[0] * shape_point[1];
    int cap_verts  = shape_face_verts[0] * shape_face_verts[1];
    int cap_bary_coords  = shape_bary_coords[0] * shape_bary_coords[1]* shape_bary_coords[2];
    int cap_out = shape_out[0] * shape_out[1];

    std::vector<float> point_value(cap_point);
    std::vector<float> tri_v0_value(cap_verts);
    std::vector<float> tri_v1_value(cap_verts);
    std::vector<float> tri_v2_value(cap_verts);
    std::vector<float> scale_value(1);
    std::vector<float> bary_coords_value(cap_bary_coords);


    // std::vector<float> golden_sign(cap_out);
    std::vector<float> golden_dist(cap_out);
    std::vector<float> golden_face_sd2(cap_out);
    std::vector<uint8_t> golden_is_inside(cap_out);

    readInput<float>(GetGoldenDir() + "/point" + bin_path, point_value);
    readInput<float>(GetGoldenDir() + "/face_verts_0" + bin_path, tri_v0_value);
    readInput<float>(GetGoldenDir() + "/face_verts_1" + bin_path, tri_v1_value);
    readInput<float>(GetGoldenDir() + "/face_verts_2" + bin_path, tri_v2_value);
    readInput<float>(GetGoldenDir() + "/scale" + bin_path, scale_value);
    readInput<float>(GetGoldenDir() + "/bary_coords" + bin_path, bary_coords_value);

    // readInput<uint8_t>(GetGoldenDir() + "/sign" + bin_path, golden_sign);

    Tensor point (DataType::DT_FP32, shape_point, "A");
    Tensor tri_v0(DataType::DT_FP32, shape_face_verts, "B");
    Tensor tri_v1(DataType::DT_FP32, shape_face_verts, "C");
    Tensor tri_v2(DataType::DT_FP32, shape_face_verts, "D");
    Tensor scale(DataType::DT_FP32, {1}, "E");
    Tensor bary_coords(DataType::DT_FP32, shape_bary_coords, "F");
    Tensor distance(DataType::DT_FP32, shape_out, "T");
    Tensor face_sd2(DataType::DT_FP32, shape_out, "Z");
    Tensor inside(DataType::DT_BOOL, shape_out, "W");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, point_value),
        RawTensorData::CreateTensor<float>(tri_v0, tri_v0_value),
        RawTensorData::CreateTensor<float>(tri_v1, tri_v1_value),
        RawTensorData::CreateTensor<float>(tri_v2, tri_v2_value),
        RawTensorData::CreateTensor<float>(scale, scale_value),
        RawTensorData::CreateTensor<float>(bary_coords, bary_coords_value),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<uint8_t>(inside, golden_is_inside),
        RawTensorData::CreateTensor<float>(distance, golden_dist),
        RawTensorData::CreateTensor<float>(face_sd2, golden_face_sd2),
    });

    readInput<float>(GetGoldenDir() + "/distance" + bin_path, golden_dist);
    readInput<float>(GetGoldenDir() + "/face_sd2" + bin_path, golden_face_sd2);
    readInput<uint8_t>(GetGoldenDir() + "/is_inside" + bin_path, golden_is_inside);

    PROGRAM("PointTriangleDistance") {
        TileShape::Current().SetVecTile({16, 8, 8});
        FUNCTION("PointTriangleDistanceFunc", {point, tri_v0, tri_v1, tri_v2, scale, bary_coords}, {inside, distance, face_sd2}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                auto square_distance = TriangleSquaredDistance( point,  tri_v0,  tri_v1,  tri_v2,  scale); // (#point, #faces)
                distance = square_distance;
                auto barycentrics_x = View(bary_coords, {point_num, face_num, 1}, {0, 0, 0});// (#point, #faces, 1)
                auto barycentrics_y = View(bary_coords, {point_num, face_num, 1}, {0, 0, 1});// (#point, #faces, 1)
                auto barycentrics_z = View(bary_coords, {point_num, face_num, 1}, {0, 0, 2});// (#point, #faces, 1)

                auto barycentrics_x_unsq = Squeeze(barycentrics_x, -1);// (#point, #faces)
                auto barycentrics_y_unsq = Squeeze(barycentrics_y, -1);// (#point, #faces)
                auto barycentrics_z_unsq = Squeeze(barycentrics_z, -1);// (#point, #faces)

                auto z_one  = Full(Element(DataType::DT_FP32, 1.0f), DT_FP32, barycentrics_x_unsq.GetShape());   // (#point, #faces)
                auto z_zero = Full(Element(DataType::DT_FP32, 0.0f), DT_FP32, barycentrics_x_unsq.GetShape());   // (#point, #faces)
                auto barycentrics_x_greater_zero = Compare(barycentrics_x_unsq, z_zero, OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(barycentrics_y_unsq, z_zero, OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(barycentrics_z_unsq, z_zero, OpType::GT, OutType::BOOL); // (#point, #faces)

                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                inside = is_inside;

                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
                face_sd2 = Where(is_inside, square_distance_neg, square_distance);
                // // auto sign = ScalarAddS(ScalarMulS(is_inside_fp32, Element(DataType::DT_FP32, -2.0f)), Element(DataType::DT_FP32, 1.0f));
                // face_sd2 = Mul(square_distance, sign);
            }
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_inside = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_distance = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_face_sd2 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);

    printf("=========start================\n");
    for (int i = 0; i < cap_out; i++) {
        if ((abs(golden_is_inside[i] - *((uint8_t *)res_inside->data() + i)) > 0.001)) {
            int point_idx = i / face_num;
            int face_idx  = i % face_num;
            printf("index: %d : golden_inside: %d,  inside_res: %d \n", i, (int)golden_is_inside[i],  (int)*((uint8_t *)res_inside->data() + i));
            printf("bary_coords[0] %f : bary_coords[1]: %f,  bary_coords:[2]: %f \n", 
                        bary_coords_value[point_idx * face_num * 3 + face_idx * 3 + 0],
                        bary_coords_value[point_idx * face_num * 3 + face_idx * 3 + 1],
                        bary_coords_value[point_idx * face_num * 3 + face_idx * 3 + 2]);
        }
    }
    printf("=========end================\n");
    for (int i = 0; i < cap_out; i++) {
        if (golden_face_sd2[i] < 0) {
            printf("index: %d : golden_dist: %f,  res_distance: %f \n", i, golden_dist[i],  *((float *)res_distance->data() + i));
            printf("index: %d : golden_face_sd2: %f,  res_face_sd2: %f \n", i, golden_face_sd2[i],  *((float *)res_face_sd2->data() + i));
        }
    }
    bool ret = true;
    ret = resultCmp(golden_dist, (float *)res_distance->data(), 0.001f);
    ret = resultCmp(golden_face_sd2, (float *)res_face_sd2->data(), 0.001f);
    EXPECT_EQ(ret, true);
}

Tensor BarycentricCoordsNoperspective(Tensor point,Tensor tri_v0, Tensor tri_v1,Tensor tri_v2) {
    auto v01 = Sub(tri_v1, tri_v0); //(#tri, 2)
    auto v02 = Sub(tri_v2, tri_v0); //(#tri, 2)
    auto v0p = Sub(Unsqueeze(point, -2), tri_v0); //(#p, #tri, 2)

    auto d00 = Dot(v01, v01, -1);//(#tri)
    auto d01 = Dot(v01, v02, -1);//(#tri)
    auto d11 = Dot(v02, v02, -1);//(#tri)
    auto d20 = Dot(v0p, v01, -1);//(#p, #tri)
    auto d21 = Dot(v0p, v02, -1);//(#p, #tri)

    // printf("d00.shape[%lu]= {%d} \n",d00.GetShape().size(), d00.GetShape()[0]);

    auto denom = Sub(Mul(Unsqueeze(d00, 0), d11), Unsqueeze(Mul(d01, d01), 0));//# (#p..., #tri)
    auto v = Div(Sub(Mul(Unsqueeze(d11, 0), d20), Mul(Unsqueeze(d01, 0), d21)), denom); //# (#p..., #tri)
    auto w = Div(Sub(Mul(Unsqueeze(d00, 0), d21), Mul(Unsqueeze(d01, 0), d20)), denom); //# (#p..., #tri)
    auto v_Unsque = Unsqueeze(v, -1); //# (#p, #tri, 1)
    auto w_Unsque = Unsqueeze(w, -1); //# (#p, #tri, 1)
    Element ele_one(DataType::DT_FP32, 1.0f);
    Tensor ones1 = Full(ele_one, DT_FP32, v_Unsque.GetShape());
    auto u = Sub(Sub(ones1, v_Unsque), w_Unsque); //# 1 - v - w)

    auto bary_coords = Cat({u, v_Unsque, w_Unsque}, -1);//# (#p, #tri, 3)

    return bary_coords;
}

TEST_F(RasterizeMeshOnBoardTest, test_barycentric_coords_noperspective) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    std::string bin_path = "_8_11.bin";
    int face_num = 976;
    int point_num = 8 * 8;

    // std::string bin_path = "_9_5.bin";
    // int face_num = 272;
    // int point_num = 8 * 8;

    // std::string bin_path = "_9_10.bin";
    // int face_num = 368;
    // int point_num = 8 * 8;

    // std::string bin_path = "_11_8.bin";
    // int face_num = 96;
    // int point_num = 8 * 8;

    // std::string bin_path = "_11_10.bin";
    // int face_num = 192;
    // int point_num = 8 * 8;

    int coord_xy = 2;
    int coord_xyz = 3;

    std::vector<int64_t> shape_point      = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts = {face_num, coord_xy};
    std::vector<int64_t> shape_out = {point_num, face_num, coord_xyz};

    int cap_point  = shape_point[0] * shape_point[1];
    int cap_verts  = shape_face_verts[0] * shape_face_verts[1];
    int cap_out = shape_out[0] * shape_out[1]* shape_out[2];

    std::vector<float> point_value(cap_point);
    std::vector<float> tri_v0_value(cap_verts);
    std::vector<float> tri_v1_value(cap_verts);
    std::vector<float> tri_v2_value(cap_verts);
    std::vector<float> bary_coords_value(cap_out);

    readInput<float>(GetGoldenDir() + "/point" + bin_path, point_value);
    readInput<float>(GetGoldenDir() + "/face_verts_0" + bin_path, tri_v0_value);
    readInput<float>(GetGoldenDir() + "/face_verts_1" + bin_path, tri_v1_value);
    readInput<float>(GetGoldenDir() + "/face_verts_2" + bin_path, tri_v2_value);

    Tensor point (DataType::DT_FP32, shape_point, "A");
    Tensor tri_v0(DataType::DT_FP32, shape_face_verts, "B");
    Tensor tri_v1(DataType::DT_FP32, shape_face_verts, "C");
    Tensor tri_v2(DataType::DT_FP32, shape_face_verts, "D");
    Tensor bary_coords(DataType::DT_FP32, shape_out, "Z");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, point_value),
        RawTensorData::CreateTensor<float>(tri_v0, tri_v0_value),
        RawTensorData::CreateTensor<float>(tri_v1, tri_v1_value),
        RawTensorData::CreateTensor<float>(tri_v2, tri_v2_value),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<float>(bary_coords, bary_coords_value),
    });

    readInput<float>(GetGoldenDir() + "/bary_coords" + bin_path, bary_coords_value);

    TileShape::Current().SetVecTile({16, 8, 8});

    PROGRAM("BarycentricCoordsNoperspective") {
        FUNCTION("BarycentricCoordsNoperspectiveFunc", {point, tri_v0, tri_v1, tri_v2}, {bary_coords}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                bary_coords = BarycentricCoordsNoperspective(point, tri_v0, tri_v1, tri_v2);
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_bary_coords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    bool ret = true;
    ret = resultCmp(bary_coords_value, (float *)res_bary_coords->data(), 0.001f);
    EXPECT_EQ(ret, true);
}

// Tensor BarycentricCoordsPespectCorrect(Tensor bary_coords, Tensor tri_v0, Tensor tri_v1, Tensor tri_v2) {
//     auto az = View(tri_v0, {tri_v0.GetShape()[0], 1}, {0, 2}); // ( #faces, 1)
//     auto bz = View(tri_v1, {tri_v1.GetShape()[0], 1}, {0, 2}); // ( #faces, 1)
//     auto cz = View(tri_v2, {tri_v2.GetShape()[0], 1}, {0, 2}); // ( #faces, 1)

//     auto bary_coords_0 = View(bary_coords, {bary_coords.GetShape()[0], bary_coords.GetShape()[1], 1}, {0, 0, 0}); // (#point, #faces, 1)
//     auto bary_coords_1 = View(bary_coords, {bary_coords.GetShape()[0], bary_coords.GetShape()[1], 1}, {0, 0, 1}); // (#point, #faces, 1)
//     auto bary_coords_2 = View(bary_coords, {bary_coords.GetShape()[0], bary_coords.GetShape()[1], 1}, {0, 0, 2}); // (#point, #faces, 1)

//     auto w0_top = Mul(Mul(bary_coords_0, bz), cz); // (#point, #faces, 1)
//     auto w1_top = Mul(az, Mul(bary_coords_1, cz)); // (#point, #faces, 1)
//     auto w2_top = Mul(Mul(az, bz), bary_coords_2); // (#point, #faces, 1)

//     auto top_sum = Add(Add(w0_top, w1_top), w2_top); // (#point, #faces, 1)

//     Element esp_one(DataType::DT_FP32, 1e-6f);
//     Tensor esp = Full(esp_one, DT_FP32, top_sum.GetShape());

//     auto denom = Maximum(top_sum, esp); // (#point, #faces, 1)
//     auto w0 = Div(w0_top, denom); // (#point, #faces, 1)
//     auto w1 = Div(w1_top, denom); // (#point, #faces, 1)
//     auto w2 = Div(w2_top, denom); // (#point, #faces, 1)
//     // auto w0_unsque = Unsqueeze(w0, -1); // (#point, #faces)
//     // auto w1_unsque = Unsqueeze(w1, -1); // (#point, #faces)
//     // auto w2_unsque = Unsqueeze(w2, -1); // (#point, #faces)
//     // auto bary_coords_correct = Cat({w0_unsque, w1_unsque, w2_unsque}, -1);
    
//     auto bary_coords_correct = Cat({w0, w1, w2}, -1);
//     return bary_coords_correct;
// }

// TEST_F(RasterizeMeshOnBoardTest, test_barycentric_coords_perspective_correct) {
//     aclInit(nullptr);
//     rtSetDevice(GetDeviceIdByEnvVar());
//     int face_num = 96;
//     int point_num = 8 * 8;
//     // int coord_xy = 2;
//     int coord_xyz = 3;

//     std::vector<int64_t> shape_bary_coords = {point_num, face_num, coord_xyz};
//     std::vector<int64_t> shape_face_verts = {face_num, coord_xyz};

//     // DataType dtype = DataType::DT_FP32;
//     int cap_bary_coords  = shape_bary_coords[0] * shape_bary_coords[1]* shape_bary_coords[2];
//     int cap_verts  = shape_face_verts[0] * shape_face_verts[1];

//     int cap_out = shape_bary_coords[0] * shape_bary_coords[1]* shape_bary_coords[2];

//     uint64_t outputSize = cap_out * sizeof(float);
//     uint8_t* out_ptr = allocDevAddr(outputSize);

//     std::string bin_path = "_12_6.bin";

//     PROGRAM("BarycentricCoordsPespectCorrect") {
//         void *bary_coords_ptr  = readToDev(GetGoldenDir() + "/bary_nopersp" + bin_path, cap_bary_coords);
//         void *verts_0 = readToDev(GetGoldenDir() + "/face_verts_0_xyz" + bin_path, cap_verts);
//         void *verts_1 = readToDev(GetGoldenDir() + "/face_verts_1_xyz" + bin_path, cap_verts);
//         void *verts_2 = readToDev(GetGoldenDir() + "/face_verts_2_xyz" + bin_path, cap_verts);

//         TileShape::Current().SetVecTile({16, 8, 8});

//         Tensor bary_coords(DataType::DT_FP32,  shape_bary_coords, (uint8_t *)bary_coords_ptr,  "A");
//         Tensor tri_v0(DataType::DT_FP32, shape_face_verts, (uint8_t *)verts_0, "B");
//         Tensor tri_v1(DataType::DT_FP32, shape_face_verts, (uint8_t *)verts_1, "C");
//         Tensor tri_v2(DataType::DT_FP32, shape_face_verts, (uint8_t *)verts_2, "D");

//         Tensor bary_coords_correct(DataType::DT_FP32, shape_bary_coords, out_ptr, "Z");
//         ConfigManager::Instance();

//         FUNCTION("BarycentricCoordsNoperspectiveFunc", {bary_coords, tri_v0, tri_v1, tri_v2, bary_coords_correct}) {
//             bary_coords_correct = BarycentricCoordsPespectCorrect(bary_coords, tri_v0, tri_v1, tri_v2);
//         }
//     }

//     std::vector<float> golden(cap_out);
//     std::vector<float> res(cap_out);
//     machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
//     readInput(GetGoldenDir() + "/bary_coords_corect" + bin_path, golden);
//     for (int i = 0; i < cap_out; i++) {
//         printf("golden[%d]: %f,  res[%d]: %f \n", i, golden[i], i, res[i]);
//     }
//     int ret = resultCmp(golden, res, 0.001f);
//     EXPECT_EQ(ret, true);
// }

TEST_F(RasterizeMeshOnBoardTest, test_rasterize) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    // std::string bin_path = "_8_11.bin";
    // int face_num = 976;
    // int point_num = 64 * 64;

    // std::string bin_path = "_9_5.bin";
    // int face_num = 272;
    // int point_num = 8 * 8;

    // std::string bin_path = "_9_10.bin";
    // int face_num = 368;
    // int point_num = 64 * 64;

    // std::string bin_path = "_11_8.bin";
    // int face_num = 96;
    // int point_num = 8 * 8;

    std::string bin_path = "_11_10.bin";
    int face_num = 192;
    int point_num = 8 * 8;

    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    std::vector<int64_t> shape_point        = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};
    std::vector<int64_t> shape_face_idx     = {face_num};


    std::vector<int64_t> shape_bary_coords      = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_point_per_frag   = {point_num, face_num};

    std::vector<int64_t> shape_pix_to_face  = {point_num, 1};
    std::vector<int64_t> shape_zbuf         = {point_num, 1};
    std::vector<int64_t> shape_dists        = {point_num, 1};
    std::vector<int64_t> shape_barycoords   = {point_num, 1, coord_xyz};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());
    int cap_face_idx  = std::accumulate(shape_face_idx.begin(), shape_face_idx.end(), 1, std::multiplies<int>());

    int cap_shape_point_per_frag  = std::accumulate(shape_point_per_frag.begin(), shape_point_per_frag.end(), 1, std::multiplies<int>());

    int cap_bary_coords  = std::accumulate(shape_bary_coords.begin(), shape_bary_coords.end(), 1, std::multiplies<int>());
    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);
    std::vector<int32_t> input_face_idx(cap_face_idx, 0);
    std::vector<float>   input_scale(1, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_bary_coords(cap_bary_coords, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_z_frags(cap_shape_point_per_frag, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    std::vector<int8_t>  golden_tmp_should_write(cap_shape_point_per_frag, 0);
    std::vector<int8_t>  golden_tmp_inside(cap_shape_point_per_frag, 0);
    std::vector<float>   golden_tmp_face_sd2(cap_shape_point_per_frag, 0);
    std::vector<float>   golden_tmp_distance(cap_shape_point_per_frag, 0);
    std::vector<int32_t> golden_tmp_sort_idex(cap_shape_point_per_frag, 0);
    std::vector<int32_t> golden_tmp_index(cap_zbuf, 0);

    readInput<float>(GetGoldenDir()  + "/point" + bin_path,      input_point);
    readInput<float>(GetGoldenDir()  + "/face_verts" + bin_path, input_face_verts);
    readInput<int32_t>(GetGoldenDir()+ "/face_idx" + bin_path,   input_face_idx);
    readInput<float>(GetGoldenDir()  + "/scale" + bin_path,      input_scale);

    Tensor point(DataType::DT_FP32,      shape_point,      "I_1");
    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_2");
    Tensor face_idx(DataType::DT_INT32,  shape_face_idx,   "I_3");
    Tensor scale(DataType::DT_FP32,      {1},              "I_4");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor bary_coords(DataType::DT_FP32,  shape_bary_coords, "O_3");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor z_frags_out(DataType::DT_FP32,  shape_point_per_frag, "O_5");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords, "O_6");

    Tensor tmp_distance(DataType::DT_FP32,     shape_bary_coords,    "T_1");
    Tensor tmp_face_sd2(DataType::DT_FP32,     shape_point_per_frag, "T_2");
    Tensor tmp_inside(DataType::DT_BOOL,       shape_point_per_frag, "T_3");
    Tensor tmp_should_write(DataType::DT_BOOL, shape_point_per_frag, "T_4");
    Tensor tmp_index(DataType::DT_INT32,       shape_zbuf,           "T_5");
    Tensor tmp_sort_idex(DataType::DT_INT32,   shape_point_per_frag, "T_6");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, input_point),
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<int32_t>(face_idx, input_face_idx),
        RawTensorData::CreateTensor<float>(scale, input_scale),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(bary_coords, golden_bary_coords),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(bary_coords, golden_barycoords),
        // RawTensorData::CreateTensor<float>(z_frags_out, golden_z_frags),
        // RawTensorData::CreateTensor<float>(tmp_face_sd2, golden_tmp_face_sd2),
        // RawTensorData::CreateTensor<int8_t>(tmp_inside, golden_tmp_inside),
        // RawTensorData::CreateTensor<int8_t>(tmp_should_write, golden_tmp_should_write),
        // RawTensorData::CreateTensor<int32_t>(tmp_index, golden_tmp_index),
        // RawTensorData::CreateTensor<float>(tmp_distance, golden_tmp_distance),
        // RawTensorData::CreateTensor<int32_t>(tmp_sort_idex, golden_tmp_sort_idex),
    });

    readInput<int32_t>(GetGoldenDir()+ "/pix_to_face" + bin_path, golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/zbuf" + bin_path,        golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/bary_coords" + bin_path, golden_bary_coords);
    readInput<float>(GetGoldenDir()  + "/dists" + bin_path,       golden_dists);
    readInput<float>(GetGoldenDir()  + "/z_frags" + bin_path,     golden_z_frags);
    readInput<float>(GetGoldenDir()  + "/barycentrics" + bin_path,golden_barycoords);

    readInput<float>(GetGoldenDir()  + "/distance" + bin_path,     golden_tmp_distance);
    readInput<float>(GetGoldenDir()  + "/face_sd2" + bin_path,     golden_tmp_face_sd2);
    readInput<int8_t>(GetGoldenDir() + "/is_inside" + bin_path,    golden_tmp_inside);
    readInput<int8_t>(GetGoldenDir() + "/should_write" + bin_path, golden_tmp_should_write);
    readInput<int32_t>(GetGoldenDir()+ "/index" + bin_path,        golden_tmp_index);
    // readInput<int32_t>(GetGoldenDir()+ "/sorted_idx" + bin_path,   golden_tmp_sort_idex);

    // float INF_Z = INFINITY;
    PROGRAM("Rasterize") {
        TileShape::Current().SetVecTile({32, 32, 8});

        FUNCTION("RasterizeFunc", {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, bary_coords, dists, barycoords}) {
        // FUNCTION("RasterizeFunc", {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, bary_coords, dists, barycoords, z_frags_out, 
        //                             tmp_face_sd2, tmp_inside, tmp_should_write, tmp_index, tmp_distance, tmp_sort_idex}) {
            LOOP("loop1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);

                auto tri_v0_xy = View(face_verts, {face_verts.GetShape()[0], 1, 2}, {0, 0, 0});// (#faces,1, 2)
                auto tri_v1_xy = View(face_verts, {face_verts.GetShape()[0], 1, 2}, {0, 1, 0});// (#faces,1, 2)
                auto tri_v2_xy = View(face_verts, {face_verts.GetShape()[0], 1, 2}, {0, 2, 0});// (#faces,1, 2)

                auto tri_v0_xy_sq = Squeeze(tri_v0_xy, -2);// (#faces, 2)
                auto tri_v1_xy_sq = Squeeze(tri_v1_xy, -2);// (#faces, 2)
                auto tri_v2_xy_sq = Squeeze(tri_v2_xy, -2);// (#faces, 2)

                // BarycentricCoordsNoperspective
                auto v01 = Sub(tri_v1_xy_sq, tri_v0_xy_sq); //(#faces, 2)
                auto v02 = Sub(tri_v2_xy_sq, tri_v0_xy_sq); //(#faces, 2)
                auto v0p = Sub(Unsqueeze(point, -2), tri_v0_xy_sq); //(#p, #faces, 2)

                auto d00 = Dot(v01, v01, -1);//(#faces)
                auto d01 = Dot(v01, v02, -1);//(#faces)
                auto d11 = Dot(v02, v02, -1);//(#faces)
                auto d20 = Dot(v0p, Unsqueeze(v01, 0), -1);//(#p, #faces)
                auto d21 = Dot(v0p, Unsqueeze(v02, 0), -1);//(#p, #faces)

                auto denom = Sub(Unsqueeze(Mul(d00, d11), 0), Unsqueeze(Mul(d01, d01), 0));//# (1, #faces)
                auto v = Div(Sub(Mul(Unsqueeze(d11, 0), d20), Mul(Unsqueeze(d01, 0), d21)), denom); //# (#p, #faces)
                auto w = Div(Sub(Mul(Unsqueeze(d00, 0), d21), Mul(Unsqueeze(d01, 0), d20)), denom); //# (#p, #faces)
                auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)
                auto v_Unsque = Unsqueeze(v, -1); //# (#p, #faces, 1)
                auto w_Unsque = Unsqueeze(w, -1); //# (#p, #faces, 1)
                auto u_Unsque = Unsqueeze(u, -1); //# (#p, #faces, 1)
                auto barycentrics= Cat({u_Unsque, v_Unsque, w_Unsque}, -1);//# (#p, #faces, 3)

                // auto bary_coords_correct = BarycentricCoordsPespectCorrect(barycentrics, tri_v0_xyz_sq, tri_v1_xyz_sq, tri_v2_xyz_sq);

                // TriangleSigned SquaredDistance
                auto square_distance = TriangleSquaredDistance(point,  tri_v0_xy_sq,  tri_v1_xy_sq,  tri_v2_xy_sq,  scale); // (#point, #faces)
                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));

                auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                // bary_coords = barycentrics;
                // tmp_distance = square_distance;
                // tmp_face_sd2 = face_sd2;
                // tmp_inside = is_inside;

                auto face_v0_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2}); // (#faces, 1, 1)
                auto face_v1_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 2}); // (#faces, 1, 1)
                auto face_v2_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 2}); // (#faces, 1, 1)
                auto face_v0_z_sq = Squeeze(face_v0_z, -1);// (#faces, 1)
                auto face_v1_z_sq = Squeeze(face_v1_z, -1);// (#faces, 1)
                auto face_v2_z_sq = Squeeze(face_v2_z, -1);// (#faces, 1)
                auto z_frags = Add(Add(Mul(u_Unsque, face_v0_z_sq), Mul(v_Unsque, face_v1_z_sq)), Mul(w_Unsque, face_v2_z_sq));  // (#point, #faces, 1)
                auto z_frags_sq = Squeeze(z_frags, -1);// (#point, #faces)

                // // auto z_frags = Sum(Mul(barycentrics, face_verts_z), -1);// (#point, #faces, 1)  // device执行失败，
                // auto face_verts_z = View(face_verts, {face_verts.GetShape()[0], face_verts.GetShape()[1], 1}, {0, 0, 2});// (#faces, 3, 1)
                // auto face_verts_z_sq = Squeeze(face_verts_z, -1);// (#faces, 3)
                // auto z_frags = Sum(Mul(barycentrics, Unsqueeze(face_verts_z_sq, 0)), -1, true);// (#point, #faces, 1)
                // auto z_frags_sq = Squeeze(z_frags, -1);// (#point, #faces)

                // z_frags_out = z_frags_sq;

                // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                Element blur_radius (DataType::DT_FP32, 0.0f);
                auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                TileShape::Current().SetVecTile({4, 128, 8}); // 64, 976

                // auto output_topk1 = TopK(zbuf_1, 1, -1, false);
                // auto min_idx = std::get<1>(output_topk1);
                // zbuf = std::get<0>(output_topk1);

                auto sorted_idx = ArgSort(zbuf_1, -1, false);
                Element ele_zero(DataType::DT_INT32, 0.0f);
                Tensor zero_index = Full(ele_zero, DT_INT32, {point_num, 1});               // (#point, 1)
                auto min_idx = GatherElements(sorted_idx, zero_index, -1);                  // (#point, 1)

                // tmp_should_write = should_write;
                // tmp_sort_idex = sorted_idx;
                // tmp_index = min_idx;

                auto face_idx_unsq = Unsqueeze(face_idx, 0);                                // (1, #faces)
                auto pix_to_face_expand = Expand(face_idx_unsq, {point_num, face_num});     // (#point, #faces)
                pix_to_face = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                zbuf  = GatherElements(zbuf_1, min_idx, -1);                                 // (#point, #1)
                dists = GatherElements(face_sd2, min_idx, -1);                               // (#point, #1)

                std::vector<int64_t> idxShape(min_idx.GetStorage()->shape);
                idxShape.insert(idxShape.end(), 3);
                auto idx_bary_coords = Expand(Unsqueeze(min_idx, -1), idxShape);            // (#point, #1, 3)
                barycoords = GatherElements(barycentrics, idx_bary_coords, -2);             // (#point, #1, 3)
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_bary_coords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);
    auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(4);
    // auto res_z_frags = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(5);
    // auto res_tmp_face_sd2 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(6);
    // auto res_tmp_inside = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(7);
    // auto res_tmp_should_write = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(8);
    // auto res_tmp_index = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(9);
    // auto res_tmp_distance = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(10);
    // auto res_tmp_sort_index = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(11);


    // printf("\n\n=================== golden_z_frags ===================\n\n");
    // (void)resultCmp(golden_z_frags,     (float *)res_z_frags->data(), 0.001f);      // OK

    // printf("\n\n=================== golden_tmp_distance  =================== \n\n"); // 0.035% error
    // (void)resultCmp(golden_tmp_distance,(float *)res_tmp_distance->data(), 0.001f); // OK

    // printf("\n\n=================== golden_tmp_inside  =================== \n\n");
    // (void)resultCmp<int8_t>(golden_tmp_inside, (int8_t *)res_tmp_inside->data(), 0, 0, 1000, false, true, 0); //ok

    // printf("\n\n=================== golden_tmp_face_sd2  =================== \n\n");  // 0.035% error
    // (void)resultCmp(golden_tmp_face_sd2,(float *)res_tmp_face_sd2->data(), 0.001f); // OK

    // // for (int i = 0; i < 256; i++) {
    // //     printf("golden_tmp_inside  [%d]: %d,  tmp_inside  [%d]: %d \n", i, (int)golden_tmp_inside[i], i, (int)(*((int8_t *)res_tmp_inside->data() + i)));
    // //     printf("golden_tmp_distance[%d]: %f,  tmp_distance[%d]: %f \n", i, golden_tmp_distance[i], i, (*((float *)res_tmp_distance->data() + i)));
    // //     printf("golden_tmp_face_sd2[%d]: %f,  tmp_face_sd2[%d]: %f \n", i, golden_tmp_face_sd2[i], i, (*((float *)res_tmp_face_sd2->data() + i)));
    // // }

    // printf("\n\n=================== golden_tmp_should_write  =================== \n\n"); // 0.022% error
    // (void)resultCmp<int8_t>(golden_tmp_should_write, (int8_t *)res_tmp_should_write->data(), 0, 0, 1000, false, true, 0); // OK
    // // for (int i = 0; i < 256; i++) {
    // //     printf("should_write Index[%d]: golden: %d, res: %d \n", i, golden_tmp_should_write[i], *((int8_t *)res_tmp_should_write->data() + i) );
    // // }

    // printf("\n\n=================== golden_tmp_index  golden_tmp_sort_idex =================== \n\n"); // 1.46% error
    // // (void)resultCmp(golden_tmp_sort_idex,  (int32_t *)res_tmp_sort_index->data(), 0);
    // (void)resultCmp(golden_tmp_index,  (int32_t *)res_tmp_index->data(), 0);

    printf("\n\n=================== golden_pix_to_face ===================\n\n");// 1.46% error
    (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    printf("\n\n=================== golden_zbuf ===================\n\n");
    (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    printf("\n\n=================== golden_bary_coords ===================\n\n");  // 1.46% error
    // (void)resultCmp(golden_bary_coords, (float *)res_bary_coords->data(), 0.001f); //OK
    (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    // for (int i = 0; i < cap_zbuf; i++) {
    //     int32_t index = golden_tmp_index[i];
    //     printf("Min Index[%d]:   golden: %d, res: %d \n", i, golden_tmp_index[i], *((int32_t *)res_tmp_index->data() + i) );
    //     printf("barycoords[%d]:  golden: %f, %f, %f, res: %f, %f, %f, ori:  %f, %f, %f,\n", i, 
    //                             golden_barycoords[i*3], golden_barycoords[i*3 + 1], golden_barycoords[i*3+2],
    //                             *((float *)res_barycoords->data() + i*3), *((float *)res_barycoords->data() + i*3 + 1),*((float *)res_barycoords->data() + i*3 +2),
    //                             golden_bary_coords[i*face_num*3 + index*3], golden_bary_coords[i*face_num*3 + index*3 + 1], golden_bary_coords[i*face_num*3 + index*3 + 2]);
    // }
    printf("\n\n=================== golden_dists ===================\n\n");
    (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    // for (int i = 0; i < cap_zbuf; i++) {
    //     printf("Min Index[%d]:   golden: %d, res: %d \n", i, golden_tmp_index[i], *((int32_t *)res_tmp_index->data() + i) );
    //     printf("pix_to_face[%d]: golden: %d, res: %d \n", i, golden_pix_to_face[i], *((int32_t *)res_pix_to_face->data() + i));
    //     printf("zbuf[%d]:        golden: %f, res: %f \n", i, golden_zbuf[i], *((float *)res_zbuf->data() + i));
    //     printf("dists[%d]:       golden: %f, res: %f \n", i, golden_dists[i], *((float *)res_dists->data() + i));
    // }

    int ret = true;
    EXPECT_EQ(ret, true);
}


void RunRasterize_bak(string path, int point_size_x, int point_size_y, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    std::string bin_path = path;
    int face_num = tri_num;
    int point_num = point_size_x * point_size_y;

    std::string func_name = "rasterize_" + std::to_string(point_size_x) + "_" + std::to_string(point_size_y) + std::to_string(face_num);
    std::string loop_name = func_name + "_loop";
    printf("out : %s, %s", func_name.c_str(), loop_name.c_str());

    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    std::vector<int64_t> shape_point        = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};
    std::vector<int64_t> shape_face_idx     = {face_num};


    std::vector<int64_t> shape_bary_coords      = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_point_per_frag   = {point_num, face_num};

    std::vector<int64_t> shape_pix_to_face  = {point_num, 1};
    std::vector<int64_t> shape_zbuf         = {point_num, 1};
    std::vector<int64_t> shape_dists        = {point_num, 1};
    std::vector<int64_t> shape_barycoords   = {point_num, 1, coord_xyz};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());
    int cap_face_idx  = std::accumulate(shape_face_idx.begin(), shape_face_idx.end(), 1, std::multiplies<int>());

    int cap_shape_point_per_frag  = std::accumulate(shape_point_per_frag.begin(), shape_point_per_frag.end(), 1, std::multiplies<int>());

    int cap_bary_coords  = std::accumulate(shape_bary_coords.begin(), shape_bary_coords.end(), 1, std::multiplies<int>());
    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);
    std::vector<int32_t> input_face_idx(cap_face_idx, 0);
    std::vector<float>   input_scale(1, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_bary_coords(cap_bary_coords, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_z_frags(cap_shape_point_per_frag, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    std::vector<int8_t>  golden_tmp_should_write(cap_shape_point_per_frag, 0);
    std::vector<int8_t>  golden_tmp_inside(cap_shape_point_per_frag, 0);
    std::vector<float>   golden_tmp_face_sd2(cap_shape_point_per_frag, 0);
    std::vector<float>   golden_tmp_distance(cap_shape_point_per_frag, 0);
    std::vector<int32_t> golden_tmp_sort_idex(cap_shape_point_per_frag, 0);
    std::vector<int32_t> golden_tmp_index(cap_zbuf, 0);

    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/point"      + bin_path, input_point);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_verts" + bin_path, input_face_verts);
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_idx" + bin_path,   input_face_idx);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/scale"    + bin_path,   input_scale);

    Tensor point(DataType::DT_FP32,      shape_point,      "I_1");
    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_2");
    Tensor face_idx(DataType::DT_INT32,  shape_face_idx,   "I_3");
    Tensor scale(DataType::DT_FP32,      {1},              "I_4");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords, "O_6");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, input_point),
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<int32_t>(face_idx, input_face_idx),
        RawTensorData::CreateTensor<float>(scale, input_scale),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(barycoords, golden_barycoords),
    });

    //共用golden
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/pix_to_face" + bin_path, golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/zbuf"  + bin_path,       golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/dists" + bin_path,       golden_dists);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/barycentrics" + bin_path,golden_barycoords);

    // float INF_Z = INFINITY;
    PROGRAM("Rasterize") {
        TileShape::Current().SetVecTile({32, 32, 8});
        FUNCTION(func_name.c_str(), {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, dists, barycoords}) {
            LOOP(loop_name.c_str(), FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);

                auto tri_v0_xy = View(face_verts, {face_verts.GetShape()[0], 1, 2}, {0, 0, 0});// (#faces,1, 2)
                auto tri_v1_xy = View(face_verts, {face_verts.GetShape()[0], 1, 2}, {0, 1, 0});// (#faces,1, 2)
                auto tri_v2_xy = View(face_verts, {face_verts.GetShape()[0], 1, 2}, {0, 2, 0});// (#faces,1, 2)

                auto tri_v0_xy_sq = Squeeze(tri_v0_xy, -2);// (#faces, 2)
                auto tri_v1_xy_sq = Squeeze(tri_v1_xy, -2);// (#faces, 2)
                auto tri_v2_xy_sq = Squeeze(tri_v2_xy, -2);// (#faces, 2)

                // BarycentricCoordsNoperspective
                auto v01 = Sub(tri_v1_xy_sq, tri_v0_xy_sq); //(#faces, 2)
                auto v02 = Sub(tri_v2_xy_sq, tri_v0_xy_sq); //(#faces, 2)
                auto v0p = Sub(Unsqueeze(point, -2), tri_v0_xy_sq); //(#p, #faces, 2)

                auto d00 = Dot(v01, v01, -1);//(#faces)
                auto d01 = Dot(v01, v02, -1);//(#faces)
                auto d11 = Dot(v02, v02, -1);//(#faces)
                auto d20 = Dot(v0p, Unsqueeze(v01, 0), -1);//(#p, #faces)
                auto d21 = Dot(v0p, Unsqueeze(v02, 0), -1);//(#p, #faces)

                auto denom = Sub(Unsqueeze(Mul(d00, d11), 0), Unsqueeze(Mul(d01, d01), 0));//# (1, #faces)
                auto v = Div(Sub(Mul(Unsqueeze(d11, 0), d20), Mul(Unsqueeze(d01, 0), d21)), denom); //# (#p, #faces)
                auto w = Div(Sub(Mul(Unsqueeze(d00, 0), d21), Mul(Unsqueeze(d01, 0), d20)), denom); //# (#p, #faces)
                auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)
                auto v_Unsque = Unsqueeze(v, -1); //# (#p, #faces, 1)
                auto w_Unsque = Unsqueeze(w, -1); //# (#p, #faces, 1)
                auto u_Unsque = Unsqueeze(u, -1); //# (#p, #faces, 1)

                // auto bary_coords_correct = BarycentricCoordsPespectCorrect(barycentrics, tri_v0_xyz_sq, tri_v1_xyz_sq, tri_v2_xyz_sq);

                // TriangleSigned SquaredDistance
                auto square_distance = TriangleSquaredDistance(point,  tri_v0_xy_sq,  tri_v1_xy_sq,  tri_v2_xy_sq,  scale); // (#point, #faces)
                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));

                auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                auto face_v0_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2}); // (#faces, 1, 1)
                auto face_v1_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 2}); // (#faces, 1, 1)
                auto face_v2_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 2}); // (#faces, 1, 1)
                auto face_v0_z_sq = Squeeze(face_v0_z, -1);// (#faces, 1)
                auto face_v1_z_sq = Squeeze(face_v1_z, -1);// (#faces, 1)
                auto face_v2_z_sq = Squeeze(face_v2_z, -1);// (#faces, 1)
                auto z_frags = Add(Add(Mul(u_Unsque, face_v0_z_sq), Mul(v_Unsque, face_v1_z_sq)), Mul(w_Unsque, face_v2_z_sq));  // (#point, #faces, 1)
                auto z_frags_sq = Squeeze(z_frags, -1);    // (#point, #faces)

                // // auto z_frags = Sum(Mul(barycentrics, face_verts_z), -1);// (#point, #faces, 1)  // device执行失败，
                // auto face_verts_z = View(face_verts, {face_verts.GetShape()[0], face_verts.GetShape()[1], 1}, {0, 0, 2});// (#faces, 3, 1)
                // auto face_verts_z_sq = Squeeze(face_verts_z, -1);// (#faces, 3)
                // auto z_frags = Sum(Mul(barycentrics, Unsqueeze(face_verts_z_sq, 0)), -1, true);// (#point, #faces, 1)
                // auto z_frags_sq = Squeeze(z_frags, -1);// (#point, #faces)

                // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                Element blur_radius (DataType::DT_FP32, 0.0f);
                auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                TileShape::Current().SetVecTile({4, 128, 8}); // 64, 976

                // auto output_topk1 = TopK(zbuf_1, 1, -1, false);
                // auto min_idx = std::get<1>(output_topk1);
                // zbuf = std::get<0>(output_topk1);

                auto sorted_idx = ArgSort(zbuf_1, -1, false);
                Element ele_zero(DataType::DT_INT32, 0.0f);
                Tensor zero_index = Full(ele_zero, DT_INT32, {point_num, 1});               // (#point, 1)
                auto min_idx = GatherElements(sorted_idx, zero_index, -1);                  // (#point, 1)

                auto face_idx_unsq = Unsqueeze(face_idx, 0);                                // (1, #faces)
                auto pix_to_face_expand = Expand(face_idx_unsq, {point_num, face_num});     // (#point, #faces)
                pix_to_face = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                zbuf  = GatherElements(zbuf_1, min_idx, -1);                                // (#point, #1)
                dists = GatherElements(face_sd2, min_idx, -1);                              // (#point, #1)

                // auto barycentrics= Cat({u_Unsque, v_Unsque, w_Unsque}, -1);              //# (#p, #faces, 3)
                // std::vector<int64_t> idxShape(min_idx.GetStorage()->shape);
                // idxShape.insert(idxShape.end(), 3);
                // auto idx_bary_coords = Expand(Unsqueeze(min_idx, -1), idxShape);         // (#point, #1, 3)
                // barycoords = GatherElements(barycentrics, idx_bary_coords, -2);          // (#point, #1, 3)

                auto barycoords_u = GatherElements(u, min_idx, -1);                         // (#point, #1)
                auto barycoords_v = GatherElements(v, min_idx, -1);                         // (#point, #1)
                auto barycoords_w = GatherElements(w, min_idx, -1);                         // (#point, #1)
                barycoords= Cat({Unsqueeze(barycoords_u, -1), Unsqueeze(barycoords_v, -1), Unsqueeze(barycoords_w, -1)}, -1);//# (#p, #faces, 3)
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);

    printf("\n\n=================== golden_pix_to_face ===================\n\n");
    (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    printf("\n\n=================== golden_zbuf ===================\n\n");
    (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    printf("\n\n=================== golden_bary_coords ===================\n\n");
    (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    printf("\n\n=================== golden_dists ===================\n\n");
    (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    int ret = true;
    EXPECT_EQ(ret, true);
}



void RunRasterize_OK(string path, int point_size_x, int point_size_y, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    std::string bin_path = path;
    int face_num = tri_num;
    int point_num = point_size_x * point_size_y;

    std::string func_name = "rasterize_" + std::to_string(point_size_x) + "_" + std::to_string(point_size_y) + "_" + std::to_string(face_num);
    std::string loop_name = func_name + "_loop";
    std::string loop_name2 = func_name + "_loop2";

    printf("out : %s, %s, %s", func_name.c_str(), loop_name.c_str(), loop_name2.c_str());

    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    std::vector<int64_t> shape_point        = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};
    std::vector<int64_t> shape_face_idx     = {face_num};


    std::vector<int64_t> shape_bary_coords      = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_point_per_frag   = {point_num, face_num};

    std::vector<int64_t> shape_pix_to_face  = {point_num, 1};
    std::vector<int64_t> shape_zbuf         = {point_num, 1};
    std::vector<int64_t> shape_dists        = {point_num, 1};
    std::vector<int64_t> shape_barycoords   = {point_num, 1, coord_xyz};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());
    int cap_face_idx  = std::accumulate(shape_face_idx.begin(), shape_face_idx.end(), 1, std::multiplies<int>());

    // int cap_shape_point_per_frag  = std::accumulate(shape_point_per_frag.begin(), shape_point_per_frag.end(), 1, std::multiplies<int>());

    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);
    std::vector<int32_t> input_face_idx(cap_face_idx, 0);
    std::vector<float>   input_scale(1, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/point"      + bin_path, input_point);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_verts" + bin_path, input_face_verts);
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_idx" + bin_path,   input_face_idx);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/scale"    + bin_path,   input_scale);

    Tensor point(DataType::DT_FP32,      shape_point,      "I_1");
    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_2");
    Tensor face_idx(DataType::DT_INT32,  shape_face_idx,   "I_3");
    Tensor scale(DataType::DT_FP32,      {1},              "I_4");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords,  "O_6");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, input_point),
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<int32_t>(face_idx, input_face_idx),
        RawTensorData::CreateTensor<float>(scale, input_scale),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(barycoords, golden_barycoords),
    });

    //共用golden
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/pix_to_face" + bin_path, golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/zbuf"  + bin_path,       golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/dists" + bin_path,       golden_dists);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/barycentrics" + bin_path,golden_barycoords);

    // float INF_Z = INFINITY;
    PROGRAM("Rasterize") {
        TileShape::Current().SetVecTile({128, 64, 8});
        FUNCTION(func_name.c_str(), {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, dists, barycoords}) {

            Tensor zbuf_ori(DT_FP32, shape_point_per_frag, "Zbuf_o");
            Tensor face_sd2_ori(DT_FP32, shape_point_per_frag, "Face_sd2");
            Tensor barycentrics_u(DT_FP32, shape_point_per_frag, "Barycentrics_u");
            Tensor barycentrics_v(DT_FP32, shape_point_per_frag, "Barycentrics_v");
            Tensor barycentrics_w(DT_FP32, shape_point_per_frag, "Barycentrics_w");

            LOOP(loop_name.c_str(), FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);

                auto tri_v0_x = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 0});// (#faces,1, 1)
                auto tri_v0_y = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 1});// (#faces,1, 1)
                auto tri_v0_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2});// (#faces,1, 1)

                auto tri_v1_x = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 0});// (#faces,1, 1)
                auto tri_v1_y = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 1});// (#faces,1, 1)
                auto tri_v1_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 2});// (#faces,1, 1)

                auto tri_v2_x = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 0});// (#faces,1, 1)
                auto tri_v2_y = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 1});// (#faces,1, 1)
                auto tri_v2_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 2});// (#faces,1, 1)

                auto point_x = View(point, {point.GetShape()[0], 1}, {0, 0});// (#point,1)
                auto point_y = View(point, {point.GetShape()[0], 1}, {0, 1});// (#point,1)

                auto tri_v0_x_re = Reshape(tri_v0_x, {face_verts.GetShape()[0]});
                auto tri_v0_y_re = Reshape(tri_v0_y, {face_verts.GetShape()[0]});
                auto tri_v0_z_re = Reshape(tri_v0_z, {face_verts.GetShape()[0]});

                auto tri_v1_x_re = Reshape(tri_v1_x, {face_verts.GetShape()[0]});
                auto tri_v1_y_re = Reshape(tri_v1_y, {face_verts.GetShape()[0]});
                auto tri_v1_z_re = Reshape(tri_v1_z, {face_verts.GetShape()[0]});

                auto tri_v2_x_re = Reshape(tri_v2_x, {face_verts.GetShape()[0]});
                auto tri_v2_y_re = Reshape(tri_v2_y, {face_verts.GetShape()[0]});
                auto tri_v2_z_re = Reshape(tri_v2_z, {face_verts.GetShape()[0]});

                auto point_x_re = Reshape(point_x,  {point.GetShape()[0]});// (#point)
                auto point_y_re = Reshape(point_y,  {point.GetShape()[0]});// (#point)

                // BarycentricCoordsNoperspective
                auto v01_x = Sub(tri_v1_x_re, tri_v0_x_re); //(#faces)
                auto v01_y = Sub(tri_v1_y_re, tri_v0_y_re); //(#faces)

                auto v02_x = Sub(tri_v2_x_re, tri_v0_x_re); //(#faces)
                auto v02_y = Sub(tri_v2_y_re, tri_v0_y_re); //(#faces)

                auto point_x_expand  = Expand(point_x, {point.GetShape()[0], face_verts.GetShape()[0]});//(#p, #faces)
                auto point_y_expand  = Expand(point_y, {point.GetShape()[0], face_verts.GetShape()[0]});//(#p, #faces)

                auto v0p_x = Sub(point_x_expand, Unsqueeze(tri_v0_x_re, 0)); //(#p, #faces)
                auto v0p_y = Sub(point_y_expand, Unsqueeze(tri_v0_y_re, 0)); //(#p, #faces)

                auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));//(#faces)
                auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));//(#faces)
                auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));//(#faces)

                auto d20 = Add(Mul(v0p_x, Unsqueeze(v01_x,0)), Mul(v0p_y, Unsqueeze(v01_y,0)));//(#p, #faces)
                auto d21 = Add(Mul(v0p_x, Unsqueeze(v02_x,0)), Mul(v0p_y, Unsqueeze(v02_y,0)));//(#p, #faces)

                auto denom = Sub(Unsqueeze(Mul(d00, d11), 0), Unsqueeze(Mul(d01, d01), 0));//# (1, #faces)
                auto v = Div(Sub(Mul(Unsqueeze(d11, 0), d20), Mul(Unsqueeze(d01, 0), d21)), denom); //# (#p, #faces)
                auto w = Div(Sub(Mul(Unsqueeze(d00, 0), d21), Mul(Unsqueeze(d01, 0), d20)), denom); //# (#p, #faces)
                auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

                // auto bary_coords_correct = BarycentricCoordsPespectCorrect(barycentrics, tri_v0_xyz_sq, tri_v1_xyz_sq, tri_v2_xyz_sq);

                // TriangleSigned SquaredDistance
                auto square_distance = TriangleSquaredDistanceSingle(point_x_re, point_y_re, 
                                        tri_v0_x_re,  tri_v0_y_re, tri_v1_x_re, tri_v1_y_re, tri_v2_x_re, tri_v2_y_re, scale); // (#point, #faces)

                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
                auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                auto z_frags_sq = Add(Add(Mul(u, tri_v0_z_re), Mul(v, tri_v1_z_re)), Mul(w, tri_v2_z_re));  // (#point, #faces)

                // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                Element blur_radius (DataType::DT_FP32, 0.0f);
                auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                barycentrics_u = u;
                barycentrics_v = v;
                barycentrics_w = w;
                zbuf_ori = zbuf_1;
                face_sd2_ori = face_sd2;
            }
            LOOP(loop_name2.c_str(), FunctionType::DYNAMIC_LOOP, idx2, LoopRange(1)) {
                UNUSED(idx2);
                // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                TileShape::Current().SetVecTile({16, 128, 8}); // 64, 976

                auto output_topk1 = TopK(zbuf_ori, 1, -1, false);
                auto min_idx = std::get<1>(output_topk1);
                zbuf = std::get<0>(output_topk1);

                auto face_idx_unsq = Unsqueeze(face_idx, 0);                                // (1, #faces)
                auto pix_to_face_expand = Expand(face_idx_unsq, {point_num, face_num});     // (#point, #faces)
                pix_to_face = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                dists = GatherElements(face_sd2_ori, min_idx, -1);                           // (#point, #1)

                auto barycoords_u = GatherElements(barycentrics_u, min_idx, -1);             // (#point, #1)
                auto barycoords_v = GatherElements(barycentrics_v, min_idx, -1);             // (#point, #1)
                auto barycoords_w = GatherElements(barycentrics_w, min_idx, -1);             // (#point, #1)
                barycoords= Cat({Unsqueeze(barycoords_u, -1), Unsqueeze(barycoords_v, -1), Unsqueeze(barycoords_w, -1)}, -1);//# (#p, 1, 3)
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);

    printf("\n\n=================== golden_pix_to_face ===================\n\n");
    (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    printf("\n\n=================== golden_zbuf ===================\n\n");
    (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    printf("\n\n=================== golden_bary_coords ===================\n\n");
    (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    printf("\n\n=================== golden_dists ===================\n\n");
    (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    int ret = true;
    EXPECT_EQ(ret, true);
}

void RunRasterize_OK2(string path, int point_size_x, int point_size_y, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    std::string bin_path = path;
    int face_num = tri_num;
    int point_num = point_size_x * point_size_y;

    std::string func_name = "rasterize_" + std::to_string(point_size_x) + "_" + std::to_string(point_size_y) + "_" + std::to_string(face_num);
    std::string loop_name = func_name + "_loop";
    std::string loop_name2 = func_name + "_loop2";
    std::string loop_name3 = func_name + "_loop3";

    printf("out : %s, %s, %s", func_name.c_str(), loop_name.c_str(), loop_name2.c_str());

    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    std::vector<int64_t> shape_point        = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};
    std::vector<int64_t> shape_face_idx     = {face_num};


    std::vector<int64_t> shape_bary_coords      = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_point_per_frag   = {point_num, face_num};

    std::vector<int64_t> shape_pix_to_face  = {point_num, 1};
    std::vector<int64_t> shape_zbuf         = {point_num, 1};
    std::vector<int64_t> shape_dists        = {point_num, 1};
    std::vector<int64_t> shape_barycoords   = {point_num, 1, coord_xyz};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());
    int cap_face_idx  = std::accumulate(shape_face_idx.begin(), shape_face_idx.end(), 1, std::multiplies<int>());

    // int cap_shape_point_per_frag  = std::accumulate(shape_point_per_frag.begin(), shape_point_per_frag.end(), 1, std::multiplies<int>());

    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);
    std::vector<int32_t> input_face_idx(cap_face_idx, 0);
    std::vector<float>   input_scale(1, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/point"      + bin_path, input_point);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_verts" + bin_path, input_face_verts);
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_idx" + bin_path,   input_face_idx);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/scale"    + bin_path,   input_scale);

    Tensor point(DataType::DT_FP32,      shape_point,      "I_1");
    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_2");
    Tensor face_idx(DataType::DT_INT32,  shape_face_idx,   "I_3");
    Tensor scale(DataType::DT_FP32,      {1},              "I_4");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords,  "O_6");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, input_point),
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<int32_t>(face_idx, input_face_idx),
        RawTensorData::CreateTensor<float>(scale, input_scale),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(barycoords, golden_barycoords),
    });

    //共用golden
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/pix_to_face" + bin_path, golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/zbuf"  + bin_path,       golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/dists" + bin_path,       golden_dists);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/barycentrics" + bin_path,golden_barycoords);

    // int tile0_shape0 = 64; // * 1024 / 4 / face_num / 8 * 8;
    // int tile1_shape0 = 32; // * 1024 / 4 / face_num / 8 * 8;

    // printf("-------tile0_shape0: %d ,tile1_shape0 %d---------", tile0_shape0, tile1_shape0);
    // float INF_Z = INFINITY;
    PROGRAM("Rasterize") {
        FUNCTION(func_name.c_str(), {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, dists, barycoords}) {
            TileShape::Current().SetVecTile({128, 128, 8});

            Tensor zbuf_ori(DT_FP32, shape_point_per_frag, "Zbuf_o");
            Tensor face_sd2_ori(DT_FP32, shape_point_per_frag, "Face_sd2");
            Tensor barycentrics_u(DT_FP32, shape_point_per_frag, "Barycentrics_u");
            Tensor barycentrics_v(DT_FP32, shape_point_per_frag, "Barycentrics_v");
            Tensor barycentrics_w(DT_FP32, shape_point_per_frag, "Barycentrics_w");

            Tensor barycoords_u(DT_FP32,  {point_num, 1}, "barycoords_u");
            Tensor barycoords_v(DT_FP32,  {point_num, 1}, "barycoords_v");
            Tensor barycoords_w(DT_FP32,  {point_num, 1}, "barycoords_w");

            LOOP(loop_name.c_str(), FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);

                auto tri_v0_x = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 0}), {1, face_verts.GetShape()[0]});// (1, #faces)
                auto tri_v0_y = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 1}), {1, face_verts.GetShape()[0]});// (1, #faces)
                auto tri_v0_z = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2}), {1, face_verts.GetShape()[0]});// (1, #faces)

                auto tri_v1_x = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 0}), {1, face_verts.GetShape()[0]});// (1, #faces)
                auto tri_v1_y = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 1}), {1, face_verts.GetShape()[0]});// (1, #faces)
                auto tri_v1_z = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 2}), {1, face_verts.GetShape()[0]});// (1, #faces)

                auto tri_v2_x = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 0}), {1, face_verts.GetShape()[0]});// (1, #faces)
                auto tri_v2_y = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 1}), {1, face_verts.GetShape()[0]});// (1, #faces)
                auto tri_v2_z = Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 2}), {1, face_verts.GetShape()[0]});// (1, #faces)

                auto point_x = Expand(View(point, {point.GetShape()[0], 1}, {0, 0}), {point.GetShape()[0], face_verts.GetShape()[0]});// (#point, #faces)
                auto point_y = Expand(View(point, {point.GetShape()[0], 1}, {0, 1}), {point.GetShape()[0], face_verts.GetShape()[0]});// (#point, #faces)

                // BarycentricCoordsNoperspective
                auto v01_x = Sub(tri_v1_x, tri_v0_x); // (1, #faces)
                auto v01_y = Sub(tri_v1_y, tri_v0_y); // (1, #faces)

                auto v02_x = Sub(tri_v2_x, tri_v0_x); // (1, #faces)
                auto v02_y = Sub(tri_v2_y, tri_v0_y); // (1, #faces)

                auto v0p_x = Sub(point_x, tri_v0_x); //(#p, #faces)
                auto v0p_y = Sub(point_y, tri_v0_y); //(#p, #faces)

                auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));// (1, #faces)
                auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));// (1, #faces)
                auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));// (1, #faces)

                auto d20 = Add(Mul(v0p_x, v01_x), Mul(v0p_y, v01_y));//(#p, #faces)
                auto d21 = Add(Mul(v0p_x, v02_x), Mul(v0p_y, v02_y));//(#p, #faces)

                auto denom = Sub(Mul(d00, d11), Mul(d01, d01));         //(1, #faces)
                auto v = Div(Sub(Mul(d11, d20), Mul(d01, d21)), denom); //(#p, #faces)
                auto w = Div(Sub(Mul(d00, d21), Mul(d01, d20)), denom); //(#p, #faces)
                auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

                // TriangleSigned SquaredDistance
                auto square_distance = TriangleSquaredDistanceSingle2(point_x, point_y, 
                                        tri_v0_x,  tri_v0_y, tri_v1_x, tri_v1_y, tri_v2_x, tri_v2_y, scale); // (#point, #faces)


                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
                auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                auto z_frags_sq = Add(Add(Mul(u, tri_v0_z), Mul(v, tri_v1_z)), Mul(w, tri_v2_z));  // (#point, #faces)

                // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                Element blur_radius (DataType::DT_FP32, 0.0f);
                auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                barycentrics_u = u;
                barycentrics_v = v;
                barycentrics_w = w;
                zbuf_ori = zbuf_1;
                face_sd2_ori = face_sd2;
            }
            LOOP(loop_name2.c_str(), FunctionType::DYNAMIC_LOOP, idx2, LoopRange(1)) {
                UNUSED(idx2);
                // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                TileShape::Current().SetVecTile({32, 128});

                auto output_topk1 = TopK(zbuf_ori, 1, -1, false);
                auto min_idx = std::get<1>(output_topk1);
                zbuf = std::get<0>(output_topk1);

                auto face_idx_unsq = Unsqueeze(face_idx, 0);                                // (1, #faces)
                auto pix_to_face_expand = Expand(face_idx_unsq, {point_num, face_num});     // (#point, #faces)
                pix_to_face = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                dists = GatherElements(face_sd2_ori, min_idx, -1);                          // (#point, #1)

                barycoords_u = GatherElements(barycentrics_u, min_idx, -1);            // (#point, #1)
                barycoords_v = GatherElements(barycentrics_v, min_idx, -1);            // (#point, #1)
                barycoords_w = GatherElements(barycentrics_w, min_idx, -1);            // (#point, #1)
            }
            LOOP(loop_name3.c_str(), FunctionType::DYNAMIC_LOOP, idx3, LoopRange(1)) {
                UNUSED(idx3);
                TileShape::Current().SetVecTile({256, 1, 8});
                barycoords= Cat({Unsqueeze(barycoords_u, -1), Unsqueeze(barycoords_v, -1), Unsqueeze(barycoords_w, -1)}, -1);//# (#p, 1, 3)
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);

    printf("\n\n=================== golden_pix_to_face ===================\n\n");
    (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    printf("\n\n=================== golden_zbuf ===================\n\n");
    (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    printf("\n\n=================== golden_bary_coords ===================\n\n");
    (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    printf("\n\n=================== golden_dists ===================\n\n");
    (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    int ret = true;
    EXPECT_EQ(ret, true);
}


void RunRasterize_OK3(string path, int point_size_x, int point_size_y, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    std::string bin_path = path;
    int face_num = tri_num;
    int point_num = point_size_x * point_size_y;

    std::string func_name = "rasterize_" + std::to_string(point_size_x) + "_" + std::to_string(point_size_y) + "_" + std::to_string(face_num);
    std::string loop_name = func_name + "_loop";
    std::string loop_name1 = func_name + "_loop1";
    std::string loop_name2 = func_name + "_loop2";
    std::string loop_name3 = func_name + "_loop3";

    printf("out : %s, %s, %s", func_name.c_str(), loop_name.c_str(), loop_name2.c_str());

    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    std::vector<int64_t> shape_point        = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};
    std::vector<int64_t> shape_face_idx     = {face_num};


    std::vector<int64_t> shape_bary_coords      = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_point_per_frag   = {point_num, face_num};

    std::vector<int64_t> shape_pix_to_face  = {point_num, 1};
    std::vector<int64_t> shape_zbuf         = {point_num, 1};
    std::vector<int64_t> shape_dists        = {point_num, 1};
    std::vector<int64_t> shape_barycoords   = {point_num, 1, coord_xyz};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());
    int cap_face_idx  = std::accumulate(shape_face_idx.begin(), shape_face_idx.end(), 1, std::multiplies<int>());

    // int cap_shape_point_per_frag  = std::accumulate(shape_point_per_frag.begin(), shape_point_per_frag.end(), 1, std::multiplies<int>());

    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);
    std::vector<int32_t> input_face_idx(cap_face_idx, 0);
    std::vector<float>   input_scale(1, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/point"      + bin_path, input_point);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_verts" + bin_path, input_face_verts);
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_idx" + bin_path,   input_face_idx);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/scale"    + bin_path,   input_scale);

    Tensor point(DataType::DT_FP32,      shape_point,      "I_1");
    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_2");
    Tensor face_idx(DataType::DT_INT32,  shape_face_idx,   "I_3");
    Tensor scale(DataType::DT_FP32,      {1},              "I_4");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords,  "O_6");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, input_point),
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<int32_t>(face_idx, input_face_idx),
        RawTensorData::CreateTensor<float>(scale, input_scale),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(barycoords, golden_barycoords),
    });

    //共用golden
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/pix_to_face" + bin_path, golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/zbuf"  + bin_path,       golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/dists" + bin_path,       golden_dists);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/barycentrics" + bin_path,golden_barycoords);

    // int tile0_shape0 = 64; // * 1024 / 4 / face_num / 8 * 8;
    // int tile1_shape0 = 32; // * 1024 / 4 / face_num / 8 * 8;

    // printf("-------tile0_shape0: %d ,tile1_shape0 %d---------", tile0_shape0, tile1_shape0);
    // float INF_Z = INFINITY;
    PROGRAM("Rasterize") {
        FUNCTION(func_name.c_str(), {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, dists, barycoords}) {

            Tensor tri_v0_x(DT_FP32, {1, face_num}, "tri_v0_x");
            Tensor tri_v0_y(DT_FP32, {1, face_num}, "tri_v0_y");
            Tensor tri_v0_z(DT_FP32, {1, face_num}, "tri_v0_z");

            Tensor tri_v1_x(DT_FP32, {1, face_num}, "tri_v1_x");
            Tensor tri_v1_y(DT_FP32, {1, face_num}, "tri_v1_y");
            Tensor tri_v1_z(DT_FP32, {1, face_num}, "tri_v1_z");
            Tensor tri_v2_x(DT_FP32, {1, face_num}, "tri_v2_x");
            Tensor tri_v2_y(DT_FP32, {1, face_num}, "tri_v2_y");
            Tensor tri_v2_z(DT_FP32, {1, face_num}, "tri_v2_z");

            Tensor point_x(DT_FP32, {point_num, face_num}, "point_x");
            Tensor point_y(DT_FP32, {point_num, face_num}, "point_y");

            Tensor zbuf_ori(DT_FP32, shape_point_per_frag, "Zbuf_o");
            Tensor face_sd2_ori(DT_FP32, shape_point_per_frag, "Face_sd2");
            Tensor barycentrics_u(DT_FP32, shape_point_per_frag, "Barycentrics_u");
            Tensor barycentrics_v(DT_FP32, shape_point_per_frag, "Barycentrics_v");
            Tensor barycentrics_w(DT_FP32, shape_point_per_frag, "Barycentrics_w");

            Tensor barycoords_u(DT_FP32,  {point_num, 1}, "barycoords_u");
            Tensor barycoords_v(DT_FP32,  {point_num, 1}, "barycoords_v");
            Tensor barycoords_w(DT_FP32,  {point_num, 1}, "barycoords_w");

            LOOP(loop_name.c_str(), FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);
                TileShape::Current().SetVecTile({128, 128, 8});

                tri_v0_x = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 0}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)
                tri_v0_y = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 1}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)
                tri_v0_z = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)

                tri_v1_x = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 0}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)
                tri_v1_y = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 1}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)
                tri_v1_z = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 2}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)

                tri_v2_x = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 0}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)
                tri_v2_y = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 1}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)
                tri_v2_z = ScalarAddS(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 2}), {1, face_verts.GetShape()[0]}), Element(DataType::DT_INT32, 0.0f));// (1, #faces)

                point_x = Expand(View(point, {point.GetShape()[0], 1}, {0, 0}), {point.GetShape()[0], face_verts.GetShape()[0]});// (#point, #faces)
                point_y = Expand(View(point, {point.GetShape()[0], 1}, {0, 1}), {point.GetShape()[0], face_verts.GetShape()[0]});// (#point, #faces)
            }
            LOOP(loop_name1.c_str(), FunctionType::DYNAMIC_LOOP, idx1, LoopRange(1)) {
                UNUSED(idx1);
                TileShape::Current().SetVecTile({128, 128});

                // BarycentricCoordsNoperspective
                auto v01_x = Sub(tri_v1_x, tri_v0_x); // (1, #faces)
                auto v01_y = Sub(tri_v1_y, tri_v0_y); // (1, #faces)

                auto v02_x = Sub(tri_v2_x, tri_v0_x); // (1, #faces)
                auto v02_y = Sub(tri_v2_y, tri_v0_y); // (1, #faces)

                auto v0p_x = Sub(point_x, tri_v0_x); //(#p, #faces)
                auto v0p_y = Sub(point_y, tri_v0_y); //(#p, #faces)

                auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));// (1, #faces)
                auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));// (1, #faces)
                auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));// (1, #faces)

                auto d20 = Add(Mul(v0p_x, v01_x), Mul(v0p_y, v01_y));//(#p, #faces)
                auto d21 = Add(Mul(v0p_x, v02_x), Mul(v0p_y, v02_y));//(#p, #faces)

                auto denom = Sub(Mul(d00, d11), Mul(d01, d01));         //(1, #faces)
                auto v = Div(Sub(Mul(d11, d20), Mul(d01, d21)), denom); //(#p, #faces)
                auto w = Div(Sub(Mul(d00, d21), Mul(d01, d20)), denom); //(#p, #faces)
                auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

                // TriangleSigned SquaredDistance
                auto square_distance = TriangleSquaredDistanceSingle2(point_x, point_y, 
                                        tri_v0_x,  tri_v0_y, tri_v1_x, tri_v1_y, tri_v2_x, tri_v2_y, scale); // (#point, #faces)


                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
                auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                auto z_frags_sq = Add(Add(Mul(u, tri_v0_z), Mul(v, tri_v1_z)), Mul(w, tri_v2_z));  // (#point, #faces)

                // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                Element blur_radius (DataType::DT_FP32, 0.0f);
                auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                barycentrics_u = u;
                barycentrics_v = v;
                barycentrics_w = w;
                zbuf_ori = zbuf_1;
                face_sd2_ori = face_sd2;
            }
            LOOP(loop_name2.c_str(), FunctionType::DYNAMIC_LOOP, idx2, LoopRange(1)) {
                UNUSED(idx2);
                // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                TileShape::Current().SetVecTile({32, 128});

                auto output_topk1 = TopK(zbuf_ori, 1, -1, false);
                auto min_idx = std::get<1>(output_topk1);
                zbuf = std::get<0>(output_topk1);

                auto face_idx_unsq = Unsqueeze(face_idx, 0);                                // (1, #faces)
                auto pix_to_face_expand = Expand(face_idx_unsq, {point_num, face_num});     // (#point, #faces)
                pix_to_face = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                dists = GatherElements(face_sd2_ori, min_idx, -1);                          // (#point, #1)

                barycoords_u = GatherElements(barycentrics_u, min_idx, -1);            // (#point, #1)
                barycoords_v = GatherElements(barycentrics_v, min_idx, -1);            // (#point, #1)
                barycoords_w = GatherElements(barycentrics_w, min_idx, -1);            // (#point, #1)
            }
            LOOP(loop_name3.c_str(), FunctionType::DYNAMIC_LOOP, idx3, LoopRange(1)) {
                UNUSED(idx3);
                TileShape::Current().SetVecTile({256, 1, 8});
                barycoords= Cat({Unsqueeze(barycoords_u, -1), Unsqueeze(barycoords_v, -1), Unsqueeze(barycoords_w, -1)}, -1);//# (#p, 1, 3)
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);

    printf("\n\n=================== golden_pix_to_face ===================\n\n");
    (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    printf("\n\n=================== golden_zbuf ===================\n\n");
    (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    printf("\n\n=================== golden_bary_coords ===================\n\n");
    (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    printf("\n\n=================== golden_dists ===================\n\n");
    (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    int ret = true;
    EXPECT_EQ(ret, true);
}

void RunRasterize(string path, int point_size_x, int point_size_y, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    std::string bin_path = path;
    int face_num = tri_num;
    int point_num = point_size_x * point_size_y;

    std::string func_name = "rasterize_" + std::to_string(point_size_x) + "_" + std::to_string(point_size_y) + "_" + std::to_string(face_num);
    std::string loop_name = func_name + "_loop";
    std::string loop_name2 = func_name + "_loop2";
    std::string loop_name3 = func_name + "_loop3";

    printf("out : %s, %s, %s", func_name.c_str(), loop_name.c_str(), loop_name2.c_str());

    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    std::vector<int64_t> shape_point        = {point_num, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};
    std::vector<int64_t> shape_face_idx     = {face_num};


    std::vector<int64_t> shape_bary_coords      = {point_num, face_num, coord_xyz};
    std::vector<int64_t> shape_point_per_frag   = {point_num, face_num};

    std::vector<int64_t> shape_pix_to_face  = {point_num, 1};
    std::vector<int64_t> shape_zbuf         = {point_num, 1};
    std::vector<int64_t> shape_dists        = {point_num, 1};
    std::vector<int64_t> shape_barycoords   = {point_num, 1, coord_xyz};

    std::vector<int64_t> shape_barycoords_tran   = {coord_xyz, point_num, 1};


    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());
    int cap_face_idx  = std::accumulate(shape_face_idx.begin(), shape_face_idx.end(), 1, std::multiplies<int>());

    // int cap_shape_point_per_frag  = std::accumulate(shape_point_per_frag.begin(), shape_point_per_frag.end(), 1, std::multiplies<int>());

    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);
    std::vector<int32_t> input_face_idx(cap_face_idx, 0);
    std::vector<float>   input_scale(1, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/point"      + bin_path, input_point);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_verts" + bin_path, input_face_verts);
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_idx" + bin_path,   input_face_idx);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/scale"    + bin_path,   input_scale);

    Tensor point(DataType::DT_FP32,      shape_point,      "I_1");
    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_2");
    Tensor face_idx(DataType::DT_INT32,  shape_face_idx,   "I_3");
    Tensor scale(DataType::DT_FP32,      {1},              "I_4");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords,  "O_6");


    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(point, input_point),
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<int32_t>(face_idx, input_face_idx),
        RawTensorData::CreateTensor<float>(scale, input_scale),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(barycoords, golden_barycoords),
    });

    //共用golden
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/pix_to_face" + bin_path, golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/zbuf"  + bin_path,       golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/dists" + bin_path,       golden_dists);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/barycentrics" + bin_path,golden_barycoords);

    // int tile0_shape0 = 64; // * 1024 / 4 / face_num / 8 * 8;
    // int tile1_shape0 = 16; // * 1024 / 4 / face_num / 8 * 8;

    // printf("-------tile0_shape0: %d ,tile1_shape0 %d---------", tile0_shape0, tile1_shape0);
    // float INF_Z = INFINITY;
    PROGRAM("Rasterize") {
        TileShape::Current().SetVecTile({128, 128, 8});
        FUNCTION(func_name.c_str(), {point, face_verts, face_idx, scale}, {pix_to_face, zbuf, dists, barycoords}) {

            Tensor zbuf_ori(DT_FP32, {point_num, face_num}, "Zbuf_o");
            Tensor face_sd2_ori(DT_FP32, {point_num, face_num}, "Face_sd2");
            Tensor barycentrics_u(DT_FP32, {point_num, face_num}, "Barycentrics_u");
            Tensor barycentrics_v(DT_FP32, {point_num, face_num}, "Barycentrics_v");
            Tensor barycentrics_w(DT_FP32, {point_num, face_num}, "Barycentrics_w");
            // Tensor topK_idx(DT_INT32, {point_num, face_num}, "topK_idx");

            Tensor barycoords_u(DT_FP32,  {point_num, 1}, "barycoords_u");
            Tensor barycoords_v(DT_FP32,  {point_num, 1}, "barycoords_v");
            Tensor barycoords_w(DT_FP32,  {point_num, 1}, "barycoords_w");

            LOOP(loop_name.c_str(), FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                UNUSED(idx);

                auto tri_v0_x = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 0}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)
                auto tri_v0_y = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 1}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)
                auto tri_v0_z = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)

                auto tri_v1_x = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 0}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)
                auto tri_v1_y = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 1}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)
                auto tri_v1_z = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 2}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)

                auto tri_v2_x = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 0}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)
                auto tri_v2_y = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 1}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)
                auto tri_v2_z = Expand(Reshape(View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 2}), {1, face_verts.GetShape()[0]}), {point.GetShape()[0], face_verts.GetShape()[0]});// (point, #faces)

                auto point_x = Expand(View(point, {point.GetShape()[0], 1}, {0, 0}), {point.GetShape()[0], face_verts.GetShape()[0]});// (#point, #faces)
                auto point_y = Expand(View(point, {point.GetShape()[0], 1}, {0, 1}), {point.GetShape()[0], face_verts.GetShape()[0]});// (#point, #faces)

                // BarycentricCoordsNoperspective
                auto v01_x = Sub(tri_v1_x, tri_v0_x); // (point, #faces)
                auto v01_y = Sub(tri_v1_y, tri_v0_y); // (point, #faces)

                auto v02_x = Sub(tri_v2_x, tri_v0_x); // (point, #faces)
                auto v02_y = Sub(tri_v2_y, tri_v0_y); // (point, #faces)

                auto v0p_x = Sub(point_x, tri_v0_x); //(#p, #faces)
                auto v0p_y = Sub(point_y, tri_v0_y); //(#p, #faces)

                auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));// (point, #faces)
                auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));// (point, #faces)
                auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));// (point, #faces)

                auto d20 = Add(Mul(v0p_x, v01_x), Mul(v0p_y, v01_y));//(#p, #faces)
                auto d21 = Add(Mul(v0p_x, v02_x), Mul(v0p_y, v02_y));//(#p, #faces)

                auto denom = Sub(Mul(d00, d11), Mul(d01, d01));         //(point, #faces)
                auto v = Div(Sub(Mul(d11, d20), Mul(d01, d21)), denom); //(#p, #faces)
                auto w = Div(Sub(Mul(d00, d21), Mul(d01, d20)), denom); //(#p, #faces)
                auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

                // TriangleSigned SquaredDistance
                auto square_distance = TriangleSquaredDistanceSingle2(point_x, point_y, 
                                        tri_v0_x,  tri_v0_y, tri_v1_x, tri_v1_y, tri_v2_x, tri_v2_y, scale); // (#point, #faces)


                auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
                auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                auto z_frags_sq = Add(Add(Mul(u, tri_v0_z), Mul(v, tri_v1_z)), Mul(w, tri_v2_z));  // (#point, #faces)

                // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                Element blur_radius (DataType::DT_FP32, 0.0f);
                auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                barycentrics_u = u;
                barycentrics_v = v;
                barycentrics_w = w;
                zbuf_ori = zbuf_1;
                face_sd2_ori = face_sd2;
            }
            LOOP(loop_name2.c_str(), FunctionType::DYNAMIC_LOOP, idx2, LoopRange(1)) {
                UNUSED(idx2);
                // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                TileShape::Current().SetVecTile({32, 128});

                auto output_topk1 = TopK(zbuf_ori, 1, -1, false);
                auto min_idx = std::get<1>(output_topk1);
                zbuf = std::get<0>(output_topk1);

                auto face_idx_unsq = Unsqueeze(face_idx, 0);                                // (1, #faces)
                auto pix_to_face_expand = Expand(face_idx_unsq, {point_num, face_num});     // (#point, #faces)
                pix_to_face = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                dists = GatherElements(face_sd2_ori, min_idx, -1);                          // (#point, #1)

                barycoords_u = GatherElements(barycentrics_u, min_idx, -1);            // (#point, #1)
                barycoords_v = GatherElements(barycentrics_v, min_idx, -1);            // (#point, #1)
                barycoords_w = GatherElements(barycentrics_w, min_idx, -1);            // (#point, #1)
            }
            LOOP(loop_name3.c_str(), FunctionType::DYNAMIC_LOOP, idx3, LoopRange(1)) {
                UNUSED(idx3);
                TileShape::Current().SetVecTile({256, 1, 8});
                barycoords= Cat({Unsqueeze(barycoords_u, -1), Unsqueeze(barycoords_v, -1), Unsqueeze(barycoords_w, -1)}, -1);//# (#p, 1, 3)
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);

    printf("\n\n=================== golden_pix_to_face ===================\n\n");
    (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    printf("\n\n=================== golden_zbuf ===================\n\n");
    (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    printf("\n\n=================== golden_bary_coords ===================\n\n");
    (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    printf("\n\n=================== golden_dists ===================\n\n");
    (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    int ret = true;
    EXPECT_EQ(ret, true);
}

struct RenderSettings {
    int image_hight;
    int image_width;
    int max_blend_depth;
    int bin_size;
    float blur_radius_ndc;
    bool clip_barycentric_coords;
    bool cull_backfaces;
    bool persp_correct;
};

void RunRasterizeTotal(RenderSettings rander_setting, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, 3);

    int image_w = rander_setting.image_width;
    int image_h = rander_setting.image_hight;
    int bin_size = rander_setting.bin_size;
    double screen_to_ndc_scale = 1.0 / (std::min(image_w, image_h) * 0.5);

    int bin_w = (image_w + bin_size -1)/bin_size;
    int bin_h = (image_h + bin_size -1)/bin_size;
    int bin_num = bin_w * bin_h;
    int vertex_num = 3;
    int coord_xyz = 3;
    int coord_xy = 2;

    // int64_t point_num = image_w * image_h;
    int64_t face_num = tri_num;

    int64_t point_num_bin = bin_size * bin_size;

    std::vector<int64_t> shape_point        = {image_w, image_h, coord_xy};
    std::vector<int64_t> shape_face_verts   = {face_num, vertex_num, coord_xyz};

    std::vector<int64_t> shape_pix_to_face  = {image_w, image_h, 1};
    std::vector<int64_t> shape_zbuf         = {image_w, image_h, 1};
    std::vector<int64_t> shape_dists        = {image_w, image_h, 1};
    std::vector<int64_t> shape_barycoords   = {image_w, image_h, coord_xyz};

    // DataType dtype = DataType::DT_FP32;
    int cap_point  = std::accumulate(shape_point.begin(), shape_point.end(), 1, std::multiplies<int>());
    int cap_face_verts  = std::accumulate(shape_face_verts.begin(), shape_face_verts.end(), 1, std::multiplies<int>());

    int cap_pix_to_face  = std::accumulate(shape_pix_to_face.begin(), shape_pix_to_face.end(), 1, std::multiplies<int>());
    int cap_zbuf  = std::accumulate(shape_zbuf.begin(), shape_zbuf.end(), 1, std::multiplies<int>());
    int cap_dists  = std::accumulate(shape_dists.begin(), shape_dists.end(), 1, std::multiplies<int>());
    int cap_barycoords  = std::accumulate(shape_barycoords.begin(), shape_barycoords.end(), 1, std::multiplies<int>());

    std::vector<float>   input_point(cap_point, 0);
    std::vector<float>   input_face_verts(cap_face_verts, 0);

    std::vector<int32_t> golden_pix_to_face(cap_pix_to_face, 0);
    std::vector<float>   golden_zbuf(cap_zbuf, 0);
    std::vector<float>   golden_dists(cap_dists, 0);
    std::vector<float>   golden_barycoords(cap_barycoords, 0);

    Tensor face_verts(DataType::DT_FP32, shape_face_verts, "I_0");
    Tensor point_w_h(DataType::DT_FP32,  shape_point,      "I_1");

    Tensor pix_to_face(DataType::DT_INT32, shape_pix_to_face, "O_1");
    Tensor zbuf(DataType::DT_FP32,         shape_zbuf,        "O_2");
    Tensor dists(DataType::DT_FP32,        shape_dists,       "O_4");
    Tensor barycoords(DataType::DT_FP32,   shape_barycoords,  "O_6");

    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/face_verts.bin", input_face_verts);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/point.bin"     , input_point);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(face_verts, input_face_verts),
        RawTensorData::CreateTensor<float>(point_w_h, input_point),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensor<int32_t>(pix_to_face, golden_pix_to_face),
        RawTensorData::CreateTensor<float>(zbuf, golden_zbuf),
        RawTensorData::CreateTensor<float>(dists, golden_dists),
        RawTensorData::CreateTensor<float>(barycoords, golden_barycoords),
    });

    //共用golden
    readInput<int32_t>(GetGoldenDir()+ "/../RasterizeMeshOnBoardTest.test_rasterize" + "/pix_to_face.bin", golden_pix_to_face);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/zbuf.bin" ,       golden_zbuf);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/dists.bin",       golden_dists);
    readInput<float>(GetGoldenDir()  + "/../RasterizeMeshOnBoardTest.test_rasterize" + "/barycentrics.bin",golden_barycoords);

    int tile1_shape0 = 128;
    int tile1_shape1 = 128;
    // int tile1_shape2 = 8;

    int tile2_shape0 = 128;
    int tile2_shape1 = 64;
    // int tile2_shape2 = 8;

    int verts_cnt = 256;
    std::string func_name = "rasterize_1024_64";
    std::string loop_name_bin_h = "rasterize_bin_h";
    std::string loop_name_bin_w = "rasterize_bin_w";
    std::string loop_name_bin_l1 = "rasterize_bin_l1";
    std::string loop_name_bin_l0 = "rasterize_bin_l0";

    UNUSED(tile2_shape0);
    UNUSED(tile2_shape1);
    UNUSED(tile1_shape0);
    UNUSED(tile1_shape1);
    UNUSED(screen_to_ndc_scale);
    UNUSED(point_num_bin);
    UNUSED(verts_cnt);
    UNUSED(bin_num);

    TileShape::Current().SetVecTile({128, 128, 32});

    FUNCTION(func_name.c_str(), {face_verts, point_w_h}, {pix_to_face, zbuf, dists, barycoords}) {
        Tensor binning_mask(DT_BOOL,   {bin_w, bin_h, face_num}, "bin_mask");
        LOOP("bin_mask_gen", FunctionType::DYNAMIC_LOOP, test_idx, LoopRange(1)) {
            UNUSED(test_idx);
            auto tri_v0_x = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 0});// (#faces,1, 1)
            auto tri_v0_y = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 1});// (#faces,1, 1)
            auto tri_v0_z = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 0, 2});// (#faces,1, 1)

            auto tri_v1_x = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 0});// (#faces,1, 1)
            auto tri_v1_y = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 1, 1});// (#faces,1, 1)

            auto tri_v2_x = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 0});// (#faces,1, 1)
            auto tri_v2_y = View(face_verts, {face_verts.GetShape()[0], 1, 1}, {0, 2, 1});// (#faces,1, 1)

            // std::vector<SymbolicScalar> tri_shape = {1, face_verts.GetShape()[0]};
            auto tri_v0_x_re = Reshape(tri_v0_x, {1, face_verts.GetShape()[0]});// (1,#faces)
            auto tri_v0_y_re = Reshape(tri_v0_y, {1, face_verts.GetShape()[0]});// (1,#faces)
            auto tri_v0_z_re = Reshape(tri_v0_z, {1, face_verts.GetShape()[0]});// (1,#faces)

            auto tri_v1_x_re = Reshape(tri_v1_x, {1, face_verts.GetShape()[0]});// (1,#faces)
            auto tri_v1_y_re = Reshape(tri_v1_y, {1, face_verts.GetShape()[0]});// (1,#faces)

            auto tri_v2_x_re = Reshape(tri_v2_x, {1, face_verts.GetShape()[0]});// (1,#faces)
            auto tri_v2_y_re = Reshape(tri_v2_y, {1, face_verts.GetShape()[0]});// (1,#faces)

            auto tri_min_w = Minimum(Minimum(tri_v0_x_re, tri_v1_x_re), tri_v2_x_re);// (1,#faces)
            auto tri_min_h = Minimum(Minimum(tri_v0_y_re, tri_v1_y_re), tri_v2_y_re);// (1,#faces)

            auto tri_max_w = Maximum(Maximum(tri_v0_x_re, tri_v1_x_re), tri_v2_x_re);// (1,#faces)
            auto tri_max_h = Maximum(Maximum(tri_v0_y_re, tri_v1_y_re), tri_v2_y_re);// (1,#faces)

            auto bin_min_w = Cast(ScalarDivS(tri_min_w, Element(DataType::DT_FP32, bin_size)), DT_INT32, CAST_FLOOR); // (1, #faces)
            auto bin_min_h = Cast(ScalarDivS(tri_min_h, Element(DataType::DT_FP32, bin_size)), DT_INT32, CAST_FLOOR); // (1, #faces)

            auto bin_max_w = Cast(ScalarDivS(tri_max_w, Element(DataType::DT_FP32, bin_size)), DT_INT32, CAST_FLOOR); // (1, #faces)
            auto bin_max_h = Cast(ScalarDivS(tri_max_h, Element(DataType::DT_FP32, bin_size)), DT_INT32, CAST_FLOOR); // (1, #faces)


            auto w_grid = Unsqueeze(Range(Element(DataType::DT_INT32, 0), Element(DataType::DT_INT32, bin_w), Element(DataType::DT_INT32, 1)), -1);  // (bin_w, 1)
            auto h_grid = Unsqueeze(Range(Element(DataType::DT_INT32, 0), Element(DataType::DT_INT32, bin_h), Element(DataType::DT_INT32, 1)), -1);  // (bin_h, 1)
            auto w_grid_expand = Expand(w_grid, {w_grid.GetShape()[0], face_verts.GetShape()[0]});// (bin_w, faces)
            auto h_grid_expand = Expand(h_grid, {h_grid.GetShape()[0], face_verts.GetShape()[0]});// (bin_h, faces)

            auto bin_min_w_fp32 = Cast(bin_min_w, DT_FP32);
            auto bin_min_h_fp32 = Cast(bin_min_h, DT_FP32);

            auto bin_max_w_fp32 = Cast(bin_max_w, DT_FP32);
            auto bin_max_h_fp32 = Cast(bin_max_h, DT_FP32);

            auto w_grid_fp32 = Cast(w_grid, DT_FP32);
            auto h_grid_fp32 = Cast(h_grid, DT_FP32);
            auto w_mask = LogicalAnd(Compare(w_grid_fp32, bin_min_w_fp32, OpType::GE, OutType::BOOL), Compare(w_grid_fp32, bin_max_w_fp32, OpType::LT, OutType::BOOL));// (bin_w, faces)
            auto h_mask = LogicalAnd(Compare(h_grid_fp32, bin_min_h_fp32, OpType::GE, OutType::BOOL), Compare(h_grid_fp32, bin_max_h_fp32, OpType::LT, OutType::BOOL));// (bin_h, faces)

            auto w_mask_expand = Expand(Unsqueeze(w_mask, 1), {w_grid.GetShape()[0], h_grid.GetShape()[0], face_verts.GetShape()[0]});// (bin_w, bin_h, faces)
            auto h_mask_expand = Expand(Unsqueeze(h_mask, 0), {w_grid.GetShape()[0], h_grid.GetShape()[0], face_verts.GetShape()[0]});// (bin_w, bin_h, faces)

            binning_mask = LogicalAnd(w_mask_expand, h_mask_expand);// (bin_w, bin_h, faces)
            // binning_tri_cnt = Sum(binning_mask, -1);
        }

        LOOP(loop_name_bin_w.c_str(), FunctionType::DYNAMIC_LOOP, bin_x, LoopRange(0, bin_w, 1)) { //shape0
            LOOP(loop_name_bin_h.c_str(), FunctionType::DYNAMIC_LOOP, bin_y, LoopRange(0, bin_h, 1)) {//shape1
                Tensor face_idx_bin(DT_INT32,  {verts_cnt},    "face_idx_bin");
                Tensor face_verts_bin(DT_FP32, {verts_cnt, 9}, "face_verts_bin");
                // Tensor points_bin(DT_FP32, {point_num_bin, 2}, "points_bin");

                LOOP("loop00", FunctionType::DYNAMIC_LOOP, idx_000, LoopRange(1)) {
                    UNUSED(idx_000);
                    auto bin_mask_part = Reshape(View(binning_mask, {1, 1, binning_mask.GetShape()[2]}, {bin_x, bin_y, 0}), {binning_mask.GetShape()[2]});         // (#1 ,1, faces)
                    // auto binning_index = Squeeze(Nonzero(Reshape(bin_mask_part, {bin_mask_part.GetShape()[2]})), -1);
                    // auto face_verts_bin =  GatherElements(face_verts, binning_index, 0);
                    // auto bin_verts_num  = Sum(Reshape(bin_mask_part,  {bin_mask_part.GetShape()[2]}));
                    // SymbolicScalar verts_cnt = GetTensorData(bin_verts_num, {0});
                    // verts_cnt.AsIntermediateVariable();
                    Tensor tensor_one = Full(Element(DataType::DT_FP32, 1.0f), DT_FP32, {binning_mask.GetShape()[2]});
                    auto bin_mask_part_fp32 = Where(bin_mask_part, tensor_one, Element(DataType::DT_FP32, 0.0f));
                    auto sorted_idx     = ArgSort(bin_mask_part_fp32, -1, false);
                    face_idx_bin        = View(sorted_idx, {verts_cnt}, {0});
                    // auto gather_idx     = Reshape(Expand(Reshape(face_idx_bin, {face_idx_bin.GetShape()[0], 1}), {face_idx_bin.GetShape()[0], 9}), {face_idx_bin.GetShape()[0], 3, 3});
                    // face_verts_bin      = Reshape(GatherElements(face_verts, gather_idx, 0), {face_idx_bin.GetShape()[0], 9});
                    face_verts_bin = ScalarAddS(Reshape(View(face_verts, {verts_cnt, 3, 3}, {0, 0, 0}), {verts_cnt, 9}), Element(DataType::DT_FP32, 0.0f)); 
                }

                Tensor barycentrics_u(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_u");
                Tensor barycentrics_v(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_v");
                Tensor barycentrics_w(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_w");

                Tensor zbuf_ori(DT_FP32,       {point_num_bin, verts_cnt}, "zbuf_o");
                Tensor face_sd2_ori(DT_FP32,   {point_num_bin, verts_cnt}, "Face_sd2");
                Tensor topk_idx(DT_INT32,      {point_num_bin, verts_cnt}, "topk_idx");
                Tensor topk_idx_s1(DT_INT32,   {point_num_bin, verts_cnt}, "topk_idx_s1");

                LOOP(loop_name_bin_l0.c_str(), FunctionType::DYNAMIC_LOOP, idx_l0, LoopRange(0, (point_num_bin + tile1_shape0 - 1)/tile1_shape0, 1)) {
                    LOOP(loop_name_bin_l1.c_str(), FunctionType::DYNAMIC_LOOP, idx_l1, LoopRange(0, (verts_cnt + tile1_shape1 - 1)/tile1_shape1, 1)) {
                        std::vector<int64_t> view_shape = {tile1_shape0, tile1_shape1};
                        std::vector<SymbolicScalar> offset = {idx_l0 * tile1_shape0, idx_l1 * tile1_shape1};
                        // std::vector<SymbolicScalar> valid_shape = {
                        //     std::min(point_num_bin - idx_l0 * tile1_shape0, tile1_shape0),
                        //     std::min(verts_cnt - idx_l1 * tile1_shape1, tile1_shape1)
                        // };
                        std::vector<SymbolicScalar> total_offset0 = {bin_x * bin_size + idx_l0 * tile1_shape0 / bin_size, bin_y * bin_size + idx_l0 * tile1_shape0 % bin_size, 0};
                        std::vector<SymbolicScalar> total_offset1 = {bin_x * bin_size + idx_l0 * tile1_shape0 / bin_size, bin_y * bin_size + idx_l0 * tile1_shape0 % bin_size, 1};

                        auto tri_v0_x_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 0}), {1, tile1_shape1});// (1, tile1_shape1)
                        auto tri_v0_y_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 1}), {1, tile1_shape1});
                        auto tri_v0_z_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 2}), {1, tile1_shape1});

                        auto tri_v1_x_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 3}), {1, tile1_shape1});
                        auto tri_v1_y_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 4}), {1, tile1_shape1});
                        auto tri_v1_z_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 5}), {1, tile1_shape1});

                        auto tri_v2_x_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 6}), {1, tile1_shape1});
                        auto tri_v2_y_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 7}), {1, tile1_shape1});
                        auto tri_v2_z_tile = Reshape(View(face_verts_bin, {tile1_shape1, 1}, {idx_l1 * tile1_shape1, 8}), {1, tile1_shape1});

                        auto point_x = Expand(Reshape(View(point_w_h, {1, tile1_shape0, 1}, total_offset0), {tile1_shape0, 1}), {tile1_shape0, tile1_shape1});
                        auto point_y = Expand(Reshape(View(point_w_h, {1, tile1_shape0, 1}, total_offset1), {tile1_shape0, 1}), {tile1_shape0, tile1_shape1});

                        // BarycentricCoordsNoperspective
                        auto v01_x = Sub(tri_v1_x_tile, tri_v0_x_tile); // (1, #faces)
                        auto v01_y = Sub(tri_v1_y_tile, tri_v0_y_tile); // (1, #faces)

                        auto v02_x = Sub(tri_v2_x_tile, tri_v0_x_tile); // (1, #faces)
                        auto v02_y = Sub(tri_v2_y_tile, tri_v0_y_tile); // (1, #faces)

                        auto v0p_x = Sub(point_x, tri_v0_x_tile); //(#p, #faces)
                        auto v0p_y = Sub(point_y, tri_v0_y_tile); //(#p, #faces)

                        auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));// (1, #faces)
                        auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));// (1, #faces)
                        auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));// (1, #faces)

                        auto d20 = Add(Mul(v0p_x, v01_x), Mul(v0p_y, v01_y));//(#p, #faces)
                        auto d21 = Add(Mul(v0p_x, v02_x), Mul(v0p_y, v02_y));//(#p, #faces)

                        auto denom = Sub(Mul(d00, d11), Mul(d01, d01));         //(1, #faces)
                        auto v = Div(Sub(Mul(d11, d20), Mul(d01, d21)), denom); //(#p, #faces)
                        auto w = Div(Sub(Mul(d00, d21), Mul(d01, d20)), denom); //(#p, #faces)
                        auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

                        Tensor scale = Full(Element(DataType::DT_FP32, (double)screen_to_ndc_scale), DT_FP32, {tile1_shape0, tile1_shape1});               // (#point, 1)
                        // TriangleSigned SquaredDistance
                        auto square_distance = TriangleSquaredDistanceSingle2(point_x, point_y, 
                                                tri_v0_x_tile,  tri_v0_y_tile, tri_v1_x_tile, tri_v1_y_tile, tri_v2_x_tile, tri_v2_y_tile, scale); // (#point, #faces)

                        auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
                        auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                        auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                        auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
                        auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
                        auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

                        auto z_frags_sq = Add(Add(Mul(u, tri_v0_z_tile), Mul(v, tri_v1_z_tile)), Mul(w, tri_v2_z_tile));  // (#point, #faces)

                        // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
                        Element blur_radius (DataType::DT_FP32, 0.0f);
                        auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
                        auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
                        auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

                        auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

                        Assemble(u,        offset, barycentrics_u);
                        Assemble(v,        offset, barycentrics_v);
                        Assemble(w,        offset, barycentrics_w);

                        Assemble(zbuf_1,   offset, zbuf_ori);
                        Assemble(z_frags_sq, offset, face_sd2_ori);
                    }
                }
                LOOP("loop3_1", FunctionType::DYNAMIC_LOOP, idx_l3_1, LoopRange(0, (point_num_bin + tile2_shape0 - 1)/tile2_shape0, 1)) {
                        std::vector<int64_t> view_shape = {tile2_shape0, verts_cnt};
                        std::vector<SymbolicScalar> bin_offset = {idx_l3_1 * tile2_shape0, 0};
                        // std::vector<SymbolicScalar> valid_shape = {
                        //     std::min(point_num_bin - idx_l3_1 * tile2_shape0, tile2_shape0),
                        //     verts_cnt
                        // };
                        std::vector<SymbolicScalar> total_offset   = {bin_x * bin_size + idx_l3_1 * tile2_shape0 / bin_size, bin_y * bin_size + idx_l3_1 * tile2_shape0 % bin_size, 0};
                        std::vector<SymbolicScalar> total_offset_1 = {bin_x * bin_size + idx_l3_1 * tile2_shape0 / bin_size, bin_y * bin_size + idx_l3_1 * tile2_shape0 % bin_size, 1};
                        std::vector<SymbolicScalar> total_offset_2 = {bin_x * bin_size + idx_l3_1 * tile2_shape0 / bin_size, bin_y * bin_size + idx_l3_1 * tile2_shape0 % bin_size, 2};


                        auto face_idx_tile   = View(face_idx_bin, {view_shape[1]}, {0});
                        auto face_sd2_tile   = View(face_sd2_ori, view_shape,  bin_offset);
                        auto zbuf_out_tile   = View(zbuf_ori,     view_shape,  bin_offset);
                        // auto topk_idx_tile2  = View(topk_idx,     {view_shape[0], 1},  bin_offset);

                        auto topk_idx_tile   = TopK(zbuf_out_tile, 1, -1, false);
                        auto topk_idx_tile2  = std::get<1>(topk_idx_tile);
                        // auto topk_idx_tile2  = View(std::get<0>(topk_idx_tile), {tile2_shape0, 1},  {0, 0});

                        auto pix_to_face_expand = Expand(Unsqueeze(face_idx_tile, 0), view_shape);         // (#point, #faces)
                        auto pix_to_face_tile   = GatherElements(pix_to_face_expand, topk_idx_tile2, -1);  // (#point, #1)
                        auto dists_tile         = GatherElements(face_sd2_tile, topk_idx_tile2, -1);       // (#point, #1)
                        auto zbuf_tile          = GatherElements(zbuf_out_tile, topk_idx_tile2, -1);       // (#point, #1)

                        Assemble(Reshape(pix_to_face_tile, {1, tile2_shape0, 1}, true), total_offset, pix_to_face);
                        Assemble(Reshape(dists_tile, {1, tile2_shape0, 1}, true), total_offset, dists);
                        Assemble(Reshape(zbuf_tile, {1, tile2_shape0, 1}, true), total_offset, zbuf);

                        auto barycoords_u_tile = GatherElements(barycentrics_u, topk_idx_tile2, -1);            // (#point, #1)
                        auto barycoords_v_tile = GatherElements(barycentrics_v, topk_idx_tile2, -1);            // (#point, #1)
                        auto barycoords_w_tile = GatherElements(barycentrics_w, topk_idx_tile2, -1);            // (#point, #1)

                        Assemble(Reshape(barycoords_u_tile, {1, tile2_shape0, 1}, true), total_offset, barycoords);
                        Assemble(Reshape(barycoords_v_tile, {1, tile2_shape0, 1}, true), total_offset_1, barycoords);
                        Assemble(Reshape(barycoords_w_tile, {1, tile2_shape0, 1}, true), total_offset_2, barycoords);
                }
            }
        }
    }

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res_pix_to_face = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    auto res_zbuf = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    auto res_dists = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(2);
    // auto res_barycoords = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(3);

    // printf("\n\n=================== golden_pix_to_face ===================\n\n");
    // (void)resultCmp(golden_pix_to_face, (int32_t *)res_pix_to_face->data(), 0);

    // printf("\n\n=================== golden_zbuf ===================\n\n");
    // (void)resultCmp(golden_zbuf,        (float *)res_zbuf->data(), 0.001f);

    // printf("\n\n=================== golden_bary_coords ===================\n\n");
    // (void)resultCmp(golden_barycoords, (float *)res_barycoords->data(), 0.001f); 

    // printf("\n\n=================== golden_dists ===================\n\n");
    // (void)resultCmp(golden_dists,       (float *)res_dists->data(), 0.001f);

    int ret = true;
    EXPECT_EQ(ret, true);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_8_8_192) {
    std::string bin_path = "_11_10.bin";
    int face_num = 192;
    int point_size_x = 8;
    int point_size_y = 8;
    RunRasterize(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_64_64_192) {
    std::string bin_path = "_11_10.bin";
    int face_num = 192;
    int point_size_x = 64;
    int point_size_y = 64;
    RunRasterize(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_32_32_368) {
    std::string bin_path = "_9_10.bin";
    int face_num = 368;
    int point_size_x = 32;
    int point_size_y = 32;
    RunRasterize(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_64_64_368) {
    std::string bin_path = "_9_10.bin";
    int face_num = 368;
    int point_size_x = 64;
    int point_size_y = 64;

    RunRasterize(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_32_32_976) {
    std::string bin_path = "_8_11.bin";
    int face_num = 976;
    int point_size_x = 32;
    int point_size_y = 32;
    RunRasterize(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_128_128_368) {
    std::string bin_path = "_9_10.bin";
    int face_num = 368;
    int point_size_x = 128;
    int point_size_y = 128;

    RunRasterize_OK2(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_64_64_976) {
    std::string bin_path = "_8_11.bin";
    int face_num = 976;
    int point_size_x = 128;
    int point_size_y = 128;
    RunRasterize_OK2(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_128_128_192) {
    std::string bin_path = "_8_11.bin";
    int face_num = 976;
    int point_size_x = 128;
    int point_size_y = 128;
    RunRasterize(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_128_128_976) {
    std::string bin_path = "_8_11.bin";
    int face_num = 976;
    int point_size_x = 128;
    int point_size_y = 128;
    RunRasterize_OK3(bin_path, point_size_x, point_size_y, face_num);
}

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_1024_64_20000) {

    RenderSettings rander_setting;
    rander_setting.image_hight = 1024;
    rander_setting.image_width = 1024;
    rander_setting.bin_size = 128;
    rander_setting.max_blend_depth = 1;
    rander_setting.blur_radius_ndc = 0;
    rander_setting.clip_barycentric_coords = false;
    rander_setting.cull_backfaces = false;
    rander_setting.persp_correct = false;
    int face_num = 512;

    RunRasterizeTotal(rander_setting, face_num);
}


        // LOOP(loop_name_bin_h.c_str(), FunctionType::DYNAMIC_LOOP, bin_y, LoopRange(1)) {
        //     LOOP(loop_name_bin_w.c_str(), FunctionType::DYNAMIC_LOOP, bin_x, LoopRange(1)) {

        //         auto bin_mask_part = View(binning_mask, {1, 1, binning_mask.GetShape()[2]}, {bin_x, bin_y, 0});         // (#1 ,1, faces)

        //         // auto binning_index = Squeeze(Nonzero(Reshape(bin_mask_part, {bin_mask_part.GetShape()[2]})), -1);
        //         // auto face_verts_bin =  GatherElements(face_verts, binning_index, 0);
        //         // auto bin_verts_num  = Sum(Reshape(bin_mask_part,  {bin_mask_part.GetShape()[2]}));
        //         auto sorted_idx     = ArgSort(Reshape(bin_mask_part, {bin_mask_part.GetShape()[2]}), -1, false);
        //         // SymbolicScalar verts_cnt = GetTensorData(bin_verts_num, {0});
        //         // verts_cnt.AsIntermediateVariable();
        //         auto face_idx_bin = View(sorted_idx, {verts_cnt}, {0});
        //         auto gather_idx = Reshape(Expand(Reshape(face_idx_bin, {face_idx_bin.GetShape()[0], 1}), {face_idx_bin.GetShape()[0], 9}), {face_idx_bin.GetShape()[0], 3, 3});
        //         auto face_verts_bin = GatherElements(face_verts, gather_idx, 0);

        //         auto bin_tri_v0_x = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 0, 0}), {1, verts_cnt});// (1, #faces)
        //         auto bin_tri_v0_y = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 0, 1}), {1, verts_cnt});// (1, #faces)
        //         auto bin_tri_v0_z = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 0, 2}), {1, verts_cnt});// (1, #faces)

        //         auto bin_tri_v1_x = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 1, 0}), {1, verts_cnt});// (1, #faces)
        //         auto bin_tri_v1_y = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 1, 1}), {1, verts_cnt});// (1, #faces)
        //         auto bin_tri_v1_z = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 1, 2}), {1, verts_cnt});// (1, #faces)

        //         auto bin_tri_v2_x = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 2, 0}), {1, verts_cnt});// (1, #faces)
        //         auto bin_tri_v2_y = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 2, 1}), {1, verts_cnt});// (1, #faces)
        //         auto bin_tri_v2_z = Reshape(View(face_verts_bin, {verts_cnt, 1, 1}, {0, 2, 2}), {1, verts_cnt});// (1, #faces)
                
        //         auto points_bin  = Reshape(View(point_w_h, {bin_size, bin_size, 2}, {bin_x * bin_size, bin_y * bin_size, 0}), {bin_size*bin_size, 2});  // (bin_size, bin_size, 2)
        //         auto bin_point_x = Expand(View(points_bin, {point_num_bin, 1}, {0, 0}), {point_num_bin, verts_cnt});// (#point, #faces)
        //         auto bin_point_y = Expand(View(points_bin, {point_num_bin, 1}, {0, 1}), {point_num_bin, verts_cnt});// (#point, #faces)


        //         // Tensor barycentrics_u(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_u");
        //         // Tensor barycentrics_v(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_v");
        //         // Tensor barycentrics_w(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_w");

        //         // Tensor zbuf_ori(DT_FP32,       {point_num_bin, verts_cnt}, "zbuf_o");
        //         // Tensor face_sd2_ori(DT_FP32,   {point_num_bin, verts_cnt}, "Face_sd2");

        //         // LOOP(loop_name_bin_l0.c_str(), FunctionType::DYNAMIC_LOOP, idx_l0, LoopRange(0, (point_num_bin + tile1_shape0 - 1)/tile1_shape0, 1)) {
        //         //     LOOP(loop_name_bin_l1.c_str(), FunctionType::DYNAMIC_LOOP, idx_l1, LoopRange(0, (verts_cnt + tile1_shape1 - 1) / tile1_shape1 , 1)) {
        //         LOOP("ttttttttttt1", FunctionType::DYNAMIC_LOOP, idx_l0, LoopRange(1)) {
        //             LOOP("ttttttttttt21", FunctionType::DYNAMIC_LOOP, idx_l1, LoopRange(1)) {
        //                 // std::vector<int64_t> view_shape = {tile1_shape0, tile1_shape1};
        //                 // std::vector<SymbolicScalar> offset = {idx_l0 * tile1_shape0, idx_l1 * tile1_shape1};
        //                 // std::vector<SymbolicScalar> valid_shape = {
        //                 //     std::min(point_num_bin - idx_l0 * tile1_shape0, tile1_shape0),
        //                 //     std::min(verts_cnt - idx_l1 * tile1_shape1, tile1_shape1)
        //                 // };

        //                 // auto tri_v0_x_tile = View(bin_tri_v0_x, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)
        //                 // auto tri_v0_y_tile = View(bin_tri_v0_y, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)
        //                 // auto tri_v0_z_tile = View(bin_tri_v0_z, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)

        //                 // auto tri_v1_x_tile = View(bin_tri_v1_x, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)
        //                 // auto tri_v1_y_tile = View(bin_tri_v1_y, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)
        //                 // auto tri_v1_z_tile = View(bin_tri_v1_z, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)

        //                 // auto tri_v2_x_tile = View(bin_tri_v2_x, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)
        //                 // auto tri_v2_y_tile = View(bin_tri_v2_y, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)
        //                 // auto tri_v2_z_tile = View(bin_tri_v2_z, {1, view_shape[1]}, {1, valid_shape[1]}, {0, offset[1]});// (1, tile1_shape1)

        //                 auto point_x = View(point_w_h, {tile1_shape0, tile1_shape1 , 1}, {bin_x * bin_size + idx_l0 * tile1_shape0, bin_y * bin_size + idx_l1 * tile1_shape1, 0});// (tile1_shape0, tile1_shape1)
        //                 auto point_y = View(point_w_h, {tile1_shape0, tile1_shape1 , 1}, {bin_x * bin_size + idx_l0 * tile1_shape0, bin_y * bin_size + idx_l1 * tile1_shape1, 1});// (tile1_shape0, tile1_shape1)
        //                 // BarycentricCoordsNoperspective
        //                 // auto v01_x = Sub(tri_v1_x_tile, tri_v0_x_tile); // (1, #faces)
        //                 // auto v01_y = Sub(tri_v1_y_tile, tri_v0_y_tile); // (1, #faces)

        //                 // auto v02_x = Sub(tri_v2_x_tile, tri_v0_x_tile); // (1, #faces)
        //                 // auto v02_y = Sub(tri_v2_y_tile, tri_v0_y_tile); // (1, #faces)

        //                 // auto v0p_x = Sub(point_x, tri_v0_x_tile); //(#p, #faces)
        //                 // auto v0p_y = Sub(point_y, tri_v0_y_tile); //(#p, #faces)

        //                 // auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));// (1, #faces)
        //                 // auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));// (1, #faces)
        //                 // auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));// (1, #faces)

        //                 // auto d20 = Add(Mul(v0p_x, v01_x), Mul(v0p_y, v01_y));//(#p, #faces)
        //                 // auto d21 = Add(Mul(v0p_x, v02_x), Mul(v0p_y, v02_y));//(#p, #faces)

        //                 // auto denom = Sub(Mul(d00, d11), Mul(d01, d01));         //(1, #faces)
        //                 // auto v = Div(Sub(Mul(d11, d20), Mul(d01, d21)), denom); //(#p, #faces)
        //                 // auto w = Div(Sub(Mul(d00, d21), Mul(d01, d20)), denom); //(#p, #faces)
        //                 // auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

        //                 // Tensor scale = Full(Element(DataType::DT_FP32, (double)screen_to_ndc_scale), DT_FP32, {tile1_shape0, tile1_shape1});               // (#point, 1)
        //                 // // TriangleSigned SquaredDistance
        //                 // auto square_distance = TriangleSquaredDistanceSingle2(point_x, point_y, 
        //                 //                         tri_v0_x_tile,  tri_v0_y_tile, tri_v1_x_tile, tri_v1_y_tile, tri_v2_x_tile, tri_v2_y_tile, scale); // (#point, #faces)

        //                 // auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
        //                 // auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
        //                 // auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
        //                 // auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
        //                 // auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
        //                 // auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

        //                 // auto z_frags_sq = Add(Add(Mul(u, tri_v0_z_tile), Mul(v, tri_v1_z_tile)), Mul(w, tri_v2_z_tile));  // (#point, #faces)

        //                 // // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
        //                 // Element blur_radius (DataType::DT_FP32, 0.0f);
        //                 // auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
        //                 // auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
        //                 // auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

        //                 // auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

        //                 // Assemble(u,        offset, barycentrics_u);
        //                 // Assemble(v,        offset, barycentrics_v);
        //                 // Assemble(w,        offset, barycentrics_w);

                        // Assemble(zbuf_1,   offset, zbuf_ori);
                        // Assemble(z_frags_sq,   offset, face_sd2_ori);
        //             }
        //         }

        //         // Tensor topk_idx(DT_INT32,      {point_num_bin, verts_cnt}, "topk_idx");
        //         // Tensor topk_idx_s1(DT_INT32,   {point_num_bin, (verts_cnt + tile2_shape1 - 1) / tile2_shape1}, "topk_idx_s1");

        //         // LOOP("loop2_l1", FunctionType::DYNAMIC_LOOP, idx_l0_2, LoopRange(0, (point_num_bin + tile2_shape0 - 1)/tile2_shape0, 1)) {
        //         //     LOOP("loop2_l0", FunctionType::DYNAMIC_LOOP, idx_l1_2, LoopRange(0, (verts_cnt + tile2_shape1 - 1) / tile2_shape1 , 1)) {
        //         //         std::vector<int64_t> view_shape = {tile2_shape0, tile2_shape1};
        //         //         std::vector<SymbolicScalar> offset = {idx_l0_2 * tile2_shape0, idx_l1_2 * tile2_shape1};
        //         //         std::vector<SymbolicScalar> valid_shape = {
        //         //             std::min(point_num_bin - idx_l0_2 * tile2_shape0, tile2_shape0),
        //         //             std::min(verts_cnt - idx_l1_2 * tile2_shape1, tile2_shape1)
        //         //         };
        //         //         auto zbuf_tile = View(zbuf_ori, view_shape, valid_shape, offset);

        //         //         auto output_topk1 = TopK(zbuf_tile, 1, -1, false);
        //         //         Assemble(std::get<1>(output_topk1), {offset[0], idx_l1_2}, topk_idx_s1);
        //         //     }
        //         //     auto topk_idx_tile_s0s1 = TopK(topk_idx_s1, 1, -1, false);
        //         //     Assemble(std::get<1>(topk_idx_tile_s0s1), {idx_l0_2 * tile2_shape0, 0}, topk_idx);
        //         // }

        //         // Tensor barycoords_u(DT_FP32,  {point_num_bin, 1}, "barycoords_u");
        //         // Tensor barycoords_v(DT_FP32,  {point_num_bin, 1}, "barycoords_v");
        //         // Tensor barycoords_w(DT_FP32,  {point_num_bin, 1}, "barycoords_w");

        //         // Tensor pix_to_face_bin(DT_INT32, {point_num_bin, 1}, "Y_1");
        //         // Tensor zbuf_bin(DT_FP32,         {point_num_bin, 1}, "Y_2");
        //         // Tensor dists_bin(DT_FP32,        {point_num_bin, 1}, "Y_3");
        //         // Tensor barycoords_bin(DT_FP32,   {point_num_bin, 1}, "Y_4");

        //         // LOOP("loop3", FunctionType::DYNAMIC_LOOP, idx_l0_3, LoopRange(0, (point_num_bin + tile2_shape0 - 1)/tile2_shape0, 1)) {
        //         //         std::vector<int64_t> view_shape = {tile2_shape0, verts_cnt};
        //         //         std::vector<SymbolicScalar> offset = {idx_l0_3 * tile2_shape0, 0};
        //         //         std::vector<SymbolicScalar> valid_shape = {
        //         //             std::min(point_num_bin - idx_l0_3 * tile2_shape0, tile2_shape0),
        //         //             verts_cnt
        //         //         };
        //         //         auto face_idx_tile   = View(face_idx_bin, {view_shape[1]}, {valid_shape[1]}, {0});
        //         //         auto face_sd2_tile   = View(face_sd2_ori, view_shape, valid_shape, offset);
        //         //         auto zbuf_out_tile   = View(zbuf_ori, view_shape, valid_shape, offset);
        //         //         auto topk_idx_tile2  = View(topk_idx, {view_shape[0], 1}, {valid_shape[0], 1}, {0, 0});

        //         //         auto pix_to_face_expand = Expand(Unsqueeze(face_idx_tile, 0), view_shape, valid_shape);         // (#point, #faces)
        //         //         auto pix_to_face_tile   = GatherElements(pix_to_face_expand, topk_idx_tile2, -1);  // (#point, #1)
        //         //         auto dists_tile         = GatherElements(face_sd2_tile, topk_idx_tile2, -1);       // (#point, #1)
        //         //         auto zbuf_tile          = GatherElements(face_sd2_tile, topk_idx_tile2, -1);       // (#point, #1)

        //         //         Assemble(pix_to_face_tile, offset, pix_to_face_bin);
        //         //         Assemble(dists_tile,       offset, dists_bin);
        //         //         Assemble(zbuf_out_tile,    offset, zbuf_bin);

        //         //         auto barycoords_u_tile = GatherElements(barycentrics_u, topk_idx_tile2, -1);            // (#point, #1)
        //         //         auto barycoords_v_tile = GatherElements(barycentrics_v, topk_idx_tile2, -1);            // (#point, #1)
        //         //         auto barycoords_w_tile = GatherElements(barycentrics_w, topk_idx_tile2, -1);            // (#point, #1)

        //         //         Assemble(barycoords_u_tile, offset, barycoords_u);
        //         //         Assemble(barycoords_v_tile, offset, barycoords_v);
        //         //         Assemble(barycoords_w_tile, offset, barycoords_w);

        //         // }
        //         // LOOP("loop4", FunctionType::DYNAMIC_LOOP, idx_last, LoopRange(1)) {
        //         //     UNUSED(idx_last);
        //         //     Assemble(Reshape(pix_to_face_bin, {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , pix_to_face);
        //         //     Assemble(Reshape(dists_bin,       {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , dists);
        //         //     Assemble(Reshape(zbuf_bin,        {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , zbuf);

        //         //     Assemble(Reshape(barycoords_u,    {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , barycoords);
        //         //     Assemble(Reshape(barycoords_v,    {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 1} , barycoords);
        //         //     Assemble(Reshape(barycoords_w,    {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 2} , barycoords);
        //         // }
        //     }
        // }





        // std::vector<Tensor> face_vert_bin_list;
        // std::vector<Tensor> point_bin_list;

        // face_vert_bin_list.reserve(bin_num);
        // for (int bin_id = 0; bin_id < bin_num; bin_id++){
        //     std::string bin_name = "face_vert_bin_list" + std::to_string(bin_id);
        //     face_vert_bin_list.emplace_back(Tensor(DT_FP32, {verts_cnt, 9}, bin_name));

        //     std::string bin_name2 = "point_bin_list" + std::to_string(bin_id);
        //     point_bin_list.emplace_back(Tensor(DT_FP32, {bin_size * bin_size, 2}, bin_name2));

        //     std::string bin_name3= "face_idx_bin_list" + std::to_string(bin_id);
        //     face_idx_bin_list.emplace_back(Tensor(DT_FP32, {verts_cnt}, bin_name3));
        // }
        // for (int bin_id = 0; bin_id < bin_num; bin_id++) {
        //     int bin_x = bin_id % bin_size;
        //     int bin_y = bin_id / bin_size;
        //     LOOP("bin_Init", FunctionType::DYNAMIC_LOOP, init_idx, LoopRange(1)) {
        //         UNUSED(init_idx);
        //         auto bin_mask_part = View(binning_mask, {1, 1, binning_mask.GetShape()[2]}, {bin_x, bin_y, 0});         // (#1 ,1, faces)
        //         auto sorted_idx     = ArgSort(Reshape(bin_mask_part, {bin_mask_part.GetShape()[2]}), -1, false);
        //         auto face_idx_bin   = View(sorted_idx, {verts_cnt}, {0});
        //         auto gather_idx     = Reshape(Expand(Reshape(face_idx_bin, {face_idx_bin.GetShape()[0], 1}), {face_idx_bin.GetShape()[0], 9}), {face_idx_bin.GetShape()[0], 3, 3});
        //         face_vert_bin_list[bin_num] = Reshape(GatherElements(face_verts, gather_idx, 0), {verts_cnt, 9});
        //         point_bin_list[bin_id]      = Reshape(View(point_w_h, {bin_size, bin_size, 2}, {bin_x * bin_size, bin_y * bin_size, 0}), {bin_size*bin_size, 2});  // (bin_size, bin_size, 2)
        //         face_idx_bin_list[bin_id]   = ScalarAddS(face_idx_bin, Element(DataType::DT_INT32, 0.0f));
        //     }
        // }

        // Tensor barycentrics_u(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_u");
        // Tensor barycentrics_v(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_v");
        // Tensor barycentrics_w(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_w");

        // Tensor zbuf_ori(DT_FP32,       {point_num_bin, verts_cnt}, "zbuf_o");
        // Tensor face_sd2_ori(DT_FP32,   {point_num_bin, verts_cnt}, "Face_sd2");
        // Tensor topk_idx(DT_INT32,      {point_num_bin, verts_cnt}, "topk_idx");
        // Tensor topk_idx_s1(DT_INT32,   {point_num_bin, verts_cnt}, "topk_idx_s1");

        // for (int bin_id = 0; bin_id < bin_num; bin_id++) {
        //     int bin_x = bin_id % bin_size;
        //     int bin_y = bin_id / bin_size;
        //     LOOP(loop_name_bin_l0.c_str(), FunctionType::DYNAMIC_LOOP, idx_l0, LoopRange(0, (point_num_bin + tile1_shape0 - 1)/tile1_shape0, 1)) {
        //         LOOP(loop_name_bin_l1.c_str(), FunctionType::DYNAMIC_LOOP, idx_l1, LoopRange(0, (verts_cnt + tile1_shape1 - 1) / tile1_shape1 , 1)) {
        //             std::vector<int64_t> view_shape = {tile1_shape0, tile1_shape1};
        //             std::vector<SymbolicScalar> offset = {idx_l0 * tile1_shape0, idx_l1 * tile1_shape1};
        //             std::vector<SymbolicScalar> valid_shape = {
        //                 std::min(point_num_bin - idx_l0 * tile1_shape0, tile1_shape0),
        //                 std::min(verts_cnt - idx_l1 * tile1_shape1, tile1_shape1)
        //             };

        //             auto tri_v0_x_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 0}), {1, view_shape[1]});// (1, tile1_shape1)
        //             auto tri_v0_y_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 1}), {1, view_shape[1]});
        //             auto tri_v0_z_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 2}), {1, view_shape[1]});

        //             auto tri_v1_x_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 3}), {1, view_shape[1]});
        //             auto tri_v1_y_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 4}), {1, view_shape[1]});
        //             auto tri_v1_z_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 5}), {1, view_shape[1]});

        //             auto tri_v2_x_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 6}), {1, view_shape[1]});
        //             auto tri_v2_y_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 7}), {1, view_shape[1]});
        //             auto tri_v2_z_tile = Reshape(View(face_vert_bin_list[bin_num], {view_shape[1], 1}, {idx_l1 * tile1_shape1, 8}), {1, view_shape[1]});

        //             auto point_x = View(point_w_h, {tile1_shape0, tile1_shape1 , 1}, {bin_x * bin_size + idx_l0 * tile1_shape0, bin_y * bin_size + idx_l1 * tile1_shape1, 0});// (tile1_shape0, tile1_shape1)
        //             auto point_y = View(point_w_h, {tile1_shape0, tile1_shape1 , 1}, {bin_x * bin_size + idx_l0 * tile1_shape0, bin_y * bin_size + idx_l1 * tile1_shape1, 1});// (tile1_shape0, tile1_shape1)
        //             // BarycentricCoordsNoperspective
        //             auto v01_x = Sub(tri_v1_x_tile, tri_v0_x_tile); // (1, #faces)
        //             auto v01_y = Sub(tri_v1_y_tile, tri_v0_y_tile); // (1, #faces)

        //             auto v02_x = Sub(tri_v2_x_tile, tri_v0_x_tile); // (1, #faces)
        //             auto v02_y = Sub(tri_v2_y_tile, tri_v0_y_tile); // (1, #faces)

        //             auto v0p_x = Sub(point_x, tri_v0_x_tile); //(#p, #faces)
        //             auto v0p_y = Sub(point_y, tri_v0_y_tile); //(#p, #faces)

        //             auto d00 = Add(Mul(v01_x, v01_x), Mul(v01_y, v01_y));// (1, #faces)
        //             auto d01 = Add(Mul(v01_x, v02_x), Mul(v01_y, v02_y));// (1, #faces)
        //             auto d11 = Add(Mul(v02_x, v02_x), Mul(v02_y, v02_y));// (1, #faces)

        //             auto d20 = Add(Mul(v0p_x, v01_x), Mul(v0p_y, v01_y));//(#p, #faces)
        //             auto d21 = Add(Mul(v0p_x, v02_x), Mul(v0p_y, v02_y));//(#p, #faces)

        //             auto denom = Sub(Mul(d00, d11), Mul(d01, d01));         //(1, #faces)
        //             auto v = Div(Sub(Mul(d11, d20), Mul(d01, d21)), denom); //(#p, #faces)
        //             auto w = Div(Sub(Mul(d00, d21), Mul(d01, d20)), denom); //(#p, #faces)
        //             auto u = Sub(ScalarSubS(v, Element(DataType::DT_FP32, 1.0f), true), w); //# 1 - v - w) //# (#p, #faces)

        //             Tensor scale = Full(Element(DataType::DT_FP32, (double)screen_to_ndc_scale), DT_FP32, {tile1_shape0, tile1_shape1});               // (#point, 1)
        //             // TriangleSigned SquaredDistance
        //             auto square_distance = TriangleSquaredDistanceSingle2(point_x, point_y, 
        //                                     tri_v0_x_tile,  tri_v0_y_tile, tri_v1_x_tile, tri_v1_y_tile, tri_v2_x_tile, tri_v2_y_tile, scale); // (#point, #faces)

        //             auto square_distance_neg = ScalarMulS(square_distance, Element(DataType::DT_FP32, -1.0f));
        //             auto barycentrics_x_greater_zero = Compare(u, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
        //             auto barycentrics_y_greater_zero = Compare(v, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
        //             auto barycentrics_z_greater_zero = Compare(w, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL); // (#point, #faces)
        //             auto is_inside = LogicalAnd(LogicalAnd(barycentrics_x_greater_zero, barycentrics_y_greater_zero), barycentrics_z_greater_zero);   // (#point, #faces )
        //             auto face_sd2 = Where(is_inside, square_distance_neg, square_distance);

        //             auto z_frags_sq = Add(Add(Mul(u, tri_v0_z_tile), Mul(v, tri_v1_z_tile)), Mul(w, tri_v2_z_tile));  // (#point, #faces)

        //             // should_write = (face_sd2 < blur_radius) & (z_frags > 0) # (flat_#p, #face)
        //             Element blur_radius (DataType::DT_FP32, 0.0f);
        //             auto face_sd2_less_radius = Compare(face_sd2, blur_radius, OpType::LT, OutType::BOOL);   // (#point, #faces)
        //             auto z_frags_greater_zero = Compare(z_frags_sq, Element(DataType::DT_FP32, 0.0f), OpType::GT, OutType::BOOL);     // (#point, #faces)
        //             auto should_write = LogicalAnd(face_sd2_less_radius, z_frags_greater_zero);                 // (#point, #faces)

        //             auto zbuf_1 = Where(should_write, z_frags_sq, Element(DataType::DT_FP32, 0x7f800000));     // (#point, #faces)

        //             Assemble(u,        offset, barycentrics_u);
        //             Assemble(v,        offset, barycentrics_v);
        //             Assemble(w,        offset, barycentrics_w);

        //             Assemble(Reshape(point_x, {tile1_shape0, tile1_shape1}),   offset, zbuf_ori);
        //             Assemble(Reshape(point_y, {tile1_shape0, tile1_shape1}),   offset, face_sd2_ori);

        //             auto output_topk1 = TopK(zbuf_1, 1, -1, false);
        //             Assemble(std::get<1>(output_topk1), {offset[0], idx_l1}, topk_idx_s1);
        //         }
        //         auto topk_idx_tile_s0s1 = TopK(topk_idx_s1, 1, -1, false);
        //         Assemble(std::get<1>(topk_idx_tile_s0s1), {idx_l0 * tile2_shape0, 0}, topk_idx);
        //     }

        //     Tensor barycoords_u(DT_FP32,  {point_num_bin, 1}, "barycoords_u");
        //     Tensor barycoords_v(DT_FP32,  {point_num_bin, 1}, "barycoords_v");
        //     Tensor barycoords_w(DT_FP32,  {point_num_bin, 1}, "barycoords_w");

        //     Tensor pix_to_face_bin(DT_INT32, {point_num_bin, 1}, "Y_1");
        //     Tensor zbuf_bin(DT_FP32,         {point_num_bin, 1}, "Y_2");
        //     Tensor dists_bin(DT_FP32,        {point_num_bin, 1}, "Y_3");

        //     LOOP("loop3", FunctionType::DYNAMIC_LOOP, idx_l0_3, LoopRange(0, (point_num_bin + tile2_shape0 - 1)/tile2_shape0, 1)) {
        //             std::vector<int64_t> view_shape = {tile2_shape0, verts_cnt};
        //             std::vector<SymbolicScalar> offset = {idx_l0_3 * tile2_shape0, 0};
        //             std::vector<SymbolicScalar> valid_shape = {
        //                 std::min(point_num_bin - idx_l0_3 * tile2_shape0, tile2_shape0),
        //                 verts_cnt
        //             };
        //             auto face_idx_tile   = View(face_idx_bin_list[bin_num], {view_shape[1]}, {valid_shape[1]}, {0});
        //             auto face_sd2_tile   = View(face_sd2_ori, view_shape, valid_shape, offset);
        //             auto zbuf_out_tile   = View(zbuf_ori, view_shape, valid_shape, offset);
        //             auto topk_idx_tile2  = View(topk_idx, {view_shape[0], 1}, {valid_shape[0], 1}, offset);

        //             auto pix_to_face_expand = Expand(Unsqueeze(face_idx_tile, 0), view_shape, valid_shape);         // (#point, #faces)
        //             auto pix_to_face_tile   = GatherElements(pix_to_face_expand, topk_idx_tile2, -1);  // (#point, #1)
        //             auto dists_tile         = GatherElements(face_sd2_tile, topk_idx_tile2, -1);       // (#point, #1)
        //             auto zbuf_tile          = GatherElements(face_sd2_tile, topk_idx_tile2, -1);       // (#point, #1)

        //             Assemble(pix_to_face_tile, offset, pix_to_face_bin);
        //             Assemble(dists_tile,       offset, dists_bin);
        //             Assemble(zbuf_out_tile,    offset, zbuf_bin);

        //             auto barycoords_u_tile = GatherElements(barycentrics_u, topk_idx_tile2, -1);            // (#point, #1)
        //             auto barycoords_v_tile = GatherElements(barycentrics_v, topk_idx_tile2, -1);            // (#point, #1)
        //             auto barycoords_w_tile = GatherElements(barycentrics_w, topk_idx_tile2, -1);            // (#point, #1)

        //             Assemble(barycoords_u_tile, offset, barycoords_u);
        //             Assemble(barycoords_v_tile, offset, barycoords_v);
        //             Assemble(barycoords_w_tile, offset, barycoords_w);

        //     }
        //     LOOP("loop4", FunctionType::DYNAMIC_LOOP, idx_last, LoopRange(1)) {
        //         UNUSED(idx_last);

        //         Assemble(Reshape(ScalarAddS(pix_to_face_bin, Element(DataType::DT_INT32, 0.0f)), {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , pix_to_face);
        //         Assemble(Reshape(ScalarAddS(dists_bin, Element(DataType::DT_FP32, 0.0f)),        {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , dists);
        //         Assemble(Reshape(ScalarAddS(zbuf_bin, Element(DataType::DT_FP32, 0.0f)),         {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , zbuf);
        //         Tensor barycoords_u_all(DT_FP32,  {image_w, image_h, 1}, "barycoords_u_1");
        //         Tensor barycoords_v_all(DT_FP32,  {image_w, image_h, 1}, "barycoords_v_1");
        //         Tensor barycoords_w_all(DT_FP32,  {image_w, image_h, 1}, "barycoords_w_1");
        //         Assemble(ScalarAddS(Reshape(barycoords_u,  {bin_size, bin_size, 1}), Element(DataType::DT_INT32, 0.0f)), {bin_x * bin_size, bin_y * bin_size, 0} , barycoords_u_all);
        //         Assemble(ScalarAddS(Reshape(barycoords_v,  {bin_size, bin_size, 1}), Element(DataType::DT_INT32, 0.0f)), {bin_x * bin_size, bin_y * bin_size, 0} , barycoords_v_all);
        //         Assemble(ScalarAddS(Reshape(barycoords_w,  {bin_size, bin_size, 1}), Element(DataType::DT_INT32, 0.0f)), {bin_x * bin_size, bin_y * bin_size, 0} , barycoords_w_all);
        //         barycoords = Cat({Unsqueeze(barycoords_u_all, -1), Unsqueeze(barycoords_v_all, -1), Unsqueeze(barycoords_w_all, -1)}, -1);
        //     }
        // }        


        // LOOP("loop3", FunctionType::DYNAMIC_LOOP, idx_l0_3, LoopRange(0, (point_num_bin + tile2_shape0 - 1)/tile2_shape0, 1)) {
        //                 std::vector<int64_t> view_shape = {tile2_shape0, verts_cnt};
        //                 std::vector<SymbolicScalar> bin_offset = {idx_l0_3 * tile2_shape0, 0};
        //                 // std::vector<SymbolicScalar> valid_shape = {
        //                 //     std::min(point_num_bin - idx_l0_3 * tile2_shape0, tile2_shape0),
        //                 //     verts_cnt
        //                 // };
        //                 std::vector<SymbolicScalar> total_offset = {idx_l0_3 * tile2_shape0, 0};


        //                 auto face_idx_tile   = View(face_idx_bin, {view_shape[1]}, {0});
        //                 auto face_sd2_tile   = View(face_sd2_ori, view_shape, offset);
        //                 auto zbuf_out_tile   = View(zbuf_ori,     view_shape,  offset);
        //                 auto topk_idx_tile2  = View(topk_idx,     {view_shape[0], 1},  offset);

        //                 auto pix_to_face_expand = Expand(Unsqueeze(face_idx_tile, 0), view_shape);         // (#point, #faces)
        //                 auto pix_to_face_tile   = GatherElements(pix_to_face_expand, topk_idx_tile2, -1);  // (#point, #1)
        //                 auto dists_tile         = GatherElements(face_sd2_tile, topk_idx_tile2, -1);       // (#point, #1)
        //                 auto zbuf_tile          = GatherElements(zbuf_out_tile, topk_idx_tile2, -1);       // (#point, #1)

        //                 Assemble(pix_to_face_tile, offset, pix_to_face_bin);
        //                 Assemble(dists_tile,       offset, dists_bin);
        //                 Assemble(zbuf_tile,        offset, zbuf_bin);

        //                 auto barycoords_u_tile = GatherElements(barycentrics_u, topk_idx_tile2, -1);            // (#point, #1)
        //                 auto barycoords_v_tile = GatherElements(barycentrics_v, topk_idx_tile2, -1);            // (#point, #1)
        //                 auto barycoords_w_tile = GatherElements(barycentrics_w, topk_idx_tile2, -1);            // (#point, #1)

        //                 Assemble(barycoords_u_tile, offset, barycoords_u);
        //                 Assemble(barycoords_v_tile, offset, barycoords_v);
        //                 Assemble(barycoords_w_tile, offset, barycoords_w);

        //         }
        //         LOOP("loop4", FunctionType::DYNAMIC_LOOP, idx_last, LoopRange(1)) {
        //             UNUSED(idx_last);
        //             TileShape::Current().SetVecTile({64, 64, 8});

        //             Assemble(Reshape(ScalarAddS(pix_to_face_bin, Element(DataType::DT_INT32, 0.0f)), {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0}, pix_to_face);
        //             Assemble(Reshape(ScalarAddS(dists_bin,       Element(DataType::DT_INT32, 0.0f)), {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0}, dists);
        //             Assemble(Reshape(ScalarAddS(zbuf_bin,        Element(DataType::DT_INT32, 0.0f)), {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0}, zbuf);

        //             Assemble(Reshape(ScalarAddS(barycoords_u, Element(DataType::DT_INT32, 0.0f)),  {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 0} , barycoords);
        //             Assemble(Reshape(ScalarAddS(barycoords_v, Element(DataType::DT_INT32, 0.0f)),  {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 1} , barycoords);
        //             Assemble(Reshape(ScalarAddS(barycoords_w, Element(DataType::DT_INT32, 0.0f)),  {bin_size, bin_size, 1}), {bin_x * bin_size, bin_y * bin_size, 2} , barycoords);
        //         }


        // LOOP("loop2", FunctionType::DYNAMIC_LOOP, idx_l2_1, LoopRange(0, (point_num_bin + tile1_shape0 - 1)/tile1_shape0, 1)) {
        //     auto zbuf_tile   = View(zbuf_ori, {tile1_shape0, verts_cnt},  {idx_l2_1 * tile1_shape0, 0});
        //     auto topk_idx_tile = TopK(zbuf_tile, 1, -1, false);
        //     Assemble(std::get<0>(topk_idx_tile), {idx_l2_1 * tile2_shape0, 0}, topk_idx);
        // }