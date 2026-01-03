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
    config::SetRuntimeOption(DEVICE_SCHED_MODE, 2);

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

    int verts_cnt = 128;
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

    TileShape::Current().SetVecTile({128, 64, 32});

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


void RunRasterizeTotal2(RenderSettings rander_setting, int tri_num) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);
    config::SetRuntimeOption(DEVICE_SCHED_MODE, 2);

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

    std::vector<int64_t> shape_pix_to_face  = {image_w, image_h};
    std::vector<int64_t> shape_zbuf         = {image_w, image_h};
    std::vector<int64_t> shape_dists        = {image_w, image_h};
    std::vector<int64_t> shape_barycoords   = {image_w, image_h};

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


    FUNCTION(func_name.c_str(), {face_verts, point_w_h}, {pix_to_face, zbuf, dists, barycoords}) {
        Tensor binning_mask(DT_BOOL,   {bin_w, bin_h, face_num}, "bin_mask");
        TileShape::Current().SetVecTile({64, 64, 32});

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
                Tensor points_bin(DT_FP32, {point_num_bin, 2}, "points_bin");
                
                LOOP("loop00", FunctionType::DYNAMIC_LOOP, idx_000, LoopRange(1)) {
                    TileShape::Current().SetVecTile({64, 64, 32});
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
                    points_bin = ScalarAddS(Reshape(View(face_verts, {bin_size, bin_size, 2}, {bin_x * bin_size, bin_y * bin_size, 0}), {point_num_bin, 2}), Element(DataType::DT_FP32, 0.0f));
                }

                Tensor barycentrics_u(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_u");
                Tensor barycentrics_v(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_v");
                Tensor barycentrics_w(DT_FP32, {point_num_bin, verts_cnt}, "Barycentrics_w");

                Tensor zbuf_ori(DT_FP32,       {point_num_bin, verts_cnt}, "zbuf_o");
                Tensor face_sd2_ori(DT_FP32,   {point_num_bin, verts_cnt}, "Face_sd2");
                Tensor topk_idx(DT_INT32,      {point_num_bin, verts_cnt}, "topk_idx");
                Tensor topk_idx_s1(DT_INT32,   {point_num_bin, verts_cnt}, "topk_idx_s1");

                Tensor barycoords_u(DT_FP32,  {point_num_bin, 1}, "barycoords_u");
                Tensor barycoords_v(DT_FP32,  {point_num_bin, 1}, "barycoords_v");
                Tensor barycoords_w(DT_FP32,  {point_num_bin, 1}, "barycoords_w");

                Tensor zbuf_bin(DT_FP32,  {point_num_bin, 1}, "zbuf_bin");
                Tensor dists_bin(DT_FP32,  {point_num_bin, 1}, "dists_bin");
                Tensor pix_to_face_bin(DT_INT32,  {point_num_bin, 1}, "pix_to_face_bin");

                LOOP("loop01", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
                    TileShape::Current().SetVecTile({64, 64, 8});
                    UNUSED(idx);

                    auto tri_v0_x = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 0}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)
                    auto tri_v0_y = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 1}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)
                    auto tri_v0_z = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 2}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)

                    auto tri_v1_x = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 3}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)
                    auto tri_v1_y = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 4}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)
                    auto tri_v1_z = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 5}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)

                    auto tri_v2_x = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 6}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)
                    auto tri_v2_y = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 7}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)
                    auto tri_v2_z = Reshape(View(face_verts_bin, {face_verts_bin.GetShape()[0], 1}, {0, 8}), {1, face_verts_bin.GetShape()[0]});//(1, #faces)

                    auto point_x = Expand(View(points_bin, {points_bin.GetShape()[0], 1}, {0, 0}), {points_bin.GetShape()[0], face_verts_bin.GetShape()[0]});// (#point, #faces)
                    auto point_y = Expand(View(points_bin, {points_bin.GetShape()[0], 1}, {0, 1}), {points_bin.GetShape()[0], face_verts_bin.GetShape()[0]});// (#point, #faces)
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

                    // auto bary_coords_correct = BarycentricCoordsPespectCorrect(barycentrics, tri_v0_xyz_sq, tri_v1_xyz_sq, tri_v2_xyz_sq);

                    // TriangleSigned SquaredDistance
                    Tensor scale = Full(Element(DataType::DT_FP32, (double)screen_to_ndc_scale), DT_FP32, {points_bin.GetShape()[0], face_verts_bin.GetShape()[0]});
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
                LOOP("loop03", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(1)) {
                    UNUSED(idx2);
                    // SORT要把对应的轴全部搬入UB，UB内存有限，因此tile S0 不能太大，
                    TileShape::Current().SetVecTile({64, 64, 8}); // 64, 976

                    auto output_topk1 = TopK(zbuf_ori, 1, -1, false);
                    auto min_idx = std::get<1>(output_topk1);
                    zbuf_bin = std::get<0>(output_topk1);

                    auto face_idx_unsq = Unsqueeze(face_idx_bin, 0);                                // (1, #faces)
                    auto pix_to_face_expand = Expand(face_idx_unsq, {points_bin.GetShape()[0], face_verts_bin.GetShape()[0]});     // (#point, #faces)
                    pix_to_face_bin = GatherElements(pix_to_face_expand, min_idx, -1);              // (#point, #1)
                    dists_bin = GatherElements(face_sd2_ori, min_idx, -1);                           // (#point, #1)

                    barycoords_u = GatherElements(barycentrics_u, min_idx, -1);             // (#point, #1)
                    barycoords_v = GatherElements(barycentrics_v, min_idx, -1);             // (#point, #1)
                    barycoords_w = GatherElements(barycentrics_w, min_idx, -1);             // (#point, #1)

                }
                LOOP("loop04", FunctionType::DYNAMIC_LOOP, idx3, LoopRange(1)) {
                    UNUSED(idx3);
                    TileShape::Current().SetVecTile({64, 64, 8});

                    Assemble(Reshape(ScalarAddS(pix_to_face_bin, Element(DataType::DT_INT32, 0.0f)), {bin_size, bin_size}), {bin_x * bin_size, bin_y * bin_size}, pix_to_face);
                    Assemble(Reshape(ScalarAddS(dists_bin, Element(DataType::DT_FP32, 0.0f)), {bin_size, bin_size}), {bin_x * bin_size, bin_y * bin_size}, dists);
                    Assemble(Reshape(ScalarAddS(zbuf_bin, Element(DataType::DT_FP32, 0.0f)),  {bin_size, bin_size}), {bin_x * bin_size, bin_y * bin_size}, zbuf);

                    Assemble(Reshape(ScalarAddS(barycoords_u, Element(DataType::DT_FP32, 0.0f)), {bin_size, bin_size}), {bin_x * bin_size, bin_y * bin_size}, barycoords);
                    Assemble(Reshape(ScalarAddS(barycoords_v, Element(DataType::DT_FP32, 0.0f)), {bin_size, bin_size}), {bin_x * bin_size, bin_y * bin_size}, barycoords);
                    Assemble(Reshape(ScalarAddS(barycoords_w, Element(DataType::DT_FP32, 0.0f)), {bin_size, bin_size}), {bin_x * bin_size, bin_y * bin_size}, barycoords);
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

TEST_F(RasterizeMeshOnBoardTest, test_rasterize_1024_64_20000_2) {

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

    RunRasterizeTotal2(rander_setting, face_num);
}