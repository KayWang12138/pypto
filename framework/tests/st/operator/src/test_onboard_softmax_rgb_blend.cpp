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

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"
// #include "interpreter/ascend_tensor_data.h"
// #include "test_dynamic.h"

// using namespace ascend;

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class SoftmaxRgbBlendOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

static int GetTensorSize(Tensor &tensor) {
    auto &shape = tensor.GetShape();
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// TEST_F(SoftmaxRgbBlendOnBoardTest, test_softmax_rgb_blend2_large_shape) {

//     aclInit(nullptr);
//     rtSetDevice(0);

//     int N = 1;
//     int H = 1024;
//     int W = 1024;
//     int K = 2;
//     int C = 3;
//     int Cout = 4;

//     int total_ele = N * H * W;
//     std::vector<int64_t> colorsShape    = {total_ele, K, C};
//     std::vector<int64_t> fragmentsShape = {total_ele, K};
//     std::vector<int64_t> constShape   = {1};
//     std::vector<int64_t> bgColorShape = {C};
//     std::vector<int64_t> dstShape = {total_ele, Cout};

//     int colorsCapacity    = colorsShape[0] * colorsShape[1] * colorsShape[2];
//     int fragmentsCapacity = fragmentsShape[0] * fragmentsShape[1];
//     int bgColorCapacity =  bgColorShape[0];   
//     int dstCapacity       = dstShape[0] * dstShape[1];

//     uint64_t outputSize = dstCapacity * sizeof(DT_FP32);
//     uint8_t* out_ptr = allocDevAddr(outputSize);

//     PROGRAM("SoftmaxRgbBlend") {

//         void *x_ptr = readToDev(GetGoldenDir() + "" + "/color.bin", colorsCapacity);
//         void *y_ptr = readToDev(GetGoldenDir() + "" + "/pix_to_face.bin", fragmentsCapacity);
//         void *z_ptr = readToDev(GetGoldenDir() + "" + "/zbuf.bin", fragmentsCapacity);
//         void *u_ptr = readToDev(GetGoldenDir() + "" + "/dists.bin", fragmentsCapacity);
//         void *v_ptr = readToDev(GetGoldenDir() + "" + "/background_color.bin", bgColorCapacity);
//         void *m_ptr = readToDev(GetGoldenDir() + "" + "/mask.bin", fragmentsCapacity);
        
//         void* sigma_ptr = readToDev(GetGoldenDir() + "" + "/sigma.bin", 1);
//         void* gamma_ptr = readToDev(GetGoldenDir() + "" + "/gamma.bin", 1);
//         void* znear_ptr = readToDev(GetGoldenDir() + "" + "/znear.bin", 1);
//         void* zfar_ptr  = readToDev(GetGoldenDir() + "" + "/zfar.bin", 1);

//         Tensor colors(DataType::DT_FP32, colorsShape, (uint8_t *)x_ptr, "A");
//         Tensor pix_to_face(DataType::DT_FP32, fragmentsShape, (uint8_t *)y_ptr, "B");
//         Tensor zbuf(DataType::DT_FP32, fragmentsShape, (uint8_t *)z_ptr, "C");
//         Tensor dists(DataType::DT_FP32, fragmentsShape, (uint8_t *)u_ptr, "D");
//         Tensor background_color(DataType::DT_FP32, bgColorShape, (uint8_t *)v_ptr, "E");
//         Tensor mask_in(DataType::DT_FP32, fragmentsShape, (uint8_t *)m_ptr, "M");

//         Tensor sigma(DataType::DT_FP32, constShape, (uint8_t *)sigma_ptr, "F");
//         Tensor gamma(DataType::DT_FP32, constShape, (uint8_t *)gamma_ptr,"G");
//         Tensor znear(DataType::DT_FP32, constShape, (uint8_t *)znear_ptr,"H");
//         Tensor zfar (DataType::DT_FP32, constShape, (uint8_t *)zfar_ptr,"I");

//         Tensor output(DataType::DT_FP32, dstShape, out_ptr, "X");

//         AscendProgram::GetInstance().GetConfig().UpdateSimulateCalculation(false);  // playback need open this
    
//         FUNCTION("SoftmaxRgbBlend", FunctionType::STATIC, {colors, pix_to_face, dists, zbuf, sigma, gamma,
//                                                            background_color, znear, zfar, mask_in,
//                                                            output}) {
//         AscendProgram::GetInstance().GetTileShape().SetVecTileShapes({128, 8});
//         Element epsSingle(DataType::DT_FP32, 1e-10f);

//         // gen mask &  prob_mapalpha
//         Tensor mask =  mask_in; //GreaterS(pix_to_face, Element(DataType::DT_INT32, static_cast<int64_t>(0)));   //[N*H*W,K]
//         Tensor tmp1 = MulS(dists, Element(DataType::DT_FP32, -1.0f));        // [N*H*W,K]

//         Tensor sigmoid_in  = Div(tmp1, sigma);    // [N*H*W,K]

//         Tensor sigmoid_res = Sigmoid(sigmoid_in); // [N*H*W,K]
//         Tensor prob_map = Mul(sigmoid_res, mask); // [N*H*W,K]

//         Element ele_one(DataType::DT_FP32, 1.0f);
//         Tensor ones1 = Full(ele_one, DT_FP32, prob_map.GetShape());
//         Tensor prod_in  = Sub(ones1, prob_map);  // [N*H*W,K]
//         // Tensor alpha    = RowMaxSingle(prod_in, -1);//Prod(prod_in, -1);  [N*H*W,1]
//         Tensor alpha = Prod(prod_in, -1);    // [N*H*W,1]

//         // # z_inv = (zfar - fragments_zbuf) / (zfar - znear) * mask
//         Tensor z_inv = Mul(Div((Sub(zfar, zbuf)), Sub(zfar, znear)), mask);

//         // # z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
//         Tensor max_tmp = RowMaxSingle(z_inv, -1);
//         // Tensor z_inv_max = max_tmp; //ClipByValue(max_tmp_e, eps);  // [N*H*W,1]
//         Tensor z_inv_max = ScalarMaxS(max_tmp, epsSingle);

//         // # weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
//         Tensor weights_num = Mul(prob_map, Exp(Div(Sub(z_inv, z_inv_max), gamma))); // [N*H*W,K]

//         // # delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)
//         Tensor eps = Full(epsSingle, DT_FP32, z_inv_max.GetShape());
//         // Tensor delta = Exp(Div(Sub(eps, z_inv_max), gamma));
//         Tensor delta = ScalarMaxS(Exp(Div(Sub(eps, z_inv_max), gamma)), epsSingle); // ClipByValue(tmp10, eps);  // [N*H*W,1]

//         // # denom = weights_num.sum(dim=-1)[..., None] + delta
//         Tensor denom = Add(Sum(weights_num, -1), delta); // [N*H*W,1]

//         Tensor weighted_background = Mul(delta, background_color);// [N*H*W,3]

//         Tensor weights_num_unsqueeze = Unsqueeze(weights_num, -1);// [N*H*W,K,1]
//         // Tensor weights_num_unsqueeze = Unsqueeze(weights_num, -2);// [N*H*W,1,K]

//         AscendProgram::GetInstance().GetTileShape().SetVecTileShapes({128, 8, 8});

//         // # weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
//         Tensor tmp_co = Mul(weights_num_unsqueeze, colors);  //[N*H*W, K, 3]
//         Tensor weighted_colors = Sum(tmp_co, -2);// [N*H*W,1,3]
//         // Tensor weighted_colors = Sum(tmp_co, -1);// [N*H*W,3,K] -> [N*H*W,3,1]

//         // # pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
//         std::vector<int64_t> newShape(weighted_colors.GetStorage()->shape);
//         newShape.erase(newShape.begin() + newShape.size() - 2); //  [N*H*W,1,3]
//         // newShape.erase(newShape.begin() + newShape.size() - 1);  //[N*H*W,3,1]
//         Tensor weighted_colors_squeeze(weighted_colors->tensor->datatype, newShape);
//         weighted_colors_squeeze = Reshape(weighted_colors, newShape); // [N*H*W,1,3] -> [N*H*W,3]

//         Tensor color_rgb = Div(Add(weighted_colors_squeeze, weighted_background), denom); // [N*H*W,3]

//         Tensor ones2 = Full(ele_one, DT_FP32, alpha.GetShape());
//         Tensor color_alpha = Sub(ones2, alpha);        // [N*H*W,1]
//         output = Concat({color_rgb, color_alpha}, -1); // [N*H*W,4]
//         }
//     }

//     std::vector<float> golden(dstCapacity);
//     std::vector<float> res(dstCapacity);
//     machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputSize);
//     readInput(GetGoldenDir()  + "" + "/images2.bin", golden);
//     writeInput(GetGoldenDir() + "" + "/images_npu.bin", res);

//     for (int i = 0; i < 16; i++) {
//         printf("golden: %f,  res: %f \n", golden[i], res[i]);
//     } 
//     int ret = resultCmp(golden, res, 0.001f, 0, 1000, false, true, 0);
//     // int ret = 1;
//     EXPECT_EQ(ret, true);
// }

// TEST_F(SoftmaxRgbBlendOnBoardTest, test_softmax_rgb_blend2_1_16_128_8_dynamic) {
//     int TileS0 = 128;
//     int TileS1 = 8;
//     int TileS2 = 8;

//     AscendProgram::GetInstance().GetConfig().SetOnlyGenCodeSwitch(true);
//     AscendProgram::GetInstance().GetTileShape().SetVecTileShapes(TileS0, TileS1, TileS2);
//     AscendProgram::GetInstance().GetConfig().SetRuntimeMode(RuntimeMode::T_RUNTIME_EXECUTION);

//     int N = 2;
//     int H = 8;
//     int W = 128;
//     int K = 8;
//     int C = 3;
//     int Cout = 4;

//     int total_ele = N * H * W;
//     int sigal_batch_ele = H * W;
//     std::vector<int64_t> colorsShape    = {total_ele, K, C};
//     std::vector<int64_t> fragmentsShape = {total_ele, K};
//     std::vector<int64_t> constShape   = {1};
//     std::vector<int64_t> bgColorShape = {C};
//     std::vector<int64_t> dstShape = {total_ele, Cout};

//     int colorsCapacity    = colorsShape[0] * colorsShape[1] * colorsShape[2];
//     int fragmentsCapacity = fragmentsShape[0] * fragmentsShape[1];
//     int bgColorCapacity =  bgColorShape[0];   
//     int dstCapacity       = dstShape[0] * dstShape[1];


//     std::vector<float>colorsData(colorsCapacity);
//     std::vector<float>pixToFaceData(fragmentsCapacity);
//     std::vector<float>zbufData(fragmentsCapacity);
//     std::vector<float>distsData(fragmentsCapacity);
//     std::vector<float>bgcolorData(bgColorCapacity);
//     std::vector<float>maskData(fragmentsCapacity);
//     std::vector<float>sigmaData(1);
//     std::vector<float>gammaData(1);
//     std::vector<float>znearData(1);
//     std::vector<float>zfarData(1);


//     readInput<float>(GetGoldenDir() + "" + "/color.bin", colorsData);
//     readInput<float>(GetGoldenDir() + "" + "/pix_to_face.bin", pixToFaceData);
//     readInput<float>(GetGoldenDir() + "" + "/zbuf.bin", zbufData);
//     readInput<float>(GetGoldenDir() + "" + "/dists.bin", distsData);
//     readInput<float>(GetGoldenDir() + "" + "/background_color.bin", bgcolorData);
//     readInput<float>(GetGoldenDir() + "" + "/mask.bin", maskData);
    
//     readInput<float>(GetGoldenDir() + "" + "/sigma.bin", sigmaData);
//     readInput<float>(GetGoldenDir() + "" + "/gamma.bin", gammaData);
//     readInput<float>(GetGoldenDir() + "" + "/znear.bin", znearData);
//     readInput<float>(GetGoldenDir() + "" + "/zfar.bin", zfarData);

//     Tensor colors(DataType::DT_FP32, colorsShape, "A");
//     Tensor pix_to_face(DataType::DT_FP32, fragmentsShape, "B");
//     Tensor zbuf(DataType::DT_FP32, fragmentsShape, "C");
//     Tensor dists(DataType::DT_FP32, fragmentsShape, "D");
//     Tensor background_color(DataType::DT_FP32, bgColorShape, "E");
//     Tensor mask_in(DataType::DT_FP32, fragmentsShape, "M");

//     Tensor sigma(DataType::DT_FP32, constShape, "F");
//     Tensor gamma(DataType::DT_FP32, constShape, "G");
//     Tensor znear(DataType::DT_FP32, constShape, "H");
//     Tensor zfar (DataType::DT_FP32, constShape, "I");

//     Tensor output(DataType::DT_FP32, dstShape, "X");

//     AscendProgramData::GetInstance().Reset();
//     AscendProgramData::GetInstance().AppendInputs({
//         AscendTensorData::CreateTensor<float32_t>(colors, colorsData),
//         AscendTensorData::CreateTensor<float32_t>(pix_to_face, pixToFaceData),
//         AscendTensorData::CreateTensor<float32_t>(mask_in, maskData),
//         AscendTensorData::CreateTensor<float32_t>(dists,  distsData),
//         AscendTensorData::CreateTensor<float32_t>(zbuf,   zbufData),
//         AscendTensorData::CreateTensor<float32_t>(background_color, bgcolorData),
//         AscendTensorData::CreateTensor<float32_t>(sigma, sigmaData),
//         AscendTensorData::CreateTensor<float32_t>(gamma, gammaData),
//         AscendTensorData::CreateTensor<float32_t>(znear, znearData),
//         AscendTensorData::CreateTensor<float32_t>(zfar,  zfarData),

//     });
//     AscendProgramData::GetInstance().AppendOutputs({
//         AscendTensorData::CreateConstantTensor<float32_t>(output, 1.0f),
//     });

//     FUNCTION("SoftmaxRgbBlendDynamic", FunctionType::DYNAMIC, {colors, pix_to_face, mask_in, dists, zbuf, background_color,
//                                                                 sigma, gamma, znear, zfar}, {output}){
//         LOOP("L0", FunctionType::DYNAMIC_LOOP, bs_index, LoopRange(N)) {

//             LOOP("L0_Incore", FunctionType::DYNLOOP_INCORE_GRAPH, index, LoopRange(sigal_batch_ele / TileS0)) {
//             Tensor mask_in_l = View(mask_in, {TileS0, K}, {bs_index * sigal_batch_ele + index * TileS0, 0});
//             Tensor dists_l   = View(dists,   {TileS0, K}, {bs_index * sigal_batch_ele + index * TileS0, 0});
//             Tensor zbuf_l    = View(zbuf,    {TileS0, K}, {bs_index * sigal_batch_ele + index * TileS0, 0});
//             Tensor zfar_e  = Expand(Unsqueeze(zfar, 0),  {1,K});
//             Tensor znear_e = Expand(Unsqueeze(znear, 0), {1,K});
//             Tensor colors_l = View(colors, {TileS0, K, C}, {bs_index * sigal_batch_ele + index * TileS0, 0, 0});

//             // FUNCTION("SoftmaxRgbBlend", FunctionType::STATIC) {
//                 // AscendProgram::GetInstance().GetTileShape().SetVecTileShapes({64, 8});
//                 Element epsSingle(DataType::DT_FP32, 1e-10f);

//                 // gen mask &  prob_mapalpha
//                 Tensor mask =  mask_in_l; //GreaterS(pix_to_face, Element(DataType::DT_INT32, static_cast<int64_t>(0)));   //[N*H*W,K]
//                 Tensor tmp1 = MulS(dists_l, Element(DataType::DT_FP32, -1.0f));        // [N*H*W,K]

//                 Tensor sigmoid_in  = Div(tmp1, sigma);    // [N*H*W,K]

//                 Tensor sigmoid_res = Sigmoid(sigmoid_in); // [N*H*W,K]
//                 Tensor prob_map = Mul(sigmoid_res, mask); // [N*H*W,K]

//                 Element ele_one(DataType::DT_FP32, 1.0f);
//                 Tensor ones1 = Full(ele_one, DT_FP32, prob_map.GetShape());
//                 Tensor prod_in  = Sub(ones1, prob_map);  // [N*H*W,K]
//                 // Tensor alpha    = RowMaxSingle(prod_in, -1);//Prod(prod_in, -1);  [N*H*W,1]
//                 Tensor alpha = Prod(prod_in, -1);    // [N*H*W,1]

//                 // # z_inv = (zfar - fragments_zbuf) / (zfar - znear) * mask
//                 Tensor z_inv = Mul(Div((Sub(zfar_e, zbuf_l)), Sub(zfar_e, znear_e)), mask);

//                 // # z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
//                 Tensor max_tmp = RowMaxSingle(z_inv, -1);
//                 // Tensor z_inv_max = max_tmp; //ClipByValue(max_tmp_e, eps);  // [N*H*W,1]
//                 Tensor z_inv_max = ScalarMaxS(max_tmp, epsSingle);

//                 // # weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
//                 Tensor weights_num = Mul(prob_map, Exp(Div(Sub(z_inv, z_inv_max), gamma))); // [N*H*W,K]

//                 // # delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)
//                 Tensor eps = Full(epsSingle, DT_FP32, z_inv_max.GetShape());
//                 // Tensor delta = Exp(Div(Sub(eps, z_inv_max), gamma));
//                 Tensor delta = ScalarMaxS(Exp(Div(Sub(eps, z_inv_max), gamma)), epsSingle); // ClipByValue(tmp10, eps);  // [N*H*W,1]

//                 // # denom = weights_num.sum(dim=-1)[..., None] + delta
//                 Tensor denom = Add(Sum(weights_num, -1), delta); // [N*H*W,1]

//                 Tensor weighted_background = Mul(delta, background_color);// [N*H*W,3]

//                 Tensor weights_num_unsqueeze = Unsqueeze(weights_num, -1);// [N*H*W,K,1]

//                 // # weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
//                 Tensor tmp_co = Mul(weights_num_unsqueeze, colors_l);  //[N*H*W, K, 3]
//                 Tensor weighted_colors = Sum(tmp_co, -2);// [N*H*W,1,3]

//                 // # pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
//                 std::vector<int64_t> newShape(weighted_colors.GetStorage()->shape);
//                 newShape.erase(newShape.begin() + newShape.size() - 2); //  [N*H*W,1,3]
//                 // newShape.erase(newShape.begin() + newShape.size() - 1);  //[N*H*W,3,1]
//                 Tensor weighted_colors_squeeze(weighted_colors->tensor->datatype, newShape);
//                 weighted_colors_squeeze = Reshape(weighted_colors, newShape); // [N*H*W,1,3] -> [N*H*W,3]

//                 Tensor color_rgb = Div(Add(weighted_colors_squeeze, weighted_background), denom); // [N*H*W,3]

//                 Tensor ones2 = Full(ele_one, DT_FP32, alpha.GetShape());
//                 Tensor color_alpha = Sub(ones2, alpha);        // [N*H*W,1]
//                 Tensor output_t = Concat({color_rgb, color_alpha}, -1); // [N*H*W,4]
//                 DAssemble(output_t, {bs_index * sigal_batch_ele + index * TileS0, 0}, output);
//             }
//         }
//     }
//     auto funcop = AscendProgram::GetInstance().GetLastFunction()->GetDyndevAttribute();

//     DynFuncRunner::Run(funcop);

//     auto res = ascend::AscendProgramData::GetInstance().GetOutputData(0);

//     std::vector<float> golden(dstCapacity);
//     readInput(GetGoldenDir()  + "" + "/images2.bin", golden);


//     for (int i = 0; i < 16; i++) {
//         if(abs(*((float *)res->data() + i) - golden[i]) > 0.001)
//             printf("golden: %f,  res: %f \n", golden[i], *((float *)res->data() + i));
//     }

//     std::ofstream ascendOutFile(GetGoldenDir() + "" + "/images_npu_dynamic.bin", std::ios::out | std::ios::binary);
//     if (!ascendOutFile) {
//         std::cerr << "Can not open out file!" << std::endl;
//     }
//     ascendOutFile.write((char *)res->data(), GetTensorSize(output) * sizeof(float));
//     ascendOutFile.close();

//     int ret = resultCmp(golden, (float *)res->data(), 0.001f, 0, 1000, false, false, 0);
//     EXPECT_EQ(ret, true);
// }

// TEST_F(SoftmaxRgbBlendOnBoardTest, test_operation_prod_dynamic) {

//     // 创建PROGRAM
//     AscendProgram::GetInstance().GetConfig().SetOnlyGenCodeSwitch(true);
//     AscendProgram::GetInstance().GetTileShape().SetVecTileShapes(128, 32);
//     // AscendProgram::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});
//     AscendProgram::GetInstance().GetConfig().SetRuntimeMode(RuntimeMode::T_RUNTIME_EXECUTION);

//     // shape, capacity, ptr
//     int S0 = 128;
//     int S1 = 23;

//     Tensor inputX(DT_FP32, {S0, S1}, "inputX");
//     Tensor output(DT_FP32, {S0, 1},  "output");

//     std::vector<float> dataX(GetTensorSize(inputX), 0.0f);
//     std::vector<float> golden(GetTensorSize(output), 0.0f);
//     readInput<float>(GetGoldenDir() + "/prod_x.bin", dataX);
//     readInput<float>(GetGoldenDir() + "/prod_res.bin", golden);

//     AscendProgramData::GetInstance().Reset();
//     AscendProgramData::GetInstance().AppendInputs({
//         AscendTensorData::CreateTensor<float>(inputX, dataX),
//     });
//     AscendProgramData::GetInstance().AppendOutputs({
//         AscendTensorData::CreateConstantTensor<float>(output, 0.0),
//     });

//     FUNCTION("ScopeFunc", FunctionType::DYNAMIC, {inputX}, {output}) {
//         FUNCTION("PROD_T", FunctionType::STATIC) {
//             output = Prod(inputX, -1);
//         }
//     }

//     auto funcop = AscendProgram::GetInstance().GetLastFunction()->GetDyndevAttribute();
//     DynFuncRunner::Run(funcop);
//     auto outs = ascend::AscendProgramData::GetInstance().GetOutputData(0);

//     for (int i = 0; i < 16; i++) {
//         printf("golden[%d]: %f,  res[%d]: %f \n", i, golden[i], i, *((float *)outs->data() + i));
//     }

//     std::ofstream ascendOutFile(GetGoldenDir() + "" + "/prod_res_dynamic.bin", std::ios::out | std::ios::binary);
//     if (!ascendOutFile) {
//         std::cerr << "Can not open out file!" << std::endl;
//     }
//     ascendOutFile.write((char *)outs->data(), GetTensorSize(output) * sizeof(float));
//     ascendOutFile.close();

//     EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
// }

// Tensor Squeeze(Tensor a, int axis) {
//     std::vector<int64_t> newShape(a.GetStorage()->shape);
//     // printf("ori shape: %d, %d, %d \n", newShape[0], newShape[1], newShape[2]);
//     if (axis < 0)
//         axis = newShape.size() + axis;
//     newShape.erase(newShape.begin() + axis);
//     // printf("shape: %d, %d \n", newShape[0], newShape[1]);
//     auto out = Reshape(a, newShape);
//     return out;
// }

TEST_F(SoftmaxRgbBlendOnBoardTest, test_softmax_rgb_blend2_1_128_128_2_dynamic) {
    SetInterpreterConfig();
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, false);

    int TileS0 = 128;
    int TileS1 = 8;
    int TileS2 = 8;

    int N = 2;
    int H = 64;
    int W = 128;
    int K = 2;
    int C = 3;
    int Cout = 4;

    int total_ele = N * H * W;
    int sigal_batch_ele = H * W;
    std::vector<int64_t> colorsShape    = {total_ele, K, C};
    std::vector<int64_t> fragmentsShape = {total_ele, K};
    std::vector<int64_t> constShape   = {1};
    std::vector<int64_t> bgColorShape = {C};
    std::vector<int64_t> dstShape = {total_ele, Cout};

    int colorsCapacity    = colorsShape[0] * colorsShape[1] * colorsShape[2];
    int fragmentsCapacity = fragmentsShape[0] * fragmentsShape[1];
    int bgColorCapacity   = bgColorShape[0];   
    int dstCapacity       = dstShape[0] * dstShape[1];

    std::vector<float>colorsData(colorsCapacity);
    std::vector<float>pixToFaceData(fragmentsCapacity);
    std::vector<float>zbufData(fragmentsCapacity);
    std::vector<float>distsData(fragmentsCapacity);
    std::vector<float>bgcolorData(bgColorCapacity);
    std::vector<float>maskData(fragmentsCapacity);
    std::vector<float>sigmaData(1);
    std::vector<float>gammaData(1);
    std::vector<float>znearData(1);
    std::vector<float>zfarData(1);

    readInput<float>(GetGoldenDir() + "" + "/color.bin", colorsData);
    readInput<float>(GetGoldenDir() + "" + "/pix_to_face.bin", pixToFaceData);
    readInput<float>(GetGoldenDir() + "" + "/zbuf.bin", zbufData);
    readInput<float>(GetGoldenDir() + "" + "/dists.bin", distsData);
    readInput<float>(GetGoldenDir() + "" + "/background_color.bin", bgcolorData);
    readInput<float>(GetGoldenDir() + "" + "/mask.bin", maskData);
    
    readInput<float>(GetGoldenDir() + "" + "/sigma.bin", sigmaData);
    readInput<float>(GetGoldenDir() + "" + "/gamma.bin", gammaData);
    readInput<float>(GetGoldenDir() + "" + "/znear.bin", znearData);
    readInput<float>(GetGoldenDir() + "" + "/zfar.bin", zfarData);

    Tensor colors(DataType::DT_FP32, colorsShape, "A");
    Tensor pix_to_face(DataType::DT_FP32, fragmentsShape, "B");
    Tensor zbuf(DataType::DT_FP32, fragmentsShape, "C");
    Tensor dists(DataType::DT_FP32, fragmentsShape, "D");
    Tensor background_color(DataType::DT_FP32, bgColorShape, "E");
    Tensor mask_in(DataType::DT_FP32, fragmentsShape, "M");

    Tensor sigma(DataType::DT_FP32, constShape, "F");
    Tensor gamma(DataType::DT_FP32, constShape, "G");
    Tensor znear(DataType::DT_FP32, constShape, "H");
    Tensor zfar (DataType::DT_FP32, constShape, "I");

    Tensor output(DataType::DT_FP32, dstShape, "X");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float32_t>(colors, colorsData),
        RawTensorData::CreateTensor<float32_t>(pix_to_face, pixToFaceData),
        RawTensorData::CreateTensor<float32_t>(mask_in, maskData),
        RawTensorData::CreateTensor<float32_t>(dists,  distsData),
        RawTensorData::CreateTensor<float32_t>(zbuf,   zbufData),
        RawTensorData::CreateTensor<float32_t>(background_color, bgcolorData),
        RawTensorData::CreateTensor<float32_t>(sigma, sigmaData),
        RawTensorData::CreateTensor<float32_t>(gamma, gammaData),
        RawTensorData::CreateTensor<float32_t>(znear, znearData),
        RawTensorData::CreateTensor<float32_t>(zfar,  zfarData),

    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float32_t>(output, 1.0f),
    });

   TileShape::Current().SetVecTile({TileS0, TileS1, TileS2});

    FUNCTION("SoftmaxRgbBlendDynamic", {colors, pix_to_face, mask_in, dists, zbuf, background_color,
                                                                sigma, gamma, znear, zfar}, {output}){
        LOOP("L0", FunctionType::DYNAMIC_LOOP, bs_index, LoopRange(N)) {
            Tensor mask_in_l = View(mask_in, {sigal_batch_ele, K}, {bs_index * sigal_batch_ele, 0});
            Tensor dists_l   = View(dists,   {sigal_batch_ele, K}, {bs_index * sigal_batch_ele, 0});
            Tensor zbuf_l    = View(zbuf,    {sigal_batch_ele, K}, {bs_index * sigal_batch_ele, 0});
            Tensor zfar_e  = Expand(Unsqueeze(zfar, 0),  {1,K});
            Tensor znear_e = Expand(Unsqueeze(znear, 0), {1,K});
            Tensor colors_l = View(colors, {sigal_batch_ele, K, C}, {bs_index * sigal_batch_ele, 0, 0});

            Element epsSingle(DataType::DT_FP32, 1e-10f);
            Element ele_one(DataType::DT_FP32, 1.0f);

            // gen mask &  prob_mapalpha
            Tensor mask =  mask_in_l; //GreaterS(pix_to_face, Element(DataType::DT_INT32, static_cast<int64_t>(0)));   //[N*H*W,K]
            Tensor tmp1 = ScalarMulS(dists_l, Element(DataType::DT_FP32, -1.0f));        // [N*H*W,K]

            Tensor sigmoid_in  = Div(tmp1, sigma);    // [N*H*W,K]

            Tensor sigmoid_res = Sigmoid(sigmoid_in); // [N*H*W,K]
            Tensor prob_map = Mul(sigmoid_res, mask); // [N*H*W,K]

            Tensor prod_in  = ScalarSubS(prob_map, ele_one, true);  // [N*H*W,K]
            Tensor alpha    = Amax(prod_in, -1, true);//Prod(prod_in, -1);  [N*H*W,1]
            // Tensor alpha = Prod(prod_in, -1);    // [N*H*W,1]

            // # z_inv = (zfar - fragments_zbuf) / (zfar - znear) * mask
            Tensor z_inv = Mul(Div((Sub(zfar_e, zbuf_l)), Sub(zfar_e, znear_e)), mask);

            // # z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
            Tensor max_tmp = Amax(z_inv, -1, true);
            // Tensor z_inv_max = max_tmp; //ClipByValue(max_tmp_e, eps);  // [N*H*W,1]
            Tensor z_inv_max = ScalarMaxS(max_tmp, epsSingle);

            // # weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
            Tensor weights_num = Mul(prob_map, Exp(Div(Sub(z_inv, z_inv_max), gamma))); // [N*H*W,K]

            // # delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)
            Tensor delta = ScalarMaxS(Exp(Div(ScalarSubS(z_inv_max, epsSingle, true), gamma)), epsSingle); // ClipByValue(tmp10, eps);  // [N*H*W,1]

            // # denom = weights_num.sum(dim=-1)[..., None] + delta
            Tensor denom = Add(Sum(weights_num, -1, true), delta); // [N*H*W,1]

            Tensor weighted_background = Mul(delta, background_color);// [N*H*W,3]

            Tensor weights_num_unsqueeze = Unsqueeze(weights_num, -1);// [N*H*W,K,1]

            // # weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
            Tensor tmp_co = Mul(weights_num_unsqueeze, colors_l);  //[N*H*W, K, 3]
            Tensor weighted_colors = Sum(tmp_co, -2, false);// [N*H*W,3]

            // # pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
            // std::vector<int64_t> newShape(weighted_colors.GetStorage()->shape);
            // newShape.erase(newShape.begin() + newShape.size() - 2); //  [N*H*W,1,3]
            // Tensor weighted_colors_squeeze(weighted_colors->tensor->datatype, newShape);
            // weighted_colors_squeeze = Reshape(weighted_colors, newShape); // [N*H*W,1,3] -> [N*H*W,3]
            // Tensor weighted_colors_squeeze = Sequeue(weighted_colors, -2); // [N*H*W,1,3] -> [N*H*W,3]

            Tensor color_rgb = Div(Add(weighted_colors, weighted_background), denom); // [N*H*W,3]

            Tensor color_alpha = ScalarSubS(alpha, ele_one, true);  // [N*H*W,1]
            Tensor output_t = Cat({color_rgb, color_alpha}, -1); // [N*H*W,4]
            Assemble(output_t, {bs_index * sigal_batch_ele, 0}, output);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto res = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);

    std::vector<float> golden(dstCapacity);
    readInput(GetGoldenDir()  + "" + "/images_torch.bin", golden);

    for (int i = 0; i < 16; i++) {
        if(abs(*((float *)res->data() + i) - golden[i]) > 0.001)
            printf("golden: %f,  res: %f \n", golden[i], *((float *)res->data() + i));
    }

    std::ofstream ascendOutFile(GetGoldenDir() + "" + "/images_npu_dynamic.bin", std::ios::out | std::ios::binary);
    if (!ascendOutFile) {
        std::cerr << "Can not open out file!" << std::endl;
    }
    ascendOutFile.write((char *)res->data(), GetTensorSize(output) * sizeof(float));
    ascendOutFile.close();

    int ret = resultCmp(golden, (float *)res->data(), 0.001f, 0, 1000, false, false, 0);
    EXPECT_EQ(ret, true);
}