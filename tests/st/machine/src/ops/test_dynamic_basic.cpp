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
 * \file test_dynamic_pa.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "common/data_type.h"
#include "interface/function/function.h"
#include "operation/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dynamic.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class DynamicBasicTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
public:
    void SetUp() override {
        npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
        Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
        Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    }
};

namespace {
    // Constants to replace magic numbers
    constexpr int LOOP_COUNT = 8;
    constexpr int CONDITION_THRESHOLD = 6;
}

TEST_F(DynamicBasicTest, TestHybridLoopIf2) {
    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 11.0),
        RawTensorData::CreateConstantTensor<float>(t1, 20.0),
        RawTensorData::CreateConstantTensor<float>(t2, 30.0),
        RawTensorData::CreateConstantTensor<float>(t3, 40.0),
        RawTensorData::CreateConstantTensor<float>(t4, 50.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    //clc
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, t2, t3, t4}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT)) {
            auto r0 = Add(t0, t1);
            r0 = Mul(r0, t1); // +t0, +t1
            IF(i < CONDITION_THRESHOLD) {
                r0 = Sub(r0, t2); // +t2 * 6
            } ELSE {
                r0 = Sub(r0, t3); // +t3 * 8
            }
            out = Add(r0, t4);
        }
    }

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    // EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.004f));
}

void TestLoopDViewDAssemble(const Tensor &t0, const Tensor &t1, const Tensor &blockTable, Tensor &out, int s) {
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s)) {
            SymbolicScalar idx = GetInputDataInt32Dim2(blockTable, i, 0);
            Tensor t0s = DView(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            DAssemble(t1, {0, 0}, qi);
            DAssemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            DAssemble(t0s, {0, 0}, ki);
            DAssemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            DAssemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(DynamicBasicTest, TestDD) {
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopDViewDAssemble(t0, t1, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    std::vector<float> golden(n * s * s, 128.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, DynamicRawShape) {
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");              // [32, 32]
    Tensor out(DT_FP32, {-1, s}, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(GetInputShapeDim(t0, 0) / s)) {
            Tensor t0s = DView(t0, {s, s}, {idx * s, 0});
            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, t0s, t1);
            DAssemble(t2, {idx * s, 0}, out);
        }
    }

    int n = 8;
    Tensor arg0(DT_FP32, {n * s, s});
    Tensor out0(DT_FP32, {n * s, s});
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(arg0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out0, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    std::vector<float> golden(n * s * s, 64.0f);
    auto outs = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, DynamicRawShapeUnalign) {
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor out(DT_FP32, {-1, s}, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {t0}, {out}) {
        auto shape0 = GetInputShapeDim(t0, 0);
        auto t1 = Tensor(t0.GetDataType(), {shape0, s});
        auto loop1 = (shape0 + s - 1) / s;
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop1)) {
            Tensor t0s = DView(t0, {s, s}, {idx * s, 0});
            auto t = AddS(t0s, Element(DT_FP32, 3.0));
            DAssemble(t, {idx * s, 0}, t1);
        }

        // check t1 use dynshape from t0
        auto loop2 = (GetInputShapeDim(t1, 0) + s - 1) / s;
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop2), {}, true) {
            Tensor t1s = DView(t1, {s, s}, {idx * s, 0});
            auto t = SubS(t1s, Element(DT_FP32, 1.0));
            DAssemble(t, {idx * s, 0}, out);
        }
    }

    int s0 = 200;
    Tensor arg0(DT_FP32, {s0, s});
    Tensor out0(DT_FP32, {s0, s});
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(arg0, 3.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out0, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop, DynFuncRunnerConfig(arg0->GetDataSize()));
    std::vector<float> golden(s0, 5.0f);
    auto outs = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}


TEST_F(DynamicBasicTest, TestInplace) {
    Tensor t0(DT_FP32, {32, 32}, "t0");
    Tensor t1(DT_FP32, {32, 32}, "t1");
    Tensor t2(DT_FP32, {32, 32}, "t2");
    Tensor t3(DT_FP32, {32, 32}, "t3");

    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1}, {t3}, {{t2, t0}}) {
        LOOP("l0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            UNUSED(i);
            t3 = Add(t0, t1);
            DAssemble(t3, {0, 0}, t2);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(t3, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    std::vector<float> golden(32 * 32, 3.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetInputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestStaticUnderDynDev) {
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {n * s, s}, "t1");  // [32, 32]
    Tensor out(DT_FP32, {n * s, s}, "out");
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1}, {out}) {
        FUNCTION("S0", FunctionType::STATIC) {
            out = Sub(t1, t0);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    std::vector<float> golden(n * s, 1.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestStaticLoop) {
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {n * s, s}, "t1");  // [32, 32]
    Tensor t2(DT_FP32, {s, s}, "t2");  // [32, 32]
    Tensor out(DT_FP32, {n * s, s}, "out");
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, t2}, {out}) {
        Tensor s0Out;
        FUNCTION("S0", FunctionType::STATIC) {
            s0Out = Sub(t1, t0);
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT)) {
            Tensor t0s = DView(s0Out, {s, s}, {i * s, 0});
            Tensor t3 = Add(t0s, t2);
            DAssemble(t3, {i * s, 0}, out);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateConstantTensor<float>(t2, 3.0),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop);
    std::vector<float> outGolden(n * s, 4.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outGolden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestInnerLoopOrder) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(512, 512);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});

    int vecLen = 128;
    int loopNum = 5;
    int tileNum = 4;
    Tensor inputA(DT_FP32, {loopNum, vecLen}, "inputA");
    Tensor inputB(DT_FP32, {tileNum, vecLen}, "inputB");
    Tensor output(DT_FP32, {1, vecLen}, "out");

    FUNCTION("main", FunctionType::DYNAMIC, {inputA, inputB}, {output}) {
        LOOP("Outer", FunctionType::DYNAMIC_LOOP, i, LoopRange(tileNum)) {
            Tensor tileB(DT_FP32, {1, vecLen}, "tileB");
            LOOP("Inner", FunctionType::DYNAMIC_LOOP, j, LoopRange(1)) {
                (void)j;
                auto tile = DView(inputB, {1, vecLen}, {i, 0});
                tileB = MulS(tile, Element(DataType::DT_FP32, 1.0));
            }

            LOOP("Inner2", FunctionType::DYNAMIC_LOOP, k, LoopRange(loopNum)) {
                auto tileA = DView(inputA, {1, vecLen}, {k, 0});
                tileB = Add(tileA, tileB);
            }

            LOOP("Inner3", FunctionType::DYNAMIC_LOOP, l, LoopRange(1)) {
                (void)l;
                tileB = MulS(tileB, Element(DataType::DT_FP32, 1.0));
                DAssemble(tileB, {i, 0}, output);
            }
        }
    }

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);
}

TEST_F(DynamicBasicTest, TestDeviceMachineOnModel) {
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopDViewDAssemble(t0, t1, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    DynFuncRunner::Run(funcop, {false, 25, 5});

    std::cout << "test -> blockdim = 16, aicpunum = 4" << std::endl;
    DynFuncRunner::Run(funcop, {false, 16, 4});

    std::cout << "test -> blockdim = 9, aicpunum = 4" << std::endl;
    DynFuncRunner::Run(funcop, {false, 9, 4});

    std::cout << "test -> blockdim = 8, aicpunum = 3" << std::endl;
    DynFuncRunner::Run(funcop, {false, 8, 3});

    std::cout << "test -> blockdim = 1, aicpunum = 3" << std::endl;
    DynFuncRunner::Run(funcop, {false, 1, 3});
}

TEST_F(DynamicBasicTest, TestDeviceMachineBlockdimOnBoard) {
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopDViewDAssemble(t0, t1, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();

#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcop, {true, 15, 4});
    std::vector<float> golden(n * s * s, 128.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));

    DynFuncRunner::Run(funcop, {true, 7, 3});
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs1->data(), 0.001f));
#endif
}
