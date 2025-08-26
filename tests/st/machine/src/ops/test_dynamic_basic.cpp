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
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "operation/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "machine/device/dynamic/device_utils.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

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

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * s * s, 128.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestTT) {
    int s = 64;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [64 * 8, 64]
    Tensor t1(DT_FP32, {n * s, s}, "t1");  // [64 * 8, 64]
    Tensor out(DT_FP32, {n * s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(8)) {
            Tensor t0s = DView(t0, {s, s}, {idx * s, 0});
            Tensor t1s = DView(t1, {s, s}, {idx * s, 0});
            Tensor o = Add(t0s, t1s);
            DAssemble(o, {idx * s, 0}, out);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * s * s, 3.0f);
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), FuncRunnerConfig(arg0->GetDataSize()));
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
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

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 25, 5});

    std::cout << "test -> blockdim = 16, aicpunum = 4" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 16, 4});

    std::cout << "test -> blockdim = 9, aicpunum = 4" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 9, 4});

    std::cout << "test -> blockdim = 8, aicpunum = 3" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 8, 3});

    std::cout << "test -> blockdim = 1, aicpunum = 3" << std::endl;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {false, 1, 3});
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {true, 15, 4});
    std::vector<float> golden(n * s * s, 128.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestDeviceMachineBlockdimOnBoard1) {
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

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), {true, 7, 3});
    auto outs1 = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    std::vector<float> golden(n * s * s, 128.0f);
    EXPECT_TRUE(resultCmp(golden, (float *)outs1->data(), 0.001f));
#endif
}

namespace DynamicTest {

TEST_F(DynamicBasicTest, TestTensorExtract) {
    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = 32;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    Tensor output(DT_INT32, {1, s}, "output");
    int v = 20;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, v),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {inputA}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2));
            output = TensorExtract(t0, {0, 3});
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_EQ(v + 2, *(int32_t *)outs->data());
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorData) {
    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    Tensor inputC(DT_FP32, {n, s}, "inputC");
    Tensor output(DT_FP32, {n, n}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<float>(inputC, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {inputA, inputC}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2));
            SymbolicScalar v0 = GetTensorDataInt32(t0, 0, 1); // t0[0, 1] == 3
            SymbolicScalar v1 = GetTensorDataInt32(t0, 0, 2); // t0[0, 2] == 3
            auto t2 = DView(inputC, {n, n}, {0, v0 * n});
            auto t3 = DView(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * n, 4.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataExpr) {
    int tiling = 32;
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    Tensor inputC(DT_FP32, {n, s}, "inputC");
    Tensor output(DT_FP32, {n, n}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<float>(inputC, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {inputA, inputC}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2));
            SymbolicScalar v0 = GetTensorDataInt32(t0, 0, 1); // t0[0, 1] == 3
            SymbolicScalar v1 = GetTensorDataInt32(t0, 0, 2); // t0[0, 2] == 3
            SymbolicScalar v2 = GetInputDataInt32Dim2(inputA, 0, 1); // inputA[0, 1] == 3
            SymbolicScalar v3 = GetInputDataInt32Dim2(inputA, 0, 2); // inputA[0, 2] == 3
            auto t2 = DView(inputC, {n, n}, {0, (v0 + v2 + i / i) * n});
            auto t3 = DView(inputC, {n, n}, {0, (v1 + v3 + i / i) * n});
            output = Mul(t2, t3);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * n, 4.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}


TEST_F(DynamicBasicTest, TestVectorDup) {
    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_FP32, {n, n}, "output");

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            SymbolicScalar v = 20;
            output = VectorDuplicate(v + 30, DT_INT32, {n, n});
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden(n * n, 50);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestTensorInsert) {
    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_INT32, {n}, "output");

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            auto tmp = VectorDuplicate(20, DT_INT32, {1});
            TensorInsert(tmp, {i}, output);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden(n, 20);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSetTensorData) {
    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_INT32, {n}, "output");

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            SetTensorDataInt32(30, {i / 2 * 2 + i % 2}, output);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden(n, 30);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSetTensorDataExpr) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);

    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor output(DT_INT32, {n, n, n}, "output");

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n)) {
                for (int k = 0; k < n; k++) {
                    SetTensorDataInt32(i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        golden[i] = i;
    }
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetSetTensorDataExpr) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);

    int tiling = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tiling, tiling, tiling);

    int n = tiling * 1;
    int init = 10;
    Tensor input(DT_INT32, {n, n, n}, "input");
    Tensor output(DT_INT32, {n, n, n}, "output");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input, init),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FUNCTION("main", FunctionType::DYNAMIC, {input}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n)) {
                auto add = Add(input, input);
                for (int k = 0; k < n; k++) {
                    SymbolicScalar s = GetTensorDataInt32(add, {i, j, k});
                    SetTensorDataInt32(s + i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<int32_t> golden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        golden[i] = init + init + i;
    }
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (int32_t *)outs->data(), 0.001f));
#endif
}

}
