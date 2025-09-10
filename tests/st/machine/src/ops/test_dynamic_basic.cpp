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
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
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
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1, t2, t3, t4}, {out}) {
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

void TestLoopViewAssemble(const Tensor &t0, const Tensor &t1, const Tensor &blockTable, Tensor &out, int s) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s)) {
            SymbolicScalar idx = GetInputData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
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
    TestLoopViewAssemble(t0, t1, blockTable, out, s);

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

    std::vector<std::string> funcName = {"TENSOR_main"};
    config::SetPassConfig("FunctionUnroll", "LoopUnroll", "CONVERT_TO_STATIC", funcName);
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(8)) {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            Tensor t1s = View(t1, {s, s}, {idx * s, 0});
            Tensor o = Add(t0s, t1s);
            Assemble(o, {idx * s, 0}, out);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> golden(n * s * s, 3.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestCheckPointRestore) {
    int s = 16;
    Tensor t;
    Tensor t0(DT_FP32, {s, s}, "t0");

    FunctionConfig config;
    FUNCTION("main", config, {t}, {t0}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
            IF(idx == 0) {
                t0 = VectorDuplicate(Element(DT_FP32, 1.0f), DT_FP32, {s, s});
            }
            ELSE {
                t0 = VectorDuplicate(Element(DT_FP32, 2.0f), DT_FP32, {s, s});
            }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
            (void)idx;
            t0 = VectorDuplicate(Element(DT_FP32, 1.0f), DT_FP32, {s, s});
        }
    }
    EXPECT_EQ(t0->tensor->GetRefCount(), 1);
}

TEST_F(DynamicBasicTest, TestSlotId) {
    int s = 16;
    int id[2] = {0};
    Tensor t(DT_FP32, {s, s}, "t0");
    Tensor out(DT_FP32, {s, s}, "out");

    FunctionConfig config;
    FUNCTION("main", config, {t}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
            (void)idx;
            Tensor t0(DT_FP32, {s, s}, "t1");
            LOOP("L00", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(1)) {
                (void)idx1;
                t0 = Add(t, t);
            }
            id[0] = t0.Id();
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
            (void)idx;
            Tensor t1(DT_FP32, {s, s}, "t1");
            LOOP("L10", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(1)) {
                (void)idx1;
                t1 = Add(t, t);
            }
            id[1] = t1.Id();
        }
    }
    EXPECT_NE(id[0], id[1]);
}

TEST_F(DynamicBasicTest, DynamicRawShape) {
    int s = 32;
    Tensor t0(DT_FP32, {-1, s}, "t0"); // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");              // [32, 32]
    Tensor out(DT_FP32, {-1, s}, "out");

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(GetInputShape(t0, 0) / s)) {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, t0s, t1);
            Assemble(t2, {idx * s, 0}, out);
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

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0}, {out}) {
        auto shape0 = GetInputShape(t0, 0);
        auto t1 = Tensor(t0.GetDataType(), {shape0, s});
        auto loop1 = (shape0 + s - 1) / s;
        LOOP("L0", FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop1)) {
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});
            auto t = AddS(t0s, Element(DT_FP32, 3.0));
            Assemble(t, {idx * s, 0}, t1);
        }

        // check t1 use dynshape from t0
        auto loop2 = (GetInputShape(t1, 0) + s - 1) / s;
        LOOP("L1", FunctionType::DYNAMIC_LOOP, idx, LoopRange(loop2), {}, true) {
            Tensor t1s = View(t1, {s, s}, {idx * s, 0});
            auto t = SubS(t1s, Element(DT_FP32, 1.0));
            Assemble(t, {idx * s, 0}, out);
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
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), DeviceLauncherConfig(arg0->GetDataSize()));
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

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1}, {t3}, {{t2, t0}}) {
        LOOP("l0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            UNUSED(i);
            t3 = Add(t0, t1);
            Assemble(t3, {0, 0}, t2);
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
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1}, {out}) {
        FunctionConfig funConfig2(FunctionType::STATIC);
        ;
        FUNCTION("S0", funConfig2) {
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
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, t1, t2}, {out}) {
        Tensor s0Out;
        FunctionConfig funConfig2(FunctionType::STATIC);
        ;
        FUNCTION("S0", funConfig2) {
            s0Out = Sub(t1, t0);
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT)) {
            Tensor t0s = View(s0Out, {s, s}, {i * s, 0});
            Tensor t3 = Add(t0s, t2);
            Assemble(t3, {i * s, 0}, out);
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
    TileShape::Current().SetVecTile(512, 512);
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});

    int vecLen = 16;
    int loopNum = 4;
    int tileNum = 3;
    Tensor inputA(DT_FP32, {loopNum, vecLen}, "inputA");
    Tensor inputB(DT_FP32, {tileNum, vecLen}, "inputB");
    Tensor output(DT_FP32, {tileNum, vecLen}, "out");

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA, inputB}, {output}) {
        LOOP("Outer", FunctionType::DYNAMIC_LOOP, i, LoopRange(tileNum)) {
            Tensor tileB(DT_FP32, {1, vecLen}, "tileB");
            LOOP("Inner", FunctionType::DYNAMIC_LOOP, j, LoopRange(1)) {
                (void)j;
                auto tile = View(inputB, {1, vecLen}, {i, 0});
                tileB = MulS(tile, Element(DataType::DT_FP32, 2.0));
            }

            LOOP("Inner2", FunctionType::DYNAMIC_LOOP, k, LoopRange(loopNum)) {
                auto tileA = View(inputA, {1, vecLen}, {k, 0});
                tileB = Add(tileA, tileB);
            }

            LOOP("Inner3", FunctionType::DYNAMIC_LOOP, l, LoopRange(1)) {
                (void)l;
                tileB = MulS(tileB, Element(DataType::DT_FP32, 3.0));
                Assemble(tileB, {i, 0}, output);
            }
        }
    }

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);

    std::vector<float> inputAData(loopNum * vecLen, 0);
    std::vector<float> inputBData(tileNum * vecLen, 0);
    std::vector<float> golden(tileNum * vecLen, 0);

    readInput<float>(GetGoldenDir() + "/input_a.bin", inputAData);
    readInput<float>(GetGoldenDir() + "/input_b.bin", inputBData);
    readInput(GetGoldenDir() + "/out.bin", golden);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputB, inputBData),
    });

    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    // excute
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.005f));
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
    TestLoopViewAssemble(t0, t1, blockTable, out, s);

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
    TestLoopViewAssemble(t0, t1, blockTable, out, s);

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
    TestLoopViewAssemble(t0, t1, blockTable, out, s);

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

TEST_F(DynamicBasicTest, TestLoopIfWithRank456) {
    TileShape::Current().SetVecTile(32, 32);   //设置Tileshape大小为32*32

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    std::vector<float> golden(s * s, 12.0f);

    // Direct implementation of the function logic within the test
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {t0, r0}, {out}) {
        constexpr int LOOP_LENGTH = 10;
        npu::tile_fwk::SymbolicScalar len(LOOP_LENGTH);
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len)) {
            IF(i==0) {
                IF(i==len-1) {
                    r0 = AddS(r0, Element(DataType::DT_FP32, 1.0));
                } ELSE {
                    r0 = AddS(r0, Element(DataType::DT_FP32, 2.0));
                }
            } ELSE {
                IF(i==len-1) {
                    r0 = AddS(r0, Element(DataType::DT_FP32, 0.0));
                } ELSE {
                    Tensor t0v = View(t0, {s, s}, {s * i, 0});
                    r0 = Add(t0v, r0);
                }
            }
            out = AddS(r0, Element(DataType::DT_FP32, 0.0));
        }

        FunctionConfig funConfig2(FunctionType::STATIC);
        ;
        FUNCTION("S1", funConfig2) {
            out = AddS(r0, Element(DataType::DT_FP32, 2.0));  //静态function中增加2.0的偏移量
        }
    }
    #ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
    #endif
}

TEST_F(DynamicBasicTest, TestTensorExtract) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = 32;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (size_t k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }
    Tensor output(DT_INT32, {1, s}, "output");

    int row = 3;
    int col = 4;
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2));
            output = TensorExtract(t0, {row, col});
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_EQ(row * n + col + 0x2, *(int32_t *)outputResult->data());
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorData) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 12.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA, inputC}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            SymbolicScalar v0 = GetTensorData(t0, {0, 1}); // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            SymbolicScalar v1 = GetTensorData(t0, {0, 2}); // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            auto t2 = View(inputC, {n, n}, {0, v0 * n});
            auto t3 = View(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataCrossFunction) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }
    Tensor output(DT_FP32, {n, n}, "output");
    Tensor outsum(DT_INT32, {n, n}, "outsum");

    std::vector<float> outputGolden(n * n, 12.0f);
    std::vector<int> outsumGolden(n * n, 0);
    int d0 = 1 + 2;
    int d1 = 2 + 2;
    int d2 = d0 + d1 + 1;
    outsumGolden[0] = d0;
    outsumGolden[1] = d1;
    outsumGolden[2] = d2;
    outsumGolden[3] = d0 + d1;
    outsumGolden[4] = d0 + d2;
    outsumGolden[5] = d1 + d2;
    outsumGolden[6] = d0 + d1 + d2;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
        RawTensorData::CreateConstantTensor<int32_t>(outsum, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA, inputC}, {output, outsum}) {
        SymbolicScalar v0;
        SymbolicScalar v1;
        SymbolicScalar v2;
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = AddS(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            v0 = GetTensorData(t0, {0, 1}); // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            v1 = GetTensorData(t0, {0, 2}); // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            v2 = v0 + v1 + GetInputData(inputA, {0, 1});
        }
        LOOP("Step1", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t2 = View(inputC, {n, n}, {0, v0 * n});
            auto t3 = View(inputC, {n, n}, {0, v1 * n});
            output = Mul(t2, t3);
            SetTensorData(v0, {0, 0}, outsum);
            SetTensorData(v1, {0, 1}, outsum);
            SetTensorData(v2, {0, 2}, outsum);
            SetTensorData(v0 + v1, {0, 3}, outsum);
            SetTensorData(v0 + v2, {0, 4}, outsum);
            SetTensorData(v1 + v2, {0, 5}, outsum);
            SetTensorData(v0 + v1 + v2, {0, 6}, outsum);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float *)outputResult->data(), 0.001f));
    auto outsumResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(resultCmp(outsumGolden, (int32_t *)outsumResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataUnalign) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);
    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, true);
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int cnt = 8;
    int n = tiling * 1;
    int m = tiling * cnt;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC1(DT_FP32, {n, m}, "inputC1");
    Tensor inputC2(DT_FP32, {n, m}, "inputC2");
    std::vector<float> inputC1Data(n * m, 0); // 8 x (32 x 32 / 16 x 16)
    std::vector<float> inputC2Data(n * m, 0); // 8 x (32 x 32 / 16 x 16)
    int v0Data = 3;
    int v1Data = 4;
    for (int k = 0; k < cnt; k++) {
        for (int i = 0; i < v0Data; i++) {
            for (int j = 0; j < v1Data; j++) {
                inputC1Data[i * m + k * n + j] = k;
            }
        }
        for (int i = 0; i < v0Data; i++) {
            for (int j = 0; j < v1Data; j++) {
                inputC2Data[i * m + k * n + j] = k + 1;
            }
        }
    }
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 0);
    for (int i = 0; i < v0Data; i++) {
        for (int j = 0; j < v1Data; j++) {
            outputGolden[i * n + j] = 15;
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC1, inputC1Data),
        RawTensorData::CreateTensor<float>(inputC2, inputC2Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA, inputC1, inputC2}, {output}) {
        SymbolicScalar v0;
        SymbolicScalar v1;
        SymbolicScalar v2;
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t0 = AddS(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            v0 = GetTensorData(t0, {0, 1}); // t0[0, 1] -> 0 * n + 1 + 2 -> 3
            v1 = GetTensorData(t0, {0, 2}); // t0[0, 2] -> 0 * n + 2 + 2 -> 4
            v2 = v0 + v1 + GetInputData(inputA, {0, 1});
        }
        LOOP("Step1", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto t2 = View(inputC1, {n, n}, {v0, v1}, {0, v0Data * n}); // 16 x 16 / 3 x 4, 3
            auto t3 = View(inputC2, {n, n}, {v0, v1}, {0, v1Data * n}); // 16 x 16 / 3 x 4, 5
            output = Mul(t2, t3); // 16 x 16 / 3 x 5, 12
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataExpr) {
    int tiling = 32;
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    int s = n * 8;
    Tensor inputA(DT_INT32, {n, n}, "inputA");
    std::vector<int32_t> inputAData(n * n);
    for (int k = 0; k < n * n; k++) {
        inputAData[k] = k;
    }

    Tensor inputC(DT_FP32, {n, s}, "inputC");
    std::vector<float> inputCData(n * s);
    for (int k = 0; k < n * s; k++) {
        inputCData[k] = (float)(1.0 * ((k % s) / n));
    }

    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<float> outputGolden(n * n, 35.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(inputA, inputAData),
        RawTensorData::CreateTensor<float>(inputC, inputCData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(output, 0.0f),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {inputA, inputC}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            Tensor t0 = AddS(inputA, Element(DT_INT32, (int64_t)2)); // t0[i, j] -> inputA[i, j] + 2 -> i * n + j + 2
            SymbolicScalar v0 = GetTensorData(t0, {0, 1}); // t0[0, 1] + 2 -> 0 * n + 1 + 2 -> 3
            SymbolicScalar v1 = GetTensorData(t0, {0, 2}); // t0[0, 2] + 2 -> 0 * n + 2 + 2 -> 4
            SymbolicScalar v2 = GetInputData(inputA, {0, 1}); // inputA[0, 1] -> 1
            SymbolicScalar v3 = GetInputData(inputA, {0, 2}); // inputA[0, 2] -> 2
            auto t2 = View(inputC, {n, n}, {0, (v0 + v2 + i / i) * n}); // {0, (3 + 1 + 1) * n} -> {0, 5 * n}
            auto t3 = View(inputC, {n, n}, {0, (v1 + v3 + i / i) * n}); // {0, (4 + 2 + 1) * n} -> {0, 7 * n}
            output = Mul(t2, t3);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float *)outputResult->data(), 0.001f));
#endif
}


TEST_F(DynamicBasicTest, TestVectorDup) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_FP32, {n, n}, "output");
    std::vector<int32_t> outputGolden(n * n, 50);

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            SymbolicScalar v = 20;
            output = VectorDuplicate(v + 30, DT_INT32, {n, n});
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestTensorInsert) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_INT32, {n}, "output");
    std::vector<int32_t> outputGolden(n, 20);

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            auto tmp = VectorDuplicate(20, DT_INT32, {1});
            TensorInsert(tmp, {i}, output);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSetTensorData) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);
    TileShape::Current().SetCubeTile({tiling, tiling}, {tiling, tiling}, {tiling, tiling});

    int n = tiling * 1;
    Tensor output(DT_INT32, {n}, "output");
    std::vector<int32_t> outputGolden(n, 30);

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            SetTensorData(30, {i / 2 * 2 + i % 2}, output);
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSetTensorDataExpr) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> outputGolden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        outputGolden[i] = i;
    }

    ProgramData::GetInstance().AppendInputs({
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n)) {
                for (int k = 0; k < n; k++) {
                    SetTensorData(i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetTensorDataAndDup) {
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor input(DT_INT32, {n, n}, "input");
    std::vector<int32_t> inputData(n * n);
    for (int i = 0; i < n * n; i++) {
        inputData[i] = i;
    }

    int row = 3;
    int col = 4;
    Tensor output(DT_INT32, {n, n}, "output");
    std::vector<int32_t> outputGolden(n * n, (row * n + col) * 2 + 1);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(input, inputData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {input}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            auto add = Add(input, input);
            auto s = GetTensorData(add, {row, col});
            output = VectorDuplicate(s + 1, DT_INT32, {n, n});
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestGetAndSetTensorDataExpr) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    int init = 10;
    Tensor input(DT_INT32, {n, n, n}, "input");
    Tensor output(DT_INT32, {n, n, n}, "output");
    std::vector<int32_t> outputGolden(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        outputGolden[i] = init + init + i;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(input, init),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<int32_t>(output, outputGolden),
    });

    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {input}, {output}) {
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n)) {
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(n)) {
                auto add = Add(input, input);
                for (int k = 0; k < n; k++) {
                    SymbolicScalar s = GetTensorData(add, {i, j, k});
                    SetTensorData(s + i * tiling * tiling + j * tiling + k, {i, j, k}, output);
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (int32_t *)outputResult->data(), 0.001f));
#endif
}

TEST_F(DynamicBasicTest, TestSelectAttention) {
    ConfigManager::Instance().SetCodeGenConfig(npu::tile_fwk::KEY_CODEGEN_EXPRESSION_FUSION, true);

    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling, tiling);

    int n = tiling * 1;
    Tensor input(DT_INT32, {n, n}, "input");
    std::vector<int32_t> inputData(n * n);
    for (int i = 0; i < n * n; i++) {
        inputData[i] = i % n; // inputData[x,y] -> y
    }
    Tensor table(DT_INT32, {n, n}, "table");
    std::vector<int32_t> tableData(n * n);
    for (int i = 0; i < n * n; i++) {
        tableData[i] = i % n; // tableData[x,y] -> y
    }
    Tensor c0(DT_FP32, {n, n * n}, "c0");
    std::vector<float> c0Data(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        c0Data[i] = i % (n * n) / n; // c0Data[x, a * n + b] -> a
    }
    Tensor c1(DT_FP32, {n, n * n}, "c1");
    std::vector<float> c1Data(n * n * n);
    for (int i = 0; i < n * n * n; i++) {
        c1Data[i] = i % (n * n) / n; // c1Data[x, a * n + b] -> a
    }

    Tensor output(DT_FP32, {n, n}, "output");

    DataType dtype = DT_FP32;
    float outputGoldenCell = 0;
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            outputGoldenCell += i * i;
        }
    }
    std::vector<float> outputGolden(n * n);
    for (int i = 0; i < n * n; i++) {
        outputGolden[i] = outputGoldenCell;
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<int32_t>(input, inputData),
        RawTensorData::CreateTensor<int32_t>(table, tableData),
        RawTensorData::CreateTensor<float>(c0, c0Data),
        RawTensorData::CreateTensor<float>(c1, c1Data),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(output, outputGolden),
    });

    int topk = 16;
    FunctionConfig funConfig;
    FUNCTION("main", funConfig, {input, table, c0, c1}, {output}) {
        Tensor index;
        LOOP("Idx", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            index = Add(input, input);
            index = Sub(index, input);
        }
        LOOP("Step0", FunctionType::DYNAMIC_LOOP, i, LoopRange(n), {}, true) {
            (void)i;
            Tensor r0(dtype, {n, n * n}, "r0");
            Tensor r1(dtype, {n, n * n}, "r1");
            LOOP("Step1", FunctionType::DYNAMIC_LOOP, j, LoopRange(0, n, topk), {}, true) {
                (void)j;
                for (int k = 0; k < topk; k++) {
                    SymbolicScalar s = GetTensorData(index, {i, j + k}); // index[i, j + k] -> j + k
                    SymbolicScalar slcBlockIdx = GetInputData(table, {i, s}); // table[i, s] -> s
                    auto k0 = View(c0, {n, n}, {0, s * n});
                    auto k1 = View(c1, {n, n}, {0, slcBlockIdx * n});

                    auto k0v = AddS(k0, Element(dtype, (float)0));
                    auto k1v = AddS(k1, Element(dtype, (float)0));
                    Assemble(k0v, {0, s * n}, r0);
                    Assemble(k1v, {0, slcBlockIdx * n}, r1);
                }
            }
            LOOP("Step2", FunctionType::DYNAMIC_LOOP, j, LoopRange(n), {}, true) {
                LOOP("loop1", FunctionType::DYNAMIC_LOOP, _, LoopRange(1), {}, true) {
                    (void)_;
                    auto matmul = Matrix::Matmul<false, true>(DataType::DT_FP32, r0, r1);
                    auto d1 = DivS(matmul, Element(dtype, (float)n));
                    auto d2 = DivS(d1, Element(dtype, (float)n));
                    IF (i == 0) {
                        IF (j == 0) {
                            output = d2;
                        } ELSE {
                            output = Add(output, d2);
                        }
                    } ELSE {
                        output = Add(output, d2);
                    }
                }
            }
        }
    }

#ifdef ENABLE_BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outputResult = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(outputGolden, (float *)outputResult->data(), 0.001f));
#endif
}

}
