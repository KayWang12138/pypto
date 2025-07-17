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
 * \file test_simulation_acc.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config.h"
#include "models/deepseek/deepseek_mla.h"
#include "interface/configs/config_manager.h"
#include "models/llama/llama_def.h"
#include "models/deepseek/deepseek_spec.h"

using namespace npu::tile_fwk;

class SimulationAccTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

TEST_F(SimulationAccTest, TestAddTensorFunctionDim4) {
    std::vector<int> shape{1,1,32,32};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(1,1,32,32);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddTensorFunctionDim4_1) {
    std::vector<int> shape{2,2,32,32};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(2,2,32,32);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddTensorFunctionDim2) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(8,16);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddTensorFunctionDim2_1) {
    std::vector<int> shape{32,32};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubTensorFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Sub(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubTensorFunctionDim2_1) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(64,64);

    FUNCTION("A") {
        c = Sub(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Sub(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulTensorFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Mul(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulTensorFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Mul(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Mul(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulTensorFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = Mul(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivTensorFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Div(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivTensorFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Div(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Div(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivTensorFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Tensor b(DT_FP16, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = Div(a, b);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddScalarFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        d = AddS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddScalarFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = AddS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddScalarFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = AddS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestAddScalarFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        d = AddS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubScalarFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        d = SubS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubScalarFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = SubS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubScalarFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = SubS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSubScalarFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        d = SubS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulScalarFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        d = MulS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulScalarFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = MulS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulScalarFunctionDim2_2) {
    std::vector<int> shape{32,32};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        d = MulS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestMulScalarFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        d = MulS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivScalarFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        d = DivS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivScalarFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = DivS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivScalarFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        d = DivS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestDivScalarFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Element value(DataType::DT_FP16, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        d = DivS(a, value);
    }

    // Program::GetInstance().GraphCheck();

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSqrtTensorFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Sqrt(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSqrtTensorFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Sqrt(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSqrtTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Sqrt(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestSqrtTensorFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = Sqrt(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpTensorFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Exp(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpTensorFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Exp(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Exp(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpTensorFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = Exp(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestReciprocalTensorFunctionDim2) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Reciprocal(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestReciprocalTensorFunctionDim2_1) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Reciprocal(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestReciprocalTensorFunctionDim2_2) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Reciprocal(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestReciprocalTensorFunctionDim2_3) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = Reciprocal(a);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpandTensorFunctionDim2_1) {
    std::vector<int> shape{1,1};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    std::vector<int> vec{32, 32};

    FUNCTION("A") {
        c = Expand(a, vec);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpandTensorFunctionDim2_2) {
    std::vector<int> shape{1,1};

    Tensor a(DT_FP16, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    std::vector<int> vec{64, 64};

    FUNCTION("A") {
        c = Expand(a, vec);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestExpandTensorFunctionDim2_3) {
    std::vector<int> shape{1,1};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    std::vector<int> vec{64, 64};

    FUNCTION("A") {
        c = Expand(a, vec);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}


TEST_F(SimulationAccTest, TestExpandTensorFunctionDim2_4) {
    std::vector<int> shape{1,1};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    std::vector<int> vec{128, 128};

    FUNCTION("A") {
        c = Expand(a, vec);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowSumSingleTensorFunctionDim2_1) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = RowSumSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowSumSingleTensorFunctionDim2_2) {
    std::vector<int> shape{32,32};

    Tensor a(DT_INT32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = RowSumSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowSumSingleTensorFunctionDim2_3) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(64,64);

    FUNCTION("A") {
        c = RowSumSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowSumSingleTensorFunctionDim2_4) {
    std::vector<int> shape{64,64};

    Tensor a(DT_INT32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(64,64);

    FUNCTION("A") {
        c = RowSumSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowMaxSingleTensorFunctionDim2_1) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = RowMaxSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowMaxSingleTensorFunctionDim2_2) {
    std::vector<int> shape{32,32};

    Tensor a(DT_INT32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = RowMaxSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowMaxSingleTensorFunctionDim2_3) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(64,64);

    FUNCTION("A") {
        c = RowMaxSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestRowMaxSingleTensorFunctionDim2_4) {
    std::vector<int> shape{64,64};

    Tensor a(DT_INT32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(64,64);

    FUNCTION("A") {
        c = RowMaxSingle(a, 0);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestCastTensorFunctionDim2_1) {
    std::vector<int> shape{16,16};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(16,16);

    FUNCTION("A") {
        c = Cast(a, DT_INT32, CAST_ROUND);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestCastTensorFunctionDim2_2) {
    std::vector<int> shape{32,32};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(32,32);

    FUNCTION("A") {
        c = Cast(a, DT_INT32, CAST_ROUND);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestCastTensorFunctionDim2_3) {
    std::vector<int> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(64,64);

    FUNCTION("A") {
        c = Cast(a, DT_INT32, CAST_ROUND);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(SimulationAccTest, TestCastTensorFunctionDim2_4) {
    std::vector<int> shape{128,128};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    Program::GetInstance().GetTileShape().SetVecTileShapes(128,128);

    FUNCTION("A") {
        c = Cast(a, DT_INT32, CAST_ROUND);
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}
