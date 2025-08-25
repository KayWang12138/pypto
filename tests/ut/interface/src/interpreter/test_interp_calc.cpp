/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk {
class TorchAdaptorTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
    }

    void TearDown() override {}
};

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int> &shape, const T &val) {
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateConstantTensor(data, val));
}

template <typename T>
static LogicalTensorDataPtr makeTensorData(DataType t, const std::vector<int> &shape, const std::vector<T> &vals) {
    Tensor data(t, shape);
    return std::make_shared<LogicalTensorData>(RawTensorData::CreateTensor(data, vals));
}

#define ASSERT_ALLCLOSE(self, other) \
    ASSERT(calc::AllClose(self, other)) << "lhs:\n" << self << "\nrhs:\n" << other << "\n"

TEST_F(TorchAdaptorTest, UnaryOps) {
    {
        // sqrt
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Sqrt(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // abs
        auto self = makeTensorData(DT_FP32, {16, 16}, -4.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::Abs(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // exp
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, std::exp(2.0f));
        calc::Exp(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast
        auto self = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::Cast(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // cast
        auto self = makeTensorData(DT_FP32, {16, 16}, 1e8f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(0));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(static_cast<int>(1e8f)));
        calc::Cast(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand scalar
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::ExpandS(out, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand broadcast
        auto self = makeTensorData(DT_FP32, {16, 1}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        calc::Expand(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // expand broadcast and cast
        auto self = makeTensorData(DT_FP32, {16, 1}, 2.0f);
        auto out = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        auto golden = makeTensorData(DT_INT16, {16, 16}, static_cast<int16_t>(2));
        calc::Expand(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BinaryOps) {
    {
        // add
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // sub
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::Sub(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // mul
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::Mul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // div
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto other = makeTensorData(DT_FP32, {16, 16}, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::Div(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // add broadcast
        auto self = makeTensorData(DT_FP32, {1, 16}, 4.0f);
        auto other = makeTensorData(DT_FP32, {16, 1}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::Add(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // elementwise max
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> odata = {2.0, 2.0, 3.0, 5.0};
        std::vector<float> gdata = {2.0, 2.0, 5.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto other = makeTensorData(DT_FP32, {2, 2}, odata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::Max(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // elementwise min
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> odata = {2.0, 2.0, 3.0, 5.0};
        std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto other = makeTensorData(DT_FP32, {2, 2}, odata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::Min(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scalar min
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> gdata = {1.0, 2.0, 2.0, 2.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::MinS(out, self, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // scalar max
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> gdata = {2.0, 2.0, 3.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::MaxS(out, self, Element(DT_FP32, 2.0f));
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, BinaryOpsS) {
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        calc::AddS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 3.0f);
        calc::SubS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 8.0f);
        calc::MulS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        auto self = makeTensorData(DT_FP32, {16, 16}, 5.0f);
        auto elem = Element(DT_FP32, 2.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 2.5f);
        calc::DivS(out, self, elem);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, MulMul) {
    {
        // matmul
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other, 4);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul bt
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul<false, true>(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul bt splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul<false, true>(out, self, other, 4);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        calc::AccMatMul(out, self, other, out);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {16, 8}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        calc::AccMatMul(out, self, other, out, 4);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc bt
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        calc::AccMatMul<false, true>(out, self, other, out);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul acc bt splitk
        auto self = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto other = makeTensorData(DT_FP32, {8, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 17.0f);
        calc::AccMatMul<false, true>(out, self, other, out, 4);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // matmul cast
        auto self = makeTensorData(DT_FP16, {8, 16}, float16(1.0));
        auto other = makeTensorData(DT_FP16, {16, 8}, float16(1.0));
        auto out = makeTensorData(DT_FP32, {8, 8}, 1.0f);
        auto golden = makeTensorData(DT_FP32, {8, 8}, 16.0f);
        calc::MatMul(out, self, other);
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Reduce) {
    {
        // sum expand
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 16.0f);
        calc::RowSumExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }

    {
        // sum
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 1}, 16.0f);
        calc::RowSumSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // min expand
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {1.0, 1.0, 4.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::RowMinExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // min
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {1.0, 4.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 1}, gdata);
        calc::RowMinSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // max expand
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {2.0, 2.0, 5.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 2}, gdata);
        calc::RowMaxExpand(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // max
        std::vector<float> sdata = {1.0, 2.0, 5.0, 4.0};
        std::vector<float> gdata = {2.0, 5.0};
        auto self = makeTensorData(DT_FP32, {2, 2}, sdata);
        auto out = makeTensorData(DT_FP32, {2, 1}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {2, 1}, gdata);
        calc::RowMaxSingle(out, self, -1);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // reduce acc
        auto self = makeTensorData(DT_FP32, {16, 16}, 1.0f);
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {16, 16}, 4.0f);
        calc::ReduceAcc(out, {self, self, self, self});
        ASSERT_ALLCLOSE(out, golden);
    }
}

TEST_F(TorchAdaptorTest, Misc) {
    {
        // reshape
        std::vector<float> gdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, gdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Reshape(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // transpose
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Transpose(out, self, -1, -2);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copy
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {3, 2}, gdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Copy(out, self);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // copy trans
        std::vector<float> sdata = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<float> gdata = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
        auto self = makeTensorData(DT_FP32, {2, 3}, sdata);
        auto out = makeTensorData(DT_FP32, {3, 2}, 0.0f);
        auto golden = makeTensorData(DT_FP32, {3, 2}, gdata);
        calc::Copy(out, self, true);
        ASSERT_ALLCLOSE(out, golden);
    }
    {
        // view
        auto out = makeTensorData(DT_FP32, {16, 16}, 0.0f);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                auto v = out->View({4, 4}, {i * 4, j * 4});
                calc::ExpandS(v, Element(DT_FP32, i * 2.0f + j));
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                auto v = out->View({4, 4}, {i * 4, j * 4});
                auto g = makeTensorData(DT_FP32, {4, 4}, i * 2.0f + j);
                ASSERT(calc::AllClose(v, g)) << v << "\n" << g;
            }
        }
    }
}
} // namespace npu::tile_fwk