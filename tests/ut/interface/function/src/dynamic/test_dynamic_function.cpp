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
 * \file test_dynamic_function.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <algorithm>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
using namespace npu::tile_fwk;
using namespace std;
using Json = nlohmann::json;

class DynamicFunctionTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
    }

    void TearDown() override {}
};

TEST_F(DynamicFunctionTest, TestSymbolic) {
    {
        SymbolicScalar a("a");
        SymbolicScalar b("b");
        SymbolicScalar c = a + b;
        EXPECT_FALSE(c.ConcreteValid());
        EXPECT_EQ(c.Raw()->Kind(), SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION);

        SymbolicClosure closure;
        closure.Insert("a", 100);
        closure.Insert("b", 50);
        EXPECT_EQ(150, closure.Evaluate(c));
    }

    {
        SymbolicScalar a("a", 100);
        SymbolicScalar b("b", 50);
        {
            auto c0 = -a;
            EXPECT_EQ(c0.Concrete(), -100);

            c0 = +a;
            EXPECT_EQ(c0.Concrete(), 100);

            c0 = !a;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a + b;
            EXPECT_EQ(c0.Concrete(), 150);

            c0 = a - b;
            EXPECT_EQ(c0.Concrete(), 50);

            c0 = a * b;
            EXPECT_EQ(c0.Concrete(), 5000);

            c0 = a / b;
            EXPECT_EQ(c0.Concrete(), 2);

            c0 = a % b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a == b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a != b;
            EXPECT_EQ(c0.Concrete(), 1);

            c0 = a < b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a <= b;
            EXPECT_EQ(c0.Concrete(), 0);

            c0 = a > b;
            EXPECT_EQ(c0.Concrete(), 1);

            c0 = a >= b;
            EXPECT_EQ(c0.Concrete(), 1);

            c0 = std::min(a, b);
            EXPECT_EQ(c0.Concrete(), 50);

            c0 = std::max(a, b);
            EXPECT_EQ(c0.Concrete(), 100);

            std::cout << c0 << "\n";
        }
        {
            SymbolicScalar c0 = a + 50;
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 150);
        }
        {
            SymbolicScalar c0 = 50 + a;
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 150);
        }
        {
            SymbolicScalar c0 = std::max(a, b);
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 100);
        }
        {
            SymbolicScalar c0 = std::min(a, b);
            EXPECT_TRUE(c0.ConcreteValid());
            EXPECT_EQ(c0.Concrete(), 50);
        }
        {
            a.AsIntermediateVariable();
            EXPECT_TRUE(a.IsIntermediateVariable());
        }
        {
            auto x = SymbolicScalar("x", NotLessThan(1), NotGreaterThan(2));
            EXPECT_TRUE(x.IsSymbol());
        }
    }
}

TEST_F(DynamicFunctionTest, TestDynOffset) {
    SymbolicScalar b("b");
    std::vector<int> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {b, 0};
    FUNCTION("main", FunctionType::DYNAMIC, {}, {}) {
        Tensor t(DT_FP32, {4, 4}, "t0");
        auto v = DViewPad(t, {1, 1}, {b, 2}, {b, 0});
        v->UpdateOffset(TensorOffset(offset, dynoffset));
    }

    auto func = Program::GetInstance().GetFunctionByRawName("TENSOR_main");
    Tensor t(DT_FP32, {4, 4}, "t0");
    t->UpdateOffset(TensorOffset(offset, dynoffset));
    auto tt = LogicalTensor::LoadJson(*func, {}, t->DumpJson());
    EXPECT_EQ(tt->GetShape(), t->GetShape());
}

struct MopCall {
    static int64_t func() { return 0; }
    static int64_t func1(int64_t arg0) { return arg0; }
    static int64_t func2(int64_t arg0, int64_t arg1) { return arg0 + arg1; }
    static int64_t func3(int64_t arg0, int64_t arg1, int64_t arg2) { return arg0 + arg1 + arg2; }
};

#define PTR_TO_ULONG(p) reinterpret_cast<int64_t>(reinterpret_cast<void *> (p))

TEST_F(DynamicFunctionTest, MopCall) {
    SymbolicScalar arg0("arg0", 1);
    SymbolicScalar arg1("arg1", 2);
    SymbolicScalar arg2("arg2", 3);

    SymbolicScalar func("func", PTR_TO_ULONG(MopCall::func));
    auto ret = func();
    EXPECT_TRUE(ret.ConcreteValid());
    EXPECT_EQ(ret.Concrete(), 0);

    SymbolicScalar func1("func1", PTR_TO_ULONG(MopCall::func1));
    auto ret1 = func1(arg0);
    EXPECT_TRUE(ret1.ConcreteValid());
    EXPECT_EQ(ret1.Concrete(), 1);

    SymbolicScalar func2("func2", PTR_TO_ULONG(MopCall::func2));
    auto ret2 = func2(arg0, arg1);
    EXPECT_TRUE(ret2.ConcreteValid());
    EXPECT_EQ(ret2.Concrete(), 3);

    SymbolicScalar func3("func3", PTR_TO_ULONG(MopCall::func3));
    auto ret3 = func3(arg0, arg1, arg2);
    EXPECT_TRUE(ret3.ConcreteValid());
    EXPECT_EQ(ret3.Concrete(), 6);
}

TEST_F(DynamicFunctionTest, TestLoopRange) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(16, 16);

    std::vector<int> shape{16, 64};
    std::vector<int> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int STATIC_LOOP_COUNT = 4;
    constexpr int DYNAMIC_LOOP_START = 0;
    constexpr int DYNAMIC_LOOP_END = 4;
    constexpr int DYNAMIC_LOOP_STEP = 1;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    int count = 0;
    FUNCTION("main", FunctionType::DYNAMIC, {a, b}, {c}) {
        LOOP("D1", FunctionType::STATIC, k, LoopRange(STATIC_LOOP_COUNT)) {
            auto a0 = DView(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = DView(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            DAssemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            count++;
        }

        LOOP("D2", FunctionType::DYNAMIC_LOOP, k, LoopRange(DYNAMIC_LOOP_START, DYNAMIC_LOOP_END)) {
            auto a0 = DView(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = DView(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            DAssemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            count++;
        }

        LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(DYNAMIC_LOOP_START, DYNAMIC_LOOP_END, DYNAMIC_LOOP_STEP)) {
            auto a0 = DView(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = DView(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            DAssemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            count++;
        }
    }
    EXPECT_EQ(count, 3);
}

TEST_F(DynamicFunctionTest, TestOnlyExpression) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(16, 16);

    std::vector<int> shape{16, 64};
    std::vector<int> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END)) {
        auto a0 = DView(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
        auto b0 = DView(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
        auto c0 = Add(a0, b0);
        DAssemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
    }

    auto rootFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_D3_Unroll1_PATH0_root_4");
    EXPECT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetCallopAttrList().size(), 1);
    auto attr = rootFunc->GetCallopAttrList().front();
    ALOG_INFO(attr->DumpAttr());
    EXPECT_EQ(attr->DumpAttr(2), "attr[2][  0,  0,(k*16), 16, 16, 16, 64,  0,  0]]");
}

TEST_F(DynamicFunctionTest, TestOnlySymbol) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(1, 64);

    std::vector<int> shape{4, 64};
    std::vector<int> childShape{1, 64};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;

    LOOP("DynSymbol", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END)) {
        auto a0 = DView(a, childShape, {k, 0});
        auto b0 = DView(b, childShape, {k, 0});
        auto c0 = Add(a0, b0);
        DAssemble(c0, {k, 0}, c);
    }

    auto rootFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_DynSymbol_Unroll1_PATH0_root_4");
    EXPECT_NE(rootFunc, nullptr);
    EXPECT_EQ(rootFunc->GetCallopAttrList().size(), 1);
    auto attr = rootFunc->GetCallopAttrList().front();
    ALOG_INFO(attr->DumpAttr());
    EXPECT_EQ(attr->DumpAttr(2), "attr[2][  0,  k,  0,  1, 64,  4, 64,  0,  0]]");
}

void TestHybridLoopIf(
    const Tensor &t0, const Tensor &t1, const Tensor &t2, const Tensor &t3, const Tensor &t4, Tensor &out) {
    constexpr int MAIN_LOOP_COUNT = 8;
    constexpr int CONDITION_THRESHOLD = 6;
    constexpr int SECOND_LOOP_COUNT = 2;

    Tensor r0;
    int loopCount = 0;
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, t2, t3, t4}, {out}) {
        FUNCTION("spre", FunctionType::STATIC) {
            r0 = Add(t0, t1);
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(MAIN_LOOP_COUNT)) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
            r0 = Mul(r0, t1); // +t0, +t1
            IF(i < CONDITION_THRESHOLD) {
                r0 = Sub(r0, t3); // +t2 * 6
                r0 = Sub(r0, t3); // +t2 * 6
            }
            ELSE {
                r0 = Sub(r0, t4); // +t3 * 8
                r0 = Add(r0, t4); // +t3 * 8
            }
        }
        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(SECOND_LOOP_COUNT)) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
            loopCount += i;
            r0 = Add(r0, t4);
        }
        FUNCTION("spost", FunctionType::STATIC) {
            Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
            r0 = Add(r0, t0);
            r0 = Add(r0, t1);
            r0 = Add(r0, t2);
            r0 = Add(r0, t3);
            out = Add(r0, t4);
        }
    }
    ALOG_INFO("loopCount=", loopCount);
    for (auto &ele : Program::GetInstance().GetFunctionMap()) {
        auto func = ele.second;
        std::shared_ptr<DynloopFunctionAttribute> &attr = func->GetDynloopAttribute();
        if (attr != nullptr) {
            auto node = attr->BuildPathNode();
            node->Dump();
            attr->DumpBranch();
        }
    }
}

void TestHybridLoopIf2(
    const Tensor &t0, const Tensor &t1, const Tensor &t2, const Tensor &t3, const Tensor &t4, Tensor &out) {
    constexpr int LOOP_COUNT = 8;
    constexpr int CONDITION_THRESHOLD = 6;

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
}

void TestStaticLoopStatic(const Tensor &t0, const Tensor &t1, const Tensor &t2, const Tensor &t3, const Tensor &t4,
    Tensor &out, int s) {
    SymbolicScalar GetInt32Value1("GetInt32Value1");
    SymbolicScalar GetInt32Value2("GetInt32Value2");
    SymbolicScalar blockTableAddr("blockTableAddr");
    SymbolicScalar batchAddr("batchAddr");
    Tensor r0;
    int loopCount = 0;
    FUNCTION("main", FunctionType::DYNAMIC, {t0, t1, t2, t3, t4}, {out}) {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            loopCount += i;
            r0 = Add(t0, t1);
            r0 = Sub(r0, t2);
        }
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t3, 1) / s)) {
            loopCount += i;
            Tensor t3v = View(t3, {s, s}, {0, 0});
            r0 = Add(t3v, r0);
        }

        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDimSize(t3) / s)) {
            loopCount += i;
            Tensor t3v = View(t3, {s, s}, {0, 0});
            r0 = Add(t3v, r0);
        }

        LOOP("L1", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputDataInt32Dim1(t3, npu::tile_fwk::SymbolicScalar("0")) / s)) {
            loopCount += i;
            Tensor t3v = View(t3, {s, s}, {0, 0});
            r0 = Add(t3v, r0);
        }

        LOOP("L2", FunctionType::DYNAMIC_LOOP, i,
             LoopRange(GetInputDataInt32Dim2(t3, npu::tile_fwk::SymbolicScalar("0"), npu::tile_fwk::SymbolicScalar("1")) / s)) {
            loopCount += i;
            Tensor t3v = View(t3, {s, s}, {0, 0});
            r0 = Add(t3v, r0);
        }

        LOOP("s1", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            loopCount += i;
            out = Sub(r0, t4);
        }
    }
    ALOG_INFO("loopCount=", loopCount);
}

TEST_F(DynamicFunctionTest, TestStaticLoopStatic) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

    // Representation
    int s = 32;
    int n = 1;
    int m = 1;
    int t3CountValue = 7;
    Tensor t0(DT_FP32, {n * s, m * s}, "t0");
    Tensor t1(DT_FP32, {n * s, m * s}, "t1");
    Tensor t2(DT_FP32, {n * s, m * s}, "t2");
    Tensor t3(DT_FP32, {n * s, m * s * t3CountValue}, "t3");
    Tensor t4(DT_FP32, {n * s, m * s}, "t4");
    Tensor blockTable(DT_INT32, {16, 16}, "blockTable");
    Tensor batch(DT_INT32, {1, 1}, "batch");
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

    TestStaticLoopStatic(t0, t1, t2, t3, t4, out, s);

    std::shared_ptr<DyndevFunctionAttribute> attr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();

}

TEST_F(DynamicFunctionTest, TestHybridLoopIf) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

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

    TestHybridLoopIf(t0, t1, t2, t3, t4, out);
    Program::GetInstance().DumpStack();
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->Dump();
    slotManager->BuildIncastOutcastLink("test");
    slotManager->GetOutputIndex(out);
    slotManager->GetOutputIndex(t4);
    slotManager->LookupSlotIndexBySymbol({"t0", "t1", "tt"});
    slotManager->LookupSlotIndex({t0, t1, t2});
    TensorSlot slot;
    slotManager->TensorSlotRead(slot, t1.GetStorage());

    auto programJson = Program::GetInstance().DumpJson();
    Program::GetInstance().LoadJson(programJson);

    auto programJson2 = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJson.dump(), programJson2.dump());
    Program::GetInstance().LoadJson(programJson2);

    auto programJson3 = Program::GetInstance().DumpJson();
    EXPECT_EQ(programJson3.dump(), programJson2.dump());
}

TEST_F(DynamicFunctionTest, TestHybridLoopIf2) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

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

    TestHybridLoopIf2(t0, t1, t2, t3, t4, out);
}

Tensor TestLoopWithRank(const Tensor &t0, Tensor &r0, Tensor &out, int s, int maxRank) {
    FUNCTION("main", FunctionType::DYNAMIC, {t0, r0}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s), PowersOf2(maxRank)) {
            Tensor t0v = DView(t0, {s, s}, {s * i, 0});
            r0 = Add(t0v, r0);
        }
        FUNCTION("S1", FunctionType::STATIC) {
            out = AddS(r0, Element(DataType::DT_FP32, 3.0));
        }
    }
    return out;
}

Tensor TestLoopIfWithRank(const Tensor &t0, Tensor &r0, Tensor &out, int s, int maxRank) {
    FUNCTION("main", FunctionType::DYNAMIC, {t0, r0}, {out}) {
        auto len = GetInputShapeDim(t0, 0) / s;

        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(len), PowersOf2(maxRank)) {
            IF(IsLoopBegin(i, 0)) {
                IF(IsLoopEnd(i, len)) {
                    r0 = AddS(r0, Element(DataType::DT_FP32, 1.0));
                } ELSE {
                    r0 = AddS(r0, Element(DataType::DT_FP32, 2.0));
                }
            } ELSE {
                IF(IsLoopEnd(i, len)) {
                    r0 = AddS(r0, Element(DataType::DT_FP32, 0.0));
                } ELSE {
                    Tensor t0v = DView(t0, {s, s}, {s * i, 0});
                    r0 = Add(t0v, r0);
                }
            }
        }
        FUNCTION("S1", FunctionType::STATIC) {
            out = AddS(r0, Element(DataType::DT_FP32, 3.0));
        }
    }
    return out;
}

Tensor TestLoopWithManualRank(const Tensor &t0, Tensor &r0, Tensor &out, int s, int maxRank) {
    auto func = [](const Tensor &lt0, Tensor &lr0, Tensor &lout, int ls, const npu::tile_fwk::SymbolicScalar &i, int r) {
        Program::GetInstance().GetTileShape().SetVecTileShapes({ls * r, ls * r});
        Tensor t0v = DView(lt0, {ls * r, ls}, {ls * i, 0});
        Tensor r0v = DView(lr0, {ls * r, ls}, {ls * i, 0});
        Tensor tmp = Add(t0v, r0v);
        DAssemble(tmp, {ls * i, 0}, lout);
    };
    FUNCTION("main", FunctionType::DYNAMIC, {t0, r0}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShapeDim(t0, 0) / s), PowersOf2(maxRank)) {
            UNROLL_DEFAULT {
                func(t0, r0, out, s, i, 1);
            }
            UNROLL(2) {
                func(t0, r0, out, s, i, 2);
            }
            UNROLL(3) {
                func(t0, r0, out, s, i, 3);
            }
        }
        FUNCTION("S1", FunctionType::STATIC) {
            out = AddS(out, Element(DataType::DT_FP32, 1.0));
        }
    }
    return out;
}

TEST_F(DynamicFunctionTest, TestLoopWithRank) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {s, s}, "r0");
    Tensor out(DT_FP32, {s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 3.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int maxUnrollTimes = 16;
    TestLoopWithRank(t0, r0, out, s, maxUnrollTimes);

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCallopAttrList().size(), 6);
    for (auto &callAttr : mainFunc->GetCallopAttrList()) {
        auto subFunc = Program::GetInstance().GetFunctionByMagicName(callAttr->GetCalleeMagicName());
        EXPECT_NE(subFunc, nullptr);
        if (subFunc->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
            auto loopAttr = subFunc->GetDynloopAttribute();
            EXPECT_NE(loopAttr, nullptr);
            EXPECT_EQ(loopAttr->unrollTimes, maxUnrollTimes);
            EXPECT_EQ(loopAttr->pathList.size(), 1);
            maxUnrollTimes /= 2;
        }
    }
}

TEST_F(DynamicFunctionTest, TestLoopIfWithRank) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

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
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int maxUnrollTimes = 16;
    TestLoopIfWithRank(t0, r0, out, s, maxUnrollTimes);

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCallopAttrList().size(), 6);
    for (auto &callAttr : mainFunc->GetCallopAttrList()) {
        auto subFunc = Program::GetInstance().GetFunctionByMagicName(callAttr->GetCalleeMagicName());
        EXPECT_NE(subFunc, nullptr);
        if (subFunc->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
            auto loopAttr = subFunc->GetDynloopAttribute();
            EXPECT_NE(loopAttr, nullptr);
            EXPECT_EQ(loopAttr->unrollTimes, maxUnrollTimes);
            EXPECT_EQ(loopAttr->pathList.size(), 4);
            maxUnrollTimes /= 2;
        }
    }
}

TEST_F(DynamicFunctionTest, TestLoopWithManualRank) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 10;
    Tensor t0(DT_FP32, {n * s, s}, "t0");
    Tensor r0(DT_FP32, {n * s, s}, "r0");
    Tensor out(DT_FP32, {n * s, s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(r0, 0.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0),
    });

    int maxUnrollTimes = 16;
    TestLoopWithManualRank(t0, r0, out, s, maxUnrollTimes);

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCallopAttrList().size(), 7);

    std::vector<int> ranks = {16, 8, 4, 3, 2, 1};
    int idx = 0;
    for (auto &callAttr : mainFunc->GetCallopAttrList()) {
        auto subFunc = Program::GetInstance().GetFunctionByMagicName(callAttr->GetCalleeMagicName());
        EXPECT_NE(subFunc, nullptr);
        if (subFunc->GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
            auto loopAttr = subFunc->GetDynloopAttribute();
            EXPECT_NE(loopAttr, nullptr);
            EXPECT_EQ(loopAttr->unrollTimes, ranks[idx++]);
            ALOG_ERROR("unrollTimes: ", loopAttr->unrollTimes, " range: ",loopAttr->loopRange.Dump());
            EXPECT_EQ(loopAttr->pathList.size(), 1);
        }
    }
}

TEST_F(DynamicFunctionTest, TestSymbolicScalarDumpLoad) {
    const Json expJson = {2, 6, 2, 16, 3, 1, "RUNTIME_GetInputDataInt32Dim1", 1, "INPUT_actSeqs", 1, "bIdx", 0, 256};
    SymbolicScalar exp = LoadSymbolicScalar(expJson);
    EXPECT_EQ(expJson, ToJson(exp));
}

TEST_F(DynamicFunctionTest, TestInnerLoopOrder) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(512, 512);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({128, 128}, {128, 128}, {128, 128});

    int vecLen = 128;
    int loopNum = 5;
    int tileNum = 4;
    Tensor inputA(DT_FP32, {loopNum, vecLen}, "inputA");
    Tensor inputB(DT_FP32, {tileNum, vecLen}, "inputB");
    Tensor output(DT_FP32, {1, vecLen}, "out");

    FUNCTION("Main", FunctionType::DYNAMIC, {inputA, inputB}, {output}) {
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

    auto funcMap = Program::GetInstance().GetFunctionMap();
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_Main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1);
    auto outerLoopFunc = mainFunc->GetCalleeFunctionList()[0];
    EXPECT_EQ(outerLoopFunc->GetMagicName(), "TENSOR_Outer_Unroll1_3");
    EXPECT_EQ(outerLoopFunc->GetCalleeFunctionList().size(), 1);
    auto outerLoopPathFunc = outerLoopFunc->GetCalleeFunctionList()[0];
    EXPECT_EQ(outerLoopPathFunc->GetMagicName(), "TENSOR_Outer_Unroll1_PATH0_4");
    EXPECT_EQ(outerLoopPathFunc->GetCalleeFunctionList().size(), 3);
    int idx = 0;
    std::vector<std::string> innerLoopFuncNames = {"TENSOR_Inner_Unroll1_5", "TENSOR_Inner2_Unroll1_7", "TENSOR_Inner3_Unroll1_9"};
    for (auto &innerLoopFunc : outerLoopPathFunc->GetCalleeFunctionList()) {
        ALOG_INFO("innerLoopFunc: ", innerLoopFunc->GetMagicName());
        EXPECT_EQ(innerLoopFunc->GetMagicName(), innerLoopFuncNames[idx++]);
        EXPECT_EQ(innerLoopFunc->GetCalleeFunctionList().size(), 1);
    }
}

TEST_F(DynamicFunctionTest, TestGetInputDataInt32Dim3) {
    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    Program::GetInstance().GetTileShape().SetVecTileShapes(32, 32, 32);
    Program::GetInstance().GetTileShape().SetCubeTileShapes({32, 32}, {32, 32}, {32, 32});

    int s = 32;
    int n = 1;
    int m = 1;
    Tensor t5(DT_FP32, {2 , n * s, m * s}, "t5");
    Tensor out(DT_FP32, {n * s, m * s}, "out");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t5, 64.0),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0),
    });

    int loopCount = 0;
    FUNCTION("main", FunctionType::DYNAMIC, {t5}, {out}) {
        LOOP("s1", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputDataInt32Dim3(t5,  npu::tile_fwk::SymbolicScalar("0"), npu::tile_fwk::SymbolicScalar("1"), npu::tile_fwk::SymbolicScalar("2")) / s)) {
            loopCount += i;
            out = AddS(t5, Element(DataType::DT_FP32, static_cast<double>(1.0)));
        }
    }

    std::shared_ptr<DyndevFunctionAttribute> attr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto funcop = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
}