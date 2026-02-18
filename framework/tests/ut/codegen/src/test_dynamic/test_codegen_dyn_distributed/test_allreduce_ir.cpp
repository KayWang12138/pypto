/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_allreduce_ir.cpp
 * \brief Unit tests for AllReduce IR structure verification.
 *        Validates that OneShot and TwoShot AllReduce variants (base, v2–v5)
 *        produce the expected SHMEM operation counts, correct ordering, and
 *        identical opcode sequences — without requiring hardware or HCCL runtime.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/distributed_communicator.h"
#include "shmem_ir_test_utils.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::Distributed;

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class AllReduceIRTest : public ::testing::Test {
protected:
    static constexpr uint32_t kWorldSize = 4;
    static constexpr int64_t kRow = 64;
    static constexpr int64_t kCol = 256;
    static constexpr const char* kGroup = "hcom_ut_ir";

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void CreateShmemTensors(DataType shmemDataType, const Shape& shmemDataShape,
                            Tensor& shmemData, Tensor& shmemSignal)
    {
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemData(kGroup, kWorldSize, shmemDataType, shmemDataShape, shmemData);
            CreateShmemSignal(kGroup, shmemData, shmemSignal);
        }
    }

    static DataType PromotedType(DataType dt)
    {
        if (dt == DT_BF16 || dt == DT_FP16) return DT_FP32;
        return dt;
    }

    // -----------------------------------------------------------------------
    // Helpers: build IR, verify counts + ordering for a single variant
    // -----------------------------------------------------------------------

    template <typename BuildBodyFn>
    void RunOneShotIRTest(const char* funcTag, BuildBodyFn&& buildBody)
    {
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(funcTag, {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            buildBody(in, kGroup, shmemData, shmemSignal, out);
        }

        auto ops = ExtractShmemOpcodes(funcTag);
        VerifyOneShotCounts(CountShmemOps(ops), kWorldSize);
        VerifyOneShotOrdering(ops);
    }

    template <typename BuildBodyFn>
    void RunTwoShotIRTest(const char* funcTag, BuildBodyFn&& buildBody)
    {
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        constexpr int64_t kRowPerRank = kRow / kWorldSize;
        Shape shmemDataShape{kWorldSize, kRowPerRank, kCol};

        FUNCTION(funcTag, {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            buildBody(in, kGroup, shmemData, shmemSignal, out);
        }

        auto ops = ExtractShmemOpcodes(funcTag);
        VerifyTwoShotCounts(CountShmemOps(ops), kWorldSize);
        VerifyTwoShotOrdering(ops, kWorldSize);
    }

    // -----------------------------------------------------------------------
    // Helpers: build IR and return opcodes (for equivalence tests)
    // -----------------------------------------------------------------------

    template <typename BuildBodyFn>
    std::vector<Opcode> BuildOneShotIR(const char* funcTag, BuildBodyFn&& buildBody, bool reset = true)
    {
        if (reset) Program::GetInstance().Reset();
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(funcTag, {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            buildBody(in, kGroup, shmemData, shmemSignal, out);
        }

        return ExtractShmemOpcodes(funcTag);
    }

    template <typename BuildBodyFn>
    std::vector<Opcode> BuildTwoShotIR(const char* funcTag, BuildBodyFn&& buildBody, bool reset = true)
    {
        if (reset) Program::GetInstance().Reset();
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        constexpr int64_t kRowPerRank = kRow / kWorldSize;
        Shape shmemDataShape{kWorldSize, kRowPerRank, kCol};

        FUNCTION(funcTag, {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            buildBody(in, kGroup, shmemData, shmemSignal, out);
        }

        return ExtractShmemOpcodes(funcTag);
    }
};

// ===========================================================================
// Base OneShotAllReduce / TwoShotAllReduce IR tests
// ===========================================================================

TEST_F(AllReduceIRTest, OneShotBase_IRStructure)
{
    RunOneShotIRTest("UT_IR_OS_BASE", [](Tensor& in, const char* group,
                                          Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, TwoShotBase_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_BASE", [](Tensor& in, const char* group,
                                          Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce(in, in, group, sd, ss, out);
    });
}

// ===========================================================================
// OneShot variant IR structure tests
// ===========================================================================

TEST_F(AllReduceIRTest, V2_IRStructure)
{
    RunOneShotIRTest("UT_IR_V2", [](Tensor& in, const char* group,
                                     Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce_v2(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, V3_IRStructure)
{
    RunOneShotIRTest("UT_IR_V3", [](Tensor& in, const char* group,
                                     Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce_v3(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, V4_IRStructure)
{
    RunOneShotIRTest("UT_IR_V4", [](Tensor& in, const char* group,
                                     Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce_v4(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, V5PlusPull_IRStructure)
{
    RunOneShotIRTest("UT_IR_V5", [](Tensor& in, const char* group,
                                     Tensor& sd, Tensor& ss, Tensor& out) {
        CommunicatorV2 comm(group, kWorldSize, ss);
        auto waitToken = OneShotAllReduce_v5(in, in, sd, comm);
        out = comm.Pull(waitToken, sd);
    });
}

// ===========================================================================
// OneShot cross-variant IR equivalence
// ===========================================================================

TEST_F(AllReduceIRTest, AllVariants_ShmemOpcodeEquivalence)
{
    auto irV2 = BuildOneShotIR("UT_EQUIV_V2", [](Tensor& in, const char* g,
                                                   Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce_v2(in, in, g, sd, ss, out);
    }, false);

    auto irV3 = BuildOneShotIR("UT_EQUIV_V3", [](Tensor& in, const char* g,
                                                   Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce_v3(in, in, g, sd, ss, out);
    });

    auto irV4 = BuildOneShotIR("UT_EQUIV_V4", [](Tensor& in, const char* g,
                                                   Tensor& sd, Tensor& ss, Tensor& out) {
        OneShotAllReduce_v4(in, in, g, sd, ss, out);
    });

    auto irV5 = BuildOneShotIR("UT_EQUIV_V5", [](Tensor& in, const char* g,
                                                   Tensor& sd, Tensor& ss, Tensor& out) {
        CommunicatorV2 comm(g, kWorldSize, ss);
        auto waitToken = OneShotAllReduce_v5(in, in, sd, comm);
        out = comm.Pull(waitToken, sd);
    });

    EXPECT_EQ(irV2, irV3) << "v2 and v3 SHMEM opcode sequences differ";
    EXPECT_EQ(irV3, irV4) << "v3 and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irV4, irV5) << "v4 and v5+Pull SHMEM opcode sequences differ";
}

// ===========================================================================
// Total operation count sanity check (BF16 variant)
// ===========================================================================

TEST_F(AllReduceIRTest, V2_TotalOpCountNonZero)
{
    Tensor in(DT_BF16, {kRow, kCol}, "in");
    Tensor out(DT_BF16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_OPCOUNT_V2", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        Tensor shmemData, shmemSignal;
        CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
        OneShotAllReduce_v2(in, in, kGroup, shmemData, shmemSignal, out);
    }

    auto shmemOps = ExtractShmemOpcodes("UT_OPCOUNT_V2");
    EXPECT_GT(shmemOps.size(), 0u) << "Function should contain SHMEM operations";
}

// ===========================================================================
// TwoShot AllReduce variant IR tests
// ===========================================================================

TEST_F(AllReduceIRTest, TwoShot_V2_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V2", [](Tensor& in, const char* group,
                                        Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce_v2(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, TwoShot_V3_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V3", [](Tensor& in, const char* group,
                                        Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce_v3(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, TwoShot_V4_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V4", [](Tensor& in, const char* group,
                                        Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce_v4(in, in, group, sd, ss, out);
    });
}

TEST_F(AllReduceIRTest, TwoShot_V5_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V5", [](Tensor& in, const char* group,
                                        Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotCommunicatorV2 comm(group, kWorldSize, ss);
        TwoShotAllReduce_v5(in, in, sd, comm, out);
    });
}

// ===========================================================================
// TwoShot cross-variant IR equivalence
// ===========================================================================

TEST_F(AllReduceIRTest, TwoShot_AllVariants_ShmemOpcodeEquivalence)
{
    auto irV2 = BuildTwoShotIR("UT_TS_EQUIV_V2", [](Tensor& in, const char* g,
                                                      Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce_v2(in, in, g, sd, ss, out);
    }, false);

    auto irV3 = BuildTwoShotIR("UT_TS_EQUIV_V3", [](Tensor& in, const char* g,
                                                      Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce_v3(in, in, g, sd, ss, out);
    });

    auto irV4 = BuildTwoShotIR("UT_TS_EQUIV_V4", [](Tensor& in, const char* g,
                                                      Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotAllReduce_v4(in, in, g, sd, ss, out);
    });

    auto irV5 = BuildTwoShotIR("UT_TS_EQUIV_V5", [](Tensor& in, const char* g,
                                                      Tensor& sd, Tensor& ss, Tensor& out) {
        TwoShotCommunicatorV2 comm(g, kWorldSize, ss);
        TwoShotAllReduce_v5(in, in, sd, comm, out);
    });

    EXPECT_EQ(irV2, irV3) << "TwoShot v2 and v3 SHMEM opcode sequences differ";
    EXPECT_EQ(irV3, irV4) << "TwoShot v3 and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irV4, irV5) << "TwoShot v4 and v5 SHMEM opcode sequences differ";
}
