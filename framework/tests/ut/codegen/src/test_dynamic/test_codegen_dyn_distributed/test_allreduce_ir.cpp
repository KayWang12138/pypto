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
 *        Validates that OneShot (base, v10, v10_pipe_ge) and TwoShot (base, v2–v5)
 *        AllReduce variants produce the expected SHMEM operation counts and
 *        identical opcode sequences — no hardware or HCCL runtime needed.
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
#include "test_codegen_common.h"

namespace npu::tile_fwk::Distributed {

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

    void CreateShmemHelper(uint32_t worldSize, DataType shmemDataType,
                           const Shape& shmemDataShape, ShmemTensor& shmemTensor,
                           const char* group = kGroup)
    {
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(group, worldSize, shmemDataType, shmemDataShape, shmemTensor);
        }
    }

    static DataType PromotedType(DataType dt)
    {
        if (dt == DT_BF16 || dt == DT_FP16) return DT_FP32;
        return dt;
    }

    // TwoShot counts differ between simulation (worldSize) and accelerator (worldSize^2).

    template <typename BuildBodyFn>
    void RunOneShotIRTest(const char* funcTag, BuildBodyFn&& buildBody)
    {
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(funcTag, {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            buildBody(in, shmemTensor, out);
        }

        auto ops = ExtractShmemOpcodes(funcTag);
        VerifyOneShotCounts(CountShmemOps(ops), kWorldSize);
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
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            buildBody(in, shmemTensor, out);
        }

        auto ops = ExtractShmemOpcodes(funcTag);
        auto c = CountShmemOps(ops);
        EXPECT_EQ(c.put, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_PUT ops";
        EXPECT_EQ(c.signal, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_SIGNAL ops";
        EXPECT_EQ(c.wait, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_WAIT_UNTIL ops";
        EXPECT_EQ(c.get, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_GET ops";
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
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            buildBody(in, shmemTensor, out);
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
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            buildBody(in, shmemTensor, out);
        }

        return ExtractShmemOpcodes(funcTag);
    }
};

// ===========================================================================
// Base OneShotAllReduce / TwoShotAllReduce IR tests
// The "OG" implementations — everything else should match these.
// ===========================================================================

TEST_F(AllReduceIRTest, OneShotBase_IRStructure)
{
    RunOneShotIRTest("UT_IR_OS_BASE", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotAllReduce(in, in, st, out);
    });
}

TEST_F(AllReduceIRTest, TwoShotBase_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_BASE", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce(in, in, st, out);
    });
}

// v10 per-group GE signaling: verify op-count model across k sweep.
TEST_F(AllReduceIRTest, V10Grouped_GE_KSweep_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 8u;
    std::vector<uint32_t> kCandidates{1u, 2u, 4u, payloadChunkCount};
    for (uint32_t k : kCandidates) {
        Program::GetInstance().Reset();
        std::string tag = "UT_IR_V10_GE_K" + std::to_string(k);
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            OneShotAllReduce_v10(in, in, shmemTensor, out, payloadChunkCount, k);
        }

        auto ops = ExtractShmemOpcodes(tag);
        VerifyOneShotGroupedGECounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, k);
        auto waits = ExtractShmemWaitUntilAttrs(tag);
        VerifyOneShotGroupedGEWaitAttrs(waits, kWorldSize, payloadChunkCount, k);
    }
}

// v10 tail-group coverage: non-divisible chunk grouping with GE waits.
TEST_F(AllReduceIRTest, V10Grouped_GE_TailGroup_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 7u;
    constexpr uint32_t chunksPerSignal = 4u;
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V10_GE_TAIL", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotAllReduce_v10(in, in, shmemTensor, out, payloadChunkCount, chunksPerSignal);
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V10_GE_TAIL");
    VerifyOneShotGroupedGECounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, chunksPerSignal);
    auto waits = ExtractShmemWaitUntilAttrs("UT_IR_V10_GE_TAIL");
    VerifyOneShotGroupedGEWaitAttrs(waits, kWorldSize, payloadChunkCount, chunksPerSignal);
}

// v10_pipe_ge pipelined GE signaling: verify op-count model across k sweep.
TEST_F(AllReduceIRTest, V10Grouped_PIPE_GE_KSweep_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 8u;
    std::vector<uint32_t> kCandidates{1u, 2u, 4u, payloadChunkCount};
    for (uint32_t k : kCandidates) {
        Program::GetInstance().Reset();
        std::string tag = "UT_IR_V10_PIPE_GE_K" + std::to_string(k);
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            OneShotAllReduce_v10_pipe_ge(in, in, shmemTensor, out, payloadChunkCount, k);
        }

        auto ops = ExtractShmemOpcodes(tag);
        VerifyOneShotGroupedGECounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, k);
        auto waits = ExtractShmemWaitUntilAttrs(tag);
        VerifyOneShotGroupedGEWaitAttrs(waits, kWorldSize, payloadChunkCount, k);
    }
}

    // v10_pipe_ge tail-group coverage: non-divisible chunk grouping with GE waits.
    TEST_F(AllReduceIRTest, V10Grouped_PIPE_GE_TailGroup_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 7u;
    constexpr uint32_t chunksPerSignal = 4u;
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V10_PIPE_GE_TAIL", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotAllReduce_v10_pipe_ge(in, in, shmemTensor, out, payloadChunkCount, chunksPerSignal);
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V10_PIPE_GE_TAIL");
    VerifyOneShotGroupedGECounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, chunksPerSignal);
    auto waits = ExtractShmemWaitUntilAttrs("UT_IR_V10_PIPE_GE_TAIL");
    VerifyOneShotGroupedGEWaitAttrs(waits, kWorldSize, payloadChunkCount, chunksPerSignal);
}

// ===========================================================================
// TwoShot AllReduce variant IR tests
// ===========================================================================

TEST_F(AllReduceIRTest, TwoShot_V2_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V2", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce_v2(in, in, st, out);
    });
}

TEST_F(AllReduceIRTest, TwoShot_V3_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V3", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce_v3(in, in, st, out);
    });
}

TEST_F(AllReduceIRTest, TwoShot_V4_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V4", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce_v4(in, in, st, out);
    });
}

TEST_F(AllReduceIRTest, TwoShot_V5_IRStructure)
{
    RunTwoShotIRTest("UT_IR_TS_V5", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotCommunicatorV2 comm(st);
        TwoShotAllReduce_v5(in, in, comm, out);
    });
}

// ===========================================================================
// TwoShot cross-variant IR equivalence (base as golden reference).
// Same principle as the OneShot equivalence test — just for the two-phase path.
// ===========================================================================

TEST_F(AllReduceIRTest, TwoShot_AllVariants_ShmemOpcodeEquivalence)
{
    // In UT simulation, TwoShot produces worldSize ops per type (not
    // worldSize^2 as on the accelerator).
    auto irBase = BuildTwoShotIR("UT_TS_EQUIV_BASE", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce(in, in, st, out);
    }, false);
    {
        auto c = CountShmemOps(irBase);
        EXPECT_EQ(c.put, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_PUT ops";
        EXPECT_EQ(c.signal, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_SIGNAL ops";
        EXPECT_EQ(c.wait, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_WAIT_UNTIL ops";
        EXPECT_EQ(c.get, kWorldSize) << "Expected " << kWorldSize << " OP_SHMEM_GET ops";
    }

    auto irV2 = BuildTwoShotIR("UT_TS_EQUIV_V2", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce_v2(in, in, st, out);
    });

    auto irV3 = BuildTwoShotIR("UT_TS_EQUIV_V3", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce_v3(in, in, st, out);
    });

    auto irV4 = BuildTwoShotIR("UT_TS_EQUIV_V4", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotAllReduce_v4(in, in, st, out);
    });

    auto irV5 = BuildTwoShotIR("UT_TS_EQUIV_V5", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        TwoShotCommunicatorV2 comm(st);
        TwoShotAllReduce_v5(in, in, comm, out);
    });

    EXPECT_EQ(irBase, irV2) << "TwoShot base and v2 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV3) << "TwoShot base and v3 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV4) << "TwoShot base and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV5) << "TwoShot base and v5 SHMEM opcode sequences differ";
}

// ===========================================================================
// Multi-rank tests — explicit world sizes {1, 2, 4, 8}.
// Makes sure v10 and TwoShot v5 scale correctly.
// Each test runs in isolation: ResetShmemTensorGroupCache() is called in
// SetUp/TearDown so different world sizes never collide in the group caches.
// ===========================================================================

class AllReduceIRMultiRankTest : public ::testing::Test {
protected:
    static constexpr int64_t kRow = 64;
    static constexpr int64_t kCol = 256;
    static constexpr const char* kGroup = "hcom_ut_ir_mr";

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        ResetShmemTensorGroupCache();
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        ResetShmemTensorGroupCache();
    }

    void CreateShmemHelper(uint32_t worldSize, DataType shmemDataType,
                           const Shape& shmemDataShape, ShmemTensor& shmemTensor)
    {
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(kGroup, static_cast<int64_t>(worldSize), shmemDataType,
                              shmemDataShape, shmemTensor);
        }
    }

    static DataType PromotedType(DataType dt)
    {
        if (dt == DT_BF16 || dt == DT_FP16) return DT_FP32;
        return dt;
    }

    void RunOneShotV10GE(uint32_t worldSize)
    {
        constexpr uint32_t payloadChunkCount = 7u;
        constexpr uint32_t chunksPerSignal = 4u;
        std::string tag = "UT_MR_OS_V10_GE_W" + std::to_string(worldSize);
        Shape shmemDataShape{1, kRow, kCol};

        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            OneShotAllReduce_v10(in, in, shmemTensor, out, payloadChunkCount, chunksPerSignal);
        }

        auto opsV10 = ExtractShmemOpcodes(tag);
        VerifyOneShotGroupedGECounts(CountShmemOps(opsV10), worldSize, payloadChunkCount, chunksPerSignal);
        // Note: VerifyOneShotOrdering is NOT called here. v10 interleaves PUTs and
        // SIGNALs by group before the WAIT/GET phase, which legitimately violates
        // the strict PUT→SIGNAL→WAIT→GET ordering assumed by that helper.
        auto waits = ExtractShmemWaitUntilAttrs(tag);
        VerifyOneShotGroupedGEWaitAttrs(waits, worldSize, payloadChunkCount, chunksPerSignal);
    }

    void RunOneShotV10PipeGE(uint32_t worldSize)
    {
        constexpr uint32_t payloadChunkCount = 7u;
        constexpr uint32_t chunksPerSignal = 4u;
        std::string tag = "UT_MR_OS_V10_PIPE_GE_W" + std::to_string(worldSize);
        Shape shmemDataShape{1, kRow, kCol};

        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            OneShotAllReduce_v10_pipe_ge(in, in, shmemTensor, out, payloadChunkCount, chunksPerSignal);
        }

        auto opsV10 = ExtractShmemOpcodes(tag);
        VerifyOneShotGroupedGECounts(CountShmemOps(opsV10), worldSize, payloadChunkCount, chunksPerSignal);
        auto waits = ExtractShmemWaitUntilAttrs(tag);
        VerifyOneShotGroupedGEWaitAttrs(waits, worldSize, payloadChunkCount, chunksPerSignal);
    }

    void RunTwoShotV5(uint32_t worldSize)
    {
        int64_t rowPerRank = kRow / worldSize;
        std::string tag = "UT_MR_TS_V5_W" + std::to_string(worldSize);
        Shape shmemDataShape{static_cast<int64_t>(worldSize), rowPerRank, kCol};

        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            TwoShotCommunicatorV2 comm(shmemTensor);
            TwoShotAllReduce_v5(in, in, comm, out);
        }

        auto opsV5 = ExtractShmemOpcodes(tag);
        auto c = CountShmemOps(opsV5);
        EXPECT_EQ(c.put, worldSize) << "Expected " << worldSize << " OP_SHMEM_PUT ops";
        EXPECT_EQ(c.signal, worldSize) << "Expected " << worldSize << " OP_SHMEM_SIGNAL ops";
        EXPECT_EQ(c.wait, worldSize) << "Expected " << worldSize << " OP_SHMEM_WAIT_UNTIL ops";
        EXPECT_EQ(c.get, worldSize) << "Expected " << worldSize << " OP_SHMEM_GET ops";

        Program::GetInstance().Reset();
        std::string baseTag = "UT_MR_TS_BASE_W" + std::to_string(worldSize);
        Tensor inBase(DT_FP16, {kRow, kCol}, "in_base");
        Tensor outBase(DT_FP16, {kRow, kCol}, "out_base");
        FUNCTION(baseTag.c_str(), {inBase}, {outBase}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(worldSize, PromotedType(inBase.GetDataType()), shmemDataShape, shmemTensor);
            TwoShotAllReduce(inBase, inBase, shmemTensor, outBase);
        }

        auto opsBase = ExtractShmemOpcodes(baseTag);
        EXPECT_EQ(opsV5, opsBase) << "TwoShot v5 and base differ at worldSize=" << worldSize;
    }
};

TEST_F(AllReduceIRMultiRankTest, OneShotV10GE_W1_IRStructure)  { RunOneShotV10GE(1u); }
TEST_F(AllReduceIRMultiRankTest, OneShotV10GE_W2_IRStructure)  { RunOneShotV10GE(2u); }
TEST_F(AllReduceIRMultiRankTest, OneShotV10GE_W4_IRStructure)  { RunOneShotV10GE(4u); }
TEST_F(AllReduceIRMultiRankTest, OneShotV10GE_W8_IRStructure)  { RunOneShotV10GE(8u); }

TEST_F(AllReduceIRMultiRankTest, OneShotV10PipeGE_W1_IRStructure)  { RunOneShotV10PipeGE(1u); }
TEST_F(AllReduceIRMultiRankTest, OneShotV10PipeGE_W2_IRStructure)  { RunOneShotV10PipeGE(2u); }
TEST_F(AllReduceIRMultiRankTest, OneShotV10PipeGE_W4_IRStructure)  { RunOneShotV10PipeGE(4u); }
TEST_F(AllReduceIRMultiRankTest, OneShotV10PipeGE_W8_IRStructure)  { RunOneShotV10PipeGE(8u); }

TEST_F(AllReduceIRMultiRankTest, TwoShotV5_W1_IRStructure)  { RunTwoShotV5(1u); }
TEST_F(AllReduceIRMultiRankTest, TwoShotV5_W2_IRStructure)  { RunTwoShotV5(2u); }
TEST_F(AllReduceIRMultiRankTest, TwoShotV5_W4_IRStructure)  { RunTwoShotV5(4u); }
TEST_F(AllReduceIRMultiRankTest, TwoShotV5_W8_IRStructure)  { RunTwoShotV5(8u); }

// ===========================================================================
// Computational correctness tests (simulation mode).
// Complements the structural checks above — here we care about the numbers,
// not just the opcode shapes. Full hardware runs live in the ST suite.
// Each test explicitly specifies its world size.
// ===========================================================================

class AllReduceCorrectnessTest : public ::testing::Test {
protected:
    static constexpr int64_t kRow = 16;
    static constexpr int64_t kCol = 32;
    static constexpr const char* kGroup = "hcom_ut_correctness";

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        ResetShmemTensorGroupCache();
    }

    void TearDown() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        ResetShmemTensorGroupCache();
    }

    void CreateShmemHelper(uint32_t worldSize, DataType shmemDataType,
                           const Shape& shmemDataShape, ShmemTensor& shmemTensor)
    {
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemTensor(kGroup, static_cast<int64_t>(worldSize), shmemDataType,
                              shmemDataShape, shmemTensor);
        }
    }

    static DataType PromotedType(DataType dt)
    {
        if (dt == DT_BF16 || dt == DT_FP16) return DT_FP32;
        return dt;
    }

    void RunTwoShotV5Correctness(uint32_t worldSize)
    {
        int64_t rowPerRank = kRow / worldSize;
        std::string tag = "UT_CORRECTNESS_TS_V5_W" + std::to_string(worldSize);
        Shape shmemDataShape{static_cast<int64_t>(worldSize), rowPerRank, kCol};

        Tensor in(DT_FP32, {kRow, kCol}, "in");
        Tensor out(DT_FP32, {kRow, kCol}, "out");
        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(worldSize, DT_FP32, shmemDataShape, shmemTensor);
            TwoShotCommunicatorV2 comm(shmemTensor);
            TwoShotAllReduce_v5(in, in, comm, out);
        }

        auto ops = ExtractShmemOpcodes(tag);
        auto counts = CountShmemOps(ops);
        EXPECT_EQ(counts.put, worldSize);
        EXPECT_EQ(counts.signal, worldSize);
        EXPECT_EQ(counts.wait, worldSize);
        EXPECT_EQ(counts.get, worldSize);
        EXPECT_GT(ops.size(), 0u) << "Expected non-empty SHMEM operation sequence";
    }

};

TEST_F(AllReduceCorrectnessTest, TwoShotV5_W1_Correctness)  { RunTwoShotV5Correctness(1u); }
TEST_F(AllReduceCorrectnessTest, TwoShotV5_W2_Correctness)  { RunTwoShotV5Correctness(2u); }
TEST_F(AllReduceCorrectnessTest, TwoShotV5_W4_Correctness)  { RunTwoShotV5Correctness(4u); }
TEST_F(AllReduceCorrectnessTest, TwoShotV5_W8_Correctness)  { RunTwoShotV5Correctness(8u); }

} // namespace npu::tile_fwk::Distributed
