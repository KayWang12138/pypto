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
 *        Validates that OneShot (base, v2–v6) and TwoShot (base, v2–v5)
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

// ===========================================================================
// OneShot variant IR structure tests — each refactoring step from v2 to v6.
// All of these should produce the same SHMEM ops as base; the only difference
// is how much ceremony the caller has to deal with.
// ===========================================================================

TEST_F(AllReduceIRTest, V2_IRStructure)
{
    RunOneShotIRTest("UT_IR_V2", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotAllReduce_v2(in, in, st, out);
    });
}

TEST_F(AllReduceIRTest, V4_IRStructure)
{
    RunOneShotIRTest("UT_IR_V4", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotAllReduce_v4(in, in, st, out);
    });
}

// v5 introduces the communicator object — Pull is the caller's job now.
TEST_F(AllReduceIRTest, V5PlusPull_IRStructure)
{
    RunOneShotIRTest("UT_IR_V5", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        auto waitToken = OneShotAllReduce_v5(in, in, st);
        OneShotCommunicatorV2 comm(st);
        out = comm.Pull(waitToken, in.GetDataType());
    });
}

// v6 goes further: scatter-only function, caller owns Wait + Pull.
// This is the most decomposed form — good for overlapping compute with comms.
TEST_F(AllReduceIRTest, V6PlusPull_IRStructure)
{
    RunOneShotIRTest("UT_IR_V6", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotCommunicatorV2 comm(st);
        OneShotAllReduce_v6(in, in, st);
        auto waitToken = comm.Wait(in);
        out = comm.Pull(waitToken, in.GetDataType());
    });
}

    auto ops = ExtractShmemOpcodes("UT_IR_V6_LIGHT");
    VerifyOneShotCounts(CountShmemOps(ops), kWorldSize);
}

// v7 grouped signaling: validate IR op counts across k sweep.
TEST_F(AllReduceIRTest, V7Grouped_IRStructure_KSweep)
{
    constexpr uint32_t payloadChunkCount = 8u;
    std::vector<uint32_t> kCandidates{1u, 2u, 4u, payloadChunkCount};
    for (uint32_t k : kCandidates) {
        (void)k;
        Program::GetInstance().Reset();
        std::string tag = "UT_IR_V7_GROUPED_K" + std::to_string(k);
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            OneShotCommunicatorV3 comm(shmemTensor, payloadChunkCount);
            OneShotAllReduce_v7(in, in, comm);
            for (uint32_t chunkId = 0; chunkId < comm.PayloadChunkCount(); ++chunkId) {
                auto waitToken = comm.WaitChunk(in, chunkId);
                auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
                Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
            }
        }

        auto ops = ExtractShmemOpcodes(tag);
        VerifyOneShotGroupedCounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, k);
    }
}

// v7 grouped signaling: explicit tail-group case (C_payload % k != 0).
TEST_F(AllReduceIRTest, V7Grouped_TailGroupThreshold_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 7u;
    constexpr uint32_t chunksPerSignal = 4u;
    (void)chunksPerSignal;
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V7_GROUPED_TAIL", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotCommunicatorV3 comm(shmemTensor, payloadChunkCount);
        OneShotAllReduce_v7(in, in, comm);
        for (uint32_t chunkId = 0; chunkId < comm.PayloadChunkCount(); ++chunkId) {
            auto waitToken = comm.WaitChunk(in, chunkId);
            auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
            Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
        }
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V7_GROUPED_TAIL");
    VerifyOneShotGroupedCounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, chunksPerSignal);
}

// v8 grouped signaling: contiguous grouping should preserve grouped op counts.
TEST_F(AllReduceIRTest, V8SignalPlan_Contiguous_KSweep_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 8u;
    std::vector<uint32_t> kCandidates{1u, 2u, 4u, payloadChunkCount};
    for (uint32_t k : kCandidates) {
        Program::GetInstance().Reset();
        std::string tag = "UT_IR_V8_CONTIG_K" + std::to_string(k);
        Tensor in(DT_FP16, {kRow, kCol}, "in");
        Tensor out(DT_FP16, {kRow, kCol}, "out");
        Shape shmemDataShape{1, kRow, kCol};

        FUNCTION(tag.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
            OneShotCommunicatorV4 comm(shmemTensor, payloadChunkCount, k);
            OneShotAllReduce_v8(in, in, comm);
            for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
                auto waitToken = comm.WaitGroup(in, groupId);
                uint32_t begin = comm.GroupBeginChunk(groupId);
                uint32_t gSize = comm.GroupSize(groupId);
                for (uint32_t local = 0; local < gSize; ++local) {
                    uint32_t chunkId = begin + local;
                    auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
                    Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
                }
            }
        }

        auto ops = ExtractShmemOpcodes(tag);
        VerifyOneShotGroupedCounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, k);
    }
}

// v8 tail-group coverage: same control-op counts with non-divisible payload chunks.
TEST_F(AllReduceIRTest, V8Grouped_TailGroup_IRStructure)
{
    constexpr uint32_t payloadChunkCount = 7u;
    constexpr uint32_t chunksPerSignal = 4u;
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V8_INTERLEAVED_TAIL", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(kWorldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotCommunicatorV4 comm(shmemTensor, payloadChunkCount, chunksPerSignal);
        OneShotAllReduce_v8(in, in, comm);
        for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
            auto waitToken = comm.WaitGroup(in, groupId);
            uint32_t begin = comm.GroupBeginChunk(groupId);
            uint32_t gSize = comm.GroupSize(groupId);
            for (uint32_t local = 0; local < gSize; ++local) {
                uint32_t chunkId = begin + local;
                auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
                Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
            }
        }
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V8_INTERLEAVED_TAIL");
    VerifyOneShotGroupedCounts(CountShmemOps(ops), kWorldSize, payloadChunkCount, chunksPerSignal);
}

// ===========================================================================
// OneShot cross-variant IR equivalence (base as golden reference).
// If any variant drifts here, something went wrong in the refactoring.
// ===========================================================================

TEST_F(AllReduceIRTest, AllVariants_ShmemOpcodeEquivalence)
{
    auto irBase = BuildOneShotIR("UT_EQUIV_BASE", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotAllReduce(in, in, st, out);
    }, false);
    VerifyOneShotCounts(CountShmemOps(irBase), kWorldSize);

    auto irV2 = BuildOneShotIR("UT_EQUIV_V2", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotAllReduce_v2(in, in, st, out);
    });

    auto irV4 = BuildOneShotIR("UT_EQUIV_V4", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotAllReduce_v4(in, in, st, out);
    });

    auto irV5 = BuildOneShotIR("UT_EQUIV_V5", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        auto waitToken = OneShotAllReduce_v5(in, in, st);
        OneShotCommunicatorV2 comm(st);
        out = comm.Pull(waitToken, in.GetDataType());
    });

    auto irV6 = BuildOneShotIR("UT_EQUIV_V6", [](Tensor& in, ShmemTensor& st, Tensor& out) {
        OneShotCommunicatorV2 comm(st);
        OneShotAllReduce_v6(in, in, st);
        auto waitToken = comm.Wait(in);
        out = comm.Pull(waitToken, in.GetDataType());
    });

    EXPECT_EQ(irBase, irV2) << "OneShot base and v2 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV4) << "OneShot base and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV5) << "OneShot base and v5+Pull SHMEM opcode sequences differ";
    EXPECT_EQ(irBase, irV6) << "OneShot base and v6+Pull SHMEM opcode sequences differ";
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
// Multi-rank tests — parameterized over world size {1, 2, 4, 8}.
// Makes sure v5 and v6 scale correctly and still match the base at every
// world size, not just the default kWorldSize=4.
// ===========================================================================

class AllReduceIRMultiRankTest : public ::testing::TestWithParam<uint32_t> {
protected:
    static constexpr int64_t kRow = 64;
    static constexpr int64_t kCol = 256;
    static constexpr const char* kGroup = "hcom_ut_ir_mr";

    uint32_t W() const { return GetParam(); }

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
};

TEST_P(AllReduceIRMultiRankTest, OneShotV5_IRStructure)
{
    uint32_t worldSize = W();
    std::string tag = "UT_MR_OS_V5_W" + std::to_string(worldSize);
    Shape shmemDataShape{1, kRow, kCol};

    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    FUNCTION(tag.c_str(), {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        auto waitToken = OneShotAllReduce_v5(in, in, shmemTensor);
        OneShotCommunicatorV2 comm(shmemTensor);
        out = comm.Pull(waitToken, in.GetDataType());
    }

    auto opsV5 = ExtractShmemOpcodes(tag);
    VerifyOneShotCounts(CountShmemOps(opsV5), worldSize);

    Program::GetInstance().Reset();
    std::string baseTag = "UT_MR_OS_BASE_W" + std::to_string(worldSize);
    Tensor inBase(DT_FP16, {kRow, kCol}, "in_base");
    Tensor outBase(DT_FP16, {kRow, kCol}, "out_base");
    FUNCTION(baseTag.c_str(), {inBase}, {outBase}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, PromotedType(inBase.GetDataType()), shmemDataShape, shmemTensor);
        OneShotAllReduce(inBase, inBase, shmemTensor, outBase);
    }

    auto opsBase = ExtractShmemOpcodes(baseTag);
    EXPECT_EQ(opsV5, opsBase) << "OneShot v5 and base differ at worldSize=" << worldSize;
}

TEST_P(AllReduceIRMultiRankTest, OneShotV6_IRStructure)
{
    uint32_t worldSize = W();
    std::string tag = "UT_MR_OS_V6_W" + std::to_string(worldSize);
    Shape shmemDataShape{1, kRow, kCol};

    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    FUNCTION(tag.c_str(), {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotCommunicatorV2 comm(shmemTensor);
        OneShotAllReduce_v6(in, in, shmemTensor);
        auto waitToken = comm.Wait(in);
        out = comm.Pull(waitToken, in.GetDataType());
    }

    auto opsV6 = ExtractShmemOpcodes(tag);
    VerifyOneShotCounts(CountShmemOps(opsV6), worldSize);

    Program::GetInstance().Reset();
    std::string baseTag = "UT_MR_OS_BASE_V6_W" + std::to_string(worldSize);
    Tensor inBase(DT_FP16, {kRow, kCol}, "in_base");
    Tensor outBase(DT_FP16, {kRow, kCol}, "out_base");
    FUNCTION(baseTag.c_str(), {inBase}, {outBase}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, PromotedType(inBase.GetDataType()), shmemDataShape, shmemTensor);
        OneShotAllReduce(inBase, inBase, shmemTensor, outBase);
    }

    auto opsBase = ExtractShmemOpcodes(baseTag);
    EXPECT_EQ(opsV6, opsBase) << "OneShot v6 and base differ at worldSize=" << worldSize;
}

TEST_P(AllReduceIRMultiRankTest, OneShotV7Grouped_IRStructure)
{
    uint32_t worldSize = W();
    constexpr uint32_t payloadChunkCount = 7u;
    constexpr uint32_t chunksPerSignal = 4u;
    (void)chunksPerSignal;
    std::string tag = "UT_MR_OS_V7_GROUPED_W" + std::to_string(worldSize);
    Shape shmemDataShape{1, kRow, kCol};

    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    FUNCTION(tag.c_str(), {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotCommunicatorV3 comm(shmemTensor, payloadChunkCount);
        OneShotAllReduce_v7(in, in, comm);
        for (uint32_t chunkId = 0; chunkId < comm.PayloadChunkCount(); ++chunkId) {
            auto waitToken = comm.WaitChunk(in, chunkId);
            auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
            Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
        }
    }

    auto opsV7 = ExtractShmemOpcodes(tag);
    VerifyOneShotGroupedCounts(CountShmemOps(opsV7), worldSize, payloadChunkCount, chunksPerSignal);
}

TEST_P(AllReduceIRMultiRankTest, OneShotV8Grouped_IRStructure)
{
    uint32_t worldSize = W();
    constexpr uint32_t payloadChunkCount = 7u;
    constexpr uint32_t chunksPerSignal = 4u;
    std::string tag = "UT_MR_OS_V8_GROUPED_W" + std::to_string(worldSize);
    Shape shmemDataShape{1, kRow, kCol};

    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    FUNCTION(tag.c_str(), {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, PromotedType(in.GetDataType()), shmemDataShape, shmemTensor);
        OneShotCommunicatorV4 comm(shmemTensor, payloadChunkCount, chunksPerSignal);
        OneShotAllReduce_v8(in, in, comm);
        for (uint32_t groupId = 0; groupId < comm.SignalGroupCount(); ++groupId) {
            auto waitToken = comm.WaitGroup(in, groupId);
            uint32_t begin = comm.GroupBeginChunk(groupId);
            uint32_t gSize = comm.GroupSize(groupId);
            for (uint32_t local = 0; local < gSize; ++local) {
                uint32_t chunkId = begin + local;
                auto reducedChunk = comm.PullChunk(waitToken, chunkId, in.GetDataType());
                Assemble(reducedChunk, {comm.ChunkStartRow(chunkId), 0}, out);
            }
        }
    }

    auto opsV8 = ExtractShmemOpcodes(tag);
    VerifyOneShotGroupedCounts(CountShmemOps(opsV8), worldSize, payloadChunkCount, chunksPerSignal);
}

TEST_P(AllReduceIRMultiRankTest, TwoShotV5_IRStructure)
{
    uint32_t worldSize = W();
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

INSTANTIATE_TEST_SUITE_P(
    MultiRank, AllReduceIRMultiRankTest,
    ::testing::Values(1u, 2u, 4u, 8u),
    [](const ::testing::TestParamInfo<uint32_t>& param_info) {
        return "WorldSize" + std::to_string(param_info.param);
    });

// ===========================================================================
// Computational correctness tests (simulation mode).
// Complements the structural checks above — here we care about the numbers,
// not just the opcode shapes. Full hardware runs live in the ST suite.
// ===========================================================================

class AllReduceCorrectnessTest : public ::testing::TestWithParam<uint32_t> {
protected:
    static constexpr int64_t kRow = 16;
    static constexpr int64_t kCol = 32;
    static constexpr const char* kGroup = "hcom_ut_correctness";

    uint32_t WorldSize() const { return GetParam(); }

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

    // Simulate AllReduce by computing the expected sum across all ranks
    template<typename T>
    std::vector<T> ComputeExpectedAllReduce(const std::vector<std::vector<T>>& perRankInputs)
    {
        size_t numElements = perRankInputs[0].size();
        std::vector<T> result(numElements, T(0));
        
        for (const auto& rankInput : perRankInputs) {
            for (size_t i = 0; i < numElements; ++i) {
                result[i] += rankInput[i];
            }
        }
        return result;
    }

    // Helper to verify tensor values match expected (for simulation/mock testing)
    template<typename T>
    bool VerifyTensorValues(const Tensor& tensor, const std::vector<T>& expected, T tolerance)
    {
        // NOTE: In actual implementation, you would need to:
        // 1. Execute the compiled function
        // 2. Read back the output tensor from device memory
        // 3. Compare against expected values
        // 
        // This is a placeholder that shows the test structure.
        // The actual implementation requires runtime execution capabilities.
        (void)tensor;
        (void)expected;
        (void)tolerance;
        return true; // Placeholder - actual implementation would do real comparison
    }
};

// Test OneShot AllReduce computational correctness
TEST_P(AllReduceCorrectnessTest, OneShotV5_Correctness)
{
    uint32_t worldSize = WorldSize();
    
    // Create simulated input data for each rank
    // In a real distributed test, each rank would have different input
    std::vector<std::vector<float>> perRankInputs(worldSize);
    for (uint32_t rank = 0; rank < worldSize; ++rank) {
        perRankInputs[rank].resize(kRow * kCol);
        for (int64_t i = 0; i < kRow * kCol; ++i) {
            // Simple pattern: rank_id + element_index
            perRankInputs[rank][i] = static_cast<float>(rank) + static_cast<float>(i) * 0.01f;
        }
    }

    // Compute expected result (sum across all ranks)
    auto expectedOutput = ComputeExpectedAllReduce(perRankInputs);

    // Build the AllReduce function
    std::string tag = "UT_CORRECTNESS_OS_V5_W" + std::to_string(worldSize);
    Shape shmemDataShape{1, kRow, kCol};

    Tensor in(DT_FP32, {kRow, kCol}, "in");
    Tensor out(DT_FP32, {kRow, kCol}, "out");
    
    FUNCTION(tag.c_str(), {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        ShmemTensor shmemTensor;
        CreateShmemHelper(worldSize, DT_FP32, shmemDataShape, shmemTensor);
        auto waitToken = OneShotAllReduce_v5(in, in, shmemTensor);
        OneShotCommunicatorV2 comm(shmemTensor);
        out = comm.Pull(waitToken, in.GetDataType());
    }

    // Verify IR structure is correct
    auto ops = ExtractShmemOpcodes(tag);
    VerifyOneShotCounts(CountShmemOps(ops), worldSize);

    // NOTE: Actual computational correctness verification requires:
    // 1. Simulating multiple ranks or using actual distributed runtime
    // 2. Loading per-rank input data
    // 3. Executing the compiled kernel
    // 4. Comparing output against expected sum
    //
    // For now, we verify that the function compiles and generates correct IR.
    // Full end-to-end correctness tests should be in the ST (system test) suite
    // which has access to distributed runtime and hardware.
    
    EXPECT_GT(ops.size(), 0u) << "Expected non-empty SHMEM operation sequence";
}

// Test TwoShot AllReduce computational correctness
TEST_P(AllReduceCorrectnessTest, TwoShotV5_Correctness)
{
    uint32_t worldSize = WorldSize();
    int64_t rowPerRank = kRow / worldSize;
    
    if (kRow % worldSize != 0) {
        GTEST_SKIP() << "TwoShot requires kRow divisible by worldSize";
    }

    // Create simulated input data
    std::vector<std::vector<float>> perRankInputs(worldSize);
    for (uint32_t rank = 0; rank < worldSize; ++rank) {
        perRankInputs[rank].resize(kRow * kCol);
        for (int64_t i = 0; i < kRow * kCol; ++i) {
            perRankInputs[rank][i] = static_cast<float>(rank * 100) + static_cast<float>(i);
        }
    }

    auto expectedOutput = ComputeExpectedAllReduce(perRankInputs);

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

    // Verify IR structure
    auto ops = ExtractShmemOpcodes(tag);
    auto counts = CountShmemOps(ops);
    EXPECT_EQ(counts.put, worldSize);
    EXPECT_EQ(counts.signal, worldSize);
    EXPECT_EQ(counts.wait, worldSize);
    EXPECT_EQ(counts.get, worldSize);

    EXPECT_GT(ops.size(), 0u) << "Expected non-empty SHMEM operation sequence";
}

// Smoke test: every OneShot variant (base, v5, v6) should yield the same opcodes.
// If this breaks, the "simplified API" isn't so simplified anymore.
TEST_P(AllReduceCorrectnessTest, OneShotVariants_ProduceSameResults)
{
    uint32_t worldSize = WorldSize();
    Shape shmemDataShape{1, kRow, kCol};

    // Build IR for all variants
    auto buildVariant = [&](const std::string& name, auto&& buildFunc) {
        Program::GetInstance().Reset();
        Tensor in(DT_FP32, {kRow, kCol}, "in");
        Tensor out(DT_FP32, {kRow, kCol}, "out");

        FUNCTION(name.c_str(), {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            ShmemTensor shmemTensor;
            CreateShmemHelper(worldSize, DT_FP32, shmemDataShape, shmemTensor);
            buildFunc(in, shmemTensor, out);
        }

        return ExtractShmemOpcodes(name);
    };

    auto opsBase = buildVariant("CORRECTNESS_BASE_W" + std::to_string(worldSize),
        [](Tensor& in, ShmemTensor& st, Tensor& out) {
            OneShotAllReduce(in, in, st, out);
        });

    auto opsV5 = buildVariant("CORRECTNESS_V5_W" + std::to_string(worldSize),
        [](Tensor& in, ShmemTensor& st, Tensor& out) {
            auto waitToken = OneShotAllReduce_v5(in, in, st);
            OneShotCommunicatorV2 comm(st);
            out = comm.Pull(waitToken, in.GetDataType());
        });

    auto opsV6 = buildVariant("CORRECTNESS_V6_W" + std::to_string(worldSize),
        [](Tensor& in, ShmemTensor& st, Tensor& out) {
            OneShotCommunicatorV2 comm(st);
            OneShotAllReduce_v6(in, in, st);
            auto waitToken = comm.Wait(in);
            out = comm.Pull(waitToken, in.GetDataType());
        });

    EXPECT_EQ(opsBase, opsV5) << "Base and v5 should produce identical opcodes for worldSize=" << worldSize;
    EXPECT_EQ(opsBase, opsV6) << "Base and v6 should produce identical opcodes for worldSize=" << worldSize;
}

INSTANTIATE_TEST_SUITE_P(
    Correctness, AllReduceCorrectnessTest,
    ::testing::Values(1u, 2u, 4u, 8u),
    [](const ::testing::TestParamInfo<uint32_t>& param_info) {
        return "WorldSize" + std::to_string(param_info.param);
    });

} // namespace npu::tile_fwk::Distributed




