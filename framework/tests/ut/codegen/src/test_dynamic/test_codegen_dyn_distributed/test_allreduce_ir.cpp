/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 *        Validates that OneShot AllReduce variants (v2, v3, v4, v5+Pull)
 *        produce the expected SHMEM operation counts and identical opcode
 *        sequences — without requiring hardware or HCCL runtime.
 */

#include <vector>
#include <string>
#include <algorithm>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/distributed_communicator.h"

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

    // Extract SHMEM opcodes from a specific FUNCTION block.
    // Searches Program's function map for entries whose raw name starts with
    // "TENSOR_<funcName>" and contains SHMEM operations.  When ENABLE_HIDDENLOOP
    // is active the same ops appear in both "path" and "hidden" sub-functions;
    // we prefer the "hiddenfunc" version to avoid double-counting.
    std::vector<Opcode> ExtractShmemOpcodes(const std::string& funcName)
    {
        const std::string prefix = "TENSOR_" + funcName;
        std::vector<Opcode> fallback;

        for (const auto& [name, funcPtr] : Program::GetInstance().GetFunctionMap()) {
            if (name.rfind(prefix, 0) != 0) continue;

            std::vector<Opcode> ops;
            for (auto& op : funcPtr->Operations()) {
                Opcode code = op.GetOpcode();
                if (code == Opcode::OP_SHMEM_PUT ||
                    code == Opcode::OP_SHMEM_SIGNAL ||
                    code == Opcode::OP_SHMEM_WAIT_UNTIL ||
                    code == Opcode::OP_SHMEM_GET) {
                    ops.push_back(code);
                }
            }
            if (!ops.empty()) {
                if (name.find("hiddenfunc") != std::string::npos) {
                    return ops;
                }
                if (fallback.empty()) {
                    fallback = std::move(ops);
                }
            }
        }
        EXPECT_FALSE(fallback.empty())
            << "No SHMEM operations found in any function matching prefix: " << prefix;
        return fallback;
    }

    void VerifyOneShotCounts(const std::vector<Opcode>& ops)
    {
        uint32_t put = 0, signal = 0, wait = 0, get = 0;
        for (Opcode c : ops) {
            if (c == Opcode::OP_SHMEM_PUT) put++;
            else if (c == Opcode::OP_SHMEM_SIGNAL) signal++;
            else if (c == Opcode::OP_SHMEM_WAIT_UNTIL) wait++;
            else if (c == Opcode::OP_SHMEM_GET) get++;
        }
        EXPECT_EQ(put, kWorldSize) << "Expected " << kWorldSize << " PUT ops";
        EXPECT_EQ(signal, kWorldSize) << "Expected " << kWorldSize << " SIGNAL ops";
        EXPECT_EQ(wait, 1u) << "Expected 1 WAIT_UNTIL op";
        EXPECT_EQ(get, 1u) << "Expected 1 GET op";
    }
};

// ---------------------------------------------------------------------------
// Individual variant IR structure tests
// ---------------------------------------------------------------------------

TEST_F(AllReduceIRTest, V2_IRStructure)
{
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V2", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        Tensor shmemData, shmemSignal;
        CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
        OneShotAllReduce_v2(in, in, kGroup, shmemData, shmemSignal, out);
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V2");
    VerifyOneShotCounts(ops);
}

TEST_F(AllReduceIRTest, V3_IRStructure)
{
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V3", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        Tensor shmemData, shmemSignal;
        CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
        OneShotAllReduce_v3(in, in, kGroup, shmemData, shmemSignal, out);
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V3");
    VerifyOneShotCounts(ops);
}

TEST_F(AllReduceIRTest, V4_IRStructure)
{
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V4", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        Tensor shmemData, shmemSignal;
        CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
        OneShotAllReduce_v4(in, in, kGroup, shmemData, shmemSignal, out);
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V4");
    VerifyOneShotCounts(ops);
}

TEST_F(AllReduceIRTest, V5PlusPull_IRStructure)
{
    Tensor in(DT_FP16, {kRow, kCol}, "in");
    Tensor out(DT_FP16, {kRow, kCol}, "out");
    Shape shmemDataShape{1, kRow, kCol};

    FUNCTION("UT_IR_V5", {in}, {out}) {
        TileShape::Current().SetVecTile({kRow, kCol});
        Tensor shmemData, shmemSignal;
        CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
        CommunicatorV2 comm(kGroup, kWorldSize, shmemSignal);
        auto waitToken = OneShotAllReduce_v5(in, in, shmemData, comm);
        out = comm.Pull(waitToken, shmemData);
    }

    auto ops = ExtractShmemOpcodes("UT_IR_V5");
    VerifyOneShotCounts(ops);
}

// ---------------------------------------------------------------------------
// Cross-variant IR equivalence test
// Each variant uses a unique FUNCTION name, so GetFunctionByRawName()
// finds the correct leaf function without cross-contamination.
// ---------------------------------------------------------------------------

TEST_F(AllReduceIRTest, AllVariants_ShmemOpcodeEquivalence)
{
    Shape shmemDataShape{1, kRow, kCol};

    // ── v2 ──
    std::vector<Opcode> irV2;
    {
        Tensor in(DT_FP16, {kRow, kCol}, "in_v2");
        Tensor out(DT_FP16, {kRow, kCol}, "out_v2");
        FUNCTION("UT_EQUIV_V2", {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            OneShotAllReduce_v2(in, in, kGroup, shmemData, shmemSignal, out);
        }
        irV2 = ExtractShmemOpcodes("UT_EQUIV_V2");
    }

    // ── v3 ──
    std::vector<Opcode> irV3;
    {
        Program::GetInstance().Reset();
        Tensor in(DT_FP16, {kRow, kCol}, "in_v3");
        Tensor out(DT_FP16, {kRow, kCol}, "out_v3");
        FUNCTION("UT_EQUIV_V3", {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            OneShotAllReduce_v3(in, in, kGroup, shmemData, shmemSignal, out);
        }
        irV3 = ExtractShmemOpcodes("UT_EQUIV_V3");
    }

    // ── v4 ──
    std::vector<Opcode> irV4;
    {
        Program::GetInstance().Reset();
        Tensor in(DT_FP16, {kRow, kCol}, "in_v4");
        Tensor out(DT_FP16, {kRow, kCol}, "out_v4");
        FUNCTION("UT_EQUIV_V4", {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            OneShotAllReduce_v4(in, in, kGroup, shmemData, shmemSignal, out);
        }
        irV4 = ExtractShmemOpcodes("UT_EQUIV_V4");
    }

    // ── v5 + Pull ──
    std::vector<Opcode> irV5;
    {
        Program::GetInstance().Reset();
        Tensor in(DT_FP16, {kRow, kCol}, "in_v5");
        Tensor out(DT_FP16, {kRow, kCol}, "out_v5");
        FUNCTION("UT_EQUIV_V5", {in}, {out}) {
            TileShape::Current().SetVecTile({kRow, kCol});
            Tensor shmemData, shmemSignal;
            CreateShmemTensors(PromotedType(in.GetDataType()), shmemDataShape, shmemData, shmemSignal);
            CommunicatorV2 comm(kGroup, kWorldSize, shmemSignal);
            auto waitToken = OneShotAllReduce_v5(in, in, shmemData, comm);
            out = comm.Pull(waitToken, shmemData);
        }
        irV5 = ExtractShmemOpcodes("UT_EQUIV_V5");
    }

    // ── Cross-variant equivalence ──
    EXPECT_EQ(irV2, irV3) << "v2 and v3 SHMEM opcode sequences differ";
    EXPECT_EQ(irV3, irV4) << "v3 and v4 SHMEM opcode sequences differ";
    EXPECT_EQ(irV4, irV5) << "v4 and v5+Pull SHMEM opcode sequences differ";
}

// ---------------------------------------------------------------------------
// Total operation count sanity check
// ---------------------------------------------------------------------------

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
