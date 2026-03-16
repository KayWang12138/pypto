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
 * \file shmem_ir_test_utils.h
 * \brief SHMEM IR verification helpers shared between UT and ST.
 */

#ifndef SHMEM_IR_TEST_UTILS_H
#define SHMEM_IR_TEST_UTILS_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "interface/program/program.h"

namespace npu::tile_fwk {

// FUNCTION macro naming suffixes.

const std::string SUB_FUNC_SUFFIX = "_Unroll1_PATH0";
const std::string HIDDEN_FUNC_SUFFIX = "_hiddenfunc0";

struct ShmemOpCounts {
    uint32_t put = 0;
    uint32_t signal = 0;
    uint32_t wait = 0;
    uint32_t get = 0;
};

inline bool IsShmemOpcode(Opcode code)
{
    return code == Opcode::OP_SHMEM_PUT ||
           code == Opcode::OP_SHMEM_SIGNAL ||
           code == Opcode::OP_SHMEM_WAIT_UNTIL ||
           code == Opcode::OP_SHMEM_GET;
}

inline ShmemOpCounts CountShmemOps(const std::vector<Opcode>& ops)
{
    ShmemOpCounts c;
    for (Opcode code : ops) {
        if (code == Opcode::OP_SHMEM_PUT)               c.put++;
        else if (code == Opcode::OP_SHMEM_SIGNAL)        c.signal++;
        else if (code == Opcode::OP_SHMEM_WAIT_UNTIL)    c.wait++;
        else if (code == Opcode::OP_SHMEM_GET)           c.get++;
    }
    return c;
}

// Build the raw name the compiler gives to LOOP sub-functions:
// TENSOR_{name}_Unroll1_PATH0[_hiddenfunc0]
inline std::string GetFunctionRawName(const std::string& funcName)
{
    std::string rawName = FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX;
#if ENABLE_HIDDENLOOP
    rawName += HIDDEN_FUNC_SUFFIX;
#endif
    return rawName;
}

// Debug helpers — call DumpAllIR() to print the full IR graph.
inline std::string OpcodeName(Opcode code)
{
    if (code == Opcode::OP_SHMEM_PUT)            return "OP_SHMEM_PUT";
    if (code == Opcode::OP_SHMEM_SIGNAL)         return "OP_SHMEM_SIGNAL";
    if (code == Opcode::OP_SHMEM_WAIT_UNTIL)     return "OP_SHMEM_WAIT_UNTIL";
    if (code == Opcode::OP_SHMEM_GET)            return "OP_SHMEM_GET";
    std::string n = pto::GetOpcodeName(static_cast<pto::Opcode>(code));
    if (!n.empty()) return n;
    return "OPCODE(" + std::to_string(static_cast<int>(code)) + ")";
}

inline void DumpAllIR(const std::string& label = "")
{
    if (!label.empty()) std::cout << "\n===== " << label << " =====\n";
    for (const auto& [name, funcPtr] : Program::GetInstance().GetFunctionMap()) {
        auto ops = funcPtr->Operations();
        if (ops.IsEmpty()) continue;
        bool hasShmem = false;
        for (auto& op : ops) {
            if (IsShmemOpcode(op.GetOpcode())) { hasShmem = true; break; }
        }
        std::cout << "-- " << name
                  << (hasShmem ? " [SHMEM]" : "") << " --\n";
        for (auto& op : ops) {
            Opcode code = op.GetOpcode();
            std::cout << "  " << OpcodeName(code);
            if (IsShmemOpcode(code)) std::cout << "  <===";
            std::cout << "\n";
        }
    }
    std::cout << std::flush;
}

// Extract SHMEM opcodes for funcName from the IR function map.
// Prefers the "hiddenfunc" variant to avoid double-counting.
inline std::vector<Opcode> ExtractShmemOpcodes(const std::string& funcName)
{
    std::vector<Opcode> hiddenOps;
    std::vector<Opcode> fallback;

    for (const auto& [name, funcPtr] : Program::GetInstance().GetFunctionMap()) {
        if (name.find(funcName) == std::string::npos) continue;

        bool isHidden = (name.find("hiddenfunc") != std::string::npos) &&
                        (name.find("leaf") == std::string::npos) &&
                        (name.find("root") == std::string::npos);
        for (auto& op : funcPtr->Operations()) {
            Opcode code = op.GetOpcode();
            if (IsShmemOpcode(code)) {
                if (isHidden) {
                    hiddenOps.push_back(code);
                } else {
                    fallback.push_back(code);
                }
            }
        }
    }
    if (!hiddenOps.empty()) return hiddenOps;
    EXPECT_FALSE(fallback.empty())
        << "No SHMEM operations found in any function containing: " << funcName;
    return fallback;
}

// Check that all four SHMEM op types are present (count-agnostic).
inline void VerifyShmemOpsPresent(const ShmemOpCounts& c)
{
    EXPECT_GT(c.put, 0u) << "Expected at least 1 OP_SHMEM_PUT";
    EXPECT_GT(c.signal, 0u) << "Expected at least 1 OP_SHMEM_SIGNAL";
    EXPECT_GT(c.wait, 0u) << "Expected at least 1 OP_SHMEM_WAIT_UNTIL";
    EXPECT_GT(c.get, 0u) << "Expected at least 1 OP_SHMEM_GET";
}

// OneShot: W puts, W signals, 1 wait, 1 get.
inline void VerifyOneShotCounts(const ShmemOpCounts& c, uint32_t worldSize)
{
    EXPECT_EQ(c.put, worldSize) << "Expected " << worldSize << " OP_SHMEM_PUT ops";
    EXPECT_EQ(c.signal, worldSize) << "Expected " << worldSize << " OP_SHMEM_SIGNAL ops";
    EXPECT_EQ(c.wait, 1u) << "Expected 1 OP_SHMEM_WAIT_UNTIL op";
    EXPECT_EQ(c.get, 1u) << "Expected 1 OP_SHMEM_GET op";
}

// OneShot grouped (v7): W*C puts/signals, ceil(C/k) waits, C gets.
inline void VerifyOneShotGroupedCounts(const ShmemOpCounts& c,
    uint32_t worldSize, uint32_t payloadChunkCount, uint32_t chunksPerSignal)
{
    ASSERT_GT(payloadChunkCount, 0u);
    ASSERT_GT(chunksPerSignal, 0u);
    uint32_t groupCount = (payloadChunkCount + chunksPerSignal - 1u) / chunksPerSignal;
    uint32_t expectedScatter = worldSize * payloadChunkCount;
    EXPECT_EQ(c.put, expectedScatter) << "Expected " << expectedScatter << " OP_SHMEM_PUT ops";
    EXPECT_EQ(c.signal, expectedScatter) << "Expected " << expectedScatter << " OP_SHMEM_SIGNAL ops";
    EXPECT_EQ(c.wait, groupCount) << "Expected " << groupCount << " OP_SHMEM_WAIT_UNTIL ops";
    EXPECT_EQ(c.get, payloadChunkCount) << "Expected " << payloadChunkCount << " OP_SHMEM_GET ops";
}

// TwoShot: W^2 of each op type (W chunks x W ops per chunk).
inline void VerifyTwoShotCounts(const ShmemOpCounts& c, uint32_t worldSize)
{
    uint32_t expected = worldSize * worldSize;
    EXPECT_EQ(c.put, expected) << "Expected " << expected << " OP_SHMEM_PUT ops";
    EXPECT_EQ(c.signal, expected) << "Expected " << expected << " OP_SHMEM_SIGNAL ops";
    EXPECT_EQ(c.wait, expected) << "Expected " << expected << " OP_SHMEM_WAIT_UNTIL ops";
    EXPECT_EQ(c.get, expected) << "Expected " << expected << " OP_SHMEM_GET ops";
}

// OneShot ordering: PUT*N, SIGNAL*N, WAIT, GET — phases must not go backwards.
inline void VerifyOneShotOrdering(const std::vector<Opcode>& ops)
{
    enum Phase { PHASE_PUT, PHASE_SIGNAL, PHASE_WAIT, PHASE_GET };
    Phase current = PHASE_PUT;

    for (size_t i = 0; i < ops.size(); ++i) {
        Phase opPhase = current;
        if (ops[i] == Opcode::OP_SHMEM_PUT)               opPhase = PHASE_PUT;
        else if (ops[i] == Opcode::OP_SHMEM_SIGNAL)        opPhase = PHASE_SIGNAL;
        else if (ops[i] == Opcode::OP_SHMEM_WAIT_UNTIL)    opPhase = PHASE_WAIT;
        else if (ops[i] == Opcode::OP_SHMEM_GET)           opPhase = PHASE_GET;
        else continue;

        EXPECT_GE(static_cast<int>(opPhase), static_cast<int>(current))
            << "OneShot ordering violation at index " << i
            << ": expected phase >= " << current << ", got " << opPhase;
        current = opPhase;
    }
}

// TwoShot ordering: PUT*N^2, SIGNAL*N^2, WAIT*N^2, GET*N^2 in strict blocks.
inline void VerifyTwoShotOrdering(const std::vector<Opcode>& ops, uint32_t worldSize)
{
    uint32_t n = worldSize * worldSize;
    uint32_t totalExpected = n * 4u;
    ASSERT_EQ(static_cast<uint32_t>(ops.size()), totalExpected)
        << "Expected " << totalExpected << " SHMEM ops for TwoShot, got " << ops.size();

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ops[i], Opcode::OP_SHMEM_PUT)
            << "Expected PUT at position " << i;
    }
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ops[n + i], Opcode::OP_SHMEM_SIGNAL)
            << "Expected SIGNAL at position " << (n + i);
    }
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ops[2u * n + i], Opcode::OP_SHMEM_WAIT_UNTIL)
            << "Expected WAIT_UNTIL at position " << (2u * n + i);
    }
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ops[3u * n + i], Opcode::OP_SHMEM_GET)
            << "Expected GET at position " << (3u * n + i);
    }
}

// Extract + count in one call. No ordering check — compiler
// may reorder ops across hidden functions.
inline void VerifyOneShotAllReduceIR(const std::string& funcName, uint32_t worldSize)
{
    auto ops = ExtractShmemOpcodes(funcName);
    VerifyOneShotCounts(CountShmemOps(ops), worldSize);
}

inline void VerifyTwoShotAllReduceIR(const std::string& funcName, uint32_t worldSize)
{
    auto ops = ExtractShmemOpcodes(funcName);
    VerifyTwoShotCounts(CountShmemOps(ops), worldSize);
}

} // namespace npu::tile_fwk

#endif // SHMEM_IR_TEST_UTILS_H
