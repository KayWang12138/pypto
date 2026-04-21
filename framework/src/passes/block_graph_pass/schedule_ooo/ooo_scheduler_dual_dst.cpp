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
 * \file ooo_scheduler_dual_dst.cpp
 * \brief DualDst feature detection implementation
 */

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"
#include "interface/configs/config_manager.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOScheduler.DualDst"
#endif

namespace npu::tile_fwk {

// Check if L0C tensor is suitable for dual_dst (must be 2D)
Status OoOScheduler::CheckL0CTensorForDualDst(LogicalTensorPtr l0cTensor)
{
    if (l0cTensor == nullptr) {
        return FAILED;
    }

    // Must be 2D tensor
    if (static_cast<int>(l0cTensor->GetShape().size()) != 2) {
        return FAILED;
    }

    // Check shape validity
    auto shape = l0cTensor->GetShape();
    if (shape[0] <= 0 || shape[1] <= 0) {
        return FAILED;
    }

    return SUCCESS;
}

// Check if two OP_L0C_COPY_UB ops can form a SplitM pair
Status OoOScheduler::CheckSplitMPair(Operation* op1, Operation* op2, LogicalTensorPtr l0cTensor)
{
    if (op1 == nullptr || op2 == nullptr || l0cTensor == nullptr) {
        return FAILED;
    }

    // Get output UB tensors
    auto ubTensor1 = op1->GetOOperands()[0];
    auto ubTensor2 = op2->GetOOperands()[0];

    if (ubTensor1 == nullptr || ubTensor2 == nullptr) {
        return FAILED;
    }

    // Check if output UB tile shapes are identical
    auto shape1 = ubTensor1->GetShape();
    auto shape2 = ubTensor2->GetShape();
    if (shape1 != shape2) {
        return FAILED;
    }

    // Check if validShapes are identical
    auto validShape1 = ubTensor1->GetDynValidShape();
    auto validShape2 = ubTensor2->GetDynValidShape();
    if (validShape1.size() != validShape2.size()) {
        return FAILED;
    }
    for (size_t i = 0; i < validShape1.size(); i++) {
        if (validShape1[i].Dump() != validShape2[i].Dump()) {
            return FAILED;
        }
    }

    // Get source offsets from CopyOpAttribute
    auto attr1 = op1->GetOpAttribute<CopyOpAttribute>();
    auto attr2 = op2->GetOpAttribute<CopyOpAttribute>();
    if (attr1 == nullptr || attr2 == nullptr) {
        return FAILED;
    }

    auto fromOffset1 = attr1->GetFromOffset();
    auto fromOffset2 = attr2->GetFromOffset();

    if (fromOffset1.size() < 2 || fromOffset2.size() < 2) {
        return FAILED;
    }

    int64_t offset1_M = fromOffset1[0].GetValue();
    int64_t offset1_N = fromOffset1[1].GetValue();
    int64_t offset2_M = fromOffset2[0].GetValue();
    int64_t offset2_N = fromOffset2[1].GetValue();

    // For SplitM: M axis must be continuous
    // op1 should have smaller M offset, op2 should have larger M offset
    // and offset2_M - offset1_M should equal shape[0] (the M dimension of the tile)
    int64_t tileM = shape1[0];

    if (offset1_M >= offset2_M) {
        return FAILED;  // op1 must have smaller M offset
    }

    if (offset2_M - offset1_M != tileM) {
        return FAILED;  // M offsets must be continuous
    }

    // N offsets must be the same
    if (offset1_N != offset2_N) {
        return FAILED;
    }

    // Check core assignment: op1 (lower M offset) should be on AIV0, op2 on AIV1
    auto core1 = opCoreLocationMap[op1];
    auto core2 = opCoreLocationMap[op2];

    if (core1 != CoreLocationType::AIV0 || core2 != CoreLocationType::AIV1) {
        return FAILED;
    }

    return SUCCESS;
}

// Check if two OP_L0C_COPY_UB ops can form a SplitN pair
Status OoOScheduler::CheckSplitNPair(Operation* op1, Operation* op2, LogicalTensorPtr l0cTensor)
{
    if (op1 == nullptr || op2 == nullptr || l0cTensor == nullptr) {
        return FAILED;
    }

    // Get output UB tensors
    auto ubTensor1 = op1->GetOOperands()[0];
    auto ubTensor2 = op2->GetOOperands()[0];

    if (ubTensor1 == nullptr || ubTensor2 == nullptr) {
        return FAILED;
    }

    // Check if output UB tile shapes are identical
    auto shape1 = ubTensor1->GetShape();
    auto shape2 = ubTensor2->GetShape();
    if (shape1 != shape2) {
        return FAILED;
    }

    // Check if validShapes are identical
    auto validShape1 = ubTensor1->GetDynValidShape();
    auto validShape2 = ubTensor2->GetDynValidShape();
    if (validShape1.size() != validShape2.size()) {
        return FAILED;
    }
    for (size_t i = 0; i < validShape1.size(); i++) {
        if (validShape1[i].Dump() != validShape2[i].Dump()) {
            return FAILED;
        }
    }

    // Get source offsets from CopyOpAttribute
    auto attr1 = op1->GetOpAttribute<CopyOpAttribute>();
    auto attr2 = op2->GetOpAttribute<CopyOpAttribute>();
    if (attr1 == nullptr || attr2 == nullptr) {
        return FAILED;
    }

    auto fromOffset1 = attr1->GetFromOffset();
    auto fromOffset2 = attr2->GetFromOffset();

    if (fromOffset1.size() < 2 || fromOffset2.size() < 2) {
        return FAILED;
    }

    int64_t offset1_M = fromOffset1[0].GetValue();
    int64_t offset1_N = fromOffset1[1].GetValue();
    int64_t offset2_M = fromOffset2[0].GetValue();
    int64_t offset2_N = fromOffset2[1].GetValue();

    // For SplitN: N axis must be continuous
    // op1 should have smaller N offset, op2 should have larger N offset
    // and offset2_N - offset1_N should equal shape[1] (the N dimension of the tile)
    int64_t tileN = shape1[1];

    if (offset1_N >= offset2_N) {
        return FAILED;  // op1 must have smaller N offset
    }

    if (offset2_N - offset1_N != tileN) {
        return FAILED;  // N offsets must be continuous
    }

    // M offsets must be the same
    if (offset1_M != offset2_M) {
        return FAILED;
    }

    // Check core assignment: op1 (lower N offset) should be on AIV0, op2 on AIV1
    auto core1 = opCoreLocationMap[op1];
    auto core2 = opCoreLocationMap[op2];

    if (core1 != CoreLocationType::AIV0 || core2 != CoreLocationType::AIV1) {
        return FAILED;
    }

    return SUCCESS;
}

// Main detection function: iterate over all L0C tensors and find dual_dst pairs
Status OoOScheduler::DetectDualDstMode(const std::vector<Operation*>& opList)
{
    // Check if dual_dst feature is enabled
    enableDualDst_ = config::GetPlatformConfig(KEY_ENABLE_DUAL_DST, false);

    if (!enableDualDst_) {
        APASS_LOG_DEBUG_F(Elements::Operation, "DualDst feature is disabled, skip detection.");
        return SUCCESS;
    }

    APASS_LOG_INFO_F(Elements::Operation, "=============== START DetectDualDstMode ===============");

    // Collect all L0C tensors and their L0C_COPY_UB consumers
    std::unordered_map<LogicalTensorPtr, std::vector<Operation*>> l0cToConsumers;

    for (auto* op : opList) {
        if (op->GetOpcode() != Opcode::OP_L0C_COPY_UB) {
            continue;
        }

        // Skip if not single consumer per tensor (for now)
        auto* l0cTensor = op->GetInputOperand(0);
        if (l0cTensor == nullptr) {
            continue;
        }

        l0cToConsumers[l0cTensor].push_back(op);
    }

    // Process each L0C tensor
    std::unordered_set<LogicalTensorPtr> processedTensors;
    std::unordered_set<Operation*> usedOps;

    for (auto& entry : l0cToConsumers) {
        auto* l0cTensor = entry.first;
        auto& consumers = entry.second;

        // Skip if already processed
        if (processedTensors.count(l0cTensor) > 0) {
            continue;
        }
        processedTensors.insert(l0cTensor);

        // Check if tensor is suitable for dual_dst
        if (CheckL0CTensorForDualDst(l0cTensor) != SUCCESS) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                "L0C tensor %p is not suitable for dual_dst (not 2D or invalid shape)", l0cTensor);
            continue;
        }

        // Skip if not exactly 2 consumers (we need pairs)
        if (consumers.size() != 2) {
            APASS_LOG_DEBUG_F(Elements::Operation,
                "L0C tensor %p has %zu consumers (expected 2 for dual_dst)", l0cTensor, consumers.size());
            continue;
        }

        // Skip if either op is already used
        if (usedOps.count(consumers[0]) > 0 || usedOps.count(consumers[1]) > 0) {
            continue;
        }

        // Try SplitM first
        DualDstPair pair;
        pair.l0cTensor = l0cTensor;

        // Determine which op has lower offset (should be on AIV0)
        Operation *op1 = consumers[0];
        Operation *op2 = consumers[1];

        auto attr1 = op1->GetOpAttribute<CopyOpAttribute>();
        auto attr2 = op2->GetOpAttribute<CopyOpAttribute>();

        if (attr1 != nullptr && attr2 != nullptr) {
            auto fromOffset1 = attr1->GetFromOffset();
            auto fromOffset2 = attr2->GetFromOffset();

            if (fromOffset1.size() >= 2 && fromOffset2.size() >= 2) {
                // Sort by M offset primarily, N offset secondarily
                int64_t m1 = fromOffset1[0].GetValue();
                int64_t m2 = fromOffset2[0].GetValue();
                int64_t n1 = fromOffset1[1].GetValue();
                int64_t n2 = fromOffset2[1].GetValue();

                if (m1 > m2 || (m1 == m2 && n1 > n2)) {
                    std::swap(op1, op2);
                }
            }
        }

        pair.copyUbOp1 = op1;
        pair.copyUbOp2 = op2;

        // Try SplitM
        if (CheckSplitMPair(op1, op2, l0cTensor) == SUCCESS) {
            pair.mode = DualDstMode::SPLIT_M;
            dualDstPairs_.push_back(pair);
            usedOps.insert(op1);
            usedOps.insert(op2);
            dualDstOps_[op1] = &dualDstPairs_.back();
            dualDstOps_[op2] = &dualDstPairs_.back();
            APASS_LOG_INFO_F(Elements::Operation,
                "Found SplitM dual_dst pair: op1=%p (AIV0), op2=%p (AIV1), l0cTensor=%p",
                op1, op2, l0cTensor);
            continue;
        }

        // Try SplitN
        if (CheckSplitNPair(op1, op2, l0cTensor) == SUCCESS) {
            pair.mode = DualDstMode::SPLIT_N;
            dualDstPairs_.push_back(pair);
            usedOps.insert(op1);
            usedOps.insert(op2);
            dualDstOps_[op1] = &dualDstPairs_.back();
            dualDstOps_[op2] = &dualDstPairs_.back();
            APASS_LOG_INFO_F(Elements::Operation,
                "Found SplitN dual_dst pair: op1=%p (AIV0), op2=%p (AIV1), l0cTensor=%p",
                op1, op2, l0cTensor);
            continue;
        }

        APASS_LOG_DEBUG_F(Elements::Operation,
            "No valid dual_dst pair found for L0C tensor %p", l0cTensor);
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "Detected %zu dual_dst pairs (%d SplitM, %d SplitN)",
        dualDstPairs_.size(),
        std::count_if(dualDstPairs_.begin(), dualDstPairs_.end(),
            [](const DualDstPair& p) { return p.mode == DualDstMode::SPLIT_M; }),
        std::count_if(dualDstPairs_.begin(), dualDstPairs_.end(),
            [](const DualDstPair& p) { return p.mode == DualDstMode::SPLIT_N; }));

    APASS_LOG_INFO_F(Elements::Operation, "=============== END DetectDualDstMode ===============");
    return SUCCESS;
}

} // namespace npu::tile_fwk
