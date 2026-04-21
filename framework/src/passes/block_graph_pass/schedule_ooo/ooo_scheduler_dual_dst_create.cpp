/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ooo_scheduler_dual_dst_create.cpp
 * \brief DualDst feature graph transformation implementation
 */

#include "ooo_scheduler.h"
#include "passes/pass_log/pass_log.h"
#include "interface/configs/config_manager.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOScheduler.DualDst.Create"
#endif

namespace npu::tile_fwk {

// Set CopyOpAttribute for L0C2UBDualDst operation
Status OoOScheduler::SetL0C2UBDualDstCopyAttr(
    Operation &op,
    const std::vector<SymbolicScalar> &validShape,
    const std::vector<std::vector<OpImmediate>> &fromOffsets,
    const std::vector<std::vector<OpImmediate>> &toOffsets,
    const std::vector<std::vector<OpImmediate>> &srcValidShapes,
    const std::vector<std::vector<OpImmediate>> &dstValidShapes,
    bool isSplitM)
{
    if (fromOffsets.size() < 2 || toOffsets.size() < 2) {
        APASS_LOG_ERROR_F(Elements::Operation, "Invalid offsets size for dual_dst op");
        return FAILED;
    }

    // Create CopyOpAttribute for dual_dst
    // fromOffset: source L0C tensor offsets for both outputs
    // toOffset: destination UB tensor offsets
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        fromOffsets[0],  // fromOffset for first output
        MemoryType::MEM_UB,
        OpImmediate::Specified(validShape),
        OpImmediate::Specified(op.iOperand.front()->tensor->GetDynRawShape()),
        OpImmediate::Specified(validShape)
    );

    // Set toOffset for first output
    copyAttr->SetToOffset(toOffsets[0]);

    // Set dual_dst specific attributes
    copyAttr->SetDualDstEnabled(true);
    copyAttr->SetDualDstSplitM(isSplitM);

    // Set second output parameters
    if (fromOffsets.size() >= 2) {
        copyAttr->SetFromOffset1(fromOffsets[1]);
        copyAttr->SetToOffset1(toOffsets[1]);
        copyAttr->SetSrcValidShape1(srcValidShapes[1]);
        copyAttr->SetDstValidShape1(dstValidShapes[1]);
    }

    op.SetOpAttribute(copyAttr);
    op.SetAttribute(OpAttributeKey::isCube, true);

    // Mark the alloc op with dual_dst tag
    auto* l0cTensor = op.GetInputOperand(0);
    if (l0cTensor != nullptr) {
        for (auto* producer : l0cTensor->GetProducers()) {
            if (producer->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                producer->SetAttribute("is_dual_dst", true);
                APASS_LOG_DEBUG_F(Elements::Operation, "Marked ALLOC op %p as dual_dst", producer);
            }
        }
    }

    return SUCCESS;
}

// Create OP_L0C_COPY_UB_DUAL_DST nodes and update the graph
Status OoOScheduler::CreateDualDstOpAndGraphUpdate()
{
    if (!enableDualDst_ || dualDstPairs_.empty()) {
        APASS_LOG_DEBUG_F(Elements::Operation, "DualDst feature disabled or no pairs found, skip graph update.");
        return SUCCESS;
    }

    APASS_LOG_INFO_F(Elements::Operation, "=============== START CreateDualDstOpAndGraphUpdate ===============");

    // Process each dual_dst pair
    for (auto& pair : dualDstPairs_) {
        if (pair.copyUbOp1 == nullptr || pair.copyUbOp2 == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid dual_dst pair: op1=%p, op2=%p",
                pair.copyUbOp1, pair.copyUbOp2);
            continue;
        }

        // Verify both ops have the same input L0C tensor
        auto* l0cTensor1 = pair.copyUbOp1->GetInputOperand(0);
        auto* l0cTensor2 = pair.copyUbOp2->GetInputOperand(0);

        if (l0cTensor1 != l0cTensor2) {
            APASS_LOG_ERROR_F(Elements::Operation,
                "dual_dst ops have different L0C inputs: %p vs %p", l0cTensor1, l0cTensor2);
            continue;
        }

        // Get output UB tensors
        auto* ubTensor1 = pair.copyUbOp1->GetOOperands()[0];
        auto* ubTensor2 = pair.copyUbOp2->GetOOperands()[0];

        if (ubTensor1 == nullptr || ubTensor2 == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Invalid output tensors for dual_dst pair");
            continue;
        }

        // Get attributes from original ops
        auto attr1 = pair.copyUbOp1->GetOpAttribute<CopyOpAttribute>();
        auto attr2 = pair.copyUbOp2->GetOpAttribute<CopyOpAttribute>();

        if (attr1 == nullptr || attr2 == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Missing CopyOpAttribute for dual_dst ops");
            continue;
        }

        // Prepare parameters for the new dual_dst op
        auto validShape = ubTensor1->GetDynValidShape();
        auto fromOffsets = std::vector<std::vector<OpImmediate>>{
            attr1->GetFromOffset(),
            attr2->GetFromOffset()
        };
        auto toOffsets = std::vector<std::vector<OpImmediate>>{
            attr1->GetToOffset(),
            attr2->GetToOffset()
        };

        // Get source and destination valid shapes
        std::vector<std::vector<OpImmediate>> srcValidShapes;
        std::vector<std::vector<OpImmediate>> dstValidShapes;

        for (auto* ubTensor : {ubTensor1, ubTensor2}) {
            std::vector<OpImmediate> srcShape;
            std::vector<OpImmediate> dstShape;
            for (auto& dim : ubTensor->GetDynValidShape()) {
                srcShape.push_back(OpImmediate::Specified(dim.GetValue()));
                dstShape.push_back(OpImmediate::Specified(dim.GetValue()));
            }
            srcValidShapes.push_back(srcShape);
            dstValidShapes.push_back(dstShape);
        }

        bool isSplitM = (pair.mode == DualDstMode::SPLIT_M);

        // Create new OP_L0C_COPY_UB_DUAL_DST operation
        // The new op has 1 input (L0C tensor) and 2 outputs (UB tensors)
        Operation* dualDstOp = new Operation();
        dualDstOp->SetOpcode(Opcode::OP_L0C_COPY_UB_DUAL_DST);

        // Set inputs (single L0C tensor)
        dualDstOp->AddIOperand(l0cTensor1);

        // Set outputs (both UB tensors)
        dualDstOp->AddOOperand(ubTensor1);
        dualDstOp->AddOOperand(ubTensor2);

        // Set the CopyOpAttribute with dual_dst parameters
        if (SetL0C2UBDualDstCopyAttr(*dualDstOp,
            {SymbolicScalar(validShape[0].GetValue()), SymbolicScalar(validShape[1].GetValue())},
            fromOffsets, toOffsets, srcValidShapes, dstValidShapes, isSplitM) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to set CopyOpAttribute for dual_dst op");
            delete dualDstOp;
            continue;
        }

        // Copy other attributes from original ops
        dualDstOp->SetAttribute(OpAttributeKey::isCube, true);
        dualDstOp->SetAttribute("dual_dst_split_m", isSplitM ? 1 : 0);
        dualDstOp->SetAttribute("dual_dst_enabled", 1);

        // Update tensor producers/consumers
        // L0C tensor: remove old consumers, add new one
        l0cTensor1->RemoveConsumer(pair.copyUbOp1);
        l0cTensor1->RemoveConsumer(pair.copyUbOp2);
        l0cTensor1->AddConsumer(dualDstOp);

        // UB tensors: update their producers
        ubTensor1->RemoveProducer(pair.copyUbOp1);
        ubTensor1->AddProducer(dualDstOp);
        ubTensor2->RemoveProducer(pair.copyUbOp2);
        ubTensor2->AddProducer(dualDstOp);

        // Update dependencies for consumers of the old ops
        // Consumers of op1's output now depend on dualDstOp
        // Consumers of op2's output now depend on dualDstOp
        for (auto* consumer : ubTensor1->GetConsumers()) {
            for (size_t i = 0; i < consumer->GetIOperands().size(); i++) {
                if (consumer->GetInputOperand(i) == ubTensor1) {
                    // Already correct, dualDstOp is now the producer
                }
            }
        }

        // Store the new op for insertion
        newOperations_.push_back(dualDstOp);

        // Mark old ops for removal (they'll be removed from opList)
        // Note: We don't delete them immediately, just remove from the schedule
        pair.copyUbOp1->SetAttribute("dual_dst_replaced", true);
        pair.copyUbOp2->SetAttribute("dual_dst_replaced", true);

        APASS_LOG_INFO_F(Elements::Operation,
            "Created dual_dst op %p (mode=%s) replacing op1=%p and op2=%p",
            dualDstOp, isSplitM ? "SplitM" : "SplitN", pair.copyUbOp1, pair.copyUbOp2);
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "Created %zu dual_dst operations", newOperations_.size());
    APASS_LOG_INFO_F(Elements::Operation, "=============== END CreateDualDstOpAndGraphUpdate ===============");
    return SUCCESS;
}

} // namespace npu::tile_fwk
