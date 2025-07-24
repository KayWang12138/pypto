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
 * \file insert_convert_op.cpp
 * \brief
 */

#include "insert_convert_op.h"
#include "passes/pass_utils/parallel_tool.h"
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {
Status InsertConvertOp::PreCheck(Function &function) {
    Status baseStatus = Pass::PreCheck(function);
    if (baseStatus != SUCCESS) {
        return baseStatus;
    }
    auto operations = function.Operations();

    // check convert path of tensor does not include L1 to L0A/L0B
    for (auto &operation : operations) {
        for (auto oTensor : operation.GetOOperands()) {
            auto originalType = oTensor->GetMemoryTypeOriginal();
            auto tobeType = oTensor->GetMemoryTypeToBe();
            if (tobeType == MemoryType::MEM_L1){
                assert(originalType != MemoryType::MEM_L0A && originalType != MemoryType::MEM_L0B);
            }
            if (originalType == MemoryType::MEM_L1){
                assert(tobeType != MemoryType::MEM_L0A && tobeType != MemoryType::MEM_L0B);
            }
        }
    }

    ALOG_INFO_F("InsertConvertOp PreCheck completed successfully!");
    return SUCCESS;
}

Status InsertConvertOp::PostCheck(Function &function) {
    Status baseStatus = Pass::PostCheck(function);
    if (baseStatus != SUCCESS) {
        return baseStatus;
    }
    auto operations = function.Operations();

    // Check iOperand and oOperand of OP_CONVERT
    for (auto &operation : operations) {
        if (operation.GetOpcode() == Opcode::OP_CONVERT) {
            auto iOperand = operation.GetIOperands();
            auto oOperand = operation.GetOOperands();
            assert(iOperand.size() == 1);
            assert(oOperand.size() == 1);
            assert(iOperand[0]->GetMemoryTypeOriginal() != oOperand[0]->GetMemoryTypeToBe());
            assert(iOperand[0]->GetShape() == oOperand[0]->GetShape());
            assert(iOperand[0]->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR || oOperand[0]->GetMemoryTypeToBe() == MemoryType::MEM_DEVICE_DDR);
        }
    }
    return SUCCESS;
}

void InsertConvertOp::CheckUnknown(Function &function) const {
    auto opList = function.Operations();
    ParallelTool::Instance().Parallel_for(0, opList.size(),1,[&](int st,int et,int tid) {
        (void) tid;
        for(int opIdx=st; opIdx<et; opIdx++){
            auto& op = opList[opIdx];
            std::unordered_set<MemoryType> supportedMemType = {
                MemoryType::MEM_UB,
                MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C,
                MemoryType::MEM_L2, MemoryType::MEM_L3,
                MemoryType::MEM_DEVICE_DDR,
                MemoryType::MEM_HOST1, MemoryType::MEM_FAR1, MemoryType::MEM_FAR2
            };
            switch (op.GetCoreType()) {
                case CoreType::AIC:
                    supportedMemType = {MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C,
                        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT, MemoryType::MEM_FIX, MemoryType::MEM_FIX_QUANT_PRE,
                        MemoryType::MEM_FIX_RELU_PRE, MemoryType::MEM_FIX_RELU_POST, MemoryType::MEM_FIX_QUANT_POST,
                        MemoryType::MEM_FIX_ELT_ANTIQ, MemoryType::MEM_FIX_MTE2_ANTIQ};
                    break;
                case CoreType::AIV:
                    supportedMemType = {MemoryType::MEM_UB};
                    break;
                case CoreType::GMATOMIC:
                    supportedMemType = {MemoryType::MEM_DEVICE_DDR};
                    break;
                default:
                    break;
            }
            for (auto &i : op.GetIOperands()) {
                ASSERT(supportedMemType.count(i->GetMemoryTypeToBe()) != 0) <<
                    "Op " + std::to_string(op.opmagic) + " has unsupported mem type " + MemoryTypeToString(i->GetMemoryTypeToBe());
            }
            for (auto &o : op.GetOOperands()) {
                ASSERT(supportedMemType.count(o->GetMemoryTypeToBe()) != 0) <<
                    "Op " + std::to_string(op.opmagic) + " has unsupported mem type " + MemoryTypeToString(o->GetMemoryTypeToBe());
                ASSERT(o->GetMemoryTypeOriginal() == o->GetMemoryTypeToBe()) <<
                    "Op " + std::to_string(op.opmagic) + " has two mem type " <<
                    MemoryTypeToString(o->GetMemoryTypeOriginal()) << " and " << MemoryTypeToString(o->GetMemoryTypeToBe());
            }
        }
    });
}

bool InsertConvertOp::CrossCore(const std::shared_ptr<LogicalTensor>& tensor) const {
    std::vector<MemoryType> paths;
    PassConfigManager::Instance().GetPlatformConfig().FindNearestPath(
        tensor->GetMemoryTypeOriginal(), tensor->GetMemoryTypeToBe(), paths);

    return std::find(paths.begin(), paths.end(), MemoryType::MEM_DEVICE_DDR) != paths.end();
}

void InsertConvertOp::UpdateConsumerAndReconnect(std::shared_ptr<LogicalTensor> oldTensor,
    std::shared_ptr<LogicalTensor> newTensor, Operation* op) const {
    newTensor->AddConsumer(op);
    for (size_t i = 0; i < op->iOperand.size(); ++i) {
        if ((op->iOperand[i]->magic == oldTensor->magic) &&
            (op->iOperand[i]->tensor->rawmagic == oldTensor->tensor->rawmagic)) {
            op->ReplaceIOperand(i, newTensor);
        }
    }
}

Status InsertConvertOp::RunOnFunction(Function &function) {
    oldRawToNewRaw.clear();
    converts.clear();
    for (const auto &op : function.Operations()) {
        RunOnOperation(function, op);
    }

    for (auto &c : converts) {
        auto &convertOp = function.AddOperation(Opcode::OP_CONVERT, {c.input}, {c.output});
        convertOp.SetOpAttribute(std::make_shared<ConvertOpAttribute>(c.from, c.to));
    }
    CheckUnknown(function);
    EliminateDeadOperationBackward(function);
    return SUCCESS;
}

void InsertConvertOp::RunOnOperation(Function &function, const npu::tile_fwk::Operation &operation) {
    for (auto &oOperand : operation.oOperand) {
        if (oOperand->MemoryConflict()) {
            bool crossCore = CrossCore(oOperand);

            auto producers = oOperand->GetProducers();
            bool producedByAssemble = std::all_of(producers.begin(), producers.end(),
                [](const Operation *op) { return op->GetOpcode() == Opcode::OP_ASSEMBLE; });
            if (producedByAssemble && crossCore) {
                oOperand->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
            }

            auto consumers = oOperand->GetConsumers();
            bool canSetBoth = true;
            for (auto consumer : consumers) {
                if (consumer->GetOpcode() != Opcode::OP_VIEW && consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                    canSetBoth = false;
                    break;
                }
                if (consumer->GetOpcode() == Opcode::OP_ASSEMBLE && (consumer->GetOOperands().size() != 1 || consumer->GetOOperands().front()->nodetype != NodeType::OUTCAST)) {
                    canSetBoth = false;
                    break;
                }
                bool needCopy = false;
                (void)consumer->GetAttr("NeedCopy", needCopy);
                if (needCopy) {
                    canSetBoth = false;
                }
            }
            if (canSetBoth && crossCore) {
                oOperand->SetMemoryTypeToBe(MEM_DEVICE_DDR);
            }
            std::vector<MemoryType> paths;
            PassConfigManager::Instance().GetPlatformConfig().FindNearestPath(
                oOperand->GetMemoryTypeOriginal(), oOperand->GetMemoryTypeToBe(), paths);

            std::shared_ptr<LogicalTensor> input = oOperand;
            std::shared_ptr<LogicalTensor> output;
            if (paths.size() <= 1) {
                continue;
            }
            for (size_t i = 0; i < paths.size() - 1; ++i) {
                std::shared_ptr<RawTensor> newRawTensor;
                if (oldRawToNewRaw.count(input->tensor->rawmagic) != 0) {
                    newRawTensor = oldRawToNewRaw[input->tensor->rawmagic];
                } else {
                    newRawTensor = std::make_shared<RawTensor>(input->Datatype(), input->tensor->rawshape);
                    oldRawToNewRaw.insert({input->tensor->rawmagic, newRawTensor});
                }
                input->SetMemoryTypeToBe(paths[i]);
                output = std::make_shared<LogicalTensor>(function, newRawTensor, input->offset, input->shape);
                output->SetMemoryTypeBoth(paths[i + 1]);
                converts.emplace_back(ConvertOp{paths[i], paths[i + 1], input, output});
                input = output;
            }
            for (auto &consumer : consumers) {
                if (consumer->BelongTo() == &function) {
                    UpdateConsumerAndReconnect(oOperand, output, consumer);
                }
            }
        }
    }
}
} // namespace