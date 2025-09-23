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
 * \file intra_subgraph_adapter.cpp
 * \brief
 */

#include <unordered_set>
#include "passes/pass_utils/graph_utils.h"
#include "intra_subgraph_adapter.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
template <typename T>
inline std::string IntSetToStr(const std::set<T> &colorSet) {
    std::stringstream ss;
    ss << "(";

    if (!colorSet.empty()) {
        for (auto &x : colorSet) {
            ss << x << ", ";
        }
    }

    ss << ")";
    return ss.str();
}

Status IntraSubgraphAdapter::RunOnFunction(Function &function) {
    LogicalTensors boundaryTensors = CollectBoundaryTensors(function);

    for (size_t i = 0; i < boundaryTensors.size(); i++) {
        LogicalTensorPtr tensor = boundaryTensors[i];
        if (CheckBoundaryTensor(tensor) != SUCCESS) {
            ALOG_ERROR_F("Check boundary tensor failed!");
            return FAILED;
        }

        // If the boundary tensor is already in the ddr, then skip processing this tensor
        if (tensor->GetMemoryTypeOriginal() == MEM_DEVICE_DDR) {
            continue;
        }

        std::set<int> producerColors, consumerColors;
        CollectProducerColors(tensor, producerColors);
        CollectConsumerColors(tensor, consumerColors);

        std::set<int> commonColors = SetIntersection(producerColors, consumerColors);
        if (commonColors.size() > 1) {
            ALOG_ERROR_F("The producers and consumers cannot simultaneously appear in more than one subgraph.");
            ALOG_ERROR_F("Process boundary tensor failed, tensor magic : %d.", tensor->GetMagic());
            return FAILED;
        }
        if (commonColors.size() == 0) {
            if (ProcessBoundaryTensor(function, tensor) == FAILED) {
                ALOG_ERROR_F("Process boundary tensor failed, tensor magic : %d.", tensor->GetMagic());
                return FAILED;
            }
        }
        if (commonColors.size() == 1) {
            // For boundary tensor that have both producer and consumer in a single subgraph,
            // we split it to multiple boundary tensors, whose producers and consumers do not share same subgraph.
            int mainSubgraphID = *(commonColors.begin());  // the only subgraph id that has both producers and consumers.
            LogicalTensors newBoundaryTensors;

            ALOG_DEBUG_F("********** %s requires SplitBoundaryTensor. **********", function.GetMagicName().c_str());
            ALOG_DEBUG_F("Boundary tensor: %s, mainSubgraphID: %d, producerColors: %s, consumerColors: %s",
                tensor->Dump().c_str(), mainSubgraphID, IntSetToStr(producerColors).c_str(), IntSetToStr(consumerColors).c_str());
            for (auto &producer: tensor->GetProducers()) {
                ALOG_DEBUG_F("producer: %s", producer->Dump().c_str());
            }
            for (auto &consumer: tensor->GetConsumers()) {
                ALOG_DEBUG_F("consumer: %s", consumer->Dump().c_str());
            }

            if (SplitBoundaryTensor(function, tensor, mainSubgraphID, newBoundaryTensors) == FAILED) {
                ALOG_ERROR_F("Split boundary tensor failed, tensor magic : %d.", tensor->GetMagic());
                return FAILED;
            }
            if (ProcessBoundaryTensors(function, newBoundaryTensors) == FAILED) {
                ALOG_ERROR_F("Process boundary tensors failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::CheckBoundaryTensor(LogicalTensorPtr tensor) {
    const static std::unordered_set<MemoryType> validBoundaryTensorMemType = {MEM_UB, MEM_L1,
        MEM_L0C, MEM_DEVICE_DDR};
    if (tensor->GetMemoryTypeOriginal() != tensor->GetMemoryTypeToBe()) {
        ALOG_ERROR_F("The type of boundary tensor does conflict!");
        ALOG_ERROR_F("Tensor magic : %d, tensor original memory type : %s, tensor tobe memory type : %s.",
            tensor->GetMagic(), MemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(),
            MemoryTypeToString(tensor->GetMemoryTypeToBe()).c_str());
        return FAILED;
    }
    if (validBoundaryTensorMemType.find(tensor->GetMemoryTypeOriginal()) == validBoundaryTensorMemType.end()) {
        ALOG_ERROR_F("The original type of boundary tensor should be in [UB, L1, DDR, L0C].");
        ALOG_ERROR_F("Tensor magic : %d, tensor memory type : %s.", tensor->GetMagic(),
            MemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::SplitBoundaryTensor(Function &function, LogicalTensorPtr tensor,
        int mainSubgraphID, LogicalTensors& newBoundaryTensors) {
    // if the tensor has multiple producers in different subgraph, then the producers must be OP_ASSEMBLE/OP_COPY_OUT.
    for (auto& producer : tensor->GetProducers()) {
        if (producer->GetSubgraphID() != mainSubgraphID) {
            Opcode producerOpcode = producer->GetOpcode();
            if (OpcodeManager::Inst().GetOpCalcType(producerOpcode) != OpCalcType::MOVE_OUT &&
                OpcodeManager::Inst().GetOpCalcType(producerOpcode) != OpCalcType::MOVE_LOCAL) {
                ALOG_ERROR_F("If the tensor has multiple producers, then the producers can only be OP_ASSEMBLE/OP_COPY_OUT.");
                ALOG_ERROR_F("Boundary tensor magic : %d, producer op magic : %d, producer op : %s.",
                    tensor->GetMagic(), producer->GetOpMagic(), producer->GetOpcodeStr().c_str());
                return FAILED;
            }
            LogicalTensors& producerInputs = producer->GetIOperands();
            if (producerInputs.size() != 1) {
                ALOG_ERROR_F("The OP_ASSEMBLE should have one input operand.");
                ALOG_ERROR_F("Boundary tensor magic : %d, producer op magic : %d, producer op : %s.",
                    tensor->GetMagic(), producer->GetOpMagic(), producer->GetOpcodeStr().c_str());
                return FAILED;
            }
            // When there are multiple consumers, for producer from other subgraph, we insert a new ASSEMBLE
            // to the other subgraph, and change the producer to main subgraph.
            if (tensor->GetConsumers().size() != 1) {
                LogicalTensorPtr assembleInput = producer->GetIOperands()[0];
                ALOG_DEBUG_F("SplitBoundaryTensor output of %s[%d]", producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
                LogicalTensorPtr newTensor = InsertOpBetween(function, Opcode::OP_ASSEMBLE, assembleInput, {producer});
                producer->UpdateSubgraphID(mainSubgraphID);
                ALOG_INFO_F("Adjust OP_ASSEMBLE(magic : %d) to subgraph %d.", producer->GetOpMagic(), mainSubgraphID);
                // The intermediate tensor become a new boundary tensor.
                newBoundaryTensors.push_back(newTensor);
                ALOG_INFO_F("Add new tensor(magic : %d) to boundary tensors.", assembleInput->GetMagic());
            }
        }
    }
    // When there is only one consumer, we process the boundary tensor as it doesn't have a main subgraph.
    if (tensor->GetConsumers().size() == 1) {
        newBoundaryTensors.push_back(tensor);
        return SUCCESS;
    }

    std::vector<Operation*> subsidiaryConsumers;
    for (auto& consumer : tensor->GetConsumers()) {
        if (consumer->GetSubgraphID() != mainSubgraphID) {
            subsidiaryConsumers.push_back(consumer);
            ALOG_DEBUG_F("Insert %s[%d] to the subsidiaryConsumers.", consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
        }
    }

    // For consumers from other subgraph, we insert a new ASSEMBLE before them,
    // the intermediate tensor become a new boundary tensor.
    if (subsidiaryConsumers.size() != 0) {
        ALOG_DEBUG_F("=========== multi subsidiaryConsumers size: %d ==========", subsidiaryConsumers.size());
        LogicalTensorPtr newTensor =
            InsertOpBetween(function, Opcode::OP_ASSEMBLE, tensor, subsidiaryConsumers, mainSubgraphID);
        newBoundaryTensors.push_back(newTensor);
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::ProcessBoundaryTensors(Function &function, LogicalTensors tensors) {
    for (auto& tensor : tensors) {
        if (ProcessBoundaryTensor(function, tensor) == FAILED) {
            ALOG_ERROR_F("Process boundary tensor failed, tensor magic : %d.", tensor->GetMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status IntraSubgraphAdapter::ProcessBoundaryTensor(Function &function, LogicalTensorPtr tensor) {
    ALOG_INFO_F("Process boundary tensor, tensor magic : %d, info: %s, mem: %s",
        tensor->GetMagic(), tensor->Dump().c_str(), BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str());
    // Insert OP_ASSEMBLE before the boundary tensor, if the producer is not OP_ASSEMBLE/OP_COPY_OUT
    if (AdapteTensorProducers(function, tensor) == FAILED) {
        ALOG_ERROR_F("Adapter tensor producer failed, tensor magic : %d.", tensor->GetMagic());
        return FAILED;
    }
    // Insert OP_VIEW after the boundary tensor, if the consumer is not OP_VIEW/OP_COPY_IN
    if (AdapteTensorConsumers(function, tensor) == FAILED) {
        ALOG_ERROR_F("Adapter tensor consumer failed, tensor magic : %d.", tensor->GetMagic());
        return FAILED;
    }
    // Set the boundary tensor's mem type to be DDR
    tensor->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    return SUCCESS;
}

Status IntraSubgraphAdapter::AdapteTensorProducers(Function &function, LogicalTensorPtr tensor) {
    if (tensor->GetProducers().size() > 1) {
        for (Operation* producer : tensor->GetProducers()) {
            ALOG_DEBUG_F("|---- Producer %s[%d].", producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
            if (producer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                ALOG_DEBUG_F("|---- Op Attr: %s.", producer->Dump().c_str());
            }
            if (OpcodeManager::Inst().GetOpCalcType(producer->GetOpcode()) != OpCalcType::MOVE_OUT &&
                OpcodeManager::Inst().GetOpCalcType(producer->GetOpcode()) != OpCalcType::MOVE_LOCAL) {
                ALOG_ERROR_F("If the tensor has multiple producers, then the producers can only be OP_ASSEMBLE/OP_COPY_OUT.");
                return FAILED;
            }
        }
        return SUCCESS;
    } 
    if (tensor->GetProducers().size() == 1) {
        Operation* producer = *(tensor->GetProducers().begin());
        ALOG_DEBUG_F("|---- Producer %s[%d].", producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
        if (producer->GetOpcode() == Opcode::OP_ASSEMBLE) {
            ALOG_DEBUG_F("|---- Op Attr: %s.", producer->Dump().c_str());
        }
        if (crossCoreMoveOps.find(producer->GetOpcode()) != crossCoreMoveOps.end()) {
            producer->SetOpCode(Opcode::OP_COPY_OUT);
            ALOG_DEBUG_F("change %s[%d] opcode to OP_COPY_OUT.", producer->GetOpcodeStr().c_str(), producer->GetOpMagic());
        }
        if (producer->GetOpcode() != Opcode::OP_ASSEMBLE && producer->GetOpcode() != Opcode::OP_COPY_OUT) {
            InsertOpBetween(function, Opcode::OP_ASSEMBLE, producer, tensor);
        }
        return SUCCESS;
    }
    ALOG_INFO_F("Boundary tensor has no producer, tensor magic : %d", tensor->GetMagic());
    return SUCCESS;
}

Status IntraSubgraphAdapter::AdapteTensorConsumers(Function &function, LogicalTensorPtr tensor) {
    std::unordered_map<int, std::vector<Operation*>> consumerColor2OpsMap;
    for (auto& consumer : tensor->GetConsumers()) {
        ALOG_DEBUG_F("|---- Consumer %s[%d].", consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
        if (consumer->GetOpcode() == Opcode::OP_VIEW) {
            ALOG_DEBUG_F("|---- Op Attr: %s.", consumer->Dump().c_str());
        }
        if (crossCoreMoveOps.find(consumer->GetOpcode()) != crossCoreMoveOps.end()) {
            consumer->SetOpCode(Opcode::OP_COPY_IN);
            ALOG_DEBUG_F("change %s[%d] opcode to OP_COPY_IN.", consumer->GetOpcodeStr().c_str(), consumer->GetOpMagic());
            continue;
        }
        if (consumer->GetOpcode() != Opcode::OP_VIEW && consumer->GetOpcode() != Opcode::OP_COPY_IN) {
            consumerColor2OpsMap[consumer->GetSubgraphID()].push_back(consumer);
        }
    }

    for (auto& [color, consumers] : consumerColor2OpsMap) {
        (void)color;
        InsertOpBetween(function, Opcode::OP_VIEW, tensor, consumers);
    }
    return SUCCESS;
}

// For Assemble
LogicalTensorPtr IntraSubgraphAdapter::InsertOpBetween(Function &function, Opcode opcode,
        Operation *op, LogicalTensorPtr tensor) {
    ALOG_DEBUG_F("intraSubgraphAdapter::InsertOpBetween %s 1.", OpcodeManager::Inst().GetOpcodeStr(opcode).c_str());
    ASSERT(opcode == Opcode::OP_ASSEMBLE);
    auto newRawTensor = std::make_shared<RawTensor>(tensor->Datatype(), tensor->GetRawTensor()->rawshape);
    LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor,
        tensor->GetOffset(), tensor->GetShape());
    GraphUtils::CopyDynStatus(newTensor, tensor);
    newTensor->SetMemoryTypeBoth(tensor->GetMemoryTypeOriginal(), true);
    function.GetTensorMap().Insert(newTensor, false);
    op->ReplaceOutputOperand(tensor, newTensor);
    ALOG_DEBUG_F("Compare Ori: %s", tensor->Dump().c_str());
    ALOG_DEBUG_F("Compare New: %s", newTensor->Dump().c_str());

    std::vector<int64_t> offset(tensor->GetShape().size(), 0);
    Operation* newOp = &function.AddRawOperation(opcode, {newTensor}, {tensor});
    if (opcode == Opcode::OP_ASSEMBLE) {
        newOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(newTensor->GetMemoryTypeOriginal(), offset, tensor->GetDynOffset(), tensor->GetDynValidShape()));
    }
    if (opcode == Opcode::OP_VIEW) {
        newOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, newTensor->GetMemoryTypeToBe(), tensor->GetDynOffset(), tensor->GetDynValidShape()));
    }
    ALOG_DEBUG_F("Insert New Op %s[%d], info: %s", newOp->GetOpcodeStr().c_str(), newOp->GetOpMagic(), newOp->Dump().c_str());
    newOp->UpdateSubgraphID(op->GetSubgraphID());
    newTensor->AddProducer(op);
    newTensor->AddConsumer(newOp);
    tensor->RemoveProducer(op);
    tensor->AddProducer(newOp);
    return newTensor;
}

// For View
LogicalTensorPtr IntraSubgraphAdapter::InsertOpBetween(Function &function, Opcode opcode,
        LogicalTensorPtr tensor, const std::vector<Operation*>& ops, int newOpSubgraphID) {
    /*
        before: tensor -> op
        after: tensor -> newOp -> newTensor ->op
    */
    ALOG_DEBUG_F("IntraSubgraphAdapter::InsertOpBetween %s 2.", OpcodeManager::Inst().GetOpcodeStr(opcode).c_str());
    if (ops.size() == 0) {
        ALOG_ERROR_F("Insert op between tensor and ops failed! ops is empty.");
        return nullptr;
    }
    ASSERT(opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE);
    auto newRawTensor = std::make_shared<RawTensor>(tensor->Datatype(), tensor->GetRawTensor()->rawshape);
    LogicalTensorPtr newTensor = std::make_shared<LogicalTensor>(function, newRawTensor,
        tensor->GetOffset(), tensor->GetShape(), tensor->GetDynValidShape());
    newTensor->UpdateOffset(tensor->GetTensorOffset());
    GraphUtils::CopyDynStatus(newTensor, tensor);
    newTensor->SetMemoryTypeBoth(tensor->GetMemoryTypeOriginal(), true);
    function.GetTensorMap().Insert(newTensor, false);
    ALOG_DEBUG_F("Compare Ori: %s", tensor->Dump().c_str());
    ALOG_DEBUG_F("Compare New: %s", newTensor->Dump().c_str());

    for (Operation* op : ops) {
        op->ReplaceInputOperand(tensor, newTensor);
    }

    std::vector<int64_t> offset(tensor->GetShape().size(), 0);
    Operation* newOp = &function.AddRawOperation(opcode, {tensor}, {newTensor});
    if (opcode == Opcode::OP_ASSEMBLE) {
        newOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(newTensor->GetMemoryTypeOriginal(), offset, tensor->GetDynOffset(), tensor->GetDynValidShape()));
    }
    if (opcode == Opcode::OP_VIEW) {
        newOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, newTensor->GetMemoryTypeToBe(), tensor->GetDynOffset(), tensor->GetDynValidShape()));
    }
    ALOG_DEBUG_F("Insert New Op %s[%d], info: %s", newOp->GetOpcodeStr().c_str(), newOp->GetOpMagic(), newOp->Dump().c_str());
    if (newOpSubgraphID == -1) {
        newOp->UpdateSubgraphID(ops[0]->GetSubgraphID());
    } else {
        newOp->UpdateSubgraphID(newOpSubgraphID);
    }
    newTensor->AddProducer(newOp);
    for (Operation* op : ops) {
        newTensor->AddConsumer(op);
        tensor->RemoveConsumer(op);
    }
    tensor->AddConsumer(newOp);
    return newTensor;
}

void IntraSubgraphAdapter::CollectProducerColors(LogicalTensorPtr tensor, std::set<int>& colors) {
    for (Operation* producer : tensor->GetProducers()) {
        colors.insert(producer->GetSubgraphID());
    }
}

void IntraSubgraphAdapter::CollectConsumerColors(LogicalTensorPtr tensor, std::set<int>& colors) {
    for (Operation* consumer : tensor->GetConsumers()) {
        colors.insert(consumer->GetSubgraphID());
    }
}

std::set<int> IntraSubgraphAdapter::SetIntersection(std::set<int>& a, std::set<int>& b) {
    std::set<int> c;
    for (auto i : a) {
        if (b.find(i) != b.end()) {
            c.insert(i);
        }
    }
    return c;
}

LogicalTensors IntraSubgraphAdapter::CollectBoundaryTensors(Function &function) {
    LogicalTensors ret;
    for (auto &[magic, tensor] : function.GetTensorMap().inverseMap_) {
        (void)magic;
        std::set<int> colors;
        CollectProducerColors(tensor, colors);
        CollectConsumerColors(tensor, colors);
        if (colors.size() > 1) {
            ret.push_back(tensor);
        }
    }
    return ret;
}
} // namespace npu::tile_fwk