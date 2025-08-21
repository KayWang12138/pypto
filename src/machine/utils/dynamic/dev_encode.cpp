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
 * \file dev_encode.cpp
 * \brief
 */

#include "machine/utils/dynamic/dev_encode.h"

#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/tensor_slot.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <queue>

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
namespace dynamic {
#define ONFILLCONTENT if (fillContent)

constexpr int32_t CALLOP_ARG_ATTR_BASE_INDEX = 1;
constexpr int32_t MINI_TILE_LIST_SIZE_THRESHOLD = 16;
constexpr int32_t MAX_WORKSPACE_MUL_SIZE = 75;
constexpr int32_t SLOTS_NEED_ALLOC_SIZE = 2;

struct EncodeRawTensorAttr {
    std::shared_ptr<Storage> storage;
    uint64_t storageOffset = 0;
};

void DevAscendFunction::InitIncastOutcastAttr(
        uintdevptr_t &initOffset,
        const std::vector<std::shared_ptr<LogicalTensor>> &iList,
        const std::vector<std::shared_ptr<LogicalTensor>> &oList, bool /* fillContent */) {
    incastAddressList.HostInitDataSizeOffset(initOffset, iList.size());
    outcastAddressList.HostInitDataSizeOffset(initOffset, oList.size());
}

void DevAscendFunction::InitOperationDynamicField(
        uintdevptr_t &initOffset,
        DevAscendFunctionPredInfo predInfo,
        uint32_t outcastStitchCount,
        [[maybe_unused]]const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
        const SymbolicExpressionTable *expressionTable,
        const OrderedSet<Operation *> &callList,
        const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const std::unordered_map<Operation *, OrderedSet<Operation *>> & /* callOpSuccDict */, bool fillContent) {
    opDynamicFieldList.HostInitDataSizeOffset(initOffset, callList.size());
    ONFILLCONTENT {
        for (size_t i = 0; i < callList.size(); i++) {
            auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
        }
    };
    expressionList.HostInitDataSizeOffset(initOffset, expressionTable->GetPrimaryExpressionSize());

    uint64_t operationSize = callList.size();
    uint64_t incastSize = incastTensorList.size();
    uint64_t outcastSize = outcastTensorList.size();
    uint64_t expressionSize = expressionTable->GetPrimaryExpressionSize();

    uint64_t predCountListDataSize = ALIGN_UP(operationSize * sizeof(predcount_t), sizeof(uint64_t));
    uint64_t incastDataSize = ALIGN_UP(incastSize * sizeof(void *), sizeof(uint64_t));
    uint64_t outcastDataSize = ALIGN_UP(outcastSize * sizeof(void *), sizeof(uint64_t));
    uint64_t expressionDataSize = ALIGN_UP(expressionSize * sizeof(uint64_t), sizeof(uint64_t));
    uint64_t stitchDataSize = ALIGN_UP(outcastStitchCount * sizeof(DevAscendFunctionDuppedStitch *), sizeof(uint64_t));
    uint64_t totalDataSize = predCountListDataSize + incastDataSize + outcastDataSize + expressionDataSize + stitchDataSize;
    duppedData_.HostInitDataSizeOffset(initOffset, sizeof(DevAscendFunctionDuppedData) + totalDataSize);
    duppedDataAllocSize_ = sizeof(DevAscendFunctionDuppedData) + totalDataSize;
    duppedDataCopySize_ = sizeof(DevAscendFunctionDuppedData) + predCountListDataSize;
    predInfo_ = predInfo;
    ALOG_INFO("Pred: zero=", predInfo.totalZeroPred, " aiv=", predInfo.totalZeroPredAIV, " aic=", predInfo.totalZeroPredAIC, " hub=", predInfo.totalZeroPredHub);

    ONFILLCONTENT {
        DevAscendFunctionDuppedData *dupData = reinterpret_cast<DevAscendFunctionDuppedData *>(&At(duppedData_, 0));
        dupData->operationList_.size = operationSize;

        uint64_t offset = 0;
        dupData->operationList_.predCountBase = offset;
        offset += predCountListDataSize;

        dupData->incastList_.size = incastSize;
        dupData->incastList_.base = offset;
        offset += incastDataSize;

        dupData->outcastList_.size = outcastSize;
        dupData->outcastList_.base = offset;
        offset += outcastDataSize;

        dupData->expressionList_.size = expressionSize;
        dupData->expressionList_.base = offset;
        offset += expressionDataSize;

        dupData->operationList_.stitchBase = offset;
        offset += stitchDataSize;
        ASSERT(offset == totalDataSize);

        memset_s(dupData->data_, totalDataSize, 0, totalDataSize);

        uint8_t *dataBegin = &dupData->data_[0];
        uint8_t *incastBegin = &dupData->data_[dupData->incastList_.base];
        uint8_t *outcastBegin = &dupData->data_[dupData->outcastList_.base];
        uint8_t *expressionBegin = &dupData->data_[dupData->expressionList_.base];
        uint8_t *stitchBegin = &dupData->data_[dupData->operationList_.stitchBase];
        uint8_t *dataEnd = &dupData->data_[totalDataSize];
        uint8_t *dataEndAlloc = &dupData->data_[duppedDataAllocSize_ - sizeof(DevAscendFunctionDuppedData)];
        ASSERT(dataEnd == dataEndAlloc);
        for (uint64_t i = 0; i < operationSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetOperationCurrPredCount(i));
            ASSERT(dataBegin <= ptr && ptr < incastBegin);
        }
        for (uint64_t i = 0; i < incastSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetIncastAddress(i));
            ASSERT(incastBegin <= ptr && ptr < outcastBegin);
        }
        for (uint64_t i = 0; i < outcastSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetOutcastAddress(i));
            ASSERT(outcastBegin <= ptr && ptr < expressionBegin);
        }
        for (uint64_t i = 0; i < expressionSize; i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetExpression(i));
            ASSERT(expressionBegin <= ptr && ptr < stitchBegin);
        }
    };
}

void HandleActualRaw(const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
    const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
    const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
    const std::shared_ptr<RawTensor> &rawTensor,
    DevAscendRawTensor &encoded) {
    auto iter = rawMagicToRawTensor.find(rawTensor->actualRawmagic);
    if (iter != rawMagicToRawTensor.end()) {
        if (iter->second->addrOffset == UINT64_MAX) {
            ALOG_ERROR_F("addrOffset is invalid actual raw magic %d, original raw magic %d",
                rawTensor->actualRawmagic, rawTensor->rawmagic);
            encoded.addrOffset = 0;
        } else {
            encoded.addrOffset = iter->second->addrOffset;
            if (outcastRawList.count(iter->second)) {
                encoded.ioIndex = outcastRawList.GetIndex(iter->second);
                encoded.ioProperty = DevIOProperty::ROOT_OUTCAST;
            } else if (incastRawList.count(iter->second)) {
                encoded.ioIndex = incastRawList.GetIndex(iter->second);
                encoded.ioProperty = DevIOProperty::ROOT_INCAST;
            } else {
                encoded.ioIndex = -1;
                encoded.ioProperty = DevIOProperty::NONE;
            }
            ALOG_ERROR_F("Tensor %d use tensor %d's addr io index %d", rawTensor->rawmagic, rawTensor->actualRawmagic,
                encoded.ioIndex);
        }
    }
}

void DevAscendFunction::UpdateRawTensorDesc(const std::shared_ptr<RawTensor> &rawTensor, size_t i, size_t incastRawListSize,
    DevAscendRawTensor &encoded) {
    if (rawTensor->actualRawmagic != -1) {
        ALOG_ERROR_F("[%3zu] raw %d, actualRaw %d, IOType <%s>, addrOffset 0x%lx, ioIndex %d",
                i, rawTensor->rawmagic, rawTensor->actualRawmagic, DevIOProperty2String(encoded.ioProperty).c_str(),
                encoded.addrOffset, encoded.ioIndex);
    }
    uint32_t location = 0;
    uint32_t offsetOrIndex = 0;
    switch (encoded.ioProperty) {
    case DevIOProperty::ROOT_INCAST:
        location = RAW_TENSOR_LOCATION_INCAST;
        offsetOrIndex = encoded.ioIndex;
        break;
    case DevIOProperty::ROOT_OUTCAST:
        location = RAW_TENSOR_LOCATION_OUTCAST;
        offsetOrIndex = encoded.ioIndex + incastRawListSize;
        break;
    case DevIOProperty::NONE:
        location = RAW_TENSOR_LOCATION_LOCAL;
        offsetOrIndex = encoded.addrOffset;
        break;
    default:
        ASSERT(false);
        break;
    }
    DevRawTensorDesc *desc = GetRawTensorDesc(i);
    desc->location = location;
    desc->offsetOrIndex = offsetOrIndex;
}

static void EncodeRawShape(const SymbolicExpressionTable *expressionTable,
    DevAscendRawTensor *encoded, std::shared_ptr<RawTensor> rawTensor) {
    std::vector<SymInt> shape;
    bool isDyn = false;
    for (auto x : rawTensor->GetDynRawShape()) {
        if (x.IsImmediate()) {
            shape.emplace_back(x.Concrete());
        } else {
            shape.emplace_back(true, expressionTable->LookupPrimaryExpressionIndex(x));
            isDyn = true;
        }
    }
    encoded->shape.SetShape(shape);
    encoded->dataType = rawTensor->GetDataType();
    encoded->memoryRequirement = isDyn ? 0: rawTensor->GetRawDataSize();
}

void DevAscendFunction::InitRawTensorAndMemoryRequirement(
        uintdevptr_t &initOffset,
        const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
        const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
        const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
        const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
        const std::vector<EncodeRawTensorAttr> &rawAttrs,
        const EncodeDevAscendFunctionParam &param,
        const SymbolicExpressionTable *expressionTable,
        bool fillContent) {
    auto inoutLink = param.inoutLink;
    auto slot = param.slot;
    rawTensorList_.HostInitDataSizeOffset(initOffset, rawList.size());
    rawTensorDescList_.HostInitDataSizeOffset(initOffset, rawList.size());

    ONFILLCONTENT {
        std::vector<bool> isOutputSlotMarks(inoutLink->totalSlot);
        for (int slotIdx : inoutLink->outputSlotIndexList) {
            isOutputSlotMarks[slotIdx] = true;
        }
        for (int slotIdx: inoutLink->assembleSlotIndexList) {
            isOutputSlotMarks[slotIdx] = true;
        }
        ALOG_DEBUG_F("incast raw size %zu, outcast raw size %zu, rawlist size %zu", incastRawList.size(), outcastRawList.size(),
            rawList.size());
        rawTensorWsMemoryRequirement = 0;
        outcastWsMemoryRequirement = 0;
        for (size_t i = 0; i < rawList.size(); i++) {
            const auto &rawTensor = rawList[i];
            if (rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic) {
                continue;
            }
            auto &encoded = *GetRawTensor(i);
            EncodeRawShape(expressionTable, &encoded, rawTensor);

            if (incastRawList.count(rawTensor)) {
                // No need to allocate memory for root incasts
                encoded.ioProperty = DevIOProperty::ROOT_INCAST;
                encoded.ioIndex = incastRawList.GetIndex(rawTensor);
                rawTensor->addrOffset = 0;
            } else if (outcastRawList.count(rawTensor)) {
                encoded.ioProperty = DevIOProperty::ROOT_OUTCAST;
                encoded.ioIndex = outcastRawList.GetIndex(rawTensor);
                rawTensor->addrOffset = 0;
                bool isOutput = false;
                for (int slotIdx : slot->outcastSlot[encoded.ioIndex]) {
                    if (isOutputSlotMarks[slotIdx]) {
                        isOutput = true;
                        break;
                    }
                }
                if (!isOutput) {
                    encoded.addrOffset = outcastWsMemoryRequirement;
                    rawTensor->addrOffset = outcastWsMemoryRequirement;
                    outcastWsMemoryRequirement += encoded.memoryRequirement;
                }
            } else {
                // For workspace tensors, the memoryRequirement property is deprecated, please don't use its value
                encoded.ioProperty = DevIOProperty::NONE;
                encoded.ioIndex = -1;
                encoded.addrOffset = rawAttrs[i].storage->start_ + rawAttrs[i].storageOffset;
                rawTensor->addrOffset = encoded.addrOffset;
                rawTensorWsMemoryRequirement = std::max(rawTensorWsMemoryRequirement,
                    rawAttrs[i].storage->start_ + rawAttrs[i].storage->length_);
            }

            UpdateRawTensorDesc(rawTensor, i, incastRawList.size(), encoded);
        }

        for (size_t i = 0; i < rawList.size(); i++) {
            const auto &rawTensor = rawList[i];
            if (rawTensor->actualRawmagic == -1 || rawTensor->actualRawmagic == rawTensor->rawmagic) {
                continue;
            }
            auto &encoded = *GetRawTensor(i);
            EncodeRawShape(expressionTable, &encoded, rawTensor);
            HandleActualRaw(incastRawList, outcastRawList, rawMagicToRawTensor, rawTensor, encoded);
            UpdateRawTensorDesc(rawTensor, i, incastRawList.size(), encoded);
        }

        for (size_t i = 0; i < rawList.size(); i++) {
            const auto &rawTensor = rawList[i];
            if (rawTensor->actualRawmagic != -1 && rawTensor->actualRawmagic != rawTensor->rawmagic) {
                auto it = rawMagicToRawTensor.find(rawTensor->actualRawmagic);
                ASSERT(it != rawMagicToRawTensor.end());
                auto &actualRaw = it->second;
                ASSERT(rawTensor->GetRawShapeSize() == actualRaw->GetRawShapeSize());
                ASSERT(rawTensor->GetRawDataSize() == actualRaw->GetRawDataSize());
            }
        }

        // file linkedIncastId
        auto outIncastLinkMap = param.devRoot->outIncastLinkMap;
        ALOG_ERROR_F("devRoot is %s", param.devRoot->GetRawName().c_str());
        for (size_t i = 0; i < rawList.size(); i++) {
            auto &encoded = *GetRawTensor(i);
            if (outIncastLinkMap.find(rawList[i]) != outIncastLinkMap.end()) {
                encoded.linkedIncastId = incastRawList.GetIndex(outIncastLinkMap[rawList[i]]); //换成incast的下标 ioidx
                ALOG_ERROR_F("linkedIncastId is %d", encoded.linkedIncastId);
            } else {
                encoded.linkedIncastId = -1;
                ALOG_ERROR_F("linkedIncastId is %d", encoded.linkedIncastId);
            }
        }
    }; // ONFILLCONTENT
}

void DevAscendFunction::InitTensor(
        uintdevptr_t &initOffset,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const OrderedSet<std::shared_ptr<RawTensor>> &rawList, bool fillContent) {
    tensorList_.HostInitDataSizeOffset(initOffset, tlist.size());
    for (size_t i = 0; i < tlist.size(); i++) {
        ONFILLCONTENT {
            GetTensor(i)->rawIndex = rawList.find(tlist[i]->tensor)->second;
        };
    }
}

void DevAscendFunction::InitOperation(
        uintdevptr_t &initOffset,
        const SymbolicExpressionTable *expressionTable,
        const OrderedSet<Operation *> &callList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
        const std::unordered_map<Operation *, uint64_t> &callOpPredDict,
        const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict,
        const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
        const std::vector<int32_t> &outcastStitchIndexList,
        bool fillContent) {
    std::vector<int> noPredOpList;
    std::vector<int> noSuccOpList;
    for (size_t i = 0; i < callList.size(); i++) {
        Operation *op = callList[i];
        if (!callOpPredDict.count(op) || callOpPredDict.at(op) == 0) {
            noPredOpList.push_back(i);
        }
        if (!callOpSuccDict.count(op) || callOpSuccDict.at(op).empty()) {
            noSuccOpList.push_back(i);
        }
    }
    noPredOpList_.HostInitDataSizeOffset(initOffset, noPredOpList.size());
    noSuccOpList_.HostInitDataSizeOffset(initOffset, noSuccOpList.size());

    ONFILLCONTENT {
        memcpy_s(&At(noPredOpList_, 0), noPredOpList_.ByteSize(),
            noPredOpList.data(), noPredOpList.size() * sizeof(int));
        memcpy_s(&At(noSuccOpList_, 0), noSuccOpList_.ByteSize(),
            noSuccOpList.data(), noSuccOpList.size() * sizeof(int));

        ASSERT(noPredOpList_.size() == noPredOpList.size());
        for (size_t i = 0; i < noPredOpList.size(); i++) {
            int opIdx = At(noPredOpList_, i);
            auto *op = callList[opIdx];
            ASSERT(!callOpPredDict.count(op) || callOpPredDict.at(op) == 0);
        }
        ASSERT(noSuccOpList_.size() == noSuccOpList.size());
        for (size_t i = 0; i < noSuccOpList.size(); i++) {
            int opIdx = At(noSuccOpList_, i);
            auto *op = callList[opIdx];
            ASSERT(!callOpSuccDict.count(op) || callOpSuccDict.at(op).empty());
        }
    }

    operationList_.HostInitDataSizeOffset(initOffset, callList.size());

    int operandSize = 0;
    int staticAttrSize = 0;
    int succSize = 0;
    for (size_t i = 0; i < callList.size(); i++) {
        Operation *op = callList[i];
        auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());

        operandSize += op->GetIOperands().size() + op->GetOOperands().size();
        staticAttrSize += callop->GetLinearArgList().size();
        succSize += callOpSuccDict.find(op)->second.size();
    }
    operationOperandInfoList_.HostInitDataSizeOffset(initOffset, operandSize);
    operationAttrList_.HostInitDataSizeOffset(initOffset, staticAttrSize);
    opAttrOffsetList_.HostInitDataSizeOffset(initOffset, callList.size());
    opCalleeList_.HostInitDataSizeOffset(initOffset, callList.size());
    operationSuccList_.HostInitDataSizeOffset(initOffset, succSize);

    ONFILLCONTENT {
        DevAscendFunctionDuppedData *dupData = reinterpret_cast<DevAscendFunctionDuppedData *>(&At(duppedData_, 0));
        operandSize = 0;
        staticAttrSize = 0;
        succSize = 0;
        for (size_t i = 0; i < callList.size(); i++) {
            Operation *op = callList[i];
            auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
            DevAscendOperation &staticField = At(operationList_, i);

            staticField.outcastStitchIndex = outcastStitchIndexList[i];
            staticField.opmagic = op->GetOpMagic();
            staticField.ioperandList.AssignRangeOffsetSize(
                operationOperandInfoList_, operandSize, op->GetIOperands().size());
            std::map<int, int> rawTensorIndex;
            for (size_t j = 0; j < op->GetIOperands().size(); j++) {
                auto coaIndex = op->GetIOpAttrOffset(j);
                const std::shared_ptr<LogicalTensor> &tensor = op->GetIOperands()[j];
                rawTensorIndex[coaIndex] = rawList.find(tensor->tensor)->second;
                At(staticField.ioperandList, j) =
                    DevAscendOperationOperandInfo(tlist.GetIndex(tensor), coaIndex + COA_INDEX_DIM_BASE, tensor->GetShape().size());
            }
            operandSize += op->GetIOperands().size();

            staticField.ooperandList.AssignRangeOffsetSize(
                operationOperandInfoList_, operandSize, op->GetOOperands().size());
            for (size_t j = 0; j < op->GetOOperands().size(); j++) {
                auto coaIndex = op->GetOOpAttrOffset(j);
                const std::shared_ptr<LogicalTensor> &tensor = op->GetOOperands()[j];
                rawTensorIndex[coaIndex] = rawList.find(tensor->tensor)->second;
                At(staticField.ooperandList, j) =
                    DevAscendOperationOperandInfo(tlist.GetIndex(tensor), coaIndex + COA_INDEX_DIM_BASE, tensor->GetShape().size());
            }
            operandSize += op->GetOOperands().size();
            ALOG_DEBUG_F("Producer %zu oOperand list size is %zu", i, op->GetOOperands().size());
            // Fill attr
            auto callArgs = callop->GetLinearArgList();
            int opStaticAttrSize = callArgs.size();
            staticField.attrList.AssignRangeOffsetSize(operationAttrList_, staticAttrSize, opStaticAttrSize);
            At(staticField.attrList, 0) = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
            for (size_t j = CALLOP_ARG_ATTR_BASE_INDEX; j < (size_t)opStaticAttrSize; j++) {
                int fillValue = 0;
                if (callArgs[j].IsImmediate()) {
                    if (rawTensorIndex.count(j)) {
                        fillValue = rawTensorIndex[j];
                    } else {
                        fillValue = callArgs[j].Concrete();
                    }
                } else {
                    fillValue = expressionTable->LookupPrimaryExpressionIndex(callArgs[j]);
                }
                At(staticField.attrList, j) = SymInt(!callArgs[j].IsImmediate(), fillValue);
            }

            At(opAttrOffsetList_, i) = staticAttrSize;
            At(opCalleeList_, i) = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
            staticAttrSize += opStaticAttrSize;

            // Fill succ
            int opSuccSize = callOpSuccDict.find(op)->second.size();
            staticField.succList.AssignRangeOffsetSize(operationSuccList_, succSize, opSuccSize);
            for (int j = 0; j < opSuccSize; j++) {
                int succ = callList.GetIndex(callOpSuccDict.find(op)->second[j]);
                At(staticField.succList, j) = succ;
                At(operationList_, succ).predCount++;
                dupData->GetOperationCurrPredCount(succ)++;
            }
            succSize += opSuccSize;
        }
        for (size_t i = 0; i < callList.size(); i++) {
            Operation *op = callList[i];
            ASSERT(callOpPredDict.count(op));
            ASSERT(At(operationList_, i).predCount == callOpPredDict.find(op)->second);
            ASSERT(dupData->GetOperationCurrPredCount(i) == callOpPredDict.find(op)->second);
        }
        dupData->GetSource() = this;
        for (size_t i = 0; i < callList.size(); i++) {
            uint8_t *ptr = reinterpret_cast<uint8_t *>(&dupData->GetOperationStitch(i));
            uint8_t *stitchBegin = &dupData->data_[dupData->operationList_.stitchBase];
            uint8_t *dataEndAlloc = &dupData->data_[duppedDataAllocSize_ - sizeof(DevAscendFunctionDuppedData)];
            ASSERT(stitchBegin <= ptr && ptr < dataEndAlloc);
        }
        dupData->GetSource() = nullptr;
    }
}

void DevAscendFunction::InitIncastOutcast(
        uintdevptr_t &initOffset,
        const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &inoutOpAttrs,
        const IncastOutcastSlot *slot, const std::string &initRawName, bool fillContent) {
    incastList.HostInitDataSizeOffset(initOffset, incastTensorList.size());
    outcastList.HostInitDataSizeOffset(initOffset, outcastTensorList.size());

    slotList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t slotSize = 0;
    uint64_t opSize = 0;
    uint64_t dimSize = 0;
    for (size_t i = 0; i < incastTensorList.size(); i++) {
        auto &inAttr = inoutOpAttrs.at(incastTensorList[i]);
        ONFILLCONTENT {
            At(incastList, i).tensorIndex = tlist.GetIndex(incastTensorList[i]);
        };
        ONFILLCONTENT {
            At(incastList, i).dim = inAttr.dim;
        }
        ONFILLCONTENT {
            At(incastList, i).fromSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->incastSlot[i].size());
        };
        for (size_t j = 0; j < slot->incastSlot[i].size(); j++) {
            ONFILLCONTENT {
                At(At(incastList, i).fromSlotList, j) = slot->incastSlot[i][j];
            };
        }
        slotSize += slot->incastSlot[i].size();
        opSize += inAttr.offsetAttrIdx.size();
    }
    for (size_t i = 0; i < outcastTensorList.size(); i++) {
        auto &outAttr = inoutOpAttrs.at(outcastTensorList[i]);
        ONFILLCONTENT {
            At(outcastList, i).tensorIndex = tlist.GetIndex(outcastTensorList[i]);
        };
        ONFILLCONTENT {
            At(outcastList, i).dim = outAttr.dim;
        };
        ONFILLCONTENT {
            At(outcastList, i).toSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->outcastSlot[i].size());
        };
        for (size_t j = 0; j < slot->outcastSlot[i].size(); j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).toSlotList, j) = slot->outcastSlot[i][j];
            };
        }
        slotSize += slot->outcastSlot[i].size();
        opSize += outAttr.offsetAttrIdx.size();
        dimSize += outAttr.dim;
    }
    slotList.HostInitDataSizeOffset(initOffset, slotSize);
    minimalShapeList.HostInitDataSizeOffset(initOffset, dimSize);
    dimSize = 0;
    for (size_t i = 0; i < outcastTensorList.size(); i++) {
        auto &outAttr = inoutOpAttrs.at(outcastTensorList[i]);
        ONFILLCONTENT {
            At(outcastList, i).minimalShape.AssignRangeOffsetSize(minimalShapeList, dimSize, outAttr.dim);
        };
        for (int j = 0; j < outAttr.dim; j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).minimalShape, j) = outAttr.minimalShape[j];
            };
        }
        dimSize += outAttr.dim;
    }
    offsetIdxList.HostInitDataSizeOffset(initOffset, opSize);
    shapeIdxList.HostInitDataSizeOffset(initOffset, opSize);
    producerConsumerList.HostInitDataSizeOffset(initOffset, opSize);
    inoutOperandIdxList.HostInitDataSizeOffset(initOffset, opSize);
    opSize = 0;
    for (size_t i = 0; i < incastTensorList.size(); i++) {
        auto &inAttr = inoutOpAttrs.at(incastTensorList[i]);
        ONFILLCONTENT {
            At(incastList, i).offsetAttrIdx.AssignRangeOffsetSize(offsetIdxList, opSize, inAttr.offsetAttrIdx.size());
        };
        for (size_t j = 0; j < inAttr.offsetAttrIdx.size(); j++) {
            ONFILLCONTENT {
                At(At(incastList, i).offsetAttrIdx, j) = inAttr.offsetAttrIdx[j];
            };
        }
        ONFILLCONTENT {
            At(incastList, i).shapeAttrIdx.AssignRangeOffsetSize(shapeIdxList, opSize, inAttr.shapeAttrIdx.size());
        };
        for (size_t j = 0; j < inAttr.shapeAttrIdx.size(); j++) {
            ONFILLCONTENT {
                At(At(incastList, i).shapeAttrIdx, j) = inAttr.shapeAttrIdx[j];
            };
        }
        ONFILLCONTENT {
            At(incastList, i).consumer.AssignRangeOffsetSize(producerConsumerList, opSize, inAttr.ops.size());
        };
        for (size_t j = 0; j < inAttr.ops.size(); j++) {
            ONFILLCONTENT {
                At(At(incastList, i).consumer, j) = inAttr.ops[j];
            };
        }
        ONFILLCONTENT {
            At(incastList, i).operandIdx.AssignRangeOffsetSize(inoutOperandIdxList, opSize, inAttr.ops.size());
        };
        for (size_t j = 0; j < inAttr.ops.size(); j++) {
            ONFILLCONTENT {
                At(At(incastList, i).operandIdx, j) = inAttr.operandIdx[j];
            };
        }
        opSize += inAttr.ops.size();
    }
    uint64_t tileListSize = 0;
    minimalTileIdxList.HostInitDataSizeOffset(initOffset, 0);
    for (size_t i = 0; i < outcastTensorList.size(); i++) {
        auto &outAttr = inoutOpAttrs.at(outcastTensorList[i]);
        ONFILLCONTENT {
            At(outcastList, i).offsetAttrIdx.AssignRangeOffsetSize(offsetIdxList, opSize, outAttr.offsetAttrIdx.size());
        };
        for (size_t j = 0; j < outAttr.offsetAttrIdx.size(); j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).offsetAttrIdx, j) = outAttr.offsetAttrIdx[j];
            };
        }
        ONFILLCONTENT {
            At(outcastList, i).shapeAttrIdx.AssignRangeOffsetSize(shapeIdxList, opSize, outAttr.shapeAttrIdx.size());
        };
        for (size_t j = 0; j < outAttr.shapeAttrIdx.size(); j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).shapeAttrIdx, j) = outAttr.shapeAttrIdx[j];
            };
        }
        ONFILLCONTENT {
            At(outcastList, i).producer.AssignRangeOffsetSize(producerConsumerList, opSize, outAttr.ops.size());
        };
        for (size_t j = 0; j < outAttr.ops.size(); j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).producer, j) = outAttr.ops[j];
            };
        }
        ONFILLCONTENT {
            At(outcastList, i).operandIdx.AssignRangeOffsetSize(inoutOperandIdxList, opSize, outAttr.ops.size());
        };
        for (size_t j = 0; j < outAttr.ops.size(); j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).operandIdx, j) = outAttr.operandIdx[j];
            };
        }
        ONFILLCONTENT {
            At(outcastList, i)
                .minimalTileIdx.AssignRangeOffsetSize(minimalTileIdxList, tileListSize, outAttr.minimalTileListSize);
        };
        for (int j = 0; j < outAttr.minimalTileListSize; j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).minimalTileIdx, j) = -1;
            };
        }
        opSize += outAttr.ops.size();
        tileListSize += outAttr.minimalTileListSize;
    }
    minimalTileIdxList.HostInitDataSizeOffset(initOffset, tileListSize);

    // outcast fast stitch
    tileListSize = 0;
    outcastMinimalTileIdxList.HostInitDataSizeOffset(initOffset, 0);
    for (size_t i = 0; i < outcastTensorList.size(); i++) {
        auto &outAttr = inoutOpAttrs.at(outcastTensorList[i]);
        ONFILLCONTENT {
            At(outcastList, i)
                .fastStitchTileIdx.AssignRangeOffsetSize(outcastMinimalTileIdxList, tileListSize, outAttr.minimalTileListSize);
        };
        for (int j = 0; j < outAttr.minimalTileListSize; j++) {
            ONFILLCONTENT {
                At(At(outcastList, i).fastStitchTileIdx, j) = -1;
            };
        }
        tileListSize += outAttr.minimalTileListSize;
    }
    outcastMinimalTileIdxList.HostInitDataSizeOffset(initOffset, tileListSize);
    ONFILLCONTENT {
        for (size_t i = 0; i < outcastList.size(); i++) {
            auto &outcast = At(outcastList, i);
            bool fastStitchEnable = true;
            auto &outAttr = inoutOpAttrs.at(outcastTensorList[i]);

            for (size_t j = 0; j < outAttr.ops.size(); j++) {
                auto producerIdx = outAttr.ops[j];
                auto oOperandIdx = outAttr.operandIdx[j];
                DevAscendOperation &staticField = At(operationList_, producerIdx);
                auto &oOperand = At(staticField.ooperandList, oOperandIdx);
                auto staticOffsetAttrBeginIndex = oOperand.staticOffsetAttrBeginIndex;
                uint64_t tileIndex = 0;
                if (oOperand.GetDim() != static_cast<int>(outcastTensorList[i]->shape.size())) {
                    fastStitchEnable = false;
                    break;
                }
                for (size_t dimIndex = 0; dimIndex < outcastTensorList[i]->shape.size(); ++dimIndex) {
                    auto &offset = At(staticField.attrList, staticOffsetAttrBeginIndex + dimIndex);
                    if (offset.IsExpression()) {
                        fastStitchEnable = false;
                        break;
                    } else {
                        int tileThisDim = offset.Value() / outAttr.minimalShape[dimIndex];
                        if (dimIndex == outcastTensorList[i]->shape.size() - 1) {
                            tileIndex += tileThisDim;
                        } else {
                            tileIndex += tileThisDim * outAttr.tileEachDim[dimIndex + 1];
                        }
                        ALOG_DEBUG_F("tileIndex for outcast %zu in op %zu dim %zu is %lu", i, j, dimIndex, tileIndex);
                    }
                }
                if (tileIndex < outcast.fastStitchTileIdx.size()) {
                    At(outcast.fastStitchTileIdx, tileIndex) = producerIdx;
                } else {
                    ALOG_ERROR_F("producer %d, operand %d offset not valid", producerIdx, oOperandIdx);
                    fastStitchEnable = false;
                    break;
                }
            }
            outcast.fastStitchEnable = fastStitchEnable;
        }
    }
    // incast fast stitch
    uint64_t inTileListSize = 0;
    incastMinimalTileIdxList.HostInitDataSizeOffset(initOffset, 0);
    for (size_t i = 0; i < incastList.size(); i++) {
        bool fastStitchEnable = true;
        auto &inAttr = inoutOpAttrs.at(incastTensorList[i]);
        if (inAttr.minimalTileListSize > MINI_TILE_LIST_SIZE_THRESHOLD) {
            ONFILLCONTENT {
                At(incastList, i).fastStitchEnable = false;
            }
            continue;
        }
        ONFILLCONTENT {
            At(incastList, i).fastStitchTileIdx.AssignRangeOffsetSize(incastMinimalTileIdxList,
                inTileListSize, inAttr.minimalTileListSize);
        };
        inTileListSize += inAttr.minimalTileListSize;
        for (int j = 0; j < inAttr.minimalTileListSize; j++) {
            ONFILLCONTENT {
                At(At(incastList, i).fastStitchTileIdx, j) = -1;
            };
        }
        ONFILLCONTENT {
            int minConsumerIdx = 0x7fffffff;
            for (size_t j = 0; j < inAttr.ops.size(); j++) {
                auto consumerIdx = inAttr.ops[j];
                if (consumerIdx < minConsumerIdx) {
                    minConsumerIdx = consumerIdx;
                }
                auto iOperandIdx = inAttr.operandIdx[j];
                DevAscendOperation &staticField = At(operationList_, consumerIdx);
                auto &iOperand = At(staticField.ioperandList, iOperandIdx);
                auto staticOffsetAttrBeginIndex = iOperand.staticOffsetAttrBeginIndex;
                uint64_t tileIndex = 0;
                if (iOperand.GetDim() != static_cast<int>(incastTensorList[i]->shape.size())) {
                    fastStitchEnable = false;
                    break;
                }
                for (size_t dimIndex = 0; dimIndex < incastTensorList[i]->shape.size(); ++dimIndex) {
                    auto &offset = At(staticField.attrList, staticOffsetAttrBeginIndex + dimIndex);
                    if (offset.IsExpression()) {
                        fastStitchEnable = false;
                        break;
                    } else {
                        int tileThisDim = offset.Value() / inAttr.minimalShape[dimIndex];
                        if (dimIndex == incastTensorList[i]->shape.size() - 1) {
                            tileIndex += tileThisDim;
                        } else {
                            tileIndex += tileThisDim * inAttr.tileEachDim[dimIndex + 1];
                        }
                        ALOG_DEBUG_F("tileIndex for incast %zu in op %zu dim %zu is %lu", i, j, dimIndex, tileIndex);
                    }
                }
                if (tileIndex < At(incastList, i).fastStitchTileIdx.size()) {
                    At(At(incastList, i).fastStitchTileIdx, tileIndex) = consumerIdx;
                } else {
                    ALOG_ERROR_F("consumer %d, operand %d offset not valid", consumerIdx, iOperandIdx);
                    fastStitchEnable = false;
                    break;
                }
            }
            if (fastStitchEnable) {
                At(incastList, i).fastStitchEnable = minConsumerIdx;
                ALOG_DEBUG_F("######## incast %zu is fastStitchEnable", i);
            } else {
                At(incastList, i).fastStitchEnable = -1;
                ALOG_DEBUG_F("######## incast %zu is not fastStitchEnable", i);
            }
        }
    }
    incastMinimalTileIdxList.HostInitDataSizeOffset(initOffset, inTileListSize);

    rawName_.HostInitDataSizeOffset(initOffset, (initRawName.size() / 8 + 1) * 8); // 8 byte align
    ONFILLCONTENT {
        memcpy_s(&At(rawName_, 0), rawName_.size(), initRawName.c_str(), initRawName.size());
        memset_s(&At(rawName_, initRawName.size()), rawName_.size() - initRawName.size(), 0,
            rawName_.size() - initRawName.size());
    };
}

struct EncodeDevAscendFunctionInfo {
    Function *devRoot{nullptr};

    const std::unordered_map<uint64_t, int> &calleeHashIndexDict;
    const std::vector<CceCodeInfo> &cceCodeInfoList;
    const SymbolicExpressionTable *expressionTable{nullptr};

    std::string rawName;

    OrderedSet<Operation *> callList;

    uint64_t totalZeroPred{0};
    uint64_t totalZeroPredAIV{0};
    uint64_t totalZeroPredAIC{0};
    uint64_t totalZeroPredHub{0};

    std::unordered_map<Operation *, uint64_t> callOpPredDict;
    std::unordered_map<Operation *, OrderedSet<Operation *>> callOpSuccDict;
    std::unordered_map<int, std::vector<int>> colorOutGraph;
    std::vector<std::shared_ptr<Operation>> dummyOpList;

    uint32_t outcastStitchCount{0};
    std::vector<int32_t> outcastStitchIndexList;

    OrderedSet<std::shared_ptr<RawTensor>> incastRawTensorList;
    OrderedSet<std::shared_ptr<RawTensor>> outcastRawTensorList;
    OrderedSet<std::shared_ptr<RawTensor>> rawTensorList;
    std::vector<EncodeRawTensorAttr> rawAttrs;

    std::unordered_map<int, std::shared_ptr<RawTensor>> rawMagicToRawTensor;
    OrderedSet<std::shared_ptr<LogicalTensor>> tensorList;

    std::vector<std::shared_ptr<LogicalTensor>> incastList;
    std::vector<std::shared_ptr<LogicalTensor>> outcastList;

    std::unordered_set<std::shared_ptr<LogicalTensor>> incastSet;
    std::unordered_set<std::shared_ptr<LogicalTensor>> outcastSet;

    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> inoutOpAttrs;

    void UpdateMinimalShape(InoutOperationAttr &inoutOpAttr, CallOpAttribute *callAttr, int dim, int shapeIndex) {
        for (int i = 0; i < dim; ++i) {
            if (inoutOpAttr.minimalShape[i] < 0 ||
                inoutOpAttr.minimalShape[i] > callAttr->GetLinearArgList()[shapeIndex + i]) {
                inoutOpAttr.minimalShape[i] = callAttr->GetLinearArgList()[shapeIndex + i];
                assert(inoutOpAttr.minimalShape[i]);
            }
        }
    }

    void RecordRawTensor(const std::shared_ptr<LogicalTensor> &tensor) {
        if (rawTensorList.Insert(tensor->GetRawTensor())) {
            EncodeRawTensorAttr &attr = rawAttrs.emplace_back();
            attr.storage = tensor->storage_;
            attr.storageOffset = tensor->storageOffset_;
            rawMagicToRawTensor[tensor->GetRawTensor()->rawmagic] = tensor->GetRawTensor();
        }
    }

    void EncodeOutCasts() {
        for (auto &o : outcastList) {
            tensorList.Insert(o);
            outcastRawTensorList.Insert(o->GetRawTensor());
            RecordRawTensor(o);
            InoutOperationAttr inoutOpAttr;
            auto dim = o->shape.size();
            inoutOpAttr.minimalShape = std::vector<int>(dim, -1);
            inoutOpAttr.tileEachDim = std::vector<int>(dim, -1);
            for (size_t j = 0; j < callList.size(); j++) {
                auto &op = *callList[j];
                auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
                for (size_t k = 0; k < op.GetOOperands().size(); ++k) {
                    auto &oOperand = op.GetOOperands()[k];
                    if (o->tensor->rawmagic == oOperand->tensor->rawmagic) {
                        ASSERT(oOperand->GetShape().size() == dim);
                        auto coaIndex = op.GetOOpAttrOffset(k) + COA_INDEX_DIM_BASE;
                        inoutOpAttr.dim = dim;
                        inoutOpAttr.offsetAttrIdx.push_back(coaIndex + dim * COA_INDEX_TYPE_OFFSET);
                        inoutOpAttr.shapeAttrIdx.push_back(coaIndex + dim * COA_INDEX_TYPE_SHAPE);
                        inoutOpAttr.ops.push_back(j);
                        inoutOpAttr.operandIdx.push_back(k);
                        ALOG_DEBUG_F("outcast oOperandIdx for outcast %d %d is %d", o->magic, o->GetRawMagic(), k);

                        // callopAttr does not resrve cce index, put one pos back
                        UpdateMinimalShape(inoutOpAttr, callAttr, dim, coaIndex + dim * COA_INDEX_TYPE_SHAPE);
                        ALOG_DEBUG_F("minimal shape for outcast %d raw %d op %d %d is %s\n", o->magic, o->GetRawMagic(), j,
                            op.opmagic, IntVecToStr(inoutOpAttr.minimalShape).c_str());
                    }
                }
            }

            inoutOpAttr.minimalTileListSize = 1;
            for (int l = (dim - 1); l >= 0; --l) {
                auto tiles = o->shape[l] / inoutOpAttr.minimalShape[l];
                if (o->shape[l] % inoutOpAttr.minimalShape[l] != 0) {
                    // should not happen
                    tiles += 1;
                }
                inoutOpAttr.minimalTileListSize *= tiles;
                inoutOpAttr.tileEachDim[l] = inoutOpAttr.minimalTileListSize;
            }
            ALOG_DEBUG_F("outcast %d raw %d shape %s tile %s | minimalTileListSize %d, tileEachDim %s\n", o->magic, o->GetRawMagic(),
                IntVecToStr(o->shape).c_str(),
                IntVecToStr(inoutOpAttr.minimalShape).c_str(),
                inoutOpAttr.minimalTileListSize,
                IntVecToStr(inoutOpAttr.tileEachDim).c_str());

            inoutOpAttrs.insert({o, inoutOpAttr});
        }
    }

    void EncodeIncasts() {
        for (auto &i : incastList) {
            tensorList.Insert(i);
            incastRawTensorList.Insert(i->GetRawTensor());
            RecordRawTensor(i);
            InoutOperationAttr inoutOpAttr;
            auto dim = i->shape.size();
            inoutOpAttr.minimalShape = std::vector<int>(dim, -1);
            inoutOpAttr.tileEachDim = std::vector<int>(dim, -1);
            for (size_t j = 0; j < callList.size(); j++) {
                auto &op = *callList[j];
                auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
                // add icast and oper io's relationship
                for (size_t k = 0; k < op.GetIOperands().size(); ++k) {
                    auto &iOperand = op.GetIOperands()[k];                    
                    if (i->tensor->rawmagic == iOperand->tensor->rawmagic) {
                        ASSERT(iOperand->GetShape().size() == dim);
                        auto coaIndex = op.GetIOpAttrOffset(k) + COA_INDEX_DIM_BASE;
                        inoutOpAttr.dim = dim;
                        inoutOpAttr.offsetAttrIdx.push_back(coaIndex + dim * COA_INDEX_TYPE_OFFSET);
                        inoutOpAttr.shapeAttrIdx.push_back(coaIndex + dim * COA_INDEX_TYPE_SHAPE);
                        inoutOpAttr.ops.push_back(j);
                        inoutOpAttr.operandIdx.push_back(k);

                        // callopAttr does not reserve cce index, put one pos back
                        UpdateMinimalShape(inoutOpAttr, callAttr, dim, coaIndex + dim * COA_INDEX_TYPE_SHAPE);
                        ALOG_DEBUG_F("minimal shape for incast %d raw %d op %d %d is %s\n", i->magic, i->GetRawMagic(), j,
                            op.opmagic, IntVecToStr(inoutOpAttr.minimalShape).c_str());
                    }
                }
            }

            inoutOpAttr.minimalTileListSize = 1;
            for (int l = (dim - 1); l >= 0; --l) {
                if (inoutOpAttr.minimalShape[l] != -1) {
                    auto tiles = i->shape[l] / inoutOpAttr.minimalShape[l];
                    if (i->shape[l] % inoutOpAttr.minimalShape[l] != 0) {
                        // should not happen
                        tiles += 1;
                    }
                    if (tiles < 0) {
                        tiles = 0;
                    }
                    inoutOpAttr.minimalTileListSize *= tiles;
                } else {
                    inoutOpAttr.minimalTileListSize = 0;
                }
                inoutOpAttr.tileEachDim[l] = inoutOpAttr.minimalTileListSize;
            }
            ALOG_DEBUG_F("incast %d raw %d shape %s tile %s | minimalTileListSize %d, tileEachDim %s\n", i->magic, i->GetRawMagic(),
                IntVecToStr(i->shape).c_str(),
                IntVecToStr(inoutOpAttr.minimalShape).c_str(),
                inoutOpAttr.minimalTileListSize,
                IntVecToStr(inoutOpAttr.tileEachDim).c_str());

            inoutOpAttrs.insert({i, inoutOpAttr});
        }
    }

    struct Hasher {
        template<typename T>
        std::size_t operator()(const OrderedSet<T> &ops) const {
            size_t res = 0;
            for (auto op : ops) {
                res ^= std::hash<Operation *>{}(op);
            }
            return res;
        }
    };

    Operation *MakeDummyCall() {
        LogicalTensors inputs, outputs;
        auto opAttr = std::make_shared<CallOpAttribute>();
        auto dummyOp = std::make_shared<Operation>(*devRoot, Opcode::OP_CALL);
        dummyOp->SetOpAttribute(opAttr);
        dummyOpList.push_back(dummyOp);
        ASSERT(GetCoreType(dummyOp.get()) == static_cast<int>(CoreType::HUB));
        return dummyOp.get();
    }

    int GetCoreType(Operation *callop) {
        int leafIndex = calleeHashIndexDict.at(callop->GetCalleeHash().GetHash());
        return cceCodeInfoList[leafIndex].coreType;
    }

    void removeRedundantCall(std::vector<Operation *> &/* callOpList */) {
        std::vector<Operation *> deadCallOps;
        for (auto &[callOp, succOps] : callOpSuccDict) {
            if (GetCoreType(callOp) == static_cast<int>(CoreType::HUB) && succOps.size() == 0) {
                callOpPredDict[callOp] = 0;
                deadCallOps.push_back(callOp);
            }
        }

        for (auto &[callOp, succOps] : callOpSuccDict) {
            UNUSED(callOp);
            succOps.Remove(deadCallOps);
        }
    }

    void optimizeCallopSuccs(std::vector<Operation *> &callOpList, int optimizeLimit) {
        OrderedMap<OrderedSet<Operation *>, OrderedSet<Operation *>, Hasher> predDict;
        for (auto &callOp : callOpList) {
            if (callOpSuccDict.count(callOp)) {
                auto succOps = callOpSuccDict[callOp];
                predDict[succOps].Insert(callOp);
            }
        }

        for (auto & [succSet, predSet] : predDict) {
            int optimizeCnt = succSet.size() * predSet.size() - succSet.size() - predSet.size();
            if (optimizeCnt > optimizeLimit) {
                auto dummyOp = MakeDummyCall();
                callOpList.push_back(dummyOp);

                callOpSuccDict[dummyOp] = succSet;
                for (auto &pred : predSet) {
                    callOpSuccDict[pred].Clear();
                    callOpSuccDict[pred].Insert(dummyOp);
                }

                callOpPredDict[dummyOp] = predSet.size();
                for (auto &succ: succSet) {
                    callOpPredDict[succ] -= predSet.size() - 1;
                }
            }
        }
    }

     void PrintColorGraph(int color) {
        ALOG_INFO_F("********** Call OP Graph **********\n");
        for (int i = 0; i < color; i++) {
            ALOG_INFO_F("%zu: %zu", i, colorOutGraph[i].size());
            ALOG_INFO_F("%s", IntVecToStr(colorOutGraph[i]).c_str());
        }
        int outCount = 0;
        for (int i = 0; i < color; i++) {
            outCount += colorOutGraph[i].size();
        }
        ALOG_INFO_F("total out: %d\n", outCount);
    }

    void FindRedundantEdges(int color, std::vector<std::vector<int>>& redundantColorOutGraph) {
        std::vector<int> tag(color);
        int tagValue = 2;
        for (int i = 0; i < color; i++) {
            std::vector<int> queue1, queue2;
            for (int j : colorOutGraph[i]) {
                tag[j] = tagValue;
            }
            for (int j : colorOutGraph[i]) {
                for (int k : colorOutGraph[j]) {
                    if (tag[k] == tagValue) {
                    tag[k] = 1;
                    redundantColorOutGraph[i].push_back(k);
                    } else if (tag[k] == 0) {
                    tag[k] = 1;
                    queue1.push_back(k);
                    }
                }
            }
            for (int j : queue1) {
                for (int k : colorOutGraph[j]) {
                    if (tag[k] == tagValue) {
                    tag[k] = 1;
                    redundantColorOutGraph[i].push_back(k);
                    }
                }
            }
            for (int j : colorOutGraph[i]) {
                tag[j] = 0;
            }
            for (int j : queue1) {
                tag[j] = 0;
            }
        }
    }

    void EraseRedundantColorEdges(std::vector<Operation *> &callopList) {
        int color = callopList.size();
        std::vector<std::vector<int>> redundantColorOutGraph(color);
        std::vector<int> tag(color);
        // Find redundant edges
        FindRedundantEdges(color, redundantColorOutGraph);
        // Erase redundant edges
        for (int i = 0; i < color; i++) {
            std::sort(redundantColorOutGraph[i].begin(), redundantColorOutGraph[i].end());
            // update color_out_graph
            std::vector<int> newGraph;
            size_t j = 0U;
            for (int k : redundantColorOutGraph[i]) {
                while (colorOutGraph[i][j] != k) {
                    newGraph.push_back(colorOutGraph[i][j]);
                    callOpSuccDict[callopList[i]].Insert(callopList[colorOutGraph[i][j]]);
                    callOpPredDict[callopList[colorOutGraph[i][j]]]++;
                    j++;
                }
                j++;
            }
            while (j < colorOutGraph[i].size()) {
                newGraph.push_back(colorOutGraph[i][j]);
                callOpSuccDict[callopList[i]].Insert(callopList[colorOutGraph[i][j]]);
                callOpPredDict[callopList[colorOutGraph[i][j]]]++;
                j++;
            }
            colorOutGraph[i] = newGraph;
        }
    }

    void AddDummyCallsAtBeginningAndEnding(std::vector<Operation *> &callopList) {
        static constexpr size_t OPTIMIZATION_THRESHOLD = 3;

        std::vector<Operation *> zeroPreds;
        std::vector<Operation *> zeroSuccs;
        for (auto *op : callopList) {
            if (callOpPredDict[op] == 0) {
                zeroPreds.push_back(op);
            }
            if (callOpSuccDict[op].empty()) {
                zeroSuccs.push_back(op);
            }
        }

        // Zero predecessors
        if (zeroPreds.size() >= OPTIMIZATION_THRESHOLD) {
            auto *dummyOp = MakeDummyCall();
            callopList.push_back(dummyOp);
            callOpPredDict[dummyOp] = 0;
            for (auto *op : zeroPreds) {
                ASSERT(callOpPredDict[op] == 0);
                callOpSuccDict[dummyOp].Insert(op);
                callOpPredDict[op] = 1;
            }
            ASSERT(callOpSuccDict[dummyOp].size() == zeroPreds.size());
        }

        // Zero successors
        if (zeroSuccs.size() >= OPTIMIZATION_THRESHOLD) {
            auto *dummyOp = MakeDummyCall();
            callopList.push_back(dummyOp);
            callOpSuccDict[dummyOp] = {};
            callOpPredDict[dummyOp] = zeroSuccs.size();
            for (auto *op : zeroSuccs) {
                ASSERT(callOpSuccDict[op].empty());
                callOpSuccDict[op].Insert(dummyOp);
            }
        }
    }

    EncodeDevAscendFunctionInfo(
            const std::unordered_map<uint64_t, int> &tHashIndexDict,
            const std::vector<CceCodeInfo> &tCceCodeInfoList,
            const SymbolicExpressionTable *tExpressionTable,
            Function *tdevRoot)
            : devRoot(tdevRoot),
              calleeHashIndexDict(tHashIndexDict),
              cceCodeInfoList(tCceCodeInfoList),
              expressionTable(tExpressionTable) {
        std::unordered_map<std::shared_ptr<LogicalTensor>, OrderedSet<Operation *>> consumerDict;
        std::unordered_map<Operation *, int> callopNumDict;

        rawName = devRoot->GetRawName();

        incastList = devRoot->GetIncast();
        outcastList = devRoot->GetOutcast();

        incastSet.insert(incastList.begin(), incastList.end());
        outcastSet.insert(outcastList.begin(), outcastList.end());

        std::vector<Operation *> callopList;
        for (auto &op : devRoot->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CALL) {
                callopNumDict[&op] = callopList.size();
                callopList.push_back(&op);

                for (auto &i : op.GetIOperands()) {
                    tensorList.Insert(i);
                    RecordRawTensor(i);
                    consumerDict[i].Insert(&op);
                }
                for (auto &o : op.GetOOperands()) {
                    tensorList.Insert(o);
                    RecordRawTensor(o);
                }
            }
        }
        for (auto &op : callopList) {
            callOpPredDict[op] = 0;
            callOpSuccDict[op].clear();
        }
        for (auto &op : callopList) {
            for (auto &o : op->GetOOperands()) {
                for (auto &consumer : consumerDict[o]) {
                    if (consumer->GetOpcode() != Opcode::OP_CALL) {
                        continue;
                    }
                    if (op == consumer) {
                        continue;
                    }
                    colorOutGraph[callopNumDict[op]].push_back(callopNumDict[consumer]);
                }
            }
        }
        for (size_t i = 0; i < callopList.size(); i++) {
            std::sort(colorOutGraph[i].begin(), colorOutGraph[i].end());
            colorOutGraph[i].resize(std::unique(colorOutGraph[i].begin(), colorOutGraph[i].end()) -
                                colorOutGraph[i].begin());
        }
        PrintColorGraph(callopList.size());
        EraseRedundantColorEdges(callopList);
        PrintColorGraph(callopList.size());

        removeRedundantCall(callopList);
        optimizeCallopSuccs(callopList, 10); // add dummp op at least 10 depends can be reduced

        AddDummyCallsAtBeginningAndEnding(callopList);

        std::unordered_map<Operation *, int> callopCoreTypeDict;
        for (auto &op : callopList) {
            auto callOpAttr = std::static_pointer_cast<CallOpAttribute>(op->GetOpAttribute());
            auto calleeHash = callOpAttr->GetCalleeHash().GetHash();
            ASSERT(calleeHashIndexDict.count(calleeHash));
            int cceIndex = calleeHashIndexDict.find(calleeHash)->second;
            ASSERT(cceIndex < static_cast<int>(cceCodeInfoList.size()));

            uint32_t coreType = cceCodeInfoList[cceIndex].coreType;
            ASSERT(coreType == static_cast<uint32_t>(CoreType::AIV) || coreType == static_cast<uint32_t>(CoreType::AIC) ||
                   coreType == static_cast<uint32_t>(CoreType::HUB));
            callopCoreTypeDict[op] = coreType;
        }
        static_assert(CoreType::AIV < CoreType::AIC);

        std::sort(callopList.begin(), callopList.end(), [&](Operation *lhs, Operation *rhs) {
            if (callOpPredDict[lhs] != callOpPredDict[rhs]) {
                return callOpPredDict[lhs] < callOpPredDict[rhs];
            }
            ASSERT(callopCoreTypeDict.count(lhs));
            ASSERT(callopCoreTypeDict.count(rhs));
            return callopCoreTypeDict[lhs] < callopCoreTypeDict[rhs];
        });

        totalZeroPred = callopList.size();
        for (size_t i = 0; i < callopList.size(); i++) {
            if (callOpPredDict[callopList[i]] != 0) {
                totalZeroPred = i;
                break;
            }
        }
        for (size_t i = totalZeroPred; i < callopList.size(); i++) {
            ASSERT(callOpPredDict[callopList[i]] != 0);
        }

        for (uint32_t i = 0; i < totalZeroPred; i++) {
            if (callopCoreTypeDict[callopList[i]] == static_cast<uint32_t>(CoreType::AIV)) {
                totalZeroPredAIV++;
            } else if (callopCoreTypeDict[callopList[i]] == static_cast<uint32_t>(CoreType::AIC)) {
                totalZeroPredAIC++;
            } else if (callopCoreTypeDict[callopList[i]] == static_cast<uint32_t>(CoreType::HUB)) {
                totalZeroPredHub++;
            } else {
                ASSERT(false);
            }
        }

        for (auto &op : callopList) {
            callList.Insert(op);
        }

        /*
         * As we need a reference of null, we use the 0-th element for the reference of null for DevAscendFunctionDuppedData
         * So outcastStitchCount starts from 1, as the 0-th is for the reference of null.
         */
        outcastStitchCount = 1;
        for (size_t i = 0; i < callList.size(); i++) {
            bool maybeStitch = false;
            if (callOpSuccDict[callList[i]].empty()) {
                maybeStitch = true;
            } else {
                for (auto &o : callList[i]->GetOOperands()) {
                    if (outcastSet.count(o)) {
                        maybeStitch = true;
                        break;
                    }
                }
            }
            if (maybeStitch) {
                outcastStitchIndexList.push_back(outcastStitchCount);
                outcastStitchCount++;
            } else {
                outcastStitchIndexList.push_back(0);
            }
        }

        EncodeIncasts();
        EncodeOutCasts();
    }

    void Init(DevAscendFunction *devFunc, const EncodeDevAscendFunctionParam &param, bool fillContent) {
        auto slot = param.slot;
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(&devFunc->data) - reinterpret_cast<uintdevptr_t>(devFunc);
        DevAscendFunctionPredInfo predInfo = {totalZeroPred, totalZeroPredAIV, totalZeroPredAIC, totalZeroPredHub};
        devFunc->sourceFunc = nullptr;
        devFunc->InitIncastOutcastAttr(initOffset, incastList, outcastList, fillContent);
        devFunc->InitOperationDynamicField(initOffset, predInfo, outcastStitchCount, calleeHashIndexDict,
            expressionTable, callList, incastList, outcastList, callOpSuccDict, fillContent);
        devFunc->InitRawTensorAndMemoryRequirement(initOffset, incastRawTensorList, outcastRawTensorList,
            rawTensorList, rawMagicToRawTensor, rawAttrs, param, expressionTable, fillContent);
        devFunc->InitTensor(initOffset, tensorList, rawTensorList, fillContent);
        devFunc->InitOperation(initOffset, expressionTable, callList, tensorList, rawTensorList, callOpPredDict,
            callOpSuccDict, calleeHashIndexDict, outcastStitchIndexList, fillContent);
        devFunc->InitIncastOutcast(initOffset, incastList, outcastList, tensorList, inoutOpAttrs, slot,
            rawName, fillContent);
    }
};

void EncodeDevAscendFunction(const EncodeDevAscendFunctionParam &param, uint64_t &offset, DevAscendFunction *base) {
    EncodeDevAscendFunctionInfo encodeInfo(param.calleeHashIndexDict, param.cceCodeInfoList, param.expressionTable, param.devRoot);

    if (base == nullptr) {
        DevAscendFunction devfunc;
        encodeInfo.Init(&devfunc, param, false);
        offset = devfunc.GetSize();
    } else {
        encodeInfo.Init(base, param, true);
        offset = base->GetSize();
    }
}

void DevAscendProgram::InitSymbolTable(
        uintdevptr_t &initOffset, SymbolicSymbolTable *symbolTableInput, bool fillContent) {
    symbolTable.HostInitDataSizeOffset(initOffset, symbolTableInput->GetSymbolTable().size());

    symbolTableNameList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t index = 0; index < symbolTableInput->GetSymbolTable().size(); index++) {
        std::string name = symbolTableInput->GetSymbolTable()[index];    
        ONFILLCONTENT {
            symbolTable[index].index = index;
        };
        ONFILLCONTENT {
            symbolTable[index].name.HostAssignRangeOffsetSize(symbolTableNameList, offset, name.size());
            memcpy_s(symbolTable[index].name.Data(), symbolTable[index].name.size(), name.c_str(), name.size());
        }
        offset += ALIGN_UP(name.size(), sizeof(uint64_t));
    }
    symbolTableNameList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitExpressionTableBinary(
        uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &expressionTableBinaryListInput, bool fillContent) {
    expressionTableOffsetList.HostInitDataSizeOffset(initOffset, expressionTableBinaryListInput.size());
    preGuardPage.HostInitDataSizeOffset(initOffset, PAGE_SIZE);
    ONFILLCONTENT { memset_s(preGuardPage.Data(), PAGE_SIZE, 0, PAGE_SIZE); }

    expressionTableBinary.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t i = 0; i < expressionTableBinaryListInput.size(); i++) {
        ONFILLCONTENT { expressionTableOffsetList[i] = offset; }
        ONFILLCONTENT { memcpy_s(expressionTableBinary.Data() + offset, expressionTableBinaryListInput[i].size(), expressionTableBinaryListInput[i].data(), expressionTableBinaryListInput[i].size()); }
        offset += expressionTableBinaryListInput[i].size();
    }
    expressionTableBinary.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitControlFlowBinary(
        uintdevptr_t &initOffset,
        const std::vector<uint8_t> &hostControlFlowBinaryInput, const std::vector<uint8_t> &devControlFlowBinaryInput,
        bool fillContent) {
    hostControlFlowBinary.HostInitDataSizeOffset(initOffset, hostControlFlowBinaryInput.size());
    ONFILLCONTENT { memcpy_s(hostControlFlowBinary.Data(), hostControlFlowBinaryInput.size(), hostControlFlowBinaryInput.data(), hostControlFlowBinaryInput.size()); }

    devControlFlowBinary.HostInitDataSizeOffset(initOffset, devControlFlowBinaryInput.size());
    ONFILLCONTENT { memcpy_s(devControlFlowBinary.Data(), devControlFlowBinaryInput.size(), devControlFlowBinaryInput.data(), devControlFlowBinaryInput.size()); }
}
void DevAscendProgram::InitDevEncodeList(
        uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &devEncodeListInput, bool fillContent) {
    devEncodeList.HostInitDataSizeOffset(initOffset, devEncodeListInput.size());
    devEncodeDataList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t i = 0; i < devEncodeListInput.size(); i++) {
        ONFILLCONTENT {
            devEncodeList[i].HostAssignRangeOffsetSize(devEncodeDataList, offset, devEncodeListInput[i].size());
        };
        ONFILLCONTENT {
            memcpy_s(devEncodeList[i].Data(), devEncodeList[i].size(), devEncodeListInput[i].data(),
                devEncodeListInput[i].size());
        };
        offset += ALIGN_UP(devEncodeListInput[i].size(), sizeof(uint64_t));
    }
    devEncodeDataList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitCceCodeList(
        uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &cceCodeListInput,
    const std::vector<CceCodeInfo> &cceInfo, bool fillContent) {
    cceCodeList.HostInitDataSizeOffset(initOffset, cceCodeListInput.size());
    cceCodeDataList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t i = 0; i < cceCodeListInput.size(); i++) {
        ONFILLCONTENT {
            cceCodeList[i].binary.HostAssignRangeOffsetSize(cceCodeDataList, offset, cceCodeListInput[i].size());
            cceCodeList[i].coreType = cceInfo[i].coreType;
            cceCodeList[i].psgId = cceInfo[i].psgId;
            cceCodeList[i].funcHash = cceInfo[i].funcHash;
        };
        ONFILLCONTENT {
            memcpy_s(cceCodeList[i].binary.Data(), cceCodeList[i].binary.size(), cceCodeListInput[i].data(),
                cceCodeListInput[i].size());
        };
        offset += ALIGN_UP(cceCodeListInput[i].size(), sizeof(uint64_t));
    }
    cceCodeDataList.HostInitDataSizeOffset(initOffset, offset);
}

void DevAscendProgram::InitPrefetchInfoList(uintdevptr_t &initOffset, const std::vector<L2Info> &l2InfoList,
    bool fillContent) {
    prefetchInfoList.HostInitDataSizeOffset(initOffset, l2InfoList.size());
    for (size_t i = 0; i < l2InfoList.size(); i++) {
        ONFILLCONTENT { memcpy_s(&prefetchInfoList[i], sizeof(PrefetchInfo), &l2InfoList[i], sizeof(L2Info)); };
    }
    return;
}

void DevAscendProgram::InitDisableL2List(uintdevptr_t &initOffset, const std::vector<uint8_t> &disableL2,
                                         bool fillContent) {
  disableL2List.HostInitDataSizeOffset(initOffset, disableL2.size());
  ONFILLCONTENT { (void)memcpy_s(disableL2List.Data(), disableL2.size(), disableL2.data(), disableL2.size()); };
  return;
}

void DevAscendProgram::InitStartArgsABIParamList(
        uintdevptr_t &initOffset,
        const std::vector<int> &tStartArgsInputTensorSlotIndexList,
        const std::vector<int> &tStartArgsOutputTensorSlotIndexList,
        const std::vector<int> &tStartArgsInputSymbolIndexList,
        const std::vector<SymbolHandler> &tStartArgsSymbolHandlerList,
        const std::vector<int> &tAsembleSlotIndexList,
        const std::vector<int> &tInplaceSlotIndexList,
        bool fillContent) {
    this->startArgsInputTensorSlotIndexList.HostInitDataSizeOffset(initOffset, tStartArgsInputTensorSlotIndexList.size());
    this->startArgsOutputTensorSlotIndexList.HostInitDataSizeOffset(initOffset, tStartArgsOutputTensorSlotIndexList.size());
    this->startArgsInputSymbolIndexList.HostInitDataSizeOffset(initOffset, tStartArgsInputSymbolIndexList.size());
    this->startArgsSymbolHandlerList.HostInitDataSizeOffset(initOffset, tStartArgsSymbolHandlerList.size());
    this->assembleSlotIndexList.HostInitDataSizeOffset(initOffset, tAsembleSlotIndexList.size());
    this->inplaceSlotList.HostInitDataSizeOffset(initOffset, tInplaceSlotIndexList.size());

    ONFILLCONTENT {
        for (size_t i = 0; i < tStartArgsInputTensorSlotIndexList.size(); i++) {
            this->startArgsInputTensorSlotIndexList[i] = tStartArgsInputTensorSlotIndexList[i];
        }
        for (size_t i = 0; i < tStartArgsOutputTensorSlotIndexList.size(); i++) {
            this->startArgsOutputTensorSlotIndexList[i] = tStartArgsOutputTensorSlotIndexList[i];
        }
        for (size_t i = 0; i < tStartArgsInputSymbolIndexList.size(); i++) {
            this->startArgsInputSymbolIndexList[i] = tStartArgsInputSymbolIndexList[i];
        }
        for (size_t i = 0; i < tStartArgsSymbolHandlerList.size(); i++) {
            this->startArgsSymbolHandlerList[i] = tStartArgsSymbolHandlerList[i];
        }
        for (size_t i = 0; i < tAsembleSlotIndexList.size(); i++) {
            this->assembleSlotIndexList[i] = tAsembleSlotIndexList[i];
        }
        for (size_t i = 0; i < tInplaceSlotIndexList.size(); i++) {
            this->inplaceSlotList[i] = tInplaceSlotIndexList[i];
        }
    }
}

struct EncodeDevAscendProgramInfo {
    Function *func;
    std::shared_ptr<DyndevFunctionAttribute> dyndevAttr;

    explicit EncodeDevAscendProgramInfo(Function *tfunc) : func(tfunc) {
        ASSERT(func->GetDyndevAttribute() != nullptr);
        dyndevAttr = func->GetDyndevAttribute();
    }

    void Init(DevAscendProgram *devProg, bool fillContent) {
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(devProg->data);
        devProg->slotSize = dyndevAttr->inoutLink.totalSlot;
        devProg->InitSymbolTable(initOffset, &dyndevAttr->symbolTable, fillContent);
        devProg->InitExpressionTableBinary(initOffset, dyndevAttr->expressionTableBinaryList, fillContent);
        uint64_t expressionTableSize = 0;
        for (auto &[root, exprTable] : dyndevAttr->exprTableDictGroup.devRootCoaDict) {
            (void) root;
            expressionTableSize = std::max(expressionTableSize, (uint64_t)exprTable.GetPrimaryExpressionSize());
        }
        devProg->expressionTableSize = expressionTableSize;
        devProg->InitControlFlowBinary(initOffset, dyndevAttr->hostControlFlowBinary, dyndevAttr->devControlFlowBinary,
            fillContent);
        devProg->InitDevEncodeList(initOffset, dyndevAttr->devEncodeList, fillContent);
        devProg->InitCceCodeList(initOffset, dyndevAttr->cceCodeList, dyndevAttr->cceCodeInfo, fillContent);
        devProg->InitStartArgsABIParamList(
            initOffset,
            dyndevAttr->inoutLink.inputSlotIndexList,
            dyndevAttr->inoutLink.outputSlotIndexList,
            dyndevAttr->startArgsInputSymbolIndexList,
            dyndevAttr->startArgsSymbolHandlerList,
            dyndevAttr->inoutLink.assembleSlotIndexList,
            dyndevAttr->inoutLink.inplaceSlotIndexList,
            fillContent);
        devProg->InitPrefetchInfoList(initOffset, dyndevAttr->l2InfoList, fillContent);
        devProg->InitDisableL2List(initOffset, dyndevAttr->disableL2List, fillContent);
    }
};

struct LocalWorkspaceResult {
    uint64_t memReq{0};
    uint64_t rootFuncStandardMemReq{0};
    uint64_t slotStandardMemReq{0};
    uint64_t slotPoolSize{0};
    uint64_t globalTensorMem{0};
    uint64_t standardStackWorkspacePerCore{0};
};

static int EstimatedStitchingCount() {
    static constexpr int DEFAULT_VALUE = 5;
    static int value = config::GetHostConfig("estimated_stitching_count", DEFAULT_VALUE);
    ASSERT(value > 0);
    return value;
}

static int WorkspaceRecyclePeriod() {
    static constexpr int DEFAULT_VALUE = 5;
    static int value = config::GetHostConfig("workspace_recycle_period", DEFAULT_VALUE);
    ASSERT(value > 0);
    return value;
}

static LocalWorkspaceResult CalcAicoreLocalWorkspace(DevAscendProgram &devProg) {
    LocalWorkspaceResult res;
    struct SlotInfo {
        bool isOutputSlot{false};
        bool asWriteSlot{false};
        bool isAssemble{false};
    };
    std::vector<SlotInfo> slots(devProg.slotSize);

    std::vector<int> outputSlotIdxList = devProg.GetOutputTensorSlotIndexList();
    for (int outputSlotIdx : outputSlotIdxList) {
        slots[outputSlotIdx].isOutputSlot = true;
    }
    for (auto slotIdx : devProg.GetAssembleTensorSlotIndexList()) {
        slots[slotIdx].isAssemble = true;
    }

    auto isOutputSlot = [&slots](auto func, auto idx) {
        auto &toSlotList = func->GetOutcast(idx).toSlotList;
        for (size_t j = 0; j < toSlotList.size(); j++) {
            int slotIdx = func->At(toSlotList, j);
            if (slots[slotIdx].isOutputSlot) {
                return true;
            }
        }
        return false;
    };
    auto isAssembleSlot = [&slots](auto func, auto idx) {
        auto &toSlotList = func->GetOutcast(idx).toSlotList;
        for (size_t j = 0; j < toSlotList.size(); j++) {
            int slotIdx = func->At(toSlotList, j);
            if (slots[slotIdx].isAssemble) {
                return true;
            }
        }
        return false;
    };

    uint64_t maxInnerWorkspace = 0;
    uint64_t maxOutcastWorkspace = 0;
    uint64_t maxSlotMemReq = 0;
    uint64_t globalTensorMem = 0;

    uint64_t maxStackWorkspace = 0;
    for (auto &&devEncodeData : devProg.devEncodeList) {
        DevAscendFunction *devFunc = reinterpret_cast<DevAscendFunction *>(devEncodeData.Data());
        for (size_t i = 0; i < devFunc->GetOutcastSize(); i++) {
            if (isOutputSlot(devFunc, i)) {
                continue;
            }

            if (isAssembleSlot(devFunc, i)) {
                globalTensorMem += devFunc->GetOutcastRawTensor(i)->memoryRequirement;
                continue;
            }

            auto &toSlotList = devFunc->GetOutcast(i).toSlotList;
            for (size_t j = 0; j < toSlotList.size(); j++) {
                int slotIdx = devFunc->At(toSlotList, j);
                // No output slot
                slots[slotIdx].asWriteSlot = true;
            }
            maxSlotMemReq = std::max(maxSlotMemReq, devFunc->GetOutcastRawTensor(i)->memoryRequirement);
        }

        maxInnerWorkspace = std::max(maxInnerWorkspace, devFunc->rawTensorWsMemoryRequirement);
        maxOutcastWorkspace = std::max(maxOutcastWorkspace, devFunc->outcastWsMemoryRequirement);
        maxStackWorkspace = std::max(maxStackWorkspace, static_cast<uint64_t>(devFunc->stackWorkSpaceSize));
    }

    res.slotStandardMemReq = maxSlotMemReq;
    res.globalTensorMem = globalTensorMem;
    res.rootFuncStandardMemReq = maxInnerWorkspace;

    size_t slotsNeedAlloc = std::count_if(slots.begin(), slots.end(), [](const SlotInfo &slot) {
        return slot.asWriteSlot;
    });
    res.slotPoolSize = slotsNeedAlloc * SLOTS_NEED_ALLOC_SIZE;

    res.standardStackWorkspacePerCore = maxStackWorkspace;

    uint64_t memReq = res.slotStandardMemReq * res.slotPoolSize +
        maxInnerWorkspace * WorkspaceRecyclePeriod() +
        maxOutcastWorkspace * EstimatedStitchingCount() +
        res.standardStackWorkspacePerCore * MAX_WORKSPACE_MUL_SIZE +
        res.globalTensorMem;

    static constexpr uint64_t MASK_32K = 32 * 1024 - 1;
    res.memReq = ((memReq + MASK_32K) & ~MASK_32K); // aligned to 32K

    return res;
}

static uint64_t CalcAicpuCoherentWorkspace(DevAscendProgram &devProg) {
    (void)devProg;
    static constexpr uint64_t AICPU_COHERENT_WS_SIZE = 6 * 1024 * 1024;
    return AICPU_COHERENT_WS_SIZE;
}

static uint64_t CalcStitchWorkspace(DevAscendProgram &devProg) {
    (void)devProg;
    static constexpr uint64_t AICPU_STITCH_SIZE = 2 * 1024 * 1024;
    return AICPU_STITCH_SIZE;
}

void EncodeDevAscendProgram(Function *func, uint64_t &offset, DevAscendProgram *base) {
    EncodeDevAscendProgramInfo encodeInfo(func);

    if (base == nullptr) {
        DevAscendProgram devfunc;
        encodeInfo.Init(&devfunc, false);
        offset = devfunc.GetSize();
    } else {
        encodeInfo.Init(base, true);
        offset = base->GetSize();

        // Calc workspace size
        LocalWorkspaceResult aicoreLocalWs = CalcAicoreLocalWorkspace(*base);
        base->aicoreLocalWorkspaceSize = aicoreLocalWs.memReq;
        base->rootFuncStandardMemReq = aicoreLocalWs.rootFuncStandardMemReq;
        base->slotStandardMemReq = aicoreLocalWs.slotStandardMemReq;
        base->slotPoolSize = aicoreLocalWs.slotPoolSize;
        base->globalTensorMem = aicoreLocalWs.globalTensorMem;
        base->standardStackWorkspacePerCore = aicoreLocalWs.standardStackWorkspacePerCore;

        base->aicpuCoherentWorkspaceSize = CalcAicpuCoherentWorkspace(*base);
        base->stitchPoolSize = CalcStitchWorkspace(*base);
        base->devArgs.machineConfig = func->paramConfigs_.machineConfig_;
        base->workspaceRecyclePeriod = WorkspaceRecyclePeriod();
    }
}
} // namespace dynamic
} // namespace npu::tile_fwk
