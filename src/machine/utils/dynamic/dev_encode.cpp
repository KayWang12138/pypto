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
    ALOG_INFO("Pred: zero=", predInfo.totalZeroPred, " aiv=", predInfo.totalZeroPredAIV,
        " aic=", predInfo.totalZeroPredAIC, " hub=", predInfo.totalZeroPredHub, " aicpu=", predInfo.totalZeroPredAicpu);

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

    uint64_t maxPossibleNumel = 0;
    if (std::find(rawTensor->oriRawshape.begin(), rawTensor->oriRawshape.end(), -1) == rawTensor->oriRawshape.end()) {
        maxPossibleNumel = std::max(maxPossibleNumel, std::accumulate(rawTensor->oriRawshape.begin(),
                            rawTensor->oriRawshape.end(), UINT64_C(1), std::multiplies<uint64_t>()));
    }
    if (std::find(rawTensor->rawshape.begin(), rawTensor->rawshape.end(), -1) == rawTensor->rawshape.end()) {
        maxPossibleNumel = std::max(maxPossibleNumel, std::accumulate(rawTensor->rawshape.begin(),
            rawTensor->rawshape.end(), UINT64_C(1), std::multiplies<uint64_t>()));
    }
    encoded->maxPossibleMemReq = std::max(maxPossibleNumel *BytesOf(rawTensor->GetDataType()),
                                 encoded->memoryRequirement);
}

void DevAscendFunction::FillOutputSlotMark(const IncastOutcastLink *inoutLink, std::vector<bool>& isOutputSlotMarks) {
    for (int slotIdx : inoutLink->outputSlotIndexList) {
        isOutputSlotMarks[slotIdx] = true;
    }
    for (int slotIdx: inoutLink->assembleSlotIndexList) {
        isOutputSlotMarks[slotIdx] = true;
    }
    for (int slotIdx: inoutLink->partialUpdateSlotIdexList) {
        isOutputSlotMarks[slotIdx] = true;
    }
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
        FillOutputSlotMark(inoutLink, isOutputSlotMarks);
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
            encoded.rawMagic = rawTensor->GetRawMagic();
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
                    outcastWsMemoryRequirement += encoded.maxPossibleMemReq;
                }
            } else {
                // For workspace tensors, the memoryRequirement property is deprecated, please don't use its value
                encoded.ioProperty = DevIOProperty::NONE;
                encoded.ioIndex = -1;
#if DEBUG_INFINITE_LIFETIME
                UNUSED(rawAttrs);
                encoded.addrOffset = rawTensorWsMemoryRequirement;
                rawTensor->addrOffset = encoded.addrOffset;
                rawTensorWsMemoryRequirement += encoded.maxPossibleMemReq;
#else
                encoded.addrOffset = rawAttrs[i].storage->start_ + rawAttrs[i].storageOffset;
                rawTensor->addrOffset = encoded.addrOffset;
                rawTensorWsMemoryRequirement = std::max(rawTensorWsMemoryRequirement,
                    rawAttrs[i].storage->start_ + rawAttrs[i].storage->length_);
#endif
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
                auto rawTensorRawShape = rawTensor->GetRawShape();
                bool isDynamicShape = false;
                for (auto dimShape : rawTensorRawShape) {
                    if (dimShape < 0) {
                        isDynamicShape = true;
                    }
                }
                if (isDynamicShape) continue;
                ASSERT(rawTensor->GetRawShapeSize() == actualRaw->GetRawShapeSize());
                ASSERT(rawTensor->GetRawDataSize() == actualRaw->GetRawDataSize());
            }
        }

        // file linkedIncastId
        auto outIncastLinkMap = param.devRoot->outIncastLinkMap;
        ALOG_DEBUG_F("devRoot is %s", param.devRoot->GetRawName().c_str());
        for (size_t i = 0; i < rawList.size(); i++) {
            auto &encoded = *GetRawTensor(i);
            if (outIncastLinkMap.find(rawList[i]) != outIncastLinkMap.end()) {
                encoded.linkedIncastId = incastRawList.GetIndex(outIncastLinkMap[rawList[i]]); //换成incast的下标 ioidx
                ALOG_DEBUG_F("linkedIncastId is %d", encoded.linkedIncastId);
            } else {
                encoded.linkedIncastId = -1;
                ALOG_DEBUG_F("linkedIncastId is %d", encoded.linkedIncastId);
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
        const std::vector<int> &noPredOpList,
        const std::vector<int> &noSuccOpList,
        bool fillContent) {
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
            staticField.debugOpmagic = op->GetOpMagic();
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
            staticField.depGraphSuccList.AssignRangeOffsetSize(operationSuccList_, succSize, opSuccSize);
            for (int j = 0; j < opSuccSize; j++) {
                int succ = callList.GetIndex(callOpSuccDict.find(op)->second[j]);
                At(staticField.depGraphSuccList, j) = succ;
                At(operationList_, succ).depGraphPredCount++;
                dupData->GetOperationCurrPredCount(succ)++;
            }
            succSize += opSuccSize;
        }
        for (size_t i = 0; i < callList.size(); i++) {
            Operation *op = callList[i];
            ASSERT(callOpPredDict.count(op));
            ASSERT(At(operationList_, i).depGraphPredCount == callOpPredDict.find(op)->second);
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
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &incastOpAttrDict,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &outcastOpAttrDict,
        const IncastOutcastSlot *slot, const std::string &initRawName, bool fillContent) {
    {
        // Fill metadata
        incastList.HostInitDataSizeOffset(initOffset, incastTensorList.size());
        outcastList.HostInitDataSizeOffset(initOffset, outcastTensorList.size());
        for (size_t i = 0; i < incastTensorList.size(); i++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[i]);
            auto &incast = At(incastList, i);
            ONFILLCONTENT {
                incast.tensorIndex = tlist.GetIndex(incastTensorList[i]);
                incast.dim = inAttr.dim;
                incast.cellMatchTableDesc = inAttr.cellMatchTableDesc;
            }
        }
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[i]);
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.tensorIndex = tlist.GetIndex(outcastTensorList[i]);
                outcast.dim = outAttr.dim;
                outcast.cellMatchTableDesc = outAttr.cellMatchTableDesc;
            }
        }
    }
    {
        // Fill slot list
        slotList.HostInitDataSizeOffset(initOffset, 0);
        uint64_t slotSize = 0;
        for (size_t i = 0; i < incastTensorList.size(); i++) {
            auto &incast = At(incastList, i);
            ONFILLCONTENT {
                incast.fromSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->incastSlot[i].size());
                for (size_t j = 0; j < slot->incastSlot[i].size(); j++) {
                    At(incast.fromSlotList, j) = slot->incastSlot[i][j];
                }
            }
            slotSize += slot->incastSlot[i].size();
        }
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.toSlotList.AssignRangeOffsetSize(slotList, slotSize, slot->outcastSlot[i].size());
                for (size_t j = 0; j < slot->outcastSlot[i].size(); j++) {
                    At(outcast.toSlotList, j) = slot->outcastSlot[i][j];
                }
            }
            slotSize += slot->outcastSlot[i].size();
        }
        slotList.HostInitDataSizeOffset(initOffset, slotSize);
    }
    {
        // Fill use list
        useList.HostInitDataSizeOffset(initOffset, 0);
        uint64_t useSize = 0;
        for (size_t i = 0; i < incastTensorList.size(); i++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[i]);
            auto &incast = At(incastList, i);
            ONFILLCONTENT {
                incast.consumerList.AssignRangeOffsetSize(useList, useSize, inAttr.useList.size());
                for (size_t j = 0; j < inAttr.useList.size(); j++) {
                    At(incast.consumerList, j) = inAttr.useList[j];
                }
            }
            useSize += inAttr.useList.size();
        }
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[i]);
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.producerList.AssignRangeOffsetSize(useList, useSize, outAttr.useList.size());
                for (size_t j = 0; j < outAttr.useList.size(); j++) {
                    At(outcast.producerList, j) = outAttr.useList[j];
                }
            }
            useSize += outAttr.useList.size();
        }
        useList.HostInitDataSizeOffset(initOffset, useSize);
    }
    {
        // Fill runtime full update table
        uint64_t cellMatchSizeOutcastTotal = 0;
        cellMatchRuntimeFullUpdateTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[i]);
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.cellMatchRuntimeFullUpdateTable.AssignRangeOffsetSize(cellMatchRuntimeFullUpdateTableList, cellMatchSizeOutcastTotal, outAttr.cellMatchSize);
                for (int j = 0; j < outAttr.cellMatchSize; j++) {
                    At(outcast.cellMatchRuntimeFullUpdateTable, j) = (uint32_t)-1;
                };
            }
            cellMatchSizeOutcastTotal += outAttr.cellMatchSize;
        }
        cellMatchRuntimeFullUpdateTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeOutcastTotal);
    }
    {
        // Fill stitchPolicyFullCoverProducerList
        uint64_t fullCoverTotal = 0;
        stitchPolicyFullCoverProducerList_.HostInitDataSizeOffset(initOffset, 0);
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[i]);
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.stitchPolicyFullCoverProducerHubOpIdx = outAttr.stitchPolicyFullCoverProducerHubOpIdx;
                outcast.stitchPolicyFullCoverProducerList.AssignRangeOffsetSize(stitchPolicyFullCoverProducerList_, fullCoverTotal, outAttr.stitchPolicyFullCoverProducerList.size());
                for (size_t j = 0; j < outAttr.stitchPolicyFullCoverProducerList.size(); j++) {
                    At(outcast.stitchPolicyFullCoverProducerList, j) = outAttr.stitchPolicyFullCoverProducerList[j];
                };
            }
            fullCoverTotal += outAttr.stitchPolicyFullCoverProducerList.size();
        }
        stitchPolicyFullCoverProducerList_.HostInitDataSizeOffset(initOffset, fullCoverTotal);
    }
    {
        // Fill stitchPolicyFullCoverOpList_
        uint32_t fullCoverOpTotal = 0;
        stitchPolicyFullCoverOpList_.HostInitDataSizeOffset(initOffset, 0);
        for (size_t i = 0; i < incastTensorList.size(); i++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[i]);
            auto &incast = At(incastList, i);
            ONFILLCONTENT {
                incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignRangeOffsetSize(stitchPolicyFullCoverOpList_, fullCoverOpTotal, inAttr.useOpList.size());
                for (size_t j = 0; j < inAttr.useOpList.size(); j++) {
                    At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, j) = inAttr.useOpList[j];
                }
            }
            fullCoverOpTotal += inAttr.useOpList.size();
        }
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[i]);
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.stitchPolicyFullCoverProducerAllOpIdxList.AssignRangeOffsetSize(stitchPolicyFullCoverOpList_, fullCoverOpTotal, outAttr.useOpList.size());
                for (size_t j = 0; j < outAttr.useOpList.size(); j++) {
                    At(outcast.stitchPolicyFullCoverProducerAllOpIdxList, j) = outAttr.useOpList[j];
                }
            }
            fullCoverOpTotal += outAttr.useOpList.size();
        }
        stitchPolicyFullCoverOpList_.HostInitDataSizeOffset(initOffset, fullCoverOpTotal);
    }
    {
        // Fill incast & outcast all full update table
        uint64_t cellMatchSizeIncastTotal = 0;
        cellMatchStaticIncastTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t i = 0; i < incastTensorList.size(); i++) {
            auto &inAttr = incastOpAttrDict.at(incastTensorList[i]);
            auto &incast = At(incastList, i);
            ONFILLCONTENT {
                incast.cellMatchStaticIncastTable.AssignRangeOffsetSize(cellMatchStaticIncastTableList, cellMatchSizeIncastTotal, inAttr.cellMatchSize);

                auto consumerList = &At(incast.consumerList, 0);
                auto tableData = &At(incast.cellMatchStaticIncastTable, 0);
                bool stitchByAllFullMatch = CellMatchFillIncastOutcast<true>(
                        consumerList, incast.consumerList.size(), nullptr, true, incast.cellMatchTableDesc, tableData);
                incast.stitchByAllFullMatch = stitchByAllFullMatch;
            };
            cellMatchSizeIncastTotal += inAttr.cellMatchSize;
        }
        cellMatchStaticIncastTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeIncastTotal);

        uint64_t cellMatchSizeOutcastTotal = 0;
        cellMatchStaticOutcastTableList.HostInitDataSizeOffset(initOffset, 0);
        for (size_t i = 0; i < outcastTensorList.size(); i++) {
            auto &outAttr = outcastOpAttrDict.at(outcastTensorList[i]);
            auto &outcast = At(outcastList, i);
            ONFILLCONTENT {
                outcast.cellMatchStaticOutcastTable.AssignRangeOffsetSize(cellMatchStaticOutcastTableList, cellMatchSizeOutcastTotal, outAttr.cellMatchSize);

                auto producerList = &At(outcast.producerList, 0);
                auto tableData = &At(outcast.cellMatchStaticOutcastTable, 0);
                bool stitchByAllFullMatch = CellMatchFillIncastOutcast<true>(
                        producerList, outcast.producerList.size(), nullptr, false, outcast.cellMatchTableDesc, tableData);
                outcast.stitchByAllFullMatch = stitchByAllFullMatch;
            }
            cellMatchSizeOutcastTotal += outAttr.cellMatchSize;
        }
        cellMatchStaticOutcastTableList.HostInitDataSizeOffset(initOffset, cellMatchSizeOutcastTotal);
    }

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
    uint64_t totalZeroPredAicpu{0};

    std::unordered_map<Operation *, uint64_t> callOpPredDict;
    std::unordered_map<Operation *, OrderedSet<Operation *>> callOpSuccDict;
    std::unordered_map<int, std::vector<int>> colorOutGraph;
    std::vector<std::shared_ptr<Operation>> dummyOpList;

    std::vector<int> noSuccOpList;
    std::vector<int> noPredOpList;

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

    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> incastOpAttrDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> outcastOpAttrDict;

    static DevShape InitShape(const std::vector<int64_t> &shape) {
        DevShape initShape;
        initShape.dimSize = shape.size();
        for (size_t i = 0; i < DEV_SHAPE_DIM_MAX; i++) {
            if (i < shape.size()) {
                initShape.dim[i] = shape[i];
            } else {
                initShape.dim[i] = 0;
            }
        }
        return initShape;
    }
    static DevAscendStride InitStride(const std::vector<int64_t> &stride) {
        DevAscendStride initStride;
        initStride.dimSize = stride.size();
        for (size_t i = 0; i < DEV_SHAPE_DIM_MAX; i++) {
            if (i < stride.size()) {
                initStride.dimStride[i] = stride[i];
            } else {
                initStride.dimStride[i] = 0;
            }
        }
        return initStride;
    }
    static DevCellMatchTableDesc InitCellMatchTableDesc(const std::vector<int64_t> &shape, const std::vector<int64_t> &stride) {
        DevCellMatchTableDesc desc = {
            InitShape(shape),
            InitStride(stride),
        };
        return desc;
    }

    std::vector<int> ShapeToVector(const DevShape &shape) {
        std::vector<int> data(&shape.dim[0], &shape.dim[shape.dimSize]);
        return data;
    }
    std::vector<int> StrideToVector(const DevAscendStride &stride) {
        std::vector<int> data(&stride.dimStride[0], &stride.dimStride[stride.dimSize]);
        return data;
    }

    void UpdateCellMatchShape(DevCellMatchTableDesc &cellMatchTableDesc, const std::vector<int64_t> &shape) {
        auto &cellMatchShape = cellMatchTableDesc.cellShape;
        for (size_t i = 0; i < shape.size(); ++i) {
            auto dimValue = shape[i];
            if (cellMatchShape.dim[i] > dimValue) {
                cellMatchShape.dim[i] = dimValue;
                DEV_ASSERT(cellMatchShape.dim[i]);
            }
        }
    }

    void UpdateCellMatchStrideAndSize(int &cellMatchSize, DevCellMatchTableDesc &cellMatchTableDesc, const std::shared_ptr<LogicalTensor> &tensor, int dim) {
        auto &cellMatchShape = cellMatchTableDesc.cellShape;
        auto &cellMatchStride = cellMatchTableDesc.stride;

        cellMatchSize = 1;
        cellMatchStride.dimSize = dim;
        for (int l = (dim - 1); l >= 0; --l) {
            auto tiles = tensor->shape[l] / cellMatchShape.dim[l];
            if (tensor->shape[l] % cellMatchShape.dim[l] != 0) {
                // should not happen
                tiles += 1;
            }
            cellMatchSize *= tiles;
            cellMatchStride[l] = cellMatchSize;
        }
        ALOG_DEBUG_F("incast %d raw %d shape %s | cellMatchSize %d cellMatchShape %s cellMatchStride %s\n", tensor->magic, tensor->GetRawMagic(),
            IntVecToStr(tensor->shape).c_str(),
            cellMatchSize,
            IntVecToStr(ShapeToVector(cellMatchShape)).c_str(),
            IntVecToStr(StrideToVector(cellMatchStride)).c_str());
    }

    void RecordRawTensor(const std::shared_ptr<LogicalTensor> &tensor) {
        if (rawTensorList.Insert(tensor->GetRawTensor())) {
            EncodeRawTensorAttr &attr = rawAttrs.emplace_back();
            attr.storage = tensor->storage_;
            attr.storageOffset = tensor->storageOffset_;
            rawMagicToRawTensor[tensor->GetRawTensor()->rawmagic] = tensor->GetRawTensor();
        }
    }

    void EncodeAnalysisOpUseOutCasts(const std::shared_ptr<LogicalTensor>& o, std::set<uint32_t>& allOutcastUseOpSet, InoutOperationAttr& outcastOpAttr) {
        std::set<uint32_t> outcastUseOpSet;
        int dim = outcastOpAttr.dim;
        for (size_t j = 0; j < callList.size(); j++) {
            auto &op = *callList[j];
            auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
            std::vector<DevAscendFunctionCallOperandUse> useList;
            DevAscendFunctionCallOperandUse stitchPolicyFullCoverProducer;
            for (size_t k = 0; k < op.GetOOperands().size(); ++k) {
                auto &oOperand = op.GetOOperands()[k];
                if (o->tensor->rawmagic != oOperand->tensor->rawmagic) {
                    continue;
                }

                auto coaIndex = op.GetOOpAttrOffset(k) + COA_INDEX_DIM_BASE;
                std::vector<int64_t> offset = callAttr->GetLinearImmediateArgList(coaIndex, coaIndex + dim, true);
                std::vector<int64_t> shape = callAttr->GetLinearImmediateArgList(coaIndex + dim, coaIndex + dim * 0x2, false);
                if (offset == std::vector<int64_t>(dim, 0) && shape == oOperand->GetShape()) {
                    stitchPolicyFullCoverProducer = DevAscendFunctionCallOperandUse(j, k, coaIndex, coaIndex + dim);
                } else {
                    useList.emplace_back(j, k, coaIndex, coaIndex + dim);
                }
                ALOG_DEBUG_F("outcast oOperandIdx for outcast %d %d is %d", o->magic, o->GetRawMagic(), k);
                outcastUseOpSet.insert(j);
            }

            if (stitchPolicyFullCoverProducer.operationIdx != -1) {
                outcastOpAttr.stitchPolicyFullCoverProducerList.push_back(stitchPolicyFullCoverProducer);
            } else {
                outcastOpAttr.useList.insert(outcastOpAttr.useList.end(), useList.begin(), useList.end());
                for (auto &[operationIdx, operandIdx, offsetAttrIdx, shapeAttrIdx] : useList) {
                    UNUSED(operationIdx);
                    UNUSED(operandIdx);
                    UNUSED(offsetAttrIdx);
                    auto shape = callAttr->GetLinearImmediateArgList(shapeAttrIdx, shapeAttrIdx + dim, false);
                    UpdateCellMatchShape(outcastOpAttr.cellMatchTableDesc, shape);
                    ALOG_DEBUG_F("minimal shape for outcast %d raw %d op %d %d is %s\n", o->magic, o->GetRawMagic(), j,
                        op.GetOpMagic(), IntVecToStr(ShapeToVector(outcastOpAttr.cellMatchTableDesc.cellShape)).c_str());
                }
            }
        }
        UpdateCellMatchStrideAndSize(outcastOpAttr.cellMatchSize, outcastOpAttr.cellMatchTableDesc, o, dim);
        outcastOpAttr.useOpList.insert(outcastOpAttr.useOpList.end(), outcastUseOpSet.begin(), outcastUseOpSet.end());
        allOutcastUseOpSet.insert(outcastUseOpSet.begin(), outcastUseOpSet.end());
        outcastOpAttrDict.insert({o, outcastOpAttr});
    }

    void EncodeOutCasts() {
        std::set<uint32_t> allOutcastUseOpSet;
        for (auto &o : outcastList) {
            tensorList.Insert(o);
            outcastRawTensorList.Insert(o->GetRawTensor());
            RecordRawTensor(o);
            InoutOperationAttr outcastOpAttr;
            auto dim = o->shape.size();
            outcastOpAttr.dim = dim;
            outcastOpAttr.cellMatchTableDesc = InitCellMatchTableDesc(o->GetShape(), std::vector<int64_t>(dim, 1));
            EncodeAnalysisOpUseOutCasts(o, allOutcastUseOpSet, outcastOpAttr);
        }

        // Add edge from all stitchPolicyFullCoverProducerList's node to the single node
        size_t hubEntryLeast = 2;
        for (auto &o : outcastList) {
            auto &outcastOpAttr = outcastOpAttrDict[o];
            if (outcastOpAttr.stitchPolicyFullCoverProducerList.size() > hubEntryLeast) {
                outcastOpAttr.stitchPolicyFullCoverProducerHubOpIdx = callList.size();

                auto dummyOp = MakeDummyCall();
                callList.Insert(dummyOp);

                callOpPredDict[dummyOp] = outcastOpAttr.stitchPolicyFullCoverProducerList.size();
                for (auto &producer : outcastOpAttr.stitchPolicyFullCoverProducerList) {
                    auto callOp = callList[producer.operationIdx];
                    callOpSuccDict[callOp].Insert(dummyOp);
                }
                callOpSuccDict[dummyOp].Clear();
            } else {
                outcastOpAttr.stitchPolicyFullCoverProducerHubOpIdx = -1;
            }
        }

        std::set<int> noSuccOpSet;
        for (size_t i = 0; i < callList.size(); i++) {
            Operation *op = callList[i];
            if (!callOpSuccDict.count(op) || callOpSuccDict.at(op).empty()) {
                noSuccOpList.push_back(i);
                noSuccOpSet.insert(i);
            }
        }

        /*
         * As we need a reference of null, we use the 0-th element for the reference of null for DevAscendFunctionDuppedData
         * So outcastStitchCount starts from 1, as the 0-th is for the reference of null.
         */
        outcastStitchCount = 1;
        for (size_t i = 0; i < callList.size(); i++) {
            if (allOutcastUseOpSet.count(i) || noSuccOpSet.count(i)) {
                outcastStitchIndexList.push_back(outcastStitchCount);
                outcastStitchCount++;
            } else {
                outcastStitchIndexList.push_back(0);
            }
        }
    }

    void EncodeIncasts() {
        for (auto &i : incastList) {
            tensorList.Insert(i);
            incastRawTensorList.Insert(i->GetRawTensor());
            RecordRawTensor(i);
            InoutOperationAttr incastOpAttr;
            auto dim = i->shape.size();
            incastOpAttr.dim = dim;
            incastOpAttr.cellMatchTableDesc = InitCellMatchTableDesc(i->GetShape(), std::vector<int64_t>(dim, 1));

            std::set<uint32_t> incastUseOpSet;
            for (size_t j = 0; j < callList.size(); j++) {
                auto &op = *callList[j];
                auto callAttr = dynamic_cast<CallOpAttribute *>(op.GetOpAttribute().get());
                // add icast and oper io's relationship
                for (size_t k = 0; k < op.GetIOperands().size(); ++k) {
                    auto &iOperand = op.GetIOperands()[k];
                    auto coaIndex = op.GetIOpAttrOffset(k) + COA_INDEX_DIM_BASE;
                    if (i->tensor->rawmagic == iOperand->tensor->rawmagic) {
                        ASSERT(iOperand->GetShape().size() == dim);
                        std::vector<int64_t> shape = callAttr->GetLinearImmediateArgList(coaIndex + dim, coaIndex + dim * 0x2, false);
                        incastOpAttr.useList.emplace_back(j, k, coaIndex, coaIndex + dim);
                        UpdateCellMatchShape(incastOpAttr.cellMatchTableDesc, shape);
                        ALOG_DEBUG_F("minimal shape for incast %d raw %d op %d %d is %s\n", i->magic, i->GetRawMagic(), j,
                            op.GetOpMagic(), IntVecToStr(ShapeToVector(incastOpAttr.cellMatchTableDesc.cellShape)).c_str());
                        incastUseOpSet.insert(j);
                    }
                }
            }

            UpdateCellMatchStrideAndSize(incastOpAttr.cellMatchSize, incastOpAttr.cellMatchTableDesc, i, dim);
            incastOpAttr.useOpList.insert(incastOpAttr.useOpList.end(), incastUseOpSet.begin(), incastUseOpSet.end());

            incastOpAttrDict.insert({i, incastOpAttr});
        }

        for (size_t i = 0; i < callList.size(); i++) {
            Operation *op = callList[i];
            if (!callOpPredDict.count(op) || callOpPredDict.at(op) == 0) {
                noPredOpList.push_back(i);
            }
        }
    }

    struct Hasher {
        template<typename T>
        std::size_t operator()(const OrderedSet<T> &operationSet) const {
            size_t res = 0;
            for (auto op : operationSet) {
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

    void PrintColorGraph(int colorNum) {
        ALOG_INFO_F("********** Call OP Graph **********\n");
        for (int i = 0; i < colorNum; i++) {
            ALOG_INFO_F("%zu: %zu", i, colorOutGraph[i].size());
            ALOG_INFO_F("%s", IntVecToStr(colorOutGraph[i]).c_str());
        }
        int outCount = 0;
        for (int i = 0; i < colorNum; i++) {
            outCount += colorOutGraph[i].size();
        }
        ALOG_INFO_F("total out: %d\n", outCount);
    }

    inline void findAllReachableNodes(int start_node, std::unordered_map<int, std::vector<int>>& outGraph,
                                        std::vector<std::unordered_set<int>>& reachable, std::vector<int>& visited) {
        reachable[start_node].insert(start_node);
        for (int v : outGraph[start_node]) { 
            if (visited[v] == 0) {
                findAllReachableNodes(v, outGraph, reachable, visited);
            }
            reachable[start_node].insert(reachable[v].begin(), reachable[v].end());
        }
        visited[start_node] = 1;
    }

    void FindRedundantEdges(int colorNum, std::vector<std::vector<int>>& redundantColorOutGraph) {
        std::vector<std::unordered_set<int>> reachable(colorNum);
        std::vector<int> visited(colorNum, 0);
        for (int i = 0; i < colorNum; ++i) {
            if (visited[i] == 0) {
                findAllReachableNodes(i, colorOutGraph, reachable, visited); // DFS记忆化计算
            }
        }
        for (int u = 0; u < colorNum; ++u) {
            for (int v : colorOutGraph[u]) {
                bool is_redundant = false;
                for (int w : colorOutGraph[u]) {
                    if (w == v) {
                        continue;
                    }
                    if (reachable[w].count(v)) {
                        is_redundant = true;
                        break;
                    }
                }
                if (is_redundant) {
                    redundantColorOutGraph[u].push_back(v);
                }
            }
        }
    }

    void EraseRedundantColorEdges(std::vector<Operation *> &callopList) {
        int colorNum = callopList.size();
        std::vector<std::vector<int>> redundantColorOutGraph(colorNum);
        // Find redundant edges
        FindRedundantEdges(colorNum, redundantColorOutGraph);
        // Erase redundant edges
        for (int i = 0; i < colorNum; i++) {
            std::sort(redundantColorOutGraph[i].begin(), redundantColorOutGraph[i].end());
            ALOG_INFO_F("Redundant outgraph of %d is %s", i, IntVecToStr(redundantColorOutGraph[i]).c_str());
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
                   coreType == static_cast<uint32_t>(CoreType::HUB) || coreType == static_cast<uint32_t>(CoreType::AICPU));
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
            } else if (callopCoreTypeDict[callopList[i]] == static_cast<uint32_t>(CoreType::AICPU)) {
                totalZeroPredAicpu++;
            } else {
                ASSERT(false);
            }
        }

        for (auto &op : callopList) {
            callList.Insert(op);
        }

        EncodeIncasts();
        EncodeOutCasts();
    }

    void Init(DevAscendFunction *devFunc, const EncodeDevAscendFunctionParam &param, bool fillContent) {
        auto slot = param.slot;
        uintdevptr_t initOffset = reinterpret_cast<uintdevptr_t>(&devFunc->data) - reinterpret_cast<uintdevptr_t>(devFunc);
        DevAscendFunctionPredInfo predInfo = {totalZeroPred, totalZeroPredAIV, totalZeroPredAIC, totalZeroPredHub,
            totalZeroPredAicpu};
        devFunc->sourceFunc = nullptr;
        devFunc->InitIncastOutcastAttr(initOffset, incastList, outcastList, fillContent);
        devFunc->InitOperationDynamicField(initOffset, predInfo, outcastStitchCount, calleeHashIndexDict,
            expressionTable, callList, incastList, outcastList, callOpSuccDict, fillContent);
        devFunc->InitRawTensorAndMemoryRequirement(initOffset, incastRawTensorList, outcastRawTensorList,
            rawTensorList, rawMagicToRawTensor, rawAttrs, param, expressionTable, fillContent);
        devFunc->InitTensor(initOffset, tensorList, rawTensorList, fillContent);
        devFunc->InitOperation(initOffset, expressionTable, callList, tensorList, rawTensorList, callOpPredDict, callOpSuccDict, calleeHashIndexDict, outcastStitchIndexList, noPredOpList, noSuccOpList, fillContent);
        devFunc->InitIncastOutcast(initOffset, incastList, outcastList, tensorList, incastOpAttrDict, outcastOpAttrDict, slot, rawName, fillContent);
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
    uint64_t alignedHostControlFlowBinaryInputSize = ALIGN_UP(hostControlFlowBinaryInput.size(), sizeof(uint64_t));
    hostControlFlowBinary.HostInitDataSizeOffset(initOffset, alignedHostControlFlowBinaryInputSize);
    ONFILLCONTENT { memcpy_s(hostControlFlowBinary.Data(), hostControlFlowBinaryInput.size(), hostControlFlowBinaryInput.data(), hostControlFlowBinaryInput.size()); }

    uint64_t alignedDevControlFlowBinaryInputSize = ALIGN_UP(devControlFlowBinaryInput.size(), sizeof(uint64_t));
    devControlFlowBinary.HostInitDataSizeOffset(initOffset, alignedDevControlFlowBinaryInputSize);
    ONFILLCONTENT { memcpy_s(devControlFlowBinary.Data(), devControlFlowBinaryInput.size(), devControlFlowBinaryInput.data(), devControlFlowBinaryInput.size()); }
}
void DevAscendProgram::InitDevEncodeList(
        uintdevptr_t &initOffset,
        const std::vector<std::vector<uint8_t>> &devEncodeListInput,
        bool fillContent) {
    devEncodeList.HostInitDataSizeOffset(initOffset, devEncodeListInput.size());
    devEncodeDataList.HostInitDataSizeOffset(initOffset, 0);
    uint64_t offset = 0;
    for (size_t i = 0; i < devEncodeListInput.size(); i++) {
        uint64_t alignedDevEncodeListInputSize = ALIGN_UP(devEncodeListInput[i].size(), sizeof(uint64_t));
        ONFILLCONTENT {
            devEncodeList[i].HostAssignRangeOffsetSize(devEncodeDataList, offset, alignedDevEncodeListInputSize);
        };
        ONFILLCONTENT {
            memcpy_s(devEncodeList[i].Data(), devEncodeList[i].size(), devEncodeListInput[i].data(),
                devEncodeListInput[i].size());
        };
        offset += alignedDevEncodeListInputSize;
    }
    devEncodeDataList.HostInitDataSizeOffset(initOffset, offset);
}
void DevAscendProgram::InitCceCodeList(uintdevptr_t &initOffset, const std::vector<CceCodeInfo> &cceInfo,
                                       bool fillContent) {
    cceCodeList.HostInitDataSizeOffset(initOffset, cceInfo.size());
    aicpuLeafCodeList.HostInitDataSizeOffset(initOffset, cceInfo.size());
    aicpuLeafCodeDataList.HostInitDataSizeOffset(initOffset, 0);
    size_t dataOffset = 0;
    for (size_t i = 0; i < cceInfo.size(); i++) {
        ONFILLCONTENT {
            cceCodeList[i].coreType = cceInfo[i].coreType;
            cceCodeList[i].psgId = cceInfo[i].psgId;
            cceCodeList[i].funcHash = cceInfo[i].funcHash;
            auto dataLen = cceInfo[i].aicpuLeafCode.size();
            aicpuLeafCodeList[i].aicpuLeafCode.HostAssignRangeOffsetSize(aicpuLeafCodeDataList, dataOffset, dataLen);
            (void)memcpy_s(aicpuLeafCodeList[i].aicpuLeafCode.Data(), sizeof(int32_t) * dataLen,
                cceInfo[i].aicpuLeafCode.data(), sizeof(int32_t) * dataLen);
        };
        dataOffset += cceInfo[i].aicpuLeafCode.size();
    }
    aicpuLeafCodeDataList.HostInitDataSizeOffset(initOffset, dataOffset);
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

static void InitPartialUpdateCellMatch(
        const std::vector<const DevAscendFunctionOutcast *> &outcastList,
        DevCellMatchTableDesc *partialUpdateCellMatchTableDesc) {
    std::vector<int> tensorShape;
    for (size_t i = 0; i < outcastList.size(); i++) {
        std::vector<int> outcastShape;
        for (int d = 0; d < outcastList[i]->dim; d++) {
            outcastShape.push_back(outcastList[i]->cellMatchTableDesc.GetCellShape(d) * outcastList[i]->cellMatchTableDesc.GetStrideShape(d));
            if (i > 0) {
               tensorShape[d] = std::max(tensorShape[d], outcastShape[d]);
            }
        }
        if (i == 0) {
            tensorShape = outcastShape;
        }
    }

    std::vector<int> cellShape;
    for (size_t d = 0; d < tensorShape.size(); d++) {
        int dim = 0;
        for (size_t i = 0; i < outcastList.size(); i++) {
            if (i == 0) {
                dim = outcastList[i]->cellMatchTableDesc.GetCellShape(d);
            } else if (dim != -1) {
                dim = std::gcd(dim, outcastList[i]->cellMatchTableDesc.GetCellShape(d));
            } else {
                ASSERT(outcastList[i]->cellMatchTableDesc.GetCellShape(d) == -1);
            }
        }
        cellShape.push_back(dim);
    }
    partialUpdateCellMatchTableDesc->SetCellShape(cellShape);

    std::vector<int> strideShape;
    for (size_t d = 0; d < tensorShape.size(); d++) {
        strideShape.push_back(tensorShape[d] / cellShape[d]);
    }
    partialUpdateCellMatchTableDesc->SetStrideShape(strideShape);
}

void DevAscendProgram::InitPartialUpdateSlot(
        uintdevptr_t &initOffset,
        const std::vector<std::vector<uint8_t>> &devEncodeListInput,
        const std::unordered_map<Function *, int> &rootFuncKeyDict,
        const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootIncastDict,
        const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootOutcastDict,
        const std::vector<int> &tPartialUpdateSlotIndexList,
        bool fillContent) {
    (void)slotRootIncastDict;
    (void)slotRootOutcastDict;
    this->partialUpdateList.HostInitDataSizeOffset(initOffset, slotSize);

    this->cellMatchRuntimePartialUpdateTableList.HostInitDataSizeOffset(initOffset, 0);
    int totalCellMatchSize = 0;
    for (size_t i = 0; i < tPartialUpdateSlotIndexList.size(); i++) {
        std::vector<const DevAscendFunctionOutcast *> outcastList;
        auto slotIndex = tPartialUpdateSlotIndexList[i];
        ASSERT(slotRootOutcastDict.count(slotIndex));
        for (auto &[root, outcastIndex] : slotRootOutcastDict.find(slotIndex)->second) {
            ASSERT(rootFuncKeyDict.count(root));
            int funcKey = rootFuncKeyDict.find(root)->second;
            DevAscendFunction *devFunc = reinterpret_cast<DevAscendFunction *>(const_cast<uint8_t *>(devEncodeListInput[funcKey].data()));
            outcastList.push_back(&devFunc->GetOutcast(outcastIndex));
        }
        DevCellMatchTableDesc partialUpdateCellMatchTableDesc;
        InitPartialUpdateCellMatch(outcastList, &partialUpdateCellMatchTableDesc);
        size_t tableSize = partialUpdateCellMatchTableDesc.GetStride(0);

        ONFILLCONTENT {
            auto &partialUpdate = At(partialUpdateList, slotIndex);
            partialUpdate.slotIndex = slotIndex;
            partialUpdate.cellMatchTableDesc = partialUpdateCellMatchTableDesc;
            partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignRangeOffsetSize(cellMatchRuntimePartialUpdateTableList, totalCellMatchSize, tableSize);
            auto tableData = partialUpdate.cellMatchRuntimePartialUpdateTable.Data();
            for (size_t j = 0; j < tableSize; j++) {
                tableData[j] = AICORE_TASK_INIT;
            }
        }
        totalCellMatchSize += tableSize;
    }
    totalCellMatchSize = ALIGN_UP(totalCellMatchSize, sizeof(uint64_t) * FRIENDLY_CACHE_ALIGN_U64_SIZE / sizeof(uint64_t));
    this->cellMatchRuntimePartialUpdateTableList.HostInitDataSizeOffset(initOffset, totalCellMatchSize);
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
        devProg->InitCceCodeList(initOffset, dyndevAttr->cceCodeInfo, fillContent);
        devProg->InitStartArgsABIParamList(
            initOffset,
            dyndevAttr->inoutLink.inputSlotIndexList,
            dyndevAttr->inoutLink.outputSlotIndexList,
            dyndevAttr->startArgsInputSymbolIndexList,
            dyndevAttr->startArgsSymbolHandlerList,
            dyndevAttr->inoutLink.assembleSlotIndexList,
            dyndevAttr->inoutLink.inplaceSlotIndexList,
            fillContent);
        devProg->InitPartialUpdateSlot(
                initOffset,
                dyndevAttr->devEncodeList,
                dyndevAttr->rootFuncKeyDict,
                dyndevAttr->slotRootIncastDict,
                dyndevAttr->slotRootOutcastDict,
                dyndevAttr->inoutLink.partialUpdateSlotIdexList,
                fillContent);
        devProg->InitPrefetchInfoList(initOffset, dyndevAttr->l2InfoList, fillContent);
        devProg->commGroupNum = dyndevAttr->commGroupNum;
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

static constexpr uint64_t KIBI = UINT64_C(1024);
static constexpr uint64_t MEBI = UINT64_C(1024) * 1024;
static constexpr uint64_t GIBI = UINT64_C(1024) * 1024 * 1024;

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
                globalTensorMem += devFunc->GetOutcastRawTensor(i)->maxPossibleMemReq;
                continue;
            }

            auto &toSlotList = devFunc->GetOutcast(i).toSlotList;
            for (size_t j = 0; j < toSlotList.size(); j++) {
                int slotIdx = devFunc->At(toSlotList, j);
                // No output slot
                slots[slotIdx].asWriteSlot = true;
            }
            maxSlotMemReq = std::max(maxSlotMemReq, devFunc->GetOutcastRawTensor(i)->maxPossibleMemReq);
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

    static constexpr uint64_t MASK_32K = 32 * KIBI - 1;
    res.memReq = ((memReq + MASK_32K) & ~MASK_32K); // aligned to 32K

    return res;
}

static uint64_t CalcAicpuCoherentWorkspace(DevAscendProgram &devProg) {
    (void)devProg;
    static constexpr uint64_t AICPU_COHERENT_WS_SIZE = 6 * MEBI;
    return AICPU_COHERENT_WS_SIZE;
}

static uint64_t CalcStitchWorkspace(DevAscendProgram &devProg) {
    (void)devProg;
    static constexpr uint64_t AICPU_STITCH_SIZE = 2 * MEBI;
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

#if DEBUG_INFINITE_LIFETIME
        base->debugDumpTensorMemReq = 8 * GIBI;
#else
        base->debugDumpTensorMemReq = 0;
#endif
    }
}
} // namespace dynamic
} // namespace npu::tile_fwk
