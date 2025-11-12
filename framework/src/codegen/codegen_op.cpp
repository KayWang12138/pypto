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
 * \file codegen_op.cpp
 * \brief
 */

#include "codegen_op.h"

#include <algorithm>

#include "codegen/codegen_common.h"
#include "codegen/utils/codegen_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "securec.h"

namespace npu::tile_fwk {
namespace {
bool IsCopyOpWithShapeOffsetAttr(Opcode opcode) {
    bool result =
        opcode == Opcode::OP_COPY_IN || opcode == Opcode::OP_COPY_OUT || opcode == Opcode::OP_TRANSPOSE_MOVEOUT ||
        opcode == Opcode::OP_TRANSPOSE_MOVEIN || opcode == Opcode::OP_INDEX_OUTCAST ||
        opcode == Opcode::OP_LOCAL_COPY_OUT || opcode == Opcode::OP_REMOTE_REDUCE ||
        opcode == Opcode::OP_REMOTE_GATHER || opcode == Opcode::OP_FFN_SCHED || opcode == Opcode::OP_FFN_BATCHING ||
        opcode == Opcode::OP_COPY_TO_LOCAL_EXPERT || opcode == Opcode::OP_SHMEM_PUT ||
        opcode == Opcode::OP_SHMEM_PUT_UB2GM || opcode == Opcode::OP_SHMEM_SIGNAL || opcode == Opcode::OP_SHMEM_GET ||
        opcode == Opcode::OP_SHMEM_GET_GM2UB || opcode == Opcode::OP_SHMEM_REDUCE;
    return result;
}
} // namespace

void CodeGenOp::UpdateShape(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx) {
    rawShape[operandIdx] = ToVecInt(logicalTensor.tensor->rawshape);
    ALOG_INFO_F("op code %s, operandIdx: %d, raw shape is %s", oper.GetOpcodeStr().c_str(), operandIdx,
        IntVecToStr(logicalTensor.tensor->rawshape).c_str());
    // need adapt unaligned scene after
    originShape[operandIdx] = ToVecInt(logicalTensor.oriShape);
    if (isSupportDynamicUnaligned) {
        dynamicValidShape[operandIdx] = logicalTensor.GetDynValidShape();
    }

    ASSERT(logicalTensor.shape.size() <= MAX_DIM) << "only support max dim: " << MAX_DIM;

    Opcode opcode = oper.GetOpcode();
    bool useAttrForGM = IsCopyOpWithShapeOffsetAttr(opcode);
    // Local Tensor shape just use shape from LogicalTensor
    if (!useAttrForGM || logicalTensor.GetMemoryTypeOriginal() != MEM_DEVICE_DDR) {
        shape[operandIdx] = ToVecInt(logicalTensor.shape);
        if (isSupportDynamicUnaligned) { // NEXTNEXT: stack gm should also has dynShape_ later
            ASSERT(!logicalTensor.GetDynValidShape().empty())
                << "LogicalTensor::dynShape_ can not empty in Dynamic Unaligned Scene";
        }
        return;
    }

    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    ASSERT(attr != nullptr) << ": missing OpAttr in copy op: \n" << oper.Dump();
    shape[operandIdx] = ToVecInt(attr->GetSpecifiedShape(1));
    dynShapeFromAttr[operandIdx] = attr->GetShape(); // used for spilling GM scene
    ALOG_INFO_F("attrShape(from op CopyOpAttribute) = %s", IntVecToStr(shape[operandIdx]).c_str());
}

void CodeGenOp::UpdateOffsetValueForGM(const std::vector<OpImmediate> &offsets, int operandIdx) {
    std::vector<SymbolicScalar> dynOffset(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i].IsSpecified()) {
            auto val = offsets[i].GetSpecifiedValue();
            dynOffset[i] = val;
        }
    }
    offsetGmSymbolic[operandIdx] = dynOffset;
    ALOG_INFO_F("UpdateOffsetValueForGM , offsetGmSymbolic is %s", IntVecToStr(dynOffset).c_str());
}

bool CodeGenOp::IsUpdateOffsetByAttr(const LogicalTensor &logicalTensor, bool useAttrShapeOffset){
    if ((!useAttrShapeOffset) || (((opCode != Opcode::OP_L1_TO_BT) && (opCode != Opcode::OP_L1_TO_FIX_QUANT_PRE)) &&
                                     (logicalTensor.GetMemoryTypeOriginal() != MEM_DEVICE_DDR))) {
        return false;
    }
    return true;
}

void CodeGenOp::UpdateOffsetForInput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx) {
    bool useAttrShapeOffsetForInputGM = OpcodeManager::Inst().IsCopyIn(opCode);
    if (!IsUpdateOffsetByAttr(logicalTensor, useAttrShapeOffsetForInputGM)) {
        offset[operandIdx] = ToVecInt(logicalTensor.offset); // Local Tensor offset just use offset from LogicalTensor
        ALOG_INFO_F("UpdateOffsetForInput offset is %s", IntVecToStr(offset[operandIdx]).c_str());
        return;
    }

    // only used for 1. L1 Copy; 2. spilling into gm scene(e.g., ooo spilling)
    ALOG_INFO_F("start update offset for GM input");
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    ASSERT(attr != nullptr) << ": missing OpAttr in copy in op: \n" << oper.Dump();
    UpdateOffsetValueForGM(attr->GetCopyInAttr().first, operandIdx);
}

void CodeGenOp::UpdateOffsetForOutput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx) {
    bool useAttrShapeOffsetForOutputGM = OpcodeManager::Inst().IsCopyOut(opCode);
    if (!useAttrShapeOffsetForOutputGM || logicalTensor.GetMemoryTypeOriginal() != MEM_DEVICE_DDR) {
        offset[operandIdx] = ToVecInt(logicalTensor.offset); // Local Tensor offset just use offset from LogicalTensor
        ALOG_INFO_F("UpdateOffsetForOutput offset is %s", IntVecToStr(offset[operandIdx]).c_str());
        return;
    }

    // only used for 1. L1 Copy; 2. spilling into gm scene(e.g., ooo spilling)
    ALOG_INFO_F("start update offset for GM output");
    std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(oper.GetOpAttribute());
    ASSERT(attr != nullptr) << ": missing OpAttr in copy out op: \n" << oper.Dump();
    UpdateOffsetValueForGM(attr->GetCopyOutAttr().second, operandIdx);
}

void CodeGenOp::CheckScaleValue(const npu::tile_fwk::Operation &ops) {
    auto checkValue = [&](float value) {
        if (std::isnan(value)) {
            hasNan = true;
        }
        if (std::isinf(value)) {
            if (value > 0) {
                hasPosInf = true;
            } else {
                hasNegInf = true;
            }
        }
    };

    if (ops.HasAttr(OpAttributeKey::scalar)) {
        extOperandVal = ops.GetElementAttribute(OpAttributeKey::scalar);
        float value = extOperandVal.Cast<float>();
        checkValue(value);
    }
    if (opAttrs.count(OpAttributeKey::dynScalar)) {
        extOperandValSecond = ops.GetElementAttribute(OpAttributeKey::dynScalar);
        auto scalar = opAttrs.at(OpAttributeKey::dynScalar);
        if (scalar.Type() == typeid(float)) {
            float value = extOperandValSecond.Cast<float>();
            checkValue(value);
        }
    }
}

bool CodeGenOp::Init(const npu::tile_fwk::Operation &ops) {
    ASSERT(ops.iOperand.size() + ops.oOperand.size() <= MAX_OPERANDS)
        << "can not support ops.iOperand.size: " << ops.iOperand.size()
        << ", ops.oOperand.size: " << ops.oOperand.size();

    ALOG_INFO_F("%s: init CodeGenOp from npu::tile_fwk::Operation", __FUNCTION__);

    isSupportDynamicUnaligned =
        functionType == FunctionType::DYNAMIC_LOOP_PATH && config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_UNALIGNED);
    UpdateTileOpInfo(ops);
    if (tileOpName.empty()) {
        ALOG_ERROR_F("%s: empty tileOpName for ops:\n%s", __FUNCTION__, ops.Dump().c_str());
        return false;
    }

    // opcode would be refreshed by UpdateTileOpInfo
    isSupportLayout = ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false) &&
                      SUPPORT_TILETENSOR_OPS.find(opCode) != SUPPORT_TILETENSOR_OPS.end();

    opCodeStr = OpcodeManager::Inst().GetOpcodeStr(opCode);

    int operandIdx = 0;
    for (const auto &output : ops.oOperand) {
        ALOG_INFO_F("output is %s\n", output->Dump().c_str());
        UpdateCodegenOpInfoByTensor(ops, false, output, operandIdx);
    }

    // if no output like WriteRemote OP, set operandIdx=1 for input
    if (operandIdx == 0) {
        operandIdx = 1;
    }

    for (const auto &input : ops.iOperand) {
        ALOG_INFO_F("input is %s\n", input->Dump().c_str());
        UpdateCodegenOpInfoByTensor(ops, true, input, operandIdx);
    }

    operandCnt = ops.oOperand.size() + ops.iOperand.size();

    GetGmParamIdx(ops);
    syncQueue = ops.syncQueue_;
    CheckScaleValue(ops);
    UpdateOpAttribute(ops);

    return true;
}

void CodeGenOp::UpdateCodegenOpInfoByTensor(
    const Operation &ops, bool isInput, const std::shared_ptr<LogicalTensor> &tensor, int &operandIdx) {
    operand[operandIdx] = tensor->GetMemoryTypeOriginal() == MEM_DEVICE_DDR ? tensor->tensor->GetRawMagic() :
                                                                              -tensor->tensor->GetRawMagic();
    operandWithMagic[operandIdx] = tensor->GetMagic();
    UpdateShape(ops, *tensor, operandIdx);
    if (isInput) {
        UpdateOffsetForInput(ops, *tensor, operandIdx);
    } else {
        UpdateOffsetForOutput(ops, *tensor, operandIdx);
    }
    operandDtype[operandIdx] = tensor->tensor->datatype;
    auto it = OPERAND_TYPE_TO_MEMORY_TYPE.find(tensor->GetMemoryTypeOriginal());
    ASSERT(it != OPERAND_TYPE_TO_MEMORY_TYPE.end())
        << "can not support memory type: " << static_cast<size_t>(tensor->GetMemoryTypeOriginal());
    operandType[operandIdx] = it->second;
    ++operandIdx;
}

void CodeGenOp::UpdateOpAttribute(const npu::tile_fwk::Operation &ops) {
    opAttrs = ops.GetAllAttr();
    isInputForceCombineAxis = ops.HasAttr(OpAttributeKey::inputCombineAxis);

    ConvertAttribute(ops);
}

std::string CodeGenOp::GenOpAttr(bool hasExistingParam) const {
    if (opAttrs.empty()) {
        return {};
    }

    std::vector<std::string> attrList;
    for (const auto& kv : opAttrs) {
        if (kv.first.substr(0, OP_ATTR_PREFIX.size()) != OP_ATTR_PREFIX) {
            continue;
        }
        if (kv.second.Type() == typeid(int64_t)) {
            attrList.push_back(std::to_string(npu::tile_fwk::AnyCast<int64_t>(kv.second)));
        } else if (kv.second.Type() == typeid(bool)) {
            attrList.push_back(std::to_string(npu::tile_fwk::AnyCast<bool>(kv.second)));
        } else if (kv.second.Type() == typeid(std::vector<int64_t>)) {
            auto vec = npu::tile_fwk::AnyCast<std::vector<int64_t>>(kv.second);
            for (auto v : vec) {
                attrList.push_back(std::to_string(v));
            }
        }
    }

    if (attrList.empty()) {
        return {};
    }

    std::string joined = JoinString(attrList, ", ");
    return hasExistingParam ? ", " + joined : joined;
}

void CodeGenOp::ConvertPoolAttribute(const Operation &operation) {
    auto opc = operation.GetOpcode();
    if (opc != Opcode::OP_MAX_POOL) {
        return;
    }

    std::vector<std::string> intAttrStrList{
        ConvOpAttributeKey::paddingLeft,
        ConvOpAttributeKey::paddingTop,
        ConvOpAttributeKey::paddingRight,
        ConvOpAttributeKey::paddingBottom,
        ConvOpAttributeKey::strideh,
        ConvOpAttributeKey::stridew,
        PoolOpAttributeKey::poolh,
        PoolOpAttributeKey::poolw,
    };
    for (size_t i = 0; i < intAttrStrList.size(); i++) {
        poolParams.push_back(operation.GetIntAttribute(intAttrStrList[i]));
    }
}

void CodeGenOp::ConvertAttribute(const Operation &operation) {
    ASSERT(operation.iOperand.size() + operation.oOperand.size() <= MAX_OPERANDS)
        << "can not support operation.iOperand.size: " << operation.iOperand.size()
        << ", operation.oOperand.size: " << operation.oOperand.size();
    if (opCode == Opcode::OP_CONV || opCode == Opcode::OP_CONV_ADD) {
        std::vector<std::string> intAttrStrList{
            ConvOpAttributeKey::cin,
            ConvOpAttributeKey::cout,
            ConvOpAttributeKey::paddingLeft,
            ConvOpAttributeKey::paddingTop,
            ConvOpAttributeKey::paddingRight,
            ConvOpAttributeKey::paddingBottom,
            ConvOpAttributeKey::strideh,
            ConvOpAttributeKey::stridew,
            ConvOpAttributeKey::hposX,
            ConvOpAttributeKey::hsteP,
            ConvOpAttributeKey::wposX,
            ConvOpAttributeKey::wstep,
            ConvOpAttributeKey::hoffsetY,
            ConvOpAttributeKey::woffsetY,
            ConvOpAttributeKey::reluType,
            ConvOpAttributeKey::reluAlpha,
            ConvOpAttributeKey::clearFlag,
            ConvOpAttributeKey::hasAccFlag,
            ConvOpAttributeKey::hasEltFlag,
            ConvOpAttributeKey::hasBiasFlag,
            ConvOpAttributeKey::eltBrcbFlag,
            ConvOpAttributeKey::eltMode,
        };
        // (Cin, Cout, PaddingLeft, PaddingTop, PaddingRight, PaddingBottom, Stride1, Stride2, HPosX, HStep, WPosX,
        // WStep, HOffsetY, WOffsetY, reluType, relu_alpha, clearFlag, has_acc_flag, has_elt_flag, has_bias_flag,
        // elt_brcb_flag, elt_mode, hasQuantPreVector, hasQuantPostVector, hasAntiqVector)
        for (size_t i = 0; i < intAttrStrList.size(); i++) {
            convParams.push_back(operation.GetIntAttribute(intAttrStrList[i]));
        }
        std::vector<std::string> longAttrStrList{
            FixpOpAttributeKey::quantPreScalar,
            FixpOpAttributeKey::quantPostScalar,
            FixpOpAttributeKey::antiqScalar,
        };
        for (size_t i = 0; i < longAttrStrList.size(); i++) {
            convParams.push_back(operation.GetIntAttribute(longAttrStrList[i]));
        }
    }
    if (opCode == Opcode::OP_L1_COPY_IN_FRACTAL_Z) {
        convParams.push_back(operation.GetIntAttribute(ConvOpAttributeKey::fmapC0));
    }

    if (opCode == Opcode::OP_L1_TO_FIX ||
        opCode == Opcode::OP_L1_TO_FIX_RELU_PRE || opCode == Opcode::OP_L1_TO_FIX_RELU_POST ||
        opCode == Opcode::OP_L1_TO_FIX_QUANT_POST || opCode == Opcode::OP_L1_TO_FIX_ELT_ANTIQ ||
        opCode == Opcode::OP_L1_TO_FIX_MTE2_ANTIQ) {
        convParams.push_back(operation.GetIntAttribute(FixpOpAttributeKey::fbAddrSpace));
    }

    ConvertPoolAttribute(operation);
}

void CodeGenOp::UpdateTileOpInfo(const Operation &ops) {
    opCode = ops.GetOpcode();
    tileOpName = GetTileOpName(opCode);

    if (opCode == Opcode::OP_COPY_IN && !ops.oOperand.empty()) {
        npu::tile_fwk::MemoryType memtype = ops.oOperand[0]->GetMemoryTypeOriginal();
        if (memtype == npu::tile_fwk::MemoryType::MEM_UB) {
            tileOpName = "TileOp::UBCopyIn";
            opCode = Opcode::OP_UB_COPY_IN;
        } else if (memtype == npu::tile_fwk::MemoryType::MEM_L1) {
            tileOpName = "TileOp::L1CopyIn";
            opCode = Opcode::OP_L1_COPY_IN;
        }
    } else if (opCode == Opcode::OP_COPY_OUT && !ops.iOperand.empty()) {
        npu::tile_fwk::MemoryType memtype = ops.iOperand[0]->GetMemoryTypeOriginal();
        if (memtype == npu::tile_fwk::MemoryType::MEM_UB) {
            tileOpName = "TileOp::UBCopyOut";
            opCode = Opcode::OP_UB_COPY_OUT;
        } else if (memtype == npu::tile_fwk::MemoryType::MEM_L1) {
            tileOpName = "TileOp::L1CopyOut";
            opCode = Opcode::OP_L1_COPY_OUT;
        } else if (memtype == npu::tile_fwk::MemoryType::MEM_L0C) {
            tileOpName = "TileOp::L0CCopyOut";
            opCode = Opcode::OP_L0C_COPY_OUT;
        }
    }

    if ((functionType != FunctionType::DYNAMIC_LOOP_PATH) || DISTRIBUTED_OPS.count(opCode)) {
        return;
    }

    std::string dynPrefix = "Dyn";

    size_t nameSpaceLen = std::strlen("TileOp::");
    // NEXTNEXT: delete if after all TileOp have adapted dynamic unalinged scene
    if (isSupportDynamicUnaligned &&
        SUPPORT_DYNAMIC_UNALIGNED_OPS.find(opCode) != SUPPORT_DYNAMIC_UNALIGNED_OPS.end()) {
        tileOpName.insert(nameSpaceLen, dynPrefix);
        return;
    }

    // NEXTNEXT: delete after isSupportDynamicUnaligned can be always true
    if (OpcodeManager::Inst().IsCopyInOrOut(opCode)) {
        tileOpName.insert(nameSpaceLen, dynPrefix);
    }

    ALOG_INFO_F("after UpdateTileOpInfo: tileOpName = %s", tileOpName.c_str());
}

void CodeGenOp::GetGmParamIdx(const npu::tile_fwk::Operation &oper) {
    if (!isUnderDynamicFunction || oper.IsNeedStackGM()) {
        auto inParamLocSize = oper.inParamLocation_.size();
        auto outParamLocSize = oper.outParamLocation_.size();

        // Ops like UB_ALLOC have output operands, but does not have output
        // param locs, so here we should not assert 'outParamLocSize == outputTensors.size()' !
        ASSERT(inParamLocSize <= oper.iOperand.size()) << "size of Op.inParamLocation_ is larger than input operands";
        ASSERT(outParamLocSize <= oper.oOperand.size())
            << "size of Op.outParamLocation_ is larger than output operands";

        ALOG_INFO_F("%d: inParamLocation = %s", __FUNCTION__, IntVecToStr(oper.inParamLocation_).c_str());
        ALOG_INFO_F("%d: outParamLocation = %s", __FUNCTION__, IntVecToStr(oper.outParamLocation_).c_str());

        std::copy(oper.outParamLocation_.begin(), oper.outParamLocation_.end(), paramLocation);
        std::copy(oper.inParamLocation_.begin(), oper.inParamLocation_.end(), paramLocation + oper.oOperand.size());
        return;
    }

    if ((oper.GetOpcode() == Opcode::OP_SHMEM_PUT) || (oper.GetOpcode() == Opcode::OP_SHMEM_SIGNAL) ||
        (oper.GetOpcode() == Opcode::OP_SHMEM_GET) || (oper.GetOpcode() == Opcode::OP_SHMEM_REDUCE) ||
        (oper.GetOpcode() == Opcode::OP_SHMEM_PUT_UB2GM) || (oper.GetOpcode() == Opcode::OP_SHMEM_GET_GM2UB)) {
        for (size_t i = 0; i < oper.GetOOperands().size(); ++i) {
            if (oper.GetOOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
                paramLocation[i] = oper.GetOOpAttrOffset(i);
            }
        }
        size_t iOffset = oper.GetOOperands().size() == 0 ? 1 : oper.GetOOperands().size();
        for (size_t i = 0; i < oper.GetIOperands().size(); ++i) {
            if (oper.GetIOperands()[i]->GetMemoryTypeToBe() == MEM_DEVICE_DDR) {
                paramLocation[i + iOffset] = oper.GetIOpAttrOffset(i);
            }
        }
        return;
    }

    if (oper.GetOpcode() == Opcode::OP_LOAD) {
        paramLocation[0] = oper.GetIOpAttrOffset(0);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        return;
    }

    if (oper.GetOpcode() == Opcode::OP_GATHER_IN_L1) {
        paramLocation[0] = oper.GetIOpAttrOffset(0);
        paramLocation[1] = oper.GetIOpAttrOffset(1);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        return;
    }

    if (OpcodeManager::Inst().IsCopyIn(oper.GetOpcode())) {
        const std::shared_ptr<OpAttribute> &attr = oper.GetOpAttribute();
        ASSERT(attr != nullptr) << "Copy In attr is null";
        std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
        paramLocation[1] = oper.GetIOpAttrOffset(0);
        ALOG_INFO_F("Gm Param Index of Copy In Op %s is %d", tileOpName.c_str(), paramLocation[1]);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        ALOG_INFO_F("%s GmTensorParamIdxInCallFunc: %d", __FUNCTION__, GmTensorParamIdxInCallFunc);
        return;
    }

    if (OpcodeManager::Inst().IsCopyOut(oper.GetOpcode())) {
        const std::shared_ptr<OpAttribute> &attr = oper.GetOpAttribute();
        ASSERT(attr != nullptr) << "Copy In attr is null";
        std::shared_ptr<CopyOpAttribute> copyAttr = std::static_pointer_cast<CopyOpAttribute>(attr);
        paramLocation[0] = oper.GetOOpAttrOffset(0);
        ALOG_INFO_F("Gm Param Index of Copy Out Op %s is %d", tileOpName.c_str(), paramLocation[0]);
        GmTensorParamIdxInCallFunc = oper.GetIntAttribute("GmTensorParamIdxInCallFunc");
        ALOG_INFO_F("%s GmTensorParamIdxInCallFunc: %d", __FUNCTION__, GmTensorParamIdxInCallFunc);
        return;
    }
}

std::string CodeGenOp::GenBarrier() const {
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "pipe_barrier(%s);\n", pipeId1.c_str());
    if (ret < 0) {
        ALOG_INFO_F("genBarrier snprintf_s failed %d", ret);
    }
    return buffer;
}

std::string CodeGenOp::GenSyncSetOp() const {
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "set_flag(%s, %s, EVENT_ID%d);\n", pipeId1.c_str(),
        pipeId2.c_str(), syncQueue.eventId_);
    if (ret < 0) {
        ALOG_INFO_F("genSyncSetOp snprintf_s failed %d", ret);
    }
    return buffer;
}

std::string CodeGenOp::GenSyncWaitOp() const {
    char buffer[256] = "CG_ERROR";
    auto pipeId1 = GetPipeId(syncQueue.pipeId_);
    auto pipeId2 = GetPipeId(syncQueue.trigPipeId_);
    int ret = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, "wait_flag(%s, %s, EVENT_ID%d);\n",
        pipeId1.c_str(), pipeId2.c_str(), syncQueue.eventId_);
    if (ret < 0) {
        ALOG_INFO_F("genSyncWaitOp snprintf_s failed %d", ret);
    }
    return buffer;
}

} // namespace npu::tile_fwk
