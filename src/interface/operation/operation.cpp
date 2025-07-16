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
 * \file operation.cpp
 * \brief
 */

#include "operation.h"

#include <functional>

#include "common/data_type.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/opcode.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/cycles.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/utils/serialization.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
const std::string OpAttributeKey::color = "COLOR";
const std::string OpAttributeKey::scalar = "SCALAR";
const std::string OpAttributeKey::isGlobalInput = "IS_GLOBAL_INPUT";
const std::string OpAttributeKey::seqNo = "SEQ_NO";
const std::string OpAttributeKey::isCube = "IS_CUBE";
const std::string OpAttributeKey::blockPadding = "BLOCK_PADDING";
const std::string OpAttributeKey::tilePadding = "TILE_PADDING";
const std::string OpAttributeKey::reshapePadding = "RESHAPE_PADDING";
const std::string OpAttributeKey::shapePadded = "SHAPE_PADDED";
const std::string OpAttributeKey::needAlloc = "NEED_ALLOC";
const std::string OpAttributeKey::broadcastLastAxis = "BROADCAST_LAST_AXIS";
const std::string OpAttributeKey::dontTouch = "DONT_TOUCH";
const std::string OpAttributeKey::tag = "TAG";
const std::string OpAttributeKey::commGroupInfo = "COMM_GROUP_INFO";
const std::string OpAttributeKey::distTilingInfo = "DIST_TILING_INFO";
const std::string OpAttributeKey::sameInOut = "SAME_IN_OUT";
const std::string OpAttributeKey::inputCombineAxis = "op_attr_input_combine_axis";
const std::string OpAttributeKey::outputCombineAxis = "op_attr_output_combine_axis";
const std::string OpAttributeKey::inplaceIdx = "INPLACE_IDX";
const std::string OpAttributeKey::cacheMode = "CACHE_MODE";
const std::string OpAttributeKey::panzBlockSize = "PA_NZ_BLOCK_SIZE";
const std::string OpAttributeKey::inputCombineAxisDone = "input_combine_axis_done"; // only for flow verify tool
const std::string OpAttributeKey::outputCombineAxisDone = "output_combine_axis_done"; // flow verify tool only

const std::string ConvOpAttributeKey::cin = "CIN";
const std::string ConvOpAttributeKey::cout = "COUT";
const std::string ConvOpAttributeKey::paddingLeft = "PADDING_LEFT";
const std::string ConvOpAttributeKey::paddingTop = "PADDING_TOP";
const std::string ConvOpAttributeKey::paddingRight = "PADDING_RIGHT";
const std::string ConvOpAttributeKey::paddingBottom = "PADDING_BOTTOM";
const std::string ConvOpAttributeKey::strideh = "STRIDEH";
const std::string ConvOpAttributeKey::stridew = "STRIDEW";
const std::string ConvOpAttributeKey::hposX = "HPOS_X";
const std::string ConvOpAttributeKey::hsteP = "HSTEP";
const std::string ConvOpAttributeKey::wposX = "WPOS_X";
const std::string ConvOpAttributeKey::wstep = "WSTEP";
const std::string ConvOpAttributeKey::hoffsetY = "HOFFSET_Y";
const std::string ConvOpAttributeKey::woffsetY = "WOFFSET_Y";
const std::string ConvOpAttributeKey::reluType = "RELU_TYPE";
const std::string ConvOpAttributeKey::reluAlpha = "RELU_ALPHA";
const std::string ConvOpAttributeKey::clearFlag = "CLEAR_FLAG";
const std::string ConvOpAttributeKey::hasAccFlag = "HAS_ACC_FLAG";
const std::string ConvOpAttributeKey::hasEltFlag = "HAS_ELT_FLAG";
const std::string ConvOpAttributeKey::hasBiasFlag = "HAS_BIAS_FLAG";
const std::string ConvOpAttributeKey::eltBrcbFlag = "ELT_BRCB_FLAG";
const std::string ConvOpAttributeKey::fmapSrcNum = "FMAP_SRC_NUM";

const std::string ConvOpAttributeKey::eltMode = "ELT_MODE";

const std::string ConvOpAttributeKey::fmapC0 = "FMAP_C0";
const std::string FixpOpAttributeKey::quantPreScalar = "QUANT_PRE_SCALAR";
const std::string FixpOpAttributeKey::quantPostScalar = "QUANT_POST_SCALAR";
const std::string FixpOpAttributeKey::antiqScalar = "ANTIQ_SCALAR";
const std::string FixpOpAttributeKey::hasQuantPreVector = "HAS_QUANT_PRE_VECTOR";
const std::string FixpOpAttributeKey::hasQuantPostVector = "HAS_QUANT_POST_VECTOR";
const std::string FixpOpAttributeKey::hasAntiqVector = "HAS_ANTIQ_VECTOR";
const std::string FixpOpAttributeKey::fbAddrSpace = "FIX_BUFFER_ADDR_SPACE";

const std::string PoolOpAttributeKey::poolh = "POOL_WIN_H";
const std::string PoolOpAttributeKey::poolw = "POOL_WIN_W";
bool OperationCmp::operator()(const Operation *lhs, const Operation *rhs) const {
    return lhs->GetOpMagic() < rhs->GetOpMagic();
}

Operation::Operation(
    Function &cur, Opcode opcode, LogicalTensors iOperands, LogicalTensors oOperands, bool updateTensorMap, int opMagic)
    : iOperand(std::move(iOperands)),
      oOperand(std::move(oOperands)),
      opmagic(opMagic),
      opcode_(opcode),
      isTileOp_(!cur.IsGraphType(GraphType::TENSOR_GRAPH)),
      coreType_(CoreType::MIX),
      function_(&cur) {
    if (opcode != Opcode::OP_CALL) {
        ASSERT(cur.GetFunctionType() != FunctionType::EAGER);
    }

    auto opCoreType = OpcodeManager::Inst().GetCoreType(opcode);
    tileShape_.Reset();
    if (opmagic == -1) {
        opmagic = cur.opSeed_++;
    }

    switch (opCoreType) {
        case OpCoreType::AIC: coreType_ = CoreType::AIC; break;
        case OpCoreType::AIV: coreType_ = CoreType::AIV; break;
        case OpCoreType::AICPU: coreType_ = CoreType::AICPU; break;
        default: break;
    }

    if (!IsOpCodeSupportMultiProducers(opcode)) {
    } else if (opcode == Opcode::OP_CALL) {
        if (cur.IsCompiledFunction()) {
            for (auto &t : GetOOperands()) {
                if (cur.GetTensorMap().GetTensorByMagic(t->magic) == nullptr) {
                    ASSERT(updateTensorMap || cur.IsCompiledFunction());
                }
            }
        }
    } else {
        if (cur.GetTensorMap().GetTensorByMagic(GetOOperands()[0]->magic) == nullptr) {
            ASSERT(updateTensorMap || cur.IsCompiledFunction());
        } else {
            updateTensorMap = false;
        }
    }

    if (function_->IsGraphType({GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH})) {
        tileShape_ = cur.BelongTo().GetTileShape();
        if (!ConfigManager::Instance().GetSemanticLabel().empty()) {
            SetSemanticLabel(ConfigManager::Instance().GetSemanticLabel());
        }
    }

    for (auto &input : GetIOperands()) {
        input->AddConsumer(this);
    }

    for (auto &output : GetOOperands()) {
        output->AddProducer(this);
        if (updateTensorMap) {
            function_->GetTensorMap().Insert(output);
        }
    }

    if (!GetIOperands().empty()) {
        // Get operation latency
        std::vector<std::vector<int>> shape;
        for (auto &srcTile : GetIOperands()) {
            shape.emplace_back(srcTile->shape);
        }
        latency_ = GetCycles(GetOpcodeStr(), shape, GetIOperands()[0]->tensor->datatype);
    }
}

std::string Operation::GetStringAttribute(const std::string &key) const {
    ASSERT(HasAttr(key)) << "Operation doesn't have attribute " << key;
    std::string attrVal;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string &key, const std::string &value) {
    SetAttr(key, value);
}

std::vector<int> Operation::GetVectorIntAttribute(const std::string &key) const {
    ASSERT(HasAttr(key)) << "Operation doesn't have attribute " << key;
    std::vector<int> attrVal;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string &key, std::vector<int> value) {
    SetAttr(key, value);
}

bool Operation::GetBoolAttribute(const std::string &key) const {
    if (!HasAttr(key)) {
        return false;
    }
    bool attrVal = false;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string &key, bool value) {
    SetAttr(key, value);
}

[[nodiscard]] npu::tile_fwk::Any Operation::GetAttribute(const std::string &key) const {
    return GetRawAttr(key);
}

void Operation::SetAttribute(const std::string &key, npu::tile_fwk::Any value) {
    SetAttr(key, value);
}

int Operation::GetIntAttribute(const std::string &key) const {
    ASSERT(HasAttr(key)) << "Operation doesn't have attribute " << key;
    int attrVal = 0;
    GetAttr(key, attrVal);
    return attrVal;
}

void Operation::SetAttribute(const std::string &key, int value) {
    SetAttr(key, value);
}

CastMode Operation::GetCastModeAttribute(const std::string &key) const {
    ASSERT(HasAttr(key)) << "Operation doesn't have attribute " << key;
    int attrVal = 0;
    GetAttr(key, attrVal);
    ASSERT(attrVal >= CAST_NONE && attrVal <= CAST_ODD);
    return static_cast<CastMode>(attrVal);
}

void Operation::SetAttribute(const std::string &key, CastMode value) {
    SetAttr(key, static_cast<int>(value));
}

// std::map<std::string, npu::tile_fwk::any> Operation::GetAllAttribute() const {
void Operation::SetAttribute(const std::string &key, uint64_t value) {
    SetAttr(key, value);
}

[[nodiscard]] Element Operation::GetElementAttribute(const std::string &key) const {
    ASSERT(HasAttr(key)) << "Operation doesn't have attribute " << key;
    Element attrVal;
    GetAttr(key, attrVal);
    return attrVal;
}
void Operation::SetAttribute(const std::string &key, Element value) {
    SetAttr(key, value);
}

uint64_t Operation::GetLongAttribute(const std::string &key) const {
    ASSERT(HasAttr(key)) << "Operation doesn't have attribute " << key;
    uint64_t attrVal = 0;
    GetAttr(key, attrVal);
    return attrVal;
}

std::map<std::string, npu::tile_fwk::Any> Operation::GetAllAttribute() const {
    return GetAllAttr();
}

void DebugJson(const Json &j) {
    constexpr int32_t dumpLen = 2;
    std::string s = j.dump(dumpLen);
    printf("%s\n", s.c_str());
}

Json Operation::DumpJson(bool dumpTensor) const {
    Json opDump;
    opDump[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_OPERATION);

    Json ioperandsDump = Json::array();
    Json ooperandsDump = Json::array();
    for (auto &i : iOperand) {
        if (dumpTensor) {
            ioperandsDump.push_back(i->DumpJson());
        } else {
            ioperandsDump.push_back(i->GetMagic());
        }
    }
    for (auto &o : oOperand) {
        if (dumpTensor) {
            ooperandsDump.push_back(o->DumpJson());
        } else {
            ooperandsDump.push_back(o->GetMagic());
        }
    }
    opDump["ioperands"] = ioperandsDump;
    opDump["ooperands"] = ooperandsDump;
    opDump["opcode"] = GetOpcodeStr();
    opDump["latency"] = GetLatency();

    if (IsCall()) {
        auto calleeHash = std::static_pointer_cast<CallOpAttribute>(GetOpAttribute())->GetCalleeHash();
        Function *callee = nullptr;
        for (auto &ele : Program::GetInstance().GetFunctionMap()) {
            if (ele.second->GetFunctionHash() == calleeHash) {
                callee = ele.second.get();
            }
        }
        if (callee == nullptr) {
            ALOG_ERROR_F("Cannot find function by calleeHash %s", calleeHash.c_str());
        } else {
            if (callee->rootFunc_ == nullptr) {
                opDump["calleehash"] = calleeHash.GetHash();
            } else {
                opDump["calleehash"] = callee->rootFunc_->GetFunctionHash().GetHash();
            }
        }
    }

    opDump["opmagic"] = GetOpMagic();
    opDump["semantic_label"] = semanticLabels_;

    opDump["subgraphid"] = subgraphID_;
    Json inLocation = Json::array();
    Json outLocation = Json::array();
    for (auto &inLoc : inParamLocation_) {
        inLocation.emplace_back(inLoc);
    }
    for (auto &outLoc : outParamLocation_) {
        outLocation.emplace_back(outLoc);
    }
    if (!inParamLocation_.empty()) {
        opDump["in_param_loc"] = inLocation;
        opDump["static"]["in_param_loc"] = opDump["in_param_loc"];
    }
    if (!outParamLocation_.empty()) {
        opDump["out_param_loc"] = outLocation;
        opDump["static"]["out_param_loc"] = opDump["out_param_loc"];
    }
    if (opcode_ == Opcode::OP_CALL && BelongTo()->IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::ROOT_GRAPH)) {
        auto callAttr = dynamic_cast<CallOpAttribute *>(GetOpAttribute().get());
        auto programId = callAttr->invokeInfo_->GetProgramId();
        auto programIter = function_->programs_.find(programId);
        if (programIter != function_->programs_.end()) {
            auto programFuncMagic = programIter->second->GetFuncMagic();
            opDump["program_funcmagic"] = programFuncMagic;
        } else {
            opDump["program_funcmagic"] = programFuncMagic_;
        }
        CallOpAttribute *attr = dynamic_cast<CallOpAttribute *>(GetOpAttribute().get());
        opDump["invoke_info"] = attr->DumpInvokeInfoJson();
        opDump["static"]["invoke_info"] = opDump["invoke_info"];
    }

    if (isTileOp_) {
        HashBuffer vecBuffer, cubeBuffer, distBuffer;
        tileShape_.SerializeTo(vecBuffer, cubeBuffer, distBuffer);
        opDump["tile"]["vec"] = vecBuffer;
        opDump["tile"]["cube"] = cubeBuffer;
        opDump["tile"]["comm"] = distBuffer;
    }

    if (GetOpAttribute() != nullptr) {
        opDump["attr"] = GetOpAttribute()->DumpDynJson();
    }

    for (const auto &pair : GetAttr()) {
        opDump["op_attr"][pair.first] = DumpAttrJson(pair.first);
    }

    opDump["sync_queue"] = syncQueue_.ToJson();
    opDump["static"]["sync_queue"] = opDump["sync_queue"];
    return opDump;
}

std::shared_ptr<Operation> Operation::LoadJson(
    Function &cur, const std::unordered_map<int, std::shared_ptr<LogicalTensor>> &tensorDict, const Json &opDump) {
    ASSERT(opDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_OPERATION));

    std::vector<std::shared_ptr<LogicalTensor>> ioperands;
    std::vector<std::shared_ptr<LogicalTensor>> ooperands;
    for (auto &i : opDump["ioperands"]) {
        std::shared_ptr<LogicalTensor> tensor;
        if (i.is_number()) {
            int magic = i.get<int>();
            ASSERT(tensorDict.count(magic));
            tensor = tensorDict.find(magic)->second;
        } else {
            tensor = tensorDict.find(i["magic"].get<int>())->second;
        }
        ioperands.push_back(tensor);
    }
    for (auto &o : opDump["ooperands"]) {
        std::shared_ptr<LogicalTensor> tensor;
        if (o.is_number()) {
            int magic = o.get<int>();
            ASSERT(tensorDict.count(magic));
            tensor = tensorDict.find(magic)->second;
        } else {
            tensor = tensorDict.find(o["magic"].get<int>())->second;
        }
        ooperands.push_back(tensor);
    }

    Opcode opcode = FindOpcode(opDump["opcode"].get<std::string>());
    int opMagic = opDump["opmagic"].get<int>();
    std::shared_ptr<Operation> op = std::make_shared<Operation>(cur, opcode, ioperands, ooperands, true, opMagic);

    op->semanticLabels_ =
        opDump["semantic_label"].get<std::array<std::string, static_cast<int>(SemanticLabelType::LABEL_COUNT)>>();

    int subgraphid = opDump["subgraphid"].get<int>();
    op->subgraphID_ = subgraphid;
    int latency = opDump["latency"].get<int>();
    op->UpdateLatency(latency);
    if (opDump.count("in_param_loc")) {
        for (auto &inLoc : opDump["in_param_loc"]) {
            op->inParamLocation_.emplace_back(inLoc);
        }
    }
    if (opDump.count("out_param_loc")) {
        for (auto &outLoc : opDump["out_param_loc"]) {
            op->outParamLocation_.emplace_back(outLoc);
        }
    }

    if (opDump.count("tile")) {
        HashBuffer vecBuffer = opDump["tile"]["vec"].get<HashBuffer>();
        HashBuffer cubeBuffer = opDump["tile"]["cube"].get<HashBuffer>();
        HashBuffer distBuffer = opDump["tile"]["comm"].get<HashBuffer>();
        op->isTileOp_ = true;
        op->tileShape_ = TileShape::DeserializeFrom(vecBuffer, cubeBuffer, distBuffer);
    } else {
        op->isTileOp_ = false;
    }

    if (opDump.count("attr")) {
        auto &attrJson = opDump["attr"];
        std::shared_ptr<OpAttribute> opAttribute;
        switch (opcode) {
            case Opcode::OP_VIEW: opAttribute = DeserializeFrom<ViewOpAttribute>(attrJson); break;
            case Opcode::OP_ASSEMBLE: opAttribute = DeserializeFrom<AssembleOpAttribute>(attrJson); break;
            case Opcode::OP_CALL: opAttribute = DeserializeFrom<CallOpAttribute>(opDump, &cur); break;
            case Opcode::OP_CONVERT: opAttribute = DeserializeFrom<ConvertOpAttribute>(attrJson); break;
            case Opcode::OP_COPY_IN:
            case Opcode::OP_COPY_OUT: opAttribute = DeserializeFrom<CopyOpAttribute>(attrJson); break;
            case Opcode::OP_TRANSPOSE_DATAMOVE: opAttribute = DeserializeFrom<CopyOpAttribute>(attrJson); break;
            case Opcode::OP_INDEX_OUTCAST: opAttribute = DeserializeFrom<CopyOpAttribute>(attrJson); break;
            default: break;
        }
        op->SetOpAttribute(opAttribute);
    }

    if (opDump.count("program_funcmagic")) {
        op->programFuncMagic_ = opDump["program_funcmagic"].get<int>();
    }

    if (opDump.count("op_attr") != 0) {
        auto &opAttrJson = opDump["op_attr"];
        for (auto it = opAttrJson.begin(); it != opAttrJson.end(); ++it) {
            op->LoadAttrJson(it.key(), it.value());
        }
    }

    if (opDump.count("sync_queue") != 0) {
        op->syncQueue_.FromJson(opDump["sync_queue"]);
    }
    return op;
}

std::string Operation::DumpSSA() const {
    std::ostringstream oss;
    for (size_t i = 0; i < oOperand.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << oOperand[i]->DumpSSA(false, true, true);
        if (i == oOperand.size() - 1) {
            oss << " = ";
        }
    }
    oss << "!" << GetOpMagic() << " " << GetOpcodeStr(true);
    if (opAttribute_ != nullptr) {
        oss << " " << opAttribute_->Dump();
    }

    for (size_t i = 0; i < iOperand.size(); i++) {
        if (i == 0) {
            oss << " ";
        } else if (i > 0) {
            oss << ", ";
        }
        oss << iOperand[i]->DumpSSA(false, true, false);
    }
    oss << "\n";
    return oss.str();
}

std::string Operation::DumpASM() const {
    std::ostringstream oss;
    constexpr int WIDTH_4 = 4;
    constexpr int32_t WIDTH_8 = 8;
    constexpr int32_t WIDTH_16 = 16;
    constexpr int outCastWidth = 3;

    oss << std::setw(WIDTH_8) << std::setfill(' ') << opmagic << "  ";
    oss << std::setw(WIDTH_8) << std::setfill(' ') << "latency:" << latency_ << "  ";
    oss << std::left << std::setw(WIDTH_16) << std::setfill(' ') << GetOpcodeStr(true);

    if (opAttribute_ != nullptr) {
        oss << " " << opAttribute_->Dump();
    }
    oss << std::endl;
    oss << std::setw(WIDTH_16) << std::setfill(' ') << " {";

    for (size_t i = 0; i < iOperand.size(); ++i) {
        if (IsCall()) {
            oss << "\n            INCAST[" << std::setw(outCastWidth) << std::setfill(' ') << i << "]";
        } else if (i > 0) {
            oss << ",";
        }
        oss << std::setw(WIDTH_4) << std::setfill(' ')
            << iOperand[i]->DumpASM(false); // Assuming LogicalTensor has to_json implemented
    }
    if (IsCall()) {
        oss << "\n            }\n            {";
    } else {
        oss << "} {";
    }
    for (size_t i = 0; i < oOperand.size(); ++i) {
        if (IsCall()) {
            oss << "\n            OUTCAST[" << std::setw(outCastWidth) << std::setfill(' ') << i << "]";
        } else if (i > 0) {
            oss << ",";
        }
        oss << std::setw(WIDTH_4) << std::setfill(' ')
            << oOperand[i]->DumpASM(false); // Assuming LogicalTensor has to_json implemented
        // oss << "\n  ==   " << oOperand[i]->expectedValue;
    }

    if (IsAllocOpCode(GetOpcode())) {
        for (const auto magicOutput : GetOutCtrlOperations()) {
            for (size_t i = 0; i < magicOutput->oOperand.size(); ++i) {
                if (IsCall()) {
                    oss << "\n            OUTCAST[" << std::setw(outCastWidth) << std::setfill(' ') << i << "]";
                } else if (i > 0) {
                    oss << ",";
                }
                oss << std::setw(WIDTH_4) << std::setfill(' ')
                    << magicOutput->oOperand[i]->DumpASM(false); // Assuming LogicalTensor has to_json implemented
                // oss << "\n  ==   " << oOperand[i]->expectedValue;
            }
        }
    }
    if (IsCall()) {
        oss << "\n            }\n";
    } else {
        oss << "}\n";
    }

    return oss.str();
}

std::string Operation::Dump() const {
    if (config::GetPlatformConfig("USE_SSA", true)) {
        return DumpSSA();
    } else {
        return DumpASM();
    }
}

void Operation::ReplaceInputOperand(
    const std::shared_ptr<LogicalTensor> &originInput, const std::shared_ptr<LogicalTensor> &newInput) {
    if (originInput == nullptr || newInput == nullptr) {
        return;
    }
    for (size_t i = 0; i < iOperand.size(); ++i) {
        ASSERT(iOperand[i] != nullptr);
        if (iOperand[i] == originInput) {
            iOperand[i] = newInput;
            continue;
        }
    }
}

void Operation::ReplaceOutputOperand(
    const std::shared_ptr<LogicalTensor> &originOutput, const std::shared_ptr<LogicalTensor> &newOutput) {
    if (originOutput == nullptr || newOutput == nullptr) {
        return;
    }
    for (size_t i = 0; i < oOperand.size(); ++i) {
        ASSERT(oOperand[i] != nullptr);
        if (oOperand[i] == originOutput) {
            oOperand[i] = newOutput;
            continue;
        }
    }
}

void Operation::ReplaceIOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor) {
    ASSERT(index < GetIOperands().size());
    GetIOperands()[index]->RemoveConsumer(*this);
    GetIOperands()[index] = std::move(newTensor);
    GetIOperands()[index]->AddConsumer(*this);

    operationHash_ = 0;
}

void Operation::ReplaceOOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor) {
    ASSERT(index < GetOOperands().size());
    GetOOperands()[index]->RemoveProducer(*this);
    GetOOperands()[index] = std::move(newTensor);
    GetOOperands()[index]->AddProducer(*this);

    operationHash_ = 0;
}

LogicalTensorPtr Operation::GetInputOperand(const size_t index) const {
    if (index >= iOperand.size()) {
        return nullptr;
    }
    return iOperand[index];
}

LogicalTensorPtr Operation::GetOutputOperand(const size_t index) const {
    if (index >= oOperand.size()) {
        return nullptr;
    }
    return oOperand[index];
}

std::unordered_set<Operation *> Operation::ConsumerOps() const {
    std::unordered_set<Operation *> consumers;
    for (const auto &output : GetOOperands()) {
        for (const auto &consumer : output->GetConsumers()) {
            if (consumer->BelongTo() == function_) {
                consumers.emplace(consumer);
            }
        }
    }
    return consumers;
}

std::unordered_set<Operation *> Operation::ProducerOps() const {
    std::unordered_set<Operation *> producers;
    for (const auto &input : GetIOperands()) {
        for (const auto &producer : input->GetProducers()) {
            if (producer->BelongTo() == function_) {
                producers.emplace(producer);
            }
        }
    }
    return producers;
}

void Operation::UpdateInputOperand(const size_t index, const std::shared_ptr<LogicalTensor> &newInput) {
    if (newInput == nullptr || index >= iOperand.size()) {
        return;
    }
    iOperand[index]->RemoveConsumer(this);
    iOperand[index] = newInput;
    iOperand[index]->AddConsumer(this);
}

void Operation::UpdateOutputOperand(const size_t index, const std::shared_ptr<LogicalTensor> &newOutput) {
    if (newOutput == nullptr || index >= oOperand.size()) {
        return;
    }
    oOperand[index]->RemoveProducer(this);
    oOperand[index] = newOutput;
    oOperand[index]->AddProducer(this);
}

void Operation::AddInCtrlOperation(Operation &operation) {
    if (operation.BelongTo() != BelongTo()) {
        return;
    }
    if (this->GetOpMagic() == operation.GetOpMagic()) {
        return;
    }
    inputCtrlOps.emplace(&operation);
}

void Operation::RemoveInCtrlOperation(Operation &operation) {
    auto iter = inputCtrlOps.find(&operation);
    if (iter != inputCtrlOps.end()) {
        inputCtrlOps.erase(iter);
    }
}

void Operation::AddOutCtrlOperation(Operation &operation) {
    if (operation.BelongTo() != BelongTo()) {
        return;
    }
    if (this->GetOpMagic() == operation.GetOpMagic()) {
        return;
    }
    outputCtrlOps.emplace(&operation);
}

void Operation::RemoveOutCtrlOperation(Operation &operation) {
    auto iter = outputCtrlOps.find(&operation);
    if (iter != outputCtrlOps.end()) {
        outputCtrlOps.erase(iter);
    }
}

std::string Operation::GetOpcodeStr(bool appendTile) const {
    if (!OpcodeManager::Inst().HasOpcode(opcode_)) {
        return "";
    }

    auto result = OpcodeManager::Inst().GetOpcodeStr(opcode_);
    if (appendTile && isTileOp_) {
        result = "TILE_" + result;
    }
    return result;
}

[[nodiscard]] std::string Operation::GetCoreTypeStr() const {
    switch (coreType_) {
        case CoreType::AIC: return "AIC";
        case CoreType::AIV: return "AIV";
        case CoreType::AICPU: return "AICPU";
        default: return "MIX";
    }
}

unsigned long Operation::ComputeHash() {
    // compute has every time to avoid member changed
    operationHash_ = ComputeHashOrderless();
    return operationHash_;
}

unsigned long Operation::ComputeHashOrderless() const {
    std::stringstream ss;
    ss << GetOpcodeStr(true);
    for (const auto &incast : iOperand) {
        ss << "[i";
        ss << "$" << incast->tensor->DumpSSA(false, false);
        ss << incast->DumpType();
        ss << "(";
        for (size_t i = 0; i < incast->offset.size(); ++i) {
            ss << incast->offset[i];
            if (i != incast->offset.size() - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        ss << "]";
    }

    if (opAttribute_ != nullptr) {
        ss << " " << opAttribute_->Dump();
    }
    for (const auto &attr : OpcodeManager::Inst().GetAttrs(opcode_)) {
        ss << " attr: [" << attr << " : " << DumpAttr(attr) << "]";
    }

    ss << "id" << subgraphID_;

    std::string s = ss.str();

    std::hash<std::string> hasher;
    auto result = hasher(s);
    return result;
}

bool Operation::IsCall() const {
    return opcode_ == Opcode::OP_CALL;
}

bool Operation::IsNOP() const {
    return opcode_ == Opcode::OP_NOP;
}

bool Operation::IsIsolatedOp() const {
    return iOperand.empty() && oOperand.empty() && inputCtrlOps.empty() && outputCtrlOps.empty();
}

bool Operation::OnlyHasCtrlEdgeToOp(Operation &op) const {
    if (!iOperand.empty() || !oOperand.empty() || !inputCtrlOps.empty()) {
        return false;
    }
    return outputCtrlOps.size() == 1 && outputCtrlOps.count(&op) != 0;
}

void Operation::EraseInput(const std::shared_ptr<LogicalTensor> &input) {
    for (auto iter = iOperand.begin(); iter != iOperand.end();) {
        if (iter->get()->magic == input->magic) {
            iter = iOperand.erase(iter);
        } else {
            ++iter;
        }
    }
}

void Operation::ReplaceInput(
    const std::shared_ptr<LogicalTensor> &newInput, const std::shared_ptr<LogicalTensor> &oldInput) {
    for (size_t i = 0; i < GetIOperands().size(); i++) {
        if (iOperand[i]->magic == oldInput->magic) {
            ReplaceIOperand(i, newInput);
        }
    }
}

void Operation::ReplaceOutput(
    const std::shared_ptr<LogicalTensor> &newOutput, const std::shared_ptr<LogicalTensor> &oldOutput) {
    for (int i = 0;  static_cast<size_t>(i) < oOperand.size(); ++i) {
        if (oOperand[i]->magic == oldOutput->magic) {
            ReplaceOOperand(i, newOutput);
            break;
        }
    }
}

std::string Operation::GetSemanticLabelsStr() const
{
    if (semanticLabels_.empty()) {
        return "";
    }
    std::stringstream ss;
    size_t count = 0;
    size_t size = semanticLabels_.size();
    for (const auto &label : semanticLabels_) {
        ss << label;
        if (count < size - 1) {
            ss << "_";
        }
        count++;
    }
    return ss.str();
}

void Operation::SetSubFuncInvokeInfo(const SubfuncInvokeInfoTy &invokeInfo) {
    auto callAttr = dynamic_cast<CallOpAttribute *>(opAttribute_.get());
    ASSERT(callAttr != nullptr);
    callAttr->invokeInfo_ = std::make_shared<SubfuncInvokeInfoTy>(invokeInfo);
}

int Operation::GetProgramId() {
    auto callAttr = dynamic_cast<CallOpAttribute *>(opAttribute_.get());
    ASSERT(callAttr != nullptr);
    return callAttr->invokeInfo_->GetProgramId();
}

bool Operation::IsNeedStackGM() const {
    auto isStack = [](const std::shared_ptr<LogicalTensor> &tensor) {
        return tensor->GetRawMagic() == SYMBOL_STACK_BASE;
    };

    return (OpcodeManager::Inst().IsCopyIn(opcode_) && std::any_of(iOperand.cbegin(), iOperand.cend(), isStack)) ||
           (OpcodeManager::Inst().IsCopyOut(opcode_) && std::any_of(oOperand.cbegin(), oOperand.cend(), isStack));
}
} // namespace npu::tile_fwk
