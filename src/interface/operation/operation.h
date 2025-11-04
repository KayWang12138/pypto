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
 * \file operation.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>
#include <variant>
#include <nlohmann/json.hpp>
#include "interface/inner/any.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/config.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "interface/utils/common.h"
#include "opcode.h"
#include "attribute.h"
#include "attr_holder.h"
#include "interface/utils/log.h"
#include "interface/utils/source_location.h"
#include "interface/tensor/logical_tensor.h"
#include "operation_common.h"

using Json = nlohmann::json;

namespace npu::tile_fwk {
constexpr size_t NON_GROUP = -1;
constexpr int32_t TILE_STR_PREFIX_LEN = 5;

#define AICPU_CALL_NUM_COPYOUT_RESOLVE 1
#define AICPU_CALL_NUM_BIT 16
#define AICPU_CALL_ARG_BIT 16
#define AICPU_CALL_TASK_BIT 32

class OpAttributeKey {
public:
    static const std::string aicpuCall;
    static const std::string scalar;
    static const std::string dynScalar;
    static const std::string isGlobalInput;
    static const std::string seqNo;
    static const std::string color;
    static const std::string isCube;
    static const std::string blockPadding;
    static const std::string broadcastLastAxis;
    static const std::string tilePadding;
    static const std::string reshapePadding;
    static const std::string shapePadded;
    static const std::string needAlloc;
    static const std::string dontTouch;
    static const std::string tag;
    static const std::string commGroupInfo;
    static const std::string distTilingInfo;
    static const std::string sameInOut;
    static const std::string inputCombineAxis;
    static const std::string outputCombineAxis;
    static const std::string inputCombineAxisDone;
    static const std::string outputCombineAxisDone;
    static const std::string inplaceIdx;
    static const std::string inplaceInfo;
    static const std::string cacheMode;
    static const std::string panzBlockSize;
    static const std::string requiresBoundaryCopy;
    static const std::string excludeBufferReuse;
    static const std::string bindTensor;
    static const std::string startOffset;
};


class ConvOpAttributeKey {
public:
    static const std::string cin;
    static const std::string cout;
    static const std::string paddingLeft;
    static const std::string paddingTop;
    static const std::string paddingRight;
    static const std::string paddingBottom;
    static const std::string strideh;
    static const std::string stridew;
    static const std::string hposX;
    static const std::string hsteP;
    static const std::string wposX;
    static const std::string wstep;
    static const std::string hoffsetY;
    static const std::string woffsetY;
    static const std::string reluType;
    static const std::string reluAlpha;
    static const std::string clearFlag;
    static const std::string hasAccFlag;
    static const std::string hasEltFlag;
    static const std::string hasBiasFlag;
    static const std::string eltBrcbFlag;
    static const std::string fmapSrcNum;
    static const std::string eltMode;
    static const std::string fmapC0;
};

class FixpOpAttributeKey {
public:
    static const std::string hStart;
    static const std::string hEnd;
    static const std::string quantPreScalar;
    static const std::string quantPostScalar;
    static const std::string antiqScalar;
    static const std::string hasQuantPreVector;
    static const std::string hasQuantPostVector;
    static const std::string hasAntiqVector;
    static const std::string fbAddrSpace;
};

class PoolOpAttributeKey {
public:
    static const std::string poolh;
    static const std::string poolw;
};

enum class FbBufferSpace { QUANT_PRE = 0, RELU_PRE, RELU_POST, QUANT_POST, ANTIQ_ELT, ANTIQ_MTE2 };

class Function;
// Class to represent an operation (opcode) and its operands
class Operation : public std::enable_shared_from_this<Operation>, public AttrHolder {
public:
    friend class Function;
    LogicalTensors iOperand; // Input operands (now actual objects, not shared_ptr)
    LogicalTensors oOperand; // Output operands (now actual objects, not shared_ptr)
    int opmagic; // The magic number for the operation, default value -1
    int programFuncMagic_; // function magic of leafFunction
    int outcastRefcount{0};

    int cycles{0};
    int cycleStart{0};
    int cycleEnd{0};
    int cubeDepId{-1};

    std::vector<int> inParamLocation_;
    std::vector<int> outParamLocation_;
    OpSyncQueue syncQueue_;
    QueueType queueType;

    // Constructor to initialize the opcode, input operands, output operands, and opmagic
    Operation(Function &cur, Opcode opcode, LogicalTensors iOperands, LogicalTensors oOperands,
        bool updateTensorMap = true, int opMagic = -1);

    Operation(Function &cur, Opcode opcode): Operation(cur, opcode, {}, {}, false) {}

    Operation(Function &cur, const std::string &op, const LogicalTensors &input, const LogicalTensors &output,
        bool updateTensormap = true)
        : Operation(cur, FindOpcode(op), input, output, updateTensormap) {
        if (op.substr(0, TILE_STR_PREFIX_LEN) == "TILE_") {
            isTileOp_ = true;
        }
    };

    Operation(const Operation &other) = delete;
    Operation(Operation &&other) = delete;
    Operation &operator=(const Operation &other) = delete;
    Operation &operator=(Operation &&other) = delete;

    Function *BelongTo() const { return function_; }

    const QueueType &GetQueueType() const { return queueType; }

    const OpSyncQueue &GetSyncQueue() const { return syncQueue_; }

    const TileShape &GetTileShape() const { return tileShape_; }
    void UpdateTileShape(const TileShape newTileShape) { tileShape_ = newTileShape; }

    TileShape &GetTileShapeForSetting() { return tileShape_; }

    [[nodiscard]] std::string GetStringAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, const std::string &value);

    [[nodiscard]] bool GetBoolAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, bool value);

    [[nodiscard]] int64_t GetIntAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, int64_t value);
    void SetAttribute(const std::string &key, int value) { SetAttribute(key, static_cast<int64_t>(value)); }

    [[nodiscard]] Element GetElementAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, Element value);

    template<typename T = int64_t>
    std::vector<T> GetVectorIntAttribute(const std::string &key) const {
        static_assert(std::is_integral_v<T>);
        std::vector<int64_t> val;
        GetAttr(key, val);
        if constexpr (std::is_same_v<T, int64_t>) {
            return val;
        }
        std::vector<T> ret;
        for (auto &x : val) {
            ret.emplace_back(static_cast<T>(x));
        }
        return ret;
    }

    template<typename T = int64_t>
    void SetAttribute(const std::string &key, const std::vector<T> &value) {
        static_assert(std::is_integral_v<T>);
        if constexpr (std::is_same_v<T, int64_t>) {
            SetAttr(key, value);
        } else {
            std::vector<int64_t> nvalue;
            for (auto &x : value) {
                nvalue.emplace_back(static_cast<int64_t>(x));
            }
            SetAttr(key, nvalue);
        }
    }

    [[nodiscard]] CastMode GetCastModeAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, CastMode value);

    [[nodiscard]] SymbolicScalar GetSymbolicScalarAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, const SymbolicScalar &value);

    [[nodiscard]] std::vector<SymbolicScalar> GetVectorSymbolicScalarAttribute(const std::string &key) const;
    void SetAttribute(const std::string &key, const std::vector<SymbolicScalar> &value);

    [[nodiscard]] bool HasAttribute(const std::string &key) const {
        return HasAttr(key);
    }

    [[nodiscard]] std::map<std::string, npu::tile_fwk::Any> GetAllAttribute() const;

    Json DumpJson(bool dumpTensor = true) const;
    static std::shared_ptr<Operation> LoadJson(Function &cur,
        const std::unordered_map<int, std::shared_ptr<LogicalTensor>> &tensorDict, const Json &opDump);

    [[nodiscard]] std::string DumpSSA(const std::string &prefix="") const;

    [[nodiscard]] std::string Dump() const;

    [[nodiscard]] int GetOpMagic() const { return opmagic; }

    [[nodiscard]] const LogicalTensors &GetIOperands() const { return iOperand; }
    LogicalTensors &GetIOperands() { return iOperand; }

    [[nodiscard]] const LogicalTensors &GetOOperands() const { return oOperand; }
    LogicalTensors &GetOOperands() { return oOperand; }

    size_t GetInputOperandSize() const { return iOperand.size(); }

    size_t GetOutputOperandSize() const { return oOperand.size(); }

    LogicalTensorPtr GetInputOperand(const size_t index) const;

    LogicalTensorPtr GetOutputOperand(const size_t index) const;

    int GetIOperandIndex(const LogicalTensorPtr &ioperand) const;
    int GetOOperandIndex(const LogicalTensorPtr &ooperand) const;

    void ReplaceInputOperand(const LogicalTensorPtr &originInput, const LogicalTensorPtr &newInput);

    void ReplaceOutputOperand(const LogicalTensorPtr &originOutput, const LogicalTensorPtr &newOutput);

    void UpdateInputOperand(const size_t index, const std::shared_ptr<LogicalTensor> &newInput);

    void UpdateOutputOperand(const size_t index, const std::shared_ptr<LogicalTensor> &newOutput);

    std::unordered_set<Operation *> ConsumerOps() const;
    std::unordered_set<Operation *> ProducerOps() const;

    class OperationComparator {
    public:
        bool operator()(const Operation *lhs, const Operation *rhs) const {
            return lhs->GetOpMagic() < rhs->GetOpMagic();
        }
    };
    std::set<Operation *, OperationComparator> ConsumerOpsOrdered() const;
    std::set<Operation *, OperationComparator> ProducerOpsOrdered() const;

    [[nodiscard]] const std::unordered_set<Operation *> &GetInCtrlOperations() const { return inputCtrlOps; }

    [[nodiscard]] const std::unordered_set<Operation *> &GetOutCtrlOperations() const { return outputCtrlOps; }

    void ClearInCtrlOperations() { inputCtrlOps.clear(); }

    void ClearOutCtrlOperations() { outputCtrlOps.clear(); }

    void AddInCtrlOperation(Operation &operation);

    void RemoveInCtrlOperation(Operation &operation);

    void AddOutCtrlOperation(Operation &operation);

    void RemoveOutCtrlOperation(Operation &operation);

    Operation &CloneOperation(
        Function &func, const LogicalTensors &iOperandList, const LogicalTensors &oOperandList) const;

    [[nodiscard]] std::string GetOpcodeStr(bool appendTile = false) const;
    [[nodiscard]] CoreType GetCoreType() const { return coreType_; }
    void SetCoreType(CoreType ct) { coreType_ = ct; }
    [[nodiscard]] std::string GetCoreTypeStr() const;

    unsigned long ComputeHash();
    unsigned long ComputeHashOrderless() const;
    [[nodiscard]] bool IsCall() const;
    [[nodiscard]] bool IsNOP() const;

    bool IsIsolatedOp() const;

    bool OnlyHasCtrlEdgeToOp(Operation &op) const;

    const std::shared_ptr<OpAttribute> &GetOpAttribute() const { return opAttribute_; }
    std::shared_ptr<OpAttribute> &GetOpAttribute() { return opAttribute_; }

    void SetOpAttribute(const std::shared_ptr<OpAttribute> &attr) {
        opAttribute_ = attr;
        switch (opcode_) {
            case Opcode::OP_VIEW: {
                ASSERT(dynamic_cast<ViewOpAttribute *>(opAttribute_.get()) != nullptr);
                break;
            }
            case Opcode::OP_ASSEMBLE: {
                ASSERT(dynamic_cast<AssembleOpAttribute *>(opAttribute_.get()) != nullptr);
                break;
            }
            case Opcode::OP_CALL: {
                ASSERT(dynamic_cast<CallOpAttribute *>(opAttribute_.get()) != nullptr);
                break;
            }
            case Opcode::OP_CONVERT: {
                ASSERT(dynamic_cast<ConvertOpAttribute *>(opAttribute_.get()) != nullptr);
                break;
            }
            case Opcode::OP_L1_COPY_IN:
            case Opcode::OP_L1_COPY_OUT:
            case Opcode::OP_COPY_IN:
            case Opcode::OP_L1_TO_BT:
            case Opcode::OP_L1_TO_FB:
            case Opcode::OP_COPY_OUT:
            case Opcode::OP_RESHAPE_COPY_IN:
            case Opcode::OP_RESHAPE_COPY_OUT:
            case Opcode::OP_INDEX_OUTCAST:
            case Opcode::OP_TRANSPOSE_MOVEIN:
            case Opcode::OP_TRANSPOSE_MOVEOUT:
            case Opcode::OP_REMOTE_GATHER:
            case Opcode::OP_LOCAL_COPY_OUT:
            case Opcode::OP_REMOTE_REDUCE:
            case Opcode::OP_FFN_SCHED:
            case Opcode::OP_FFN_BATCHING:
            case Opcode::OP_SHMEM_PUT:
            case Opcode::OP_SHMEM_PUT_UB2GM:
            case Opcode::OP_SHMEM_SIGNAL:
            case Opcode::OP_SHMEM_GET:
            case Opcode::OP_SHMEM_GET_GM2UB:
            case Opcode::OP_SHMEM_REDUCE:
            case Opcode::OP_COPY_TO_LOCAL_EXPERT: {
                ASSERT(dynamic_cast<CopyOpAttribute *>(opAttribute_.get()) != nullptr);
                break;
            }
            default: ASSERT(opAttribute_ == nullptr);
        }
    }

    void SetAssembleOpAttribute(
        const std::vector<int64_t> &toOffset, const std::vector<SymbolicScalar> &toDynOffset = {}) {
        ASSERT(opcode_ == Opcode::OP_ASSEMBLE);
        SetOpAttribute(std::make_shared<AssembleOpAttribute>(toOffset, toDynOffset));
    }

    void ReplaceIOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor);
    void ReplaceOOperand(size_t index, std::shared_ptr<LogicalTensor> newTensor);

    std::string GetCalleeMagicName() const {
        ASSERT(IsCall());
        return std::static_pointer_cast<CallOpAttribute>(opAttribute_)->GetCalleeMagicName();
    }

    const std::string &GetCalleeBracketName() const {
        return std::static_pointer_cast<CallOpAttribute>(opAttribute_)->GetCalleeBracketName();
    }

    const FunctionHash &GetCalleeHash() const {
        ASSERT(IsCall() || opcode_ == Opcode::OP_CALL_NOT_EXPAND);
        CallOpAttribute *callop = static_cast<CallOpAttribute *>(opAttribute_.get());
        return callop->GetCalleeHash();
    }

    void EraseInput(const std::shared_ptr<LogicalTensor> &input);
    void ReplaceInput(const std::shared_ptr<LogicalTensor> &newInput, const std::shared_ptr<LogicalTensor> &oldInput);
    void ReplaceOutput(const std::shared_ptr<LogicalTensor> &newOutput, const std::shared_ptr<LogicalTensor> &oldOutput);

    Opcode GetOpcode() const { return opcode_; }

    void SetOpCode(Opcode opcode) { opcode_ = opcode; }
    int GetLatency() const { return latency_; }
    void UpdateLatency(int latency) { latency_ = latency; }

    int GetRemainingTime() const { return remainingTime_; }
    void UpdateRemainingTime(int remainingTime) { remainingTime_ = remainingTime; }

    int GetSubgraphID() const { return subgraphID_; }
    void UpdateSubgraphID(int subgraphID) { subgraphID_ = subgraphID; }

    auto GroupID() const { return groupID_; }
    void SetGroupID(size_t groupID) const { groupID_ = groupID; }

    void SetSemanticLabel(const std::string &label) { semanticLabel_ = label; }
    const std::string &GetSemanticLabel() const { return semanticLabel_; }

    void SetAsDeleted() { isDeleted_ = true; }
    void SetAsNotDeleted() { isDeleted_ = false; }
    [[nodiscard]] bool IsDeleted() const { return isDeleted_; }

    void SetSubFuncInvokeInfo(const SubfuncInvokeInfoTy &invokeInfo);

    SubfuncInvokeInfoTy &GetSubFuncInvokeInfo() {
        auto callAttr = dynamic_cast<CallOpAttribute *>(opAttribute_.get());
        ASSERT(callAttr != nullptr);
        return *(callAttr->invokeInfo_);
    }

    int GetProgramId();

    bool IsNeedStackGM() const;

    int GetIOpAttrOffset(int pos) const {
        return iOpAttrOffset.empty() ? -1 : iOpAttrOffset[pos];
    }
    int GetOOpAttrOffset(int pos) const {
        return oOpAttrOffset.empty() ? -1 : oOpAttrOffset[pos];
    }
    void SetIOpAttrOffset(int pos, int offset) {
        if (iOpAttrOffset.empty())
            iOpAttrOffset.resize(iOperand.size(), -1);
        iOpAttrOffset[pos] = offset;
    }
    void SetOOpAttrOffset(int pos, int offset) {
        if (oOpAttrOffset.empty())
            oOpAttrOffset.resize(oOperand.size(), -1);
        oOpAttrOffset[pos] = offset;
    }
    void SetOpOffset(const std::vector<int> &iOffset, const std::vector<int> &oOffset) {
        iOpAttrOffset = iOffset;
        oOpAttrOffset = oOffset;
    }

    std::vector<std::reference_wrapper<SymbolicScalar>> GetDynamicAttributeList();
    SourceLocationPtr GetLocation() const { return location_; }

    const std::vector<std::string> &GetCommentList() const { return commentList_; }
    std::vector<std::string> &GetCommentList() { return commentList_; }

private:
    Opcode opcode_{Opcode::OP_UNKNOWN};
    int subgraphID_{NOT_IN_SUBGRAPH};
    bool isTileOp_{false};
    TileShape tileShape_;
    std::shared_ptr<OpAttribute> opAttribute_;
    unsigned long operationHash_{0};
    int latency_{1};

    std::vector<int> iOpAttrOffset;
    std::vector<int> oOpAttrOffset;
    int remainingTime_{INVALID_TIME};
    CoreType coreType_{CoreType::MIX};
    std::unordered_set<Operation *> inputCtrlOps;
    std::unordered_set<Operation *> outputCtrlOps;
    mutable size_t groupID_{NON_GROUP};
    bool isDeleted_{false};

    SourceLocationPtr location_ {nullptr};
    std::string semanticLabel_;
    Function *function_;

    std::vector<std::string> commentList_;
};
using OperationPtr = std::shared_ptr<Operation>;

// Custom comparator for Operation in magic order
struct OperationCmp {
    bool operator()(const Operation *lhs, const Operation *rhs) const;
};

/*  ！！！！！！！！！对外开放OP接口请在 tilefwk_op.h 中添加 ！！！！！！！！！！！！！！！！！！！！*/

} // namespace npu::tile_fwk
