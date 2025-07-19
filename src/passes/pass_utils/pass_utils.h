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
 * \file pass_utils.h
 * \brief
 */

#pragma once

#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class FunctionUtils {
public:
    static void AddControlEdge(Operation &outOperation, Operation &inOperation);

    static void RemoveControlEdge(Operation &outOperation, Operation &inOperation);

    static void RelinkOperationInput(Operation *op, const size_t inputIndex, const Operation *targetOp,
                                     const size_t outputIndex);

    static bool IsContinuous(const std::vector<std::shared_ptr<LogicalTensor>> &tensors);
};

constexpr size_t INVALID_IN_OUT_INDEX = 0xFFFFFFFF;
// 每个调用子图的实参列表
class SubfuncInvokeInfoTy {
public:
    struct TensorParamPackTy {
        // 实参Loc和形参Loc需要检查一致
        int paramLoc;
        int ddrId;
        // the real offset of accessing tensor for this Subgraph
        std::vector<int> offset;
        std::vector<int> shape;
        std::vector<int> rawShape;
        DataType dType;
        bool isOutputToGM;
        LogicalTensorPtr tensor;
        int opMagic;
        int operandIdx;

        TensorParamPackTy(const int newParamLoc, const int newDdrId, const std::vector<int> &newOffset,
            const std::vector<int> &newShape, const std::vector<int> &newRawShape, const DataType newDtype,
            const bool newIsOutputToGM, const LogicalTensorPtr &newTensor, const int newOpMagic, int newOperandIdx)
            : paramLoc(newParamLoc), ddrId(newDdrId), offset(newOffset), shape(newShape), rawShape(newRawShape),
              dType(newDtype), isOutputToGM(newIsOutputToGM), tensor(newTensor), opMagic(newOpMagic),
              operandIdx(newOperandIdx) {}

        TensorParamPackTy() = default;

        void Print(std::ostream &osm = std::cout) const {
            osm << IntVecToStr(offset);
            osm << IntVecToStr(shape);
            osm << IntVecToStr(rawShape);
            osm << "$" << ddrId << " Loc[" << ParamLocToStr(paramLoc) << "]";
        }

        void DumpTensor(std::vector<int64_t> &invokeParam) const {
            invokeParam.emplace_back(static_cast<int64_t>(ddrId));
        }

        bool operator==(const TensorParamPackTy &other) const {
            if (paramLoc != other.paramLoc || ddrId != other.ddrId || offset != other.offset || shape != other.shape ||
                rawShape != other.rawShape || dType != other.dType || isOutputToGM != other.isOutputToGM ||
                tensor->GetMagic() != other.tensor->GetMagic() ||
                tensor->GetRawMagic() != other.tensor->GetRawMagic() || opMagic != other.opMagic) {
                return false;
            }
            return true;
        }

        bool operator!=(const TensorParamPackTy &other) const {
            return !(*this == other);
        }
    };

    struct IncastParamPackTy {
        int paramLoc;
        int ddrId;
        std::vector<int> shape;
        std::vector<int> rawShape;
        std::vector<int> offset;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;
        int operandIdx;

        IncastParamPackTy() = default;

        IncastParamPackTy(const int newParamLoc, const int newDdrId, const std::vector<int> &newOffset,
            const std::vector<int> &newShape, const std::vector<int> &newRawShape, const DataType newDtype,
            const LogicalTensorPtr &newTensor, const int newOpMagic, int newOperandIdx)
            : paramLoc(newParamLoc), ddrId(newDdrId), shape(newShape), rawShape(newRawShape), offset(newOffset),
              dType(newDtype), tensor(newTensor), opMagic(newOpMagic), operandIdx(newOperandIdx){}

        void Print(std::ostream &osm = std::cout) const {
            osm << IntVecToStr(offset);
            osm << IntVecToStr(shape);
            osm << IntVecToStr(rawShape);
            osm << "$" << ddrId << " Loc[" << ParamLocToStr(paramLoc) << "]";
        }

        void DumpIncastInfo(std::vector<int64_t> &invokeParam) const {
            invokeParam.emplace_back(static_cast<int64_t>(ddrId));
        }

        bool operator==(const IncastParamPackTy &other) const {
            if (paramLoc != other.paramLoc || ddrId != other.ddrId || offset != other.offset || shape != other.shape ||
                rawShape != other.rawShape || dType != other.dType || tensor->GetMagic() != other.tensor->GetMagic() ||
                tensor->GetRawMagic() != other.tensor->GetRawMagic() || opMagic != other.opMagic) {
                return false;
            }
            return true;
        }

        bool operator!=(const IncastParamPackTy &other) const {
            return !(*this == other);
        }
    };

    struct OutcastParamPackTy {
        int paramLoc;
        int ddrId;
        int refCount;
        std::vector<int> offset;
        std::vector<int> shape;
        std::vector<int> rawShape;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;
        int operandIdx;

        OutcastParamPackTy(const int newParamLoc, const int newDdrId, const int newRefCount,
            const std::vector<int> &newShape, const std::vector<int> &rawshape, const std::vector<int> &newOffset,
            const DataType newDtype, const LogicalTensorPtr &newTensor, const int newOpMagic, int newOperandIdx)
            : paramLoc(newParamLoc), ddrId(newDdrId), refCount(newRefCount), offset(newOffset), shape(newShape),
              rawShape(rawshape), dType(newDtype), tensor(newTensor), opMagic(newOpMagic), operandIdx(newOperandIdx) {}

        OutcastParamPackTy() = default;

        void Print(std::ostream &osm = std::cout) const {
            osm << "[RC:" << refCount << "]";
            osm << IntVecToStr(offset);
            osm << IntVecToStr(shape);
            osm << IntVecToStr(rawShape);
            osm << "$" << ddrId << " Loc[" << ParamLocToStr(paramLoc) << "]";
        }

        void DumpOutcastInfo(std::vector<int64_t> &invokeParam) const {
            invokeParam.emplace_back(static_cast<int64_t>(ddrId));
        }

        bool operator==(const OutcastParamPackTy &other) const {
            if (paramLoc != other.paramLoc || ddrId != other.ddrId || offset != other.offset || shape != other.shape ||
                rawShape != other.rawShape || dType != other.dType || tensor->GetMagic() != other.tensor->GetMagic() ||
                tensor->GetRawMagic() != other.tensor->GetRawMagic() || opMagic != other.opMagic) {
                return false;
            }
            return true;
        }

        bool operator!=(const OutcastParamPackTy &other) const {
            return !(*this == other);
        }
    };

public:
    inline void UpdateProgramSubgraphId(const int psgId) { programSubgraphId_ = psgId; }

    inline int GetProgramId() const { return programSubgraphId_; }

    void ConstructActualInvokeParam(int esgId);

    void PrintInvokeInfo(const std::string &extraInfo) const;

    void PrettyPrintInvokeInfo(const int subgraphId) const;

    void DumpInvokeInfo(int64_t invokeParamMemOffset, int64_t *invokeParamPtr) const;

    inline const std::vector<TensorParamPackTy> &GetTensorParamList() const { return tensorParamList_; }

    inline const std::vector<IncastParamPackTy> &GetIncastTensorParamList() const { return incastTensorParamList_; }

    inline const std::vector<OutcastParamPackTy> &GetOutcastTensorParamList() const { return outcastTensorParamList_; }

    std::tuple<int, int, int> LookupInvokeArgs(const int paramLoc) const;

    bool operator==(const SubfuncInvokeInfoTy &other) const;
    bool operator!=(const SubfuncInvokeInfoTy &other) const;
    friend class Allocator;
private:
    int programSubgraphId_; // The called merged subgraph id
    std::vector<TensorParamPackTy> tensorParamList_;
    std::vector<IncastParamPackTy> incastTensorParamList_;
    std::vector<OutcastParamPackTy> outcastTensorParamList_;

public:
    // seq_no is in called subgraph
    struct InCastInfoTy {
        int seqNo;
        int operandIdx;
        int realIncastDDRId;
        std::vector<int> offset;
        std::vector<int> shape;
        std::vector<int> rawShape;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;

        InCastInfoTy(const int newSeqNo, const int newOperandIdx, const int newRealIncastDDRId,
            const std::vector<int> &newOffset, const std::vector<int> &newShape, const std::vector<int> &newRawShape,
            const DataType dtype, const LogicalTensorPtr &newTensor, const int newOpMagic)
            : seqNo(newSeqNo), operandIdx(newOperandIdx), realIncastDDRId(newRealIncastDDRId), offset(newOffset),
              shape(newShape), rawShape(newRawShape), dType(dtype), tensor(newTensor), opMagic(newOpMagic) {}
    };

    // Input output tensors of this subgraph invoke
    struct TensorInfoTy {
        int seqNo;
        int operandIdx;
        int realDDRId;
        std::vector<int> offset;
        std::vector<int> shape;
        std::vector<int> rawShape;
        DataType dType;
        bool isOutputToGM;
        LogicalTensorPtr tensor;
        int opMagic;

        TensorInfoTy(const int newSeqNo, const int newOperandIndex, const int newRealDDRId,
            const std::vector<int> &newOffset, const std::vector<int> &newShape, const std::vector<int> &newRawShape,
            const DataType newDtype, const bool newIsOutputToGM, const LogicalTensorPtr &newTensor, const int newOpMagic)
            : seqNo(newSeqNo), operandIdx(newOperandIndex), realDDRId(newRealDDRId), offset(newOffset),
              shape(newShape), rawShape(newRawShape), dType(newDtype), isOutputToGM(newIsOutputToGM),
              tensor(newTensor), opMagic(newOpMagic) {}
    };

    using TensorArgsTy = std::vector<TensorInfoTy>;
    // Incast connections
    using ExeSubgraphEdgeTy = std::tuple<int, int, InCastInfoTy>;
    // record all the connections for input subgraphs
    using ESgConnectionsTy = std::vector<ExeSubgraphEdgeTy>;

    struct SuccessorIncastRecTy {
        int successorESgId;
        int connectedOperandIdx;
        ExeSubgraphEdgeTy *successorIncast;
        int opMagic;

        SuccessorIncastRecTy(const int esgId, const int opIdx, ExeSubgraphEdgeTy *exeSubgraphEdgeTy,
            const int newOpMagic) : successorESgId(esgId), connectedOperandIdx(opIdx),
            successorIncast(exeSubgraphEdgeTy), opMagic(newOpMagic) {}
    };

    using SuccessorIncastInfoTy = std::vector<SuccessorIncastRecTy>;
    struct OutCastInfoTy {
        int srcESgId;
        int seqNo;
        int operandIdx;
        int refCount;
        int realOutCastDDRId;
        SuccessorIncastInfoTy successorIncastInfo;
        std::vector<int> offset;
        std::vector<int> shape;
        std::vector<int> rawShape;
        DataType dType;
        LogicalTensorPtr tensor;
        int opMagic;

        OutCastInfoTy(const int newSrcESgId, const int newSeqNo, int newOperandIdx, const int newRefCount, const int newDdrId,
            const SuccessorIncastInfoTy &info, const std::vector<int> &newOffset, const std::vector<int> &newShape,
            const std::vector<int> &newRawShape, const DataType dtype, const LogicalTensorPtr &newTensor,
            const int newOpMagic)
            : srcESgId(newSrcESgId), seqNo(newSeqNo), operandIdx(newOperandIdx), refCount(newRefCount), realOutCastDDRId(newDdrId),
              successorIncastInfo(info), offset(newOffset), shape(newShape), rawShape(newRawShape), dType(dtype),
              tensor(newTensor), opMagic(newOpMagic) {}

        OutCastInfoTy() = default;
    };
    using OutCastConnectionsTy = std::vector<OutCastInfoTy>;

public:
    inline void RecordTensorArg(const int seqNo, const int operandIdx, const int realDDRId,
        const std::vector<int> &offset, const std::vector<int> &shape, const std::vector<int> &rawShape,
        const DataType dtype, const bool isOutputToGM, const LogicalTensorPtr &tensor, const int opMagic) {
        tensorArgs_.emplace_back(seqNo, operandIdx, realDDRId, offset, shape, rawShape, dtype, isOutputToGM, tensor,
                                opMagic);
    }

    // Record Incast connection, build relation shape with outcast records
    inline void RecordConnection(const int srcESgId, const int dstESgId, const int tgtSeqNo, const int operandIndex,
        const int realIncastDDRId, const std::vector<int> &offset, const std::vector<int> &shape,
        const std::vector<int> &rawShape, const DataType dtype, const LogicalTensorPtr &tensor, const int opMagic) {
        connections_.emplace_back(srcESgId, dstESgId,
            InCastInfoTy{tgtSeqNo, operandIndex, realIncastDDRId, offset, shape, rawShape, dtype, tensor, opMagic});
    }

    inline void RecordOutcast(const int srcESgId, const int srcSeqNo, int srcOperandIdx, const int refCount, const int realOutcastDDRId,
        const SuccessorIncastInfoTy &incasts, const std::vector<int> &offset, const std::vector<int> &shape,
        const std::vector<int> &rawShape, const DataType dtype, const LogicalTensorPtr &tensor, const int opMagic) {
        outCasts_.emplace_back(
            srcESgId, srcSeqNo, srcOperandIdx, refCount, realOutcastDDRId, incasts, offset, shape, rawShape, dtype, tensor, opMagic);
    }

    // do some sorting after recording all infomations
    void DoFinishRecord();

    const ESgConnectionsTy &GetIncasts() const { return connections_; }

    const OutCastConnectionsTy &GetOutcasts() const { return outCasts_; }

    const TensorArgsTy &GetTensorArgs() const { return tensorArgs_; }

    CoreType GetGraphType() const { return graphType_; }

    void SetGraphType(const CoreType graphType) { graphType_ = graphType; }

    Json ToJson() const;
    Json DumpJson() const;
    void LoadJson(const Json &invokeInfoJson, Function *belongTo);
    void Print(const std::string &extInfo) const;

private:
    CoreType graphType_{CoreType::AIV};
    TensorArgsTy tensorArgs_;
    ESgConnectionsTy connections_; // InCast
    OutCastConnectionsTy outCasts_;
    bool isFinalized_{false};
};

class SubfuncParam {
public:
    struct InCastParamTy {
        int paramLoc;
        int seqNo;
        int operandIdx;
        int symDDRId;
        std::vector<int> shape;
        std::vector<int> offset;
        std::string symName;
        std::string symbol;
        DataType dataType;

        InCastParamTy(const int newSeqNo, const int newOperandIdx, const int newSymDDRId, const std::vector<int> &newShape,
            const std::vector<int> &newOffset, const std::string &newSymName, const int newParamLoc,
            const std::string newSymbol = "", const DataType newDataType = DataType::DT_BOTTOM)
            : paramLoc(newParamLoc), seqNo(newSeqNo), operandIdx(newOperandIdx), symDDRId(newSymDDRId), shape(newShape),
              offset(newOffset), symName(newSymName), symbol(newSymbol), dataType(newDataType) {}

        void Print(std::ostream &osm = std::cout) const {
            osm << "INCAST";
            osm << IntVecToStr(offset);
            osm << IntVecToStr(shape);
            osm << symName << " Loc[" << ParamLocToStr(paramLoc) << "]\n";
        }
    };

    struct OutCastParamTy {
        int paramLoc;
        int seqNo;
        int operandIdx;
        int symDDRId;
        int refCount;
        std::vector<int> offset;
        std::vector<int> shape;
        std::string symName;
        std::string symbol;
        DataType dataType;

        OutCastParamTy(const int newSeqNo, const int newOperandIdx, const int newSymDDRId, const int newRefCount,
            const std::vector<int> &newShape, const std::vector<int> &newOffset, const std::string &newSymName,
            const int newParamLoc, const std::string newSymbol = "",
            const DataType newDataType = DataType::DT_BOTTOM)
            : paramLoc(newParamLoc), seqNo(newSeqNo), operandIdx(newOperandIdx), symDDRId(newSymDDRId),
              refCount(newRefCount), offset(newOffset), shape(newShape), symName(newSymName), symbol(newSymbol),
              dataType(newDataType) {}

        void Print(std::ostream &osm = std::cout) const {
            osm << "OUTCAST";
            osm << "[" << refCount << "]";
            osm << IntVecToStr(offset);
            osm << IntVecToStr(shape);
            osm << symName << " Loc[" << ParamLocToStr(paramLoc) << "]" << std::endl;
        }
    };

    struct TensorParamTy {
        int paramLoc;
        int seqNo;
        int operandIdx;
        int symDDRId;
        std::vector<int> symOffset;
        std::vector<int> shape;
        std::string symName;
        std::string symbol;
        DataType dataType;

        TensorParamTy(const int newSeqNo, const int newOperandIdx, const int newSymDDRId,
            const std::vector<int> &newShape, const std::vector<int> &newOffset, const std::string &newSymName,
            const int newParamLoc, const std::string newSymbol = "",
            const DataType newDataType = DataType::DT_BOTTOM)
            : paramLoc(newParamLoc), seqNo(newSeqNo), operandIdx(newOperandIdx), symDDRId(newSymDDRId),
              symOffset(newOffset), shape(newShape), symName(newSymName), symbol(newSymbol), dataType(newDataType) {}

        void Print(std::ostream &osm = std::cout) const {
            osm << IntVecToStr(symOffset);
            osm << IntVecToStr(shape);
            osm << symName << " Loc[" << ParamLocToStr(paramLoc) << "]" << std::endl;
        }
    };

    using OutCastParamListTy = std::vector<OutCastParamTy>;
    using InCastParamListTy = std::vector<InCastParamTy>;
    using TensorParamListTy = std::vector<TensorParamTy>;

public:
    void AppendIncastParam(const int seqNo, const int operandIdx, const int symDDRId, const std::vector<int> &shape,
        const std::vector<int> &offset, const std::string &symName, const int paramLoc, const std::string &symbol,
        const DataType dataType) {
        inCastArgs_.emplace_back(
            InCastParamTy(seqNo, operandIdx, symDDRId, shape, offset, symName, paramLoc, symbol, dataType));
    }

    void AppendOutcastParam(const int seqNo, const int operandIdx, const int symDDRId, const int refCount,
        const std::vector<int> &shape, const std::vector<int> &offset, const std::string &symName, const int paramLoc,
        const std::string &symbol, const DataType dataType) {
        outCastArgs_.emplace_back(
            OutCastParamTy(seqNo, operandIdx, symDDRId, refCount, shape, offset, symName, paramLoc, symbol, dataType));
    }

    void AppendTensorParam(const int seqNo, const int operandIdx, const int symDDRId, const std::vector<int> &shape,
        const std::vector<int> &offset, const std::string &symName, const int paramLoc, const std::string &symbol,
        const DataType dataType) {
        tensorsArgs_.emplace_back(
            TensorParamTy(seqNo, operandIdx, symDDRId, shape, offset, symName, paramLoc, symbol, dataType));
    }

    void Finalize() {
        isFinalized_ = true;
    }

    void PrettyPrint(const int psgId, std::ostream &osm = std::cout) const {
        osm << "PARAM_LIST[" << psgId << "]:\n";
        for (auto &tensor : tensorsArgs_) {
            osm << "|--";
            tensor.Print(osm);
        }

        for (auto &ins : inCastArgs_) {
            osm << "|--";
            ins.Print(osm);
        }

        for (auto &outs : outCastArgs_) {
            osm << "|--";
            outs.Print(osm);
        }
    }
    Json ToJson() const;
    void FromJson(const Json &params);
public:
    TensorParamListTy tensorsArgs_;
    InCastParamListTy inCastArgs_;
    OutCastParamListTy outCastArgs_;
    bool isFinalized_ = false;
};

class SubfuncTopologyInfoTy {
    struct Entry {
        int esgId;
        int readyState;
        setType outGraph;
        uint32_t extType{0};
        uint32_t extParamNum{0};
        std::vector<int> extParams;
    };

public:
    void SetTableSize(const int n) { topology_.reserve(n); }

    const std::vector<Entry> &GetTopology() const { return topology_; }

    void SetMaxM(const int maxM) { maxM_ = maxM; }

    void AddEntry(const int esgId, const int readState, const setType &succ);
    
    void UpdateEntry(const uint32_t extType, const uint32_t extParamNum, const std::vector<int> &extParams);

    std::vector<int> TopoSort();

    void Print(std::ostream &osm = std::cout) const;

    void DumpEachEntryInfo(
        int esgId, CoreType coreType, int64_t entryOffset, int64_t *entryParamPtr, int32_t *readyStatePtr) const;

    bool IsEsgReady(const int esgId) const;

    std::vector<int> GetSuccs(int esgId) const {
        std::vector<int> succs;
        for (auto &entry : topology_) {
            if (esgId == entry.esgId) {
                succs.insert(succs.end(), entry.outGraph.begin(), entry.outGraph.end());
                break;
            }
        }
        return succs;
    }

    Json DumpJson() const;
    void LoadJson(const Json &topoJson);
public:
    int maxM_;
    std::vector<Entry> topology_;
    std::vector<int> readyIds_;
};
}