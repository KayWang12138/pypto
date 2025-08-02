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
 * \file logical_tensor.h
 * \brief
 */

#pragma once
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <unordered_set>
#include <functional>
#include "tilefwk/data_type.h"
#include "common/pre_def.h"
#include "raw_tensor.h"
#include "interface/operation/attr_holder.h"
#include "symbolic_scalar.h"
#include "tensor_offset.h"

#include <nlohmann/json.hpp>
#include "storage.h"
using Json = nlohmann::json;

namespace npu::tile_fwk {
class TileRange {
public:
    size_t start;
    size_t end; // exclusive end point
    int lifeStart = -1;
    int lifeEnd = -1;
    int memId = -1; // default = tensor->raw_id

    explicit TileRange(size_t s = 0, size_t e = 0, int id = -1) : start(s), end(e), memId(id) {}

    // Helper functions
    size_t Size() const { return end - start; }
    bool IsEmpty() const { return end <= start; }
};

class LogicalTensor : public AttrHolder {
public:
    bool isSubGraphBoundary;
    int subGraphID{NOT_IN_SUBGRAPH};

    std::shared_ptr<RawTensor> tensor;
    std::vector<int> offset;
    std::vector<int> shape;
    std::vector<int> oriShape;
    std::vector<SymbolicScalar> dynOffset_;
    std::vector<SymbolicScalar> dynValidShape_;

    std::vector<int> storageShape;
    std::shared_ptr<Storage> storage_ = nullptr;
    uint64_t storageOffset_ = 0;
    int magic;
    NodeType nodetype;
    TileOpFormat tensorfmt;

    std::vector<std::weak_ptr<LogicalTensor>> conflicterTensors;
    std::vector<std::shared_ptr<LogicalTensor>> overlapper;

    std::map<int, TileRange> memorymap; // subgraphID -> memoryRange

    LogicalTensor(Function &function, DataType t, std::vector<int> tshape, std::string tname = "",
        NodeType tnodetype = NodeType::LOCAL, TileOpFormat ttensorfmt = TileOpFormat::TILEOP_ND);
    LogicalTensor(Function &function, DataType t, std::vector<int> tshape, std::vector<SymbolicScalar> tValidShape,
        std::string tname = "", NodeType tnodetype = NodeType::LOCAL, TileOpFormat ttensorfmt = TileOpFormat::TILEOP_ND);
    LogicalTensor(Function &function, std::shared_ptr<RawTensor> rawTensor, std::vector<int> toffset,
        std::vector<int> tshape, NodeType tnodetype = NodeType::LOCAL, TileOpFormat ttensorfmt = TileOpFormat::TILEOP_ND);
    LogicalTensor(Function &function, std::shared_ptr<RawTensor> rawTensor, std::vector<int> toffset,
        std::vector<int> tshape, std::vector<SymbolicScalar> tValidShape, NodeType tnodetype = NodeType::LOCAL, TileOpFormat ttensorfmt = TileOpFormat::TILEOP_ND);
    LogicalTensor(LogicalTensor &&) = default;
    LogicalTensor(const LogicalTensor &) = default;
    LogicalTensor &operator=(LogicalTensor &&) = delete;
    LogicalTensor &operator=(const LogicalTensor &) = delete;
    std::shared_ptr<LogicalTensor> Clone(Function &dstFunc) const;

    Function &BelongFunction() { return *function_; }
    const Function &BelongFunction() const { return *function_; }
    void UpdateBelongFunction(Function &function) { function_ = &function; }

    std::string DumpType() const;

    /* By default, RawTensor is dumped. In whole function dumping, we only dump the magic */
    Json DumpJson(bool dumpRawTensor = true) const;
    static std::shared_ptr<LogicalTensor> LoadJson(Function &function, const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawTensorDict, const Json &tensorDump);

    std::string DumpSSA(bool showFrom = true, bool showMem = false, bool showType = true) const;
    std::string DumpASM(bool showFrom = true, bool showMem = false) const;

    std::string Dump(bool showFrom = true, bool showMem = false) const;

    std::shared_ptr<LogicalTensor> View(
        Function &function, const std::vector<int> &newShape, const std::vector<int> &newOffset) const;

    DataType Datatype() const;
    std::string Symbol() const;
    TileOpFormat GetTileOpFormat() const { return tensorfmt; }

    MemoryType GetMemoryTypeOriginal() const;
    MemoryType GetMemoryTypeToBe() const;
    void CopyMemoryType(const std::shared_ptr<LogicalTensor> &other);
    void SetMemoryTypeBoth(MemoryType t, bool force = false);
    void SetMemoryTypeOriginal(MemoryType t, bool force = false);
    void SetMemoryTypeToBe(MemoryType t);
    bool MemoryConflict() const;
    size_t MemorySize() const;
    bool IsDummy() const;
    void SetIsDummy(bool dummy = true);

    int GetSubgraphID() const { return subGraphID; }
    void UpdateSubgraphID(int subgraphID) { subGraphID = subgraphID; }

    bool Overlap(const std::shared_ptr<LogicalTensor> &other) const;

    int GetMagic() const { return magic; }
    void SetMagic(int m) { magic = m; }
    int GetRawMagic() const { return tensor->GetRawMagic(); }
    std::shared_ptr<RawTensor> GetRawTensor() const { return tensor; }
    const std::vector<int> &GetOffset() const { return offset; }
    const std::vector<int> &GetShape() const { return shape; }
    void UpdateOffset(const std::vector<int> &newOffset) {
        ASSERT(newOffset.size() == shape.size());
        offset = newOffset;
    }
    void UpdateOffset(const TensorOffset &tensorOffset) {
        this->offset = tensorOffset.GetOffset();
        this->dynOffset_ = tensorOffset.GetDynOffset();
    }
    const TensorOffset GetTensorOffset() const {
        return TensorOffset(offset, dynOffset_);
    }
    void UpdateDynValidShape(const std::vector<SymbolicScalar> &dynValidShape) {
        dynValidShape_ = dynValidShape;
    }
    struct CompareOp {
        bool operator() (const Operation *a, const Operation *b) const;
    };

    auto &GetProducers() { return producers_; }
    auto &GetConsumers() { return consumers_; }
    const auto &GetProducers() const { return producers_; }
    const auto &GetConsumers() const { return consumers_; }
    bool HasProducer(Operation *operation) const { return producers_.count(operation) > 0; }
    bool HasConsumer(Operation *operation) const { return consumers_.count(operation) > 0; }
    bool HasProducer(Operation &operation) const { return HasProducer(&operation); }
    bool HasConsumer(Operation &operation) const { return HasConsumer(&operation); }
    void AddProducer(Operation *operation) { producers_.emplace(operation); }
    void AddConsumer(Operation *operation) { consumers_.emplace(operation); }
    void RemoveProducer(Operation *operation) { producers_.erase(operation); }
    void RemoveConsumer(Operation *operation) { consumers_.erase(operation); }
    void AddProducer(Operation &operation) { AddProducer(&operation); }
    void AddConsumer(Operation &operation) { AddConsumer(&operation); }
    void RemoveProducer(Operation &operation) { RemoveProducer(&operation); }
    void RemoveConsumer(Operation &operation) { RemoveConsumer(&operation); }
    void ClearAllProducers() { producers_.clear(); }

    void operator<<(LogicalTensor &right);

    int GetDataSize() const;

    bool IsOffsetAllZero() const {
        return std::all_of(offset.begin(), offset.end(), [](int value) { return value == 0; });
    }

    const std::vector<SymbolicScalar> &GetDynOffset() const { return dynOffset_; }
    const std::vector<SymbolicScalar> &GetDynValidShape() const { return dynValidShape_; }

    void SetPrefetch(int preloadDep = 0) {
        if (tensor != nullptr) {
          tensor->SetPrefetch(preloadDep);
        }
    }
    bool NeedPrefetch() const { return tensor->NeedPrefetch(); }
    int GetPrefetchDep() const { return tensor->PrefetchDep(); }
private:
    MemoryType memoryTypeOriginal_{MemoryType::MEM_UNKNOWN};
    MemoryType memoryTypeToBe_{MemoryType::MEM_UNKNOWN};
    int readyTime_{INVALID_TIME};
    int remainingTime_{INVALID_TIME};
    Function *function_;

    std::unordered_set<std::string> semanticLabels_;
    std::set<Operation *, CompareOp> producers_;
    std::set<Operation *, CompareOp> consumers_;
};

SymbolicScalar GetViewValidShapeDim(
    const SymbolicScalar &validShapeDim,
    const SymbolicScalar &viewOffsetDim,
    const SymbolicScalar &viewShapeDim);
std::vector<SymbolicScalar> GetViewValidShape(
    const std::vector<SymbolicScalar> &validShape,
    const std::vector<int> &viewOffset,
    const std::vector<SymbolicScalar> &viewDynOffset,
    const std::vector<int> &viewShape);

constexpr int RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX = 1;
constexpr int RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX = 2;
constexpr int RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX = 3;

} // namespace npu::tile_fwk
