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
 * \file logical_tensor.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "logical_tensor.h"

#include "raw_tensor.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/id_gen.h"
#include "interface/function/function.h"
#include "interface/utils/serialization.h"

using namespace npu::tile_fwk;
LogicalTensor::LogicalTensor(
    Function &function, DataType t, std::vector<int> tshape, std::string tname, NodeType tnodetype, TileOpFormat ttensorfmt)
    : isSubGraphBoundary(false),
      subGraphID(NOT_IN_SUBGRAPH),
      tensor(std::make_shared<RawTensor>(t, tshape, std::move(tname))),
      offset(std::vector<int>(tshape.size(), 0)),
      shape(tshape),
      oriShape(tshape),
      magic(function.magicSeed_++),
      nodetype(tnodetype),
      tensorfmt(ttensorfmt),
      function_(&function)
{
}

LogicalTensor::LogicalTensor(
    Function &function, DataType t, std::vector<int> tshape, std::vector<SymbolicScalar> tValidShape,
    std::string tname, NodeType tnodetype, TileOpFormat ttensorfmt)
    : isSubGraphBoundary(false),
      subGraphID(NOT_IN_SUBGRAPH),
      tensor(std::make_shared<RawTensor>(t, tshape, std::move(tname))),
      offset(std::vector<int>(tshape.size(), 0)),
      shape(tshape),
      oriShape(tshape),
      dynValidShape_(tValidShape),
      storageShape(tshape),
      magic(function.magicSeed_++),
      nodetype(tnodetype),
      tensorfmt(ttensorfmt),
      function_(&function)
{
    auto getTensorDataDict = GetTensorDataDict(tValidShape);
    if (!tValidShape.empty() && getTensorDataDict.size() == 0) {
        tensor->UpdateDynRawShape(tValidShape);
    }
}

LogicalTensor::LogicalTensor(Function &function, std::shared_ptr<RawTensor> rawTensor,
    std::vector<int> toffset, std::vector<int> tshape, NodeType tnodetype, TileOpFormat ttensorfmt)
    : isSubGraphBoundary(false),
      subGraphID(NOT_IN_SUBGRAPH),
      tensor(rawTensor),
      offset(toffset),
      shape(tshape),
      oriShape(tshape),
      magic(function.magicSeed_++),
      nodetype(tnodetype),
      tensorfmt(ttensorfmt),
      function_(&function) {
    // Initialize other members if necessary
    isSubGraphBoundary = false;
    ASSERT(shape.size() == offset.size());
}

LogicalTensor::LogicalTensor(Function &function, std::shared_ptr<RawTensor> rawTensor,
    std::vector<int> toffset, std::vector<int> tshape, std::vector<SymbolicScalar> tValidShape, NodeType tnodetype, TileOpFormat ttensorfmt)
    : isSubGraphBoundary(false),
      subGraphID(NOT_IN_SUBGRAPH),
      tensor(rawTensor),
      offset(toffset),
      shape(tshape),
      oriShape(tshape),
      dynValidShape_(tValidShape),
      magic(function.magicSeed_++),
      nodetype(tnodetype),
      tensorfmt(ttensorfmt),
      function_(&function) {
    // Initialize other members if necessary
    isSubGraphBoundary = false;

    ASSERT(shape.size() == offset.size());
}

std::shared_ptr<LogicalTensor> LogicalTensor::Clone(Function &dstFunc, bool create) const {
    /* Clone is only for dstFunc to simplify the process of creating OP_CALL's input and output. */
    if (!create) {
        auto cloned = dstFunc.GetTensorMap().GetTensorByMagic(magic);
        if (cloned != nullptr) {
            return cloned;
        }
    }

    std::shared_ptr<RawTensor> rawTensor = dstFunc.GetTensorMap().GetRawTensorByRawMagic(tensor->rawmagic);
    if (rawTensor == nullptr) {
        rawTensor =
            std::make_shared<RawTensor>(tensor->datatype, tensor->rawshape, tensor->symbol, tensor->rawmagic);
        rawTensor->actualRawmagic = tensor->actualRawmagic;
        rawTensor->UpdateDynRawShape(tensor->GetDynRawShape());
        rawTensor->memoryId = tensor->memoryId;
    }

    std::shared_ptr<LogicalTensor> newTensor = std::make_shared<LogicalTensor>(dstFunc, rawTensor,
        offset, shape, dynValidShape_, nodetype, tensorfmt);
    newTensor->isSubGraphBoundary = isSubGraphBoundary;
    newTensor->subGraphID = subGraphID;
    if (!create) {
        newTensor->magic = magic;
        if (magic >= dstFunc.magicSeed_) {
            dstFunc.magicSeed_ = (magic + 1);
        }
    } else {
        newTensor->magic = dstFunc.magicSeed_++;
    }
    newTensor->tensorfmt = tensorfmt;

    newTensor->memorymap = memorymap;
    newTensor->memoryTypeOriginal_ = memoryTypeOriginal_;
    newTensor->memoryTypeToBe_ = memoryTypeToBe_;
    newTensor->readyTime_ = readyTime_;
    newTensor->remainingTime_ = remainingTime_;
    newTensor->semanticLabels_ = semanticLabels_;
    newTensor->dynOffset_ = dynOffset_;
    dstFunc.GetTensorMap().Insert(newTensor, false);
    return newTensor;
}

Json LogicalTensor::DumpJson(bool dumpRawTensor) const {
    Json tensorDump;
    tensorDump[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_TENSOR);
    tensorDump["offset"] = offset;
    tensorDump["shape"] = shape;
    tensorDump["validshape"] = oriShape;
    tensorDump["nodetype"] = static_cast<int>(nodetype);
    if (dumpRawTensor) {
        tensorDump[T_FIELD_RAWTENSOR] = tensor->DumpJson();
    } else {
        tensorDump[T_FIELD_RAWTENSOR] = tensor->rawmagic;
    }
    tensorDump["magic"] = magic;
    tensorDump["format"] = static_cast<int>(tensorfmt);
    if (storage_ != nullptr) {
        tensorDump["storage"] = storage_->DumpJson();
    }
    if (HasAttr(OpAttributeKey::needAlloc)) {
        bool allocValue = false;
        GetAttr(OpAttributeKey::needAlloc, allocValue);
        tensorDump["need_alloc"] = allocValue;
    }
    std::vector<std::string> resultSemanticLabels(semanticLabels_.begin(), semanticLabels_.end());
    std::sort(resultSemanticLabels.begin(), resultSemanticLabels.end());
    tensorDump["semantic_label"] = resultSemanticLabels;
    tensorDump["subgraph_boundary"] = isSubGraphBoundary;

    if (subGraphID != NOT_IN_SUBGRAPH) {
        tensorDump["subgraphid"] = subGraphID;
    }

    if (memorymap.size() != 0) {
        Json memoryrange = Json::object();
        Json liferange = Json::object();
        Json memoryid = Json::object();
        for (auto &[sgid, range] : memorymap) {
            memoryrange[std::to_string(sgid)] = Json(std::vector<std::size_t>({range.start, range.end}));
            liferange[std::to_string(sgid)] = Json(std::vector<int>({range.lifeStart, range.lifeEnd}));
            memoryid[std::to_string(sgid)] = Json(range.memId);
        }
        tensorDump["mem_range"] = memoryrange;
        tensorDump["life_range"] = liferange;
        tensorDump["mem_id"] = memoryid;
    }

    if (GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN || GetMemoryTypeToBe() != MemoryType::MEM_UNKNOWN) {
        Json memorytype = Json::object();
        if (GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
            memorytype["asis"] = static_cast<int>(GetMemoryTypeOriginal());
        }
        if (GetMemoryTypeToBe() != MemoryType::MEM_UNKNOWN) {
            memorytype["tobe"] = static_cast<int>(GetMemoryTypeToBe());
        }
        tensorDump["mem_type"] = memorytype;
    }
    Json offsetJson = Json::array();
    for (auto dynOffset : dynOffset_) {
        auto joffset = ToJson(dynOffset);
        if (joffset.size() > 0) {
            offsetJson.push_back(joffset);
        }
    }
    if (offsetJson.size() > 0) {
        tensorDump["dynoffset"] = offsetJson;
    }
    Json dynValidShapeJson = Json::array();
    for (auto dynValidShape : dynValidShape_) {
        auto jValidShape = ToJson(dynValidShape);
        if (jValidShape.size() > 0) {
            dynValidShapeJson.push_back(jValidShape);
        }
    }
    if (dynValidShapeJson.size() > 0) {
        tensorDump["dynvalidshape"] = dynValidShapeJson;
    }
    return tensorDump;
}

std::shared_ptr<LogicalTensor> LogicalTensor::LoadJson(Function &function,
            const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawTensorDict, const Json &tensorDump) {
    ASSERT(tensorDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_TENSOR));

    std::vector<int> toffset = tensorDump["offset"].get<std::vector<int>>();
    std::vector<int> tshape = tensorDump["shape"].get<std::vector<int>>();
    NodeType tnodetype = static_cast<NodeType>(tensorDump["nodetype"].get<int>());

    std::shared_ptr<RawTensor> rawTensor;
    if (tensorDump[T_FIELD_RAWTENSOR].is_number()) {
        int rawTensorMagic = tensorDump[T_FIELD_RAWTENSOR].get<int>();
        ASSERT(rawTensorDict.count(rawTensorMagic));
        rawTensor = rawTensorDict.find(rawTensorMagic)->second;
    } else {
        rawTensor = RawTensor::LoadJson(tensorDump[T_FIELD_RAWTENSOR]);
    }
    int tensorMagic = tensorDump["magic"].get<int>();

    TileOpFormat ttensorfmt = static_cast<TileOpFormat>(tensorDump["format"].get<int>());
    std::shared_ptr<LogicalTensor> tensorJson = std::make_shared<LogicalTensor>(function, rawTensor,
        toffset, tshape, tnodetype, ttensorfmt);
    tensorJson->magic = tensorMagic;

    if (tensorDump.count("need_alloc") != 0) {
        bool needAlloc = tensorDump["need_alloc"].get<bool>();
        tensorJson->SetAttr(OpAttributeKey::needAlloc, needAlloc);
    }

    std::vector<std::string> semanticLabelData = tensorDump["semantic_label"].get<std::vector<std::string>>();
    tensorJson->semanticLabels_.insert(semanticLabelData.begin(), semanticLabelData.end());

    if (tensorDump.count("subgraphid")) {
        tensorJson->subGraphID = tensorDump["subgraphid"].get<int>();
    }
    tensorJson->isSubGraphBoundary = tensorDump["subgraph_boundary"].get<bool>();
    if (tensorDump.count("mem_range")) {
        for (auto &[sgid, range] : tensorDump["mem_range"].items()) {
            tensorJson->memorymap[std::stoll(sgid)] = TileRange(range[0].get<int>(), range[1].get<int>());
        }
    }
    if (tensorDump.count("life_range")) {
        for (auto &[sgid, range] : tensorDump["life_range"].items()) {
            tensorJson->memorymap[std::stoll(sgid)].lifeStart = range[0].get<int>();
            tensorJson->memorymap[std::stoll(sgid)].lifeEnd = range[1].get<int>();
        }
    }
    if (tensorDump.count("mem_id")) {
        for (auto &[sgid, memid] : tensorDump["mem_id"].items()) {
            tensorJson->memorymap[std::stoll(sgid)].memId = memid.get<int>();
        }
    }
    if (tensorDump.count("mem_type")) {
        auto &memorytype = tensorDump["mem_type"];
        if (memorytype.count("asis")) {
            tensorJson->memoryTypeOriginal_ = static_cast<MemoryType>(memorytype["asis"].get<int>());
        }
        if (memorytype.count("tobe")) {
            tensorJson->memoryTypeToBe_ = static_cast<MemoryType>(memorytype["tobe"].get<int>());
        }
    }
    if (tensorDump.count("storage")) {
        tensorJson->storage_ = Storage::LoadJson(tensorDump["storage"]);
    }
    if (tensorDump.count("validshape")) {
        tensorJson->oriShape = tensorDump["validshape"].get<std::vector<int>>();
    }
    if (tensorDump.count("dynoffset")) {
        auto dynoffsetJson = tensorDump["dynoffset"];
        std::vector<SymbolicScalar> dynOffset;
        for (auto offsetJson : dynoffsetJson) {
            dynOffset.push_back(LoadSymbolicScalar(offsetJson));
        }
        tensorJson->dynOffset_ = dynOffset;
    }
    if (tensorDump.count("dynvalidshape")) {
        auto dynvalidJson = tensorDump["dynvalidshape"];
        std::vector<SymbolicScalar> dynValidShape;
        for (auto validJson : dynvalidJson) {
            dynValidShape.push_back(LoadSymbolicScalar(validJson));
        }
        tensorJson->UpdateDynValidShape(dynValidShape);
    }
    return tensorJson;
}

std::string LogicalTensor::DumpType() const {
    std::string result = "<";
    for (auto &value : shape) {
        result += std::to_string(value) + " x ";
    }
    result += DataType2String(Datatype());
    if (dynValidShape_.size() != 0) {
        result += " / ";
        for (auto &value : dynValidShape_) {
            result += value.Dump() + " x ";
        }
        result += DataType2String(Datatype());
    }
    result += ">";
    return result;
}

std::string LogicalTensor::DumpSSA([[maybe_unused]]bool showFrom, bool showMem, bool showType) const {
    std::ostringstream oss;
    if (showType) {
        oss << DumpType() << " ";
    }
    oss << "%" << GetMagic() << GetRawTensor()->DumpSSA(false, false);

    if (not std::all_of(offset.begin(), offset.end(), [](int ox) { return ox == 0; })) {
        oss << "(";
        for (size_t i = 0; i < offset.size(); ++i) {
            oss << offset[i];
            if (i != offset.size() - 1) {
                oss << ", ";
            }
        }
        oss << ")";
    }
    if (dynOffset_.size() != 0) {
        oss << "(";
        for (size_t i = 0; i < dynOffset_.size(); ++i) {
            oss << dynOffset_[i].Dump();
            if (i != offset.size() - 1) {
                oss << ", ";
            }
        }
        oss << ")";
    }
    oss << "#"
        << "(" << subGraphID << ")";
    if (showMem) {
        oss << MemoryTypeToString(GetMemoryTypeOriginal()) << "::" << MemoryTypeToString(GetMemoryTypeToBe());
        if (IsDummy()) {
            oss << "::IsDummy";
        }
    }
    return oss.str();
}

std::string LogicalTensor::DumpASM(bool showFrom, bool showMem) const {
    std::ostringstream oss;
    constexpr size_t width4 = 4;
    constexpr size_t width3 = 3;
    constexpr int minus2 = -2;
    if (showFrom && (GetProducers().size() != 0)) {
        oss << magic << " FROM[";
        for (auto producer : GetProducers()) {
            oss << producer->GetOpMagic() << ", ";
        }
        if (GetProducers().size() != 0) {
            oss.seekp(minus2, std::ios_base::end);
        }
        oss << "]";
        return oss.str();
    }

    oss << std::setw(width3) << std::setfill(' ') << magic << " ";
    oss << std::setw(width4) << std::setfill(' ') << tensor->DumpASM() << ", ";
    oss << "shape:[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << std::setw(width3) << std::setfill(' ') << shape[i];
        if (i != shape.size() - 1) {
            oss << ",";
        }
    }
    oss << "], ";
    oss << "validshape:[";
    for (size_t i = 0; i < oriShape.size(); ++i) {
        oss << std::setw(width3) << std::setfill(' ') << oriShape[i];
        if (i != oriShape.size() - 1) {
            oss << ",";
        }
    }
    oss << "], ";
    oss << "offset:[";
    for (size_t i = 0; i < offset.size(); ++i) {
        oss << std::setw(width3) << std::setfill(' ') << offset[i];
        if (i != offset.size() - 1) {
            oss << ",";
        }
    }
    oss << "] \\n";
    if (showMem) {
        oss << MemoryTypeToString(GetMemoryTypeOriginal()) << "::" << MemoryTypeToString(GetMemoryTypeToBe());
        if (IsDummy()) {
            oss << "::IsDummy";
        }
    }
    return oss.str();
}

std::string LogicalTensor::Dump(bool showFrom, bool showMem) const {
    if (config::GetPlatformConfig("USE_SSA", true)) {
        return DumpSSA(showFrom, showMem);
    } else {
        return DumpASM(showFrom, showMem);
    }
}

std::shared_ptr<LogicalTensor> LogicalTensor::View(
    Function &function, const std::vector<int> &newShape, const std::vector<int> &newOffset) const {
    assert((shape.size() == newShape.size()) && ".view, shape must be the same dimension");
    assert((offset.size() == newOffset.size()) && ".view, offset must be the same dimension");

    auto view = std::make_shared<LogicalTensor>(function, this->tensor, this->offset, this->shape, this->nodetype, this->tensorfmt);
    for (size_t i = 0; i < shape.size(); i++) {
        if (!(shape[i] >= (newShape[i] + newOffset[i]))) {
            assert(shape[i] >= (newShape[i] + newOffset[i]));
        }
        assert(shape[i] >= (newShape[i] + newOffset[i]));
    }

    view->shape = newShape;
    view->oriShape = newShape;
    view->offset = newOffset;
    view->offset = TensorOffset::Add(offset, newOffset);

    if (dynOffset_.size() != 0) {
        view->dynOffset_ = TensorOffset::Add(dynOffset_, newOffset);
    }
    view->dynValidShape_ = GetViewValidShape(dynValidShape_, newOffset, {}, newShape);
    return view;
}

std::string LogicalTensor::Symbol() const {
    return tensor->symbol;
}

DataType LogicalTensor::Datatype() const {
    return tensor->datatype;
}

MemoryType LogicalTensor::GetMemoryTypeOriginal() const {
    return memoryTypeOriginal_;
}

MemoryType LogicalTensor::GetMemoryTypeToBe() const {
    return memoryTypeToBe_;
}

void LogicalTensor::CopyMemoryType(const std::shared_ptr<LogicalTensor> &other) {
    memoryTypeOriginal_ = other->GetMemoryTypeOriginal();
    memoryTypeToBe_ = other->GetMemoryTypeToBe();
}

void LogicalTensor::SetMemoryTypeBoth(MemoryType t, bool force) {
    SetMemoryTypeOriginal(t, force);
    SetMemoryTypeToBe(t);
}

void LogicalTensor::SetMemoryTypeOriginal(MemoryType t, bool force) {
    if (t == MemoryType::MEM_UNKNOWN) {
        return;
    }

    if (memoryTypeOriginal_ == MemoryType::MEM_UNKNOWN) {
        memoryTypeOriginal_ = t;
    } else if (memoryTypeOriginal_ != t) {
        if (force) {
            memoryTypeOriginal_ = t;
        }
    }
}

void LogicalTensor::SetMemoryTypeToBe(MemoryType t) {
    if (t == MemoryType::MEM_UNKNOWN) {
        return;
    }
    memoryTypeToBe_ = t;
}

bool LogicalTensor::MemoryConflict() const {
    return memoryTypeOriginal_ != MemoryType::MEM_UNKNOWN && memoryTypeToBe_ != MemoryType::MEM_UNKNOWN &&
           memoryTypeOriginal_ != memoryTypeToBe_;
}

size_t LogicalTensor::MemorySize() const {
    if (IsDummy()) {
        return 0;
    }

    size_t baseMemorySize = BytesOf(Datatype());
    for (auto n : shape) {
        baseMemorySize *= n;
    }

    switch (GetMemoryTypeToBe()) {
        case MemoryType::MEM_UB: // 32B align
            return (baseMemorySize + ALIGN_SIZE_32 - 1) / ALIGN_SIZE_32 * ALIGN_SIZE_32;
        case MemoryType::MEM_L0A:
        case MemoryType::MEM_L0B:
        case MemoryType::MEM_L0C: // 512B align
            return (baseMemorySize + ALIGN_SIZE_512 - 1) / ALIGN_SIZE_512 * ALIGN_SIZE_512;
        default: return baseMemorySize;
    }
}

bool LogicalTensor::IsDummy() const {
    return tensor->IsDummy();
}

void LogicalTensor::SetIsDummy(bool dummy) {
    tensor->SetIsDummy(dummy);
}

bool LogicalTensor::Overlap(const std::shared_ptr<LogicalTensor> &other) const {
    if (tensor->rawmagic != other->tensor->rawmagic) {
        return false;
    }
    // Check if the shape and offsets overlap
    for (size_t i = 0; i < shape.size(); ++i) {
        if (offset[i] + shape[i] <= other->offset[i] || other->offset[i] + other->shape[i] <= offset[i]) {
            return false;
        }
    }
    return true;
}

int LogicalTensor::GetDataSize() const {
    int shapeSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    return shapeSize * BytesOf(tensor->GetDataType());
}

bool LogicalTensor::CompareOp::operator()(const Operation *a, const Operation *b) const {
    int funcMagicA = a->BelongTo()->GetFuncMagic();
    int funcMagicB = b->BelongTo()->GetFuncMagic();
    if (funcMagicA != funcMagicB) {
        return funcMagicA < funcMagicB;
    }
    int opmagicA = a->opmagic;
    int opmagicB = b->opmagic;
    return opmagicA < opmagicB;
}

SymbolicScalar npu::tile_fwk::GetViewValidShapeDim(
    const SymbolicScalar &validShapeDim,
    const SymbolicScalar &viewOffsetDim,
    const SymbolicScalar &viewShapeDim) {
    SymbolicScalar result;
    if (validShapeDim.ConcreteValid() && viewOffsetDim.ConcreteValid() && viewShapeDim.ConcreteValid()) {
        auto validShapeData = validShapeDim.Concrete();
        auto viewOffsetData = viewOffsetDim.Concrete();
        auto viewShapeData = viewShapeDim.Concrete();
        if (viewShapeData == -1) {
            result = std::max(validShapeData - viewOffsetData, 0L);
        } else {
            result = std::max(std::min(validShapeData - viewOffsetData, viewShapeData), 0L);
        }
    } else if (viewShapeDim.ConcreteValid() && viewShapeDim.Concrete() == -1) {
        return std::max(validShapeDim - viewOffsetDim, 0L);
    } else {
        std::string getViewValidShapeName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetViewValidShapeDim);
        getViewValidShapeName = AddRuntimePrefix(getViewValidShapeName);
        SymbolicScalar getViewValidShape(getViewValidShapeName);
        result = getViewValidShape(validShapeDim, viewOffsetDim, viewShapeDim);
    }
    return result;
}

std::vector<SymbolicScalar> npu::tile_fwk::GetViewValidShape(
    const std::vector<SymbolicScalar> &validShape,
    const std::vector<int> &viewOffset,
    const std::vector<SymbolicScalar> &viewDynOffset,
    const std::vector<int> &viewShape) {
    if (validShape.size() == 0) {
        return {};
    }
    ASSERT(validShape.size() == viewShape.size());

    std::vector<SymbolicScalar> result;
    for (size_t i = 0; i < validShape.size(); i++) {
        SymbolicScalar validShapeDim;
        if (viewDynOffset.size() != 0) {
            validShapeDim = GetViewValidShapeDim(validShape[i], viewDynOffset[i], viewShape[i]);
        } else {
            validShapeDim = GetViewValidShapeDim(validShape[i], viewOffset[i], viewShape[i]);
        }
        result.push_back(validShapeDim);
    }
    return result;
}

namespace npu::tile_fwk {

Tensor TensorExtract(const Tensor &src, const std::vector<SymbolicScalar> &offset) {
    ASSERT(src.GetShape().size() == offset.size()) << "dim mismatch";
    auto currFunc = Program::GetInstance().GetCurrentFunction();

    std::vector<int> dstShape(src.GetShape().size(), 1);
    // minimal size is 32
    dstShape.back() = 32;
    Tensor dst(src.GetDataType(), dstShape, currFunc->GetRawName() + "_TensorExtract");

    std::vector<int> viewShape(src.GetShape().size(), 1);
    Tensor view = DView(src, viewShape, offset);
    Operation &emuopView = **view.GetStorage()->GetProducers().begin();

    // Force to UB
    Tensor mark = AddS(view, Element(view.GetDataType(), (int64_t)0));
    Operation &emuopMark = **mark.GetStorage()->GetProducers().begin();

    std::vector<int> assembleOffset(src.GetShape().size(), 0);
    Operation &emuopAssemble = currFunc->AddOperation(Opcode::OP_ASSEMBLE, {mark.GetStorage()}, {dst.GetStorage()});
    emuopAssemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(assembleOffset));

    emuopView.SetAttr<int>(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_EXTRACT);
    emuopMark.SetAttr<int>(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_EXTRACT);
    emuopAssemble.SetAttr<int>(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_EXTRACT);
    return dst;
}

void TensorInsert(const Tensor &src, const std::vector<SymbolicScalar> &offset, Tensor &dst) {
    ASSERT(src.GetShape() == std::vector<int>(src.GetShape().size(), 1));
    ASSERT(src.GetShape().size() == dst.GetShape().size());
    ASSERT(src.GetShape().size() == offset.size());

    // Force to UB
    Tensor mark = AddS(src, Element(src.GetDataType(), (int64_t)0));
    Operation &emuopMark = **mark.GetStorage()->GetProducers().begin();

    DAssemble(mark, offset, dst);
    Operation &emuopAssemble = **mark.GetStorage()->GetConsumers().begin();

    emuopMark.SetAttr<int>(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_INSERT);
    emuopAssemble.SetAttr<int>(OP_EMUOP_PREFIX + "opc", EMUOP_TENSOR_INSERT);
}

static void LookupExpressionByOpcode(std::vector<RawSymbolicScalarPtr> &exprList, SymbolicOpcode opcode, const RawSymbolicScalarPtr &raw) {
    switch (raw->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            if (raw->GetExpressionOpcode() == opcode) {
                exprList.emplace_back(raw);
            }
            for (auto &op : raw->GetExpressionOperandList()) {
                LookupExpressionByOpcode(exprList, opcode, op);
            }
        } break;
        default: ASSERT(false); break;
    }
}

static std::vector<RawSymbolicScalarPtr> LookupExpressionByOpcode(const RawSymbolicScalarPtr &value, SymbolicOpcode opcode) {
    std::vector<RawSymbolicScalarPtr> exprList;
    LookupExpressionByOpcode(exprList, opcode, value);
    return exprList;
}

static RawSymbolicScalarPtr ReplaceExpression(const RawSymbolicScalarPtr &expr, const RawSymbolicScalarPtr &src, const RawSymbolicScalarPtr &dst) {
    if (expr == src) {
        return dst;
    }
    RawSymbolicScalarPtr result;
    switch (expr->Kind()) {
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_IMMEDIATE:
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_SYMBOL:
            result = expr;
            break;
        case SymbolicScalarKind::T_SCALAR_SYMBOLIC_EXPRESSION: {
            std::vector<RawSymbolicScalarPtr> subexprList;
            bool allreuse = true;
            for (auto &subexpr : expr->GetExpressionOperandList()) {
                RawSymbolicScalarPtr sub = ReplaceExpression(subexpr, src, dst);
                subexprList.push_back(sub);
                allreuse = allreuse && (sub == subexpr);
            }
            if (allreuse) {
                result = expr;
            } else {
                result = std::make_shared<RawSymbolicExpression>(expr->GetExpressionOpcode(), subexprList);
            }
        } break;
        default: ASSERT(false); break;
    }
    return result;
}

std::map<int, RawSymbolicScalarPtr> GetTensorDataDict(const RawSymbolicScalarPtr &dimOffset) {
    std::map<int, RawSymbolicScalarPtr> getTensorDataDict;
    std::vector<RawSymbolicScalarPtr> mopCall = LookupExpressionByOpcode(dimOffset, SymbolicOpcode::T_MOP_CALL);
    for (auto mop : mopCall) {
        auto callee = mop->GetExpressionOperandList()[0];
        if (!callee->IsSymbol()) {
            continue;
        }
        auto name = callee->GetSymbolName();
        if (StringUtils::StartsWith(name, AddRuntimePrefix("GetTensorData"))) {
            auto getTensorDataIndex = mop->GetExpressionOperandList()[0x1]->GetImmediateValue();
            getTensorDataDict[getTensorDataIndex] = mop;
            break;
        }
    }
    return getTensorDataDict;    
}

std::map<int, RawSymbolicScalarPtr> GetTensorDataDict(const SymbolicScalar &dimOffset) {
    return GetTensorDataDict(dimOffset.Raw());
}

std::map<int, RawSymbolicScalarPtr> GetTensorDataDict(const std::vector<SymbolicScalar> &offset) {
    std::map<int, RawSymbolicScalarPtr> getTensorDataDict;
    for (auto &off : offset) {
        auto perOffsetDict = GetTensorDataDict(off);
        for (auto &item : perOffsetDict) {
            getTensorDataDict.emplace(item);
        }
    }
    return getTensorDataDict;
}

SymbolicScalar GetTensorDataFillIO(const std::unordered_map<int, GetTensorDataIODesc> &iodescDict, const SymbolicScalar &dimOffset) {
    RawSymbolicScalarPtr curr = dimOffset.Raw();
    bool filledFound = true;
    while (filledFound) {
        // There might be nesting GetTensorData call, so it's replaced iteratively.
        std::map<int, RawSymbolicScalarPtr> getDict = GetTensorDataDict(curr);
        filledFound = false;
        for (auto [index, ptr] : getDict) {
            if (!iodescDict.count(index)) {
                continue;
            }
            auto [ioTypeValue, ioTypeIndexValue, address] = iodescDict.find(index)->second;
            std::vector<RawSymbolicScalarPtr> operandList = ptr->GetExpressionOperandList();
            auto currIOType = operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE];
            auto currIOTypeIndex = operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX];
            ASSERT(currIOType->IsImmediate());
            ASSERT(currIOTypeIndex->IsImmediate());
            if (currIOType->GetImmediateValue() == ioTypeValue && currIOTypeIndex->GetImmediateValue() == ioTypeIndexValue) {
                continue;
            }
            operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE] = std::make_shared<RawSymbolicImmediate>(ioTypeValue);
            operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX] = std::make_shared<RawSymbolicImmediate>(ioTypeIndexValue);
            operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOINDEX] = address.Raw();
            auto ptrNext = std::make_shared<RawSymbolicExpression>(ptr->GetExpressionOpcode(), operandList);
            auto currNext = ReplaceExpression(curr, ptr, ptrNext);
            curr = currNext;
            filledFound = true;
            break;
        }
    }
    return SymbolicScalar(curr);
}

}
