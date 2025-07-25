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
 * \file tensor.cpp
 * \brief
 */

#include "tilefwk/tensor.h"
#include "logical_tensor.h"
#include "raw_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"
#include "interface/utils/assert.h"
#include "interface/utils/id_gen.h"


using namespace npu::tile_fwk;

Tensor::Tensor() : storage(nullptr), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    Program::GetInstance().InsertAliveTensor(this);
}

Tensor::~Tensor() {
    Program::GetInstance().GetTensorSlotManager()->TensorDestruct(*this);

    Program::GetInstance().EraseAliveTensor(this);
    if (storage == nullptr) {
        return;
    }
    ASSERT(storage->tensor != nullptr);
    storage->tensor->AddRefCount(-1);
}

Tensor::Tensor(std::shared_ptr<LogicalTensor> s) : storage(std::move(s)), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    ASSERT(storage != nullptr && storage->tensor != nullptr);
    Program::GetInstance().InsertAliveTensor(this);
    storage->tensor->AddRefCount(1);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
}

static std::vector<SymbolicScalar> ToDynShape(const std::string &tname, const std::vector<int> &shape) {
    auto dynShape = SymbolicScalar::FromConcrete(shape);
    for (size_t dim = 0; dim < shape.size(); dim++) {
        ASSERT(shape[dim] >= -1) << "Invalid shape " << shape[dim];
        if (shape[dim] == -1) {
            auto name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputShapeDim);
            auto handler = SymbolicScalar(AddRuntimePrefix(name));
            auto input = SymbolicScalar(AddArgPrefix(tname));
            dynShape[dim] = handler(input, dim);
        }
    }
    return dynShape;
}

Tensor::Tensor(DataType t, std::vector<int> tshape, std::string tname, NodeType tnodetype,
    TileOpFormat tensorfmt)
    : index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    auto dynShape = ToDynShape(tname, tshape);
    storage = std::make_shared<LogicalTensor>(*Program::GetInstance().GetCurrentFunction(),
        t, tshape, tname, tnodetype, tensorfmt);
    storage->tensor->AddRefCount(1);

    storage->GetRawTensor()->UpdateDynRawShape(dynShape);

    Program::GetInstance().InsertAliveTensor(this);
    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    Program::GetInstance().GetTensorSlotManager()->TensorSymbol(*this, tname);
}

Tensor::Tensor(DataType t, std::vector<SymbolicScalar> tshape, std::string tname, TileOpFormat tensorfmt)
    :Tensor(t, SymbolicScalar::Concrete(tshape, -1), tname, NodeType::LOCAL, tensorfmt)
{
    auto rawTensor = storage->GetRawTensor();
    for (size_t axis = 0; axis  < tshape.size(); axis++) {
        if (tshape[axis].ConcreteValid() && tshape[axis].Concrete() == -1) {
            tshape[axis] = rawTensor->GetDynRawShape(axis);
        }
    }
    rawTensor->UpdateDynRawShape(tshape);
}

void Tensor::SetData(BinDataPtr data) {
    data_ = data;
}

Tensor::Tensor(std::shared_ptr<RawTensor> rawtensor, std::vector<int> toffset, std::vector<int> tshape,
    NodeType tnodetype, TileOpFormat tensorfmt)
    : index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    ASSERT(rawtensor != nullptr);
    Program::GetInstance().InsertAliveTensor(this);
    storage = std::make_shared<LogicalTensor>(*Program::GetInstance().GetCurrentFunction(), std::move(rawtensor),
        toffset, tshape, tnodetype, tensorfmt);
    storage->tensor->AddRefCount(1);

    Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
}

const LogicalTensor *Tensor::operator->() const {
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return storage.get();
}

LogicalTensor *Tensor::operator->() {
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return storage.get();
}

const LogicalTensor &Tensor::operator*() const {
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return *storage;
}

LogicalTensor &Tensor::operator*() {
    Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    return *storage;
}

const std::shared_ptr<LogicalTensor> &Tensor::GetStorage(bool readSlot) const
{
    if (readSlot) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    }
    return storage;
}

std::shared_ptr<LogicalTensor> &Tensor::GetStorage(bool readSlot)
{
    if (readSlot) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(*this);
    }
    return storage;
}

namespace npu {
namespace tile_fwk {
void AssignTensorData(Tensor &lhs, const Tensor &rhs) {
    if (lhs.GetData() != nullptr) {
        if (rhs.GetData() != nullptr) {
            ASSERT(lhs.GetData() == rhs.GetData());
        }
    } else {
        lhs.SetData(rhs.GetData());
    }
}
}
}

Tensor &Tensor::operator=(const Tensor &rhs) {
    if (this == &rhs) {
        return *this;
    }
    AssignTensorData(*this, rhs);
    if (storage != nullptr && storage->tensor != nullptr) {
        rhs.GetStorage()->tensor->symbol = storage->tensor->symbol;
    }
    if (storage != nullptr) {
        storage->tensor->AddRefCount(-1);
    }
    storage = rhs.GetStorage();
    if (storage != nullptr) {
        storage->tensor->AddRefCount(1);
    }
    if (storage != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&rhs) noexcept {
    if (this == &rhs) {
        return *this;
    }
    AssignTensorData(*this, rhs);
    rhs.SetData(nullptr);
    if (storage != nullptr && storage->tensor != nullptr) {
        rhs.GetStorage()->tensor->symbol = storage->tensor->symbol;
    }
    if (storage != nullptr) {
        storage->tensor->AddRefCount(-1);
    }
    storage = std::move(rhs.GetStorage());
    if (storage != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
    return *this;
}

Tensor::Tensor(const Tensor &rhs) : storage(rhs.GetStorage()), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    if (storage != nullptr) {
        storage->tensor->AddRefCount(1);
    }
    SetData(rhs.GetData());
    Program::GetInstance().InsertAliveTensor(this);
    if (storage != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

Tensor::Tensor(Tensor &&rhs) : storage(std::move(rhs.GetStorage())), index_(IdGen<IdType::TENSOR_INDEX>::Inst().NewId()) {
    Program::GetInstance().InsertAliveTensor(this);
    SetData(rhs.GetData());
    rhs.SetData(nullptr);
    if (storage != nullptr) {
        Program::GetInstance().GetTensorSlotManager()->TensorRead(rhs);
        Program::GetInstance().GetTensorSlotManager()->TensorWrite(*this);
    }
}

DataType Tensor::GetDataType() const {
    return storage->Datatype();
 }

const std::vector<int> &Tensor::GetShape() const
{
    return storage->shape;
}

int Tensor::GetShape(int axis) const {
    const size_t dimCount = storage->shape.size();
    ASSERT(dimCount > 0) << "Tensor has no dimensions!";
    if (axis < 0) {
        axis += static_cast<int>(dimCount);
    }
    ASSERT(axis >= 0 && static_cast<size_t>(axis) < dimCount) << "Axis index " << axis <<
        " is out of range [0, " << (dimCount - 1) << "].";
    return storage->shape[axis];
}

void Tensor::Prefetch(int preloadDep) {
  if (storage != nullptr) {
      storage->SetPrefetch(preloadDep);
  }
  return;
}

SymbolicScalar npu::tile_fwk::GetInputShapeDimSize(const Tensor &t) {
    std::string getInputShapeDimSizeName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputShapeDimSize);
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputShapeDimSizeName = AddRuntimePrefix(getInputShapeDimSizeName);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputShapeDimSize(getInputShapeDimSizeName);
    SymbolicScalar input(inputName);
    return getInputShapeDimSize(input);
}

SymbolicScalar npu::tile_fwk::GetInputShapeDim(const Tensor &t, int n) {
    auto rawTensor = t.GetStorage(false)->GetRawTensor();
    return rawTensor->GetDynRawShape(n);
}

SymbolicScalar npu::tile_fwk::GetInputDataInt32Dim1(const Tensor &t, SymbolicScalar off0) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim1Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim1);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim1Name = AddRuntimePrefix(getInputDataInt32Dim1Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim1(getInputDataInt32Dim1Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim1(input, off0);
}

SymbolicScalar npu::tile_fwk::GetInputDataInt32Dim2(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim2Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim2);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim2Name = AddRuntimePrefix(getInputDataInt32Dim2Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim2(getInputDataInt32Dim2Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim2(input, off0, off1);
}

SymbolicScalar npu::tile_fwk::GetInputDataInt32Dim3(const Tensor &t, SymbolicScalar off0, SymbolicScalar off1, SymbolicScalar off2) {
    auto slotManager = Program::GetInstance().GetTensorSlotManager();
    slotManager->TensorRead(t);

    std::string getInputDataInt32Dim3Name = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::GetInputDataInt32Dim3);
    int inputIndex = slotManager->GetInputIndex(t);
    ASSERT(inputIndex >= 0 && static_cast<size_t>(inputIndex) < slotManager->GetInputNameList().size()) <<
        "Tensor " << t.GetStorage(false)->GetRawTensor()->GetSymbol() << " is not in input tensor list!";
    std::string inputName = slotManager->GetInputNameList()[inputIndex];

    getInputDataInt32Dim3Name = AddRuntimePrefix(getInputDataInt32Dim3Name);
    inputName = AddArgPrefix(inputName);

    SymbolicScalar getInputDataInt32Dim3(getInputDataInt32Dim3Name);
    SymbolicScalar input(inputName);
    return getInputDataInt32Dim3(input, off0, off1, off2);
}

SymbolicScalar npu::tile_fwk::IsLoopBegin(const SymbolicScalar &symbol, const SymbolicScalar &begin) {
    std::string isLoopBeginName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopBegin);
    isLoopBeginName = AddRuntimePrefix(isLoopBeginName);
    SymbolicScalar isLoopBegin(isLoopBeginName);
    auto result = isLoopBegin(symbol, begin);
    result.AsLoopBegin(symbol.IsLoopBegin());
    result.AsLoopEnd(symbol.IsLoopEnd());
    return result;
}

SymbolicScalar npu::tile_fwk::IsLoopEnd(const SymbolicScalar &symbol, const SymbolicScalar &end) {
    std::string isLoopEndName = SymbolHandler::GetNameByHandlerId(SymbolHandlerId::IsLoopEnd);
    isLoopEndName = AddRuntimePrefix(isLoopEndName);
    SymbolicScalar isLoopEnd(isLoopEndName);
    auto result = isLoopEnd(symbol, end);
    result.AsLoopBegin(symbol.IsLoopBegin());
    result.AsLoopEnd(symbol.IsLoopEnd());
    return result;
}
