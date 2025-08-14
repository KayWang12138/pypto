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
 * \file tensor_slot.h
 * \brief
 */

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "tilefwk/tensor.h"

#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

enum class TensorSlotKind {
    T_SLOT_INVALID,
    T_SLOT_TENSOR,
    T_SLOT_CONCRETE_ASSEMBLE,
    T_SLOT_SYMBOLIC_ASSEMBLE,
};

struct TensorSlot {
public:
    TensorSlot() {}
    TensorSlot(TensorSlotKind kind, const void *slot) : kind_(kind), slot_(slot) {}

    TensorSlotKind GetKind() const { return kind_; }
    const void *GetSlot() const { return slot_; }
    bool IsKindTensor() const { return kind_ == TensorSlotKind::T_SLOT_TENSOR; }

    std::string GetSymbolName() const;

    std::shared_ptr<LogicalTensor> GetSlotValue() const;
    void SetSlotValue(const std::shared_ptr<LogicalTensor> &value) const;

    std::string Dump() const;
    std::string DumpHead(const std::string &name) const;

    bool operator==(const TensorSlot &th) const { return kind_ == th.kind_ && slot_ == th.slot_; }

    static TensorSlot CreateTensor(const Tensor &tensor) {
        return TensorSlot(TensorSlotKind::T_SLOT_TENSOR, &tensor);
    }

    static TensorSlot CreateConcreteAssemble(
        const std::tuple<std::vector<int>, std::shared_ptr<LogicalTensor>> &assemble) {
        return TensorSlot(TensorSlotKind::T_SLOT_CONCRETE_ASSEMBLE, &assemble);
    }

    static TensorSlot CreateSymbolicAssemble(
        const std::tuple<std::vector<SymbolicScalar>, std::shared_ptr<LogicalTensor>> &assemble) {
        return TensorSlot(TensorSlotKind::T_SLOT_SYMBOLIC_ASSEMBLE, &assemble);
    }

private:
    TensorSlotKind kind_{TensorSlotKind::T_SLOT_INVALID};
    const void *slot_{nullptr};
};
} // namespace npu::tile_fwk

template <>
struct std::hash<npu::tile_fwk::TensorSlot> {
    std::size_t operator()(const npu::tile_fwk::TensorSlot &t) const {
        return static_cast<std::size_t>(reinterpret_cast<uintptr_t>(t.GetSlot()));
    }
};

namespace npu::tile_fwk {

class Function;

struct TensorSlotAccess {
public:
    TensorSlotAccess() {}

    const std::shared_ptr<LogicalTensor> &GetFirstReadTensor() const { return firstReadTensor_; }
    std::shared_ptr<LogicalTensor> GetFirstReadTensor() { return firstReadTensor_; }

    const std::shared_ptr<LogicalTensor> &GetLastWriteTensor() const { return lastWriteTensor_; }
    std::shared_ptr<LogicalTensor> GetLastWriteTensor() { return lastWriteTensor_; }

    void Read(const std::shared_ptr<LogicalTensor> &tensor) {
        if (!written_) {
            firstReadTensor_ = tensor;
        }
    }
    void Write(const std::shared_ptr<LogicalTensor> &tensor) {
        written_ = true;
        lastWriteTensor_ = tensor;
    }

    std::string Dump() const {
        std::ostringstream oss;
        oss << "<" << (written_ ? 'W' : ' ') << ","
            << (firstReadTensor_ ? firstReadTensor_->Dump() : std::string("noread")) << ","
            << (lastWriteTensor_ ? lastWriteTensor_->Dump() : std::string("nowrite")) << ">";
        return oss.str();
    }

private:
    bool written_{false};
    std::shared_ptr<LogicalTensor> firstReadTensor_; // read before write
    std::shared_ptr<LogicalTensor> lastWriteTensor_;
};

struct IncastOutcastSlot {
    /* One tensor might be passed via multiple slots. An example code:
     *
     *      Tensor a("a"), b("b");
     *      Tensor v0, v1;
     *      FUNCTION("A") {
     *          Tensor t = a + b;
     *          v0 = t; // t outcast via v0
     *          v1 = t; // t outcast via v1
     *      }
     *
     *      Tensor x0;
     *      FUNCTION("B") {
     *          // t incast via both v0 and v1
     *          x0 = v0 + v1;
     *      }
     */
    std::vector<std::vector<int>> incastSlot;
    std::vector<std::vector<int>> outcastSlot;
    std::vector<int> partialUpdateOutcastList;
};

struct TensorSlotScope {
    Function *tensorFunc = nullptr;
    std::unordered_map<TensorSlot, TensorSlotAccess> accessRecord;

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>> incastToInArgumentDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>> outcastToOutArgumentDict;

    std::vector<std::unordered_set<TensorSlot>> incastReadSlotSet;
    std::vector<std::unordered_set<TensorSlot>> outcastWriteSlotSet;

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::unordered_set<std::shared_ptr<LogicalTensor>>> incastToInOriginalDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::unordered_set<std::shared_ptr<LogicalTensor>>> outcastToOutOriginalDict;

    std::unordered_set<LogicalTensorPtr> partialUpdateOutcastSet;

    IncastOutcastSlot ioslot;

    explicit TensorSlotScope(Function *tfunc) : tensorFunc(tfunc) {}
    TensorSlotScope(TensorSlotScope &&scope) = default;
    TensorSlotScope &operator=(TensorSlotScope &&scope) = default;

    std::unordered_set<TensorSlot> LookupIncastReadFrom(const std::shared_ptr<LogicalTensor> &tensor) const {
        std::unordered_set<TensorSlot> tensorSlot;
        for (auto &[slot, access] : accessRecord) {
            /* Match by raw tensor */
            if (access.GetFirstReadTensor() && access.GetFirstReadTensor()->tensor == tensor->tensor) {
                tensorSlot.insert(slot);
            }
        }
        return tensorSlot;
    }

    std::unordered_set<TensorSlot> LookupOutcastWriteTo(const std::shared_ptr<LogicalTensor> &tensor) const {
        std::unordered_set<TensorSlot> tensorSlot;
        for (auto &[slot, access] : accessRecord) {
            /* Match by raw tensor */
            if (access.GetLastWriteTensor() && access.GetLastWriteTensor()->tensor == tensor->tensor) {
                tensorSlot.insert(slot);
            }
        }
        return tensorSlot;
    }

    void BuildSlotSet();
    void BuildIncastOutcastSlot(const std::unordered_map<TensorSlot, int> &slotIndexDict);
    std::string Dump() const;
};

struct IncastOutcastLink {
    explicit IncastOutcastLink(int slotNum = 0) : totalSlot(slotNum) {}

    int totalSlot;
    std::unordered_map<Function *, IncastOutcastSlot> ioslotDict;

    std::vector<int> inputSlotIndexList;
    std::vector<int> outputSlotIndexList;
    std::vector<int> assembleSlotIndexList;
    std::vector<int> inplaceSlotIndexList;
    std::vector<int> partialUpdateSlotIdexList;
};

struct TensorSlotCheckpoint {
    std::unordered_map<TensorSlot, std::shared_ptr<LogicalTensor>> slotDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::set<Operation *, LogicalTensor::CompareOp>> producerDict;
    std::unordered_map<std::shared_ptr<LogicalTensor>, std::set<Operation *, LogicalTensor::CompareOp>> consumerDict;
};

struct TensorSlotManager {
    std::vector<std::shared_ptr<TensorSlotScope>> scopeList;

    std::shared_ptr<TensorSlotScope> currScope;

    /* Mapping from slot to its index */
    std::unordered_map<TensorSlot, int> slotIndexDict;
    std::unordered_set<TensorSlot> liveSlotSet;
    std::unordered_set<TensorSlot> assembleSlotSet;

    std::unordered_map<std::string, TensorSlot> symbolNameDict;
    std::unordered_map<TensorSlot, std::string> slotNameDict;

    std::vector<TensorSlot> inputSlotList;
    std::unordered_map<TensorSlot, int> inputSlotDict;
    std::vector<std::string> inputNameList;

    std::vector<TensorSlot> outputSlotList;
    std::unordered_map<TensorSlot, int> outputSlotDict;
    std::vector<std::string> outputNameList;
    std::unordered_map<TensorSlot, TensorSlot> inplaceDict;

    std::set<int> partialUpdateSlotIndexSet;

    std::vector<TensorSlotCheckpoint> checkpointStack;

    void BeginScope(Function *tensorFunc);
    std::shared_ptr<TensorSlotScope> EndScope();
    void ConnectSlot(std::shared_ptr<TensorSlotScope> scope);

    void TensorSlotRead(const TensorSlot &slot, const std::shared_ptr<LogicalTensor> &tensor);
    void TensorSlotWrite(const TensorSlot &slot, const std::shared_ptr<LogicalTensor> &tensor);
    void TensorSlotDestruct(const TensorSlot &slot);

    void TensorRead(const Tensor &tensor);
    void TensorWrite(const Tensor &tensor, bool isAssemble = false);
    void TensorDestruct(const Tensor &tensor);

    void TensorSymbol(const Tensor &tensor, const std::string &symbolName);

    std::vector<int> LookupSlotIndex(const std::vector<std::reference_wrapper<Tensor>> &tensorList);
    std::vector<int> LookupSlotIndexConst(const std::vector<std::reference_wrapper<const Tensor>> &tensorList);
    std::vector<int> LookupSlotIndexBySymbol(const std::vector<std::string> &symbolNameList);

    void MarkInput(const Tensor &tensor);
    void MarkOutput(const Tensor &tensor);
    void MarkInplace(const Tensor &out , const Tensor &in);

    const std::vector<std::string> &GetInputNameList() const { return inputNameList; }
    const std::vector<std::string> &GetOutputNameList() const { return outputNameList; }

    int GetInputIndex(const Tensor &tensor);
    int GetOutputIndex(const Tensor &tensor);

    void Checkpoint();
    void Restore();

    IncastOutcastLink BuildIncastOutcastLink(const std::string &rawname = "");

    std::string Dump() const;
private:
    void LogOperation(const TensorSlot &slot, const std::string &op);
};
} // namespace npu::tile_fwk
