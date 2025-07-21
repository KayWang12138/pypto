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
 * \file dev_encode.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

#include <fstream>
#include <memory>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "machine/utils/device_log.h"
#include "machine/utils/device_switch.h"
#include "interface/cache/core_func_data.h"
#include "securec.h"
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "interface/tensor/symbol_handler.h"
#include "machine/kernel/aicore.h"
#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/utils/dynamic/vector.h"
#include "machine/utils/dynamic/codegen/codegen.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk {
class Function;
class Operation;
class LogicalTensor;
class RawTensor;
class SymbolicSymbolTable;
class SymbolicExpressionTable;
class Linker;
class IncastOutcastLink;
class IncastOutcastSlot;
struct CceCodeInfo;
struct L2Info;
constexpr uint32_t IDENT_SIZE = 2;
constexpr uint32_t IDENT2_SIZE = 4;

namespace dynamic {
struct EncodeRawTensorAttr;

using  int32v8 = int32_t __attribute__((vector_size(32)));
using  int32v4 = int32_t __attribute__((vector_size(16)));
using  uint32v8 = uint32_t __attribute__((vector_size(32)));
using  uint32v4 = uint32_t __attribute__((vector_size(16)));
using  uint16v4 = uint16_t __attribute__((vector_size(8)));
using  uint16v8 = uint16_t __attribute__((vector_size(16)));

#define ALIGN_UP(val, align)            (((val) + (align) - 1) & ~((align) - 1))

template <typename T>
struct OrderedSet : std::unordered_map<T, int> {
    bool Insert(const T &data) {
        if (this->count(data) == 0) {
            this->insert(std::make_pair(data, this->size()));
            order.push_back(data);
            return true;
        }

        return false;
    }

    typename std::vector<T>::iterator begin() { return order.begin(); }
    typename std::vector<T>::iterator end() { return order.end(); }

    typename std::vector<T>::const_iterator begin() const { return order.begin(); }
    typename std::vector<T>::const_iterator end()const { return order.end(); }

    const T &operator[](int index) const { return order[index]; }
    T &operator[](int index) { return order[index]; }

    int GetIndex(const T &data) const { return this->find(data)->second; }

    void Remove(const std::vector<T> &items) {
        bool removed = false;
        for (size_t i = 0; i < items.size(); i++) {
            if (this->count(items[i])) {
                this->erase(items[i]);
                removed = true;
            }
        }
        if (removed) {
            std::vector<T> newOrder(this->size());
            for (auto &[key, val] : dynamic_cast<std::unordered_map<T, int> &>(*this)) {
                val = newOrder.size();
                newOrder.push_back(key);
            }
            order = std::move(newOrder);
        }
    }

    void Clear() {
        order.clear();
        this->clear();
    }

    bool operator==(const OrderedSet &rhs) {
        if (order.size() != rhs.size())
            return false;
        for (auto &x : rhs.order) {
            if (this->count(x) == 0)
                return false;
        }
        return true;
    }

    std::vector<T> order;
};

constexpr int ARG_ATTR_TYPE = 4;

// flag need used bit 63, see also macro values in tileop/runtime.h
struct SymInt {
    uint64_t value : 63;
    uint64_t flag  : 1; // 0 is const, 1 is expression

    SymInt() : SymInt(0) {}
    explicit SymInt(uint64_t val) : value(val), flag(0) {}
    SymInt(bool isExpr, uint64_t val) : value(val), flag(isExpr) {}
    SymInt &operator=(uint64_t val) {
        value = val;
        flag = 0;
        return *this;
    }

    uint64_t Value() const { return value; }
    bool IsExpression() const { return flag == 1; }
};

template <typename TDst, typename TSrc>
TDst *AddOffset(TSrc *src, uint64_t count) {
    return reinterpret_cast<TDst *>(src + count);
}

using uintdevptr_t = uint64_t;
using intdevptr_t = int64_t;

template <typename T>
inline void HostAssign(T *&ptr, uintdevptr_t offset) {
    ptr = reinterpret_cast<T *>(offset);
}
template <typename T>
inline void DeviceReloc(T *&ptr, intdevptr_t shift) {
    ptr = reinterpret_cast<T *>(reinterpret_cast<uintdevptr_t>(ptr) + shift);
}

template <typename T>
struct DevRelocPtr {
    DevRelocPtr() = default;
    explicit DevRelocPtr(void *addr) : ptr_(addr) {}

    T *operator->() { return ptr_; }
    void operator=(void *addr) { ptr_ = reinterpret_cast<T *>(addr); }

    T &operator[](int32_t idx) { return ptr_[idx]; }

    void HostAssignPtr(uintdevptr_t offset) { HostAssign(ptr_, offset); }
    void DeviceRelocPtr(intdevptr_t shift) { DeviceReloc(ptr_, shift); }

private:
    /* Not assign-able, not moveable, not copyable */
    DevRelocPtr(const DevRelocPtr &other) = delete;
    DevRelocPtr &operator=(const DevRelocPtr &) = delete;
    DevRelocPtr &operator=(DevRelocPtr &&) = delete;

    T *ptr_{nullptr};
};

template <typename T>
struct DevRelocVector {
    DevRelocVector() = default;
    DevRelocVector(int size, T *data) : size_(size), data_(data) {}

    T &operator[](size_t idx) {
        DEV_ASSERT(idx < size_);
        return data_[idx];
    }
    const T &operator[](size_t idx) const {
        DEV_ASSERT(idx < size_);
        return data_[idx];
    }

    const T *begin() const { return data_; }
    T *begin() { return data_; }
    const T *end() const { return data_ + size_; }
    T *end() { return data_ + size_; }

    size_t size() const { return size_; }
    const T *Data() const { return data_; }
    T *Data() { return data_; }

    void HostAssignDataSize(uintdevptr_t offset, size_t size) {
        HostAssign(data_, offset);
        size_ = size;
    }
    void HostAssignRangeOffsetSize(const DevRelocVector<T> &base, uintdevptr_t offset, size_t size) {
        HostAssignDataSize(reinterpret_cast<uintdevptr_t>((base.Data() + offset)), size);
    }
    void HostInitDataSizeOffset(uintdevptr_t &offset, size_t size) {
        HostAssign(data_, ALIGN_UP(offset, alignof(T)));
        size_ = size;
        offset = reinterpret_cast<uintdevptr_t>(data_ + size);
    }
    void DeviceRelocData(intdevptr_t shift) { DeviceReloc(data_, ALIGN_UP(shift, alignof(T))); }
    uintdevptr_t End() const { return reinterpret_cast<uintdevptr_t>(data_ + size_); }

private:
    size_t size_{0};
    T *data_{nullptr};
};

template <typename T>
struct DevLocalVector {
    size_t size() const { return size_; }

    uintdevptr_t Offset(int idx) const { return offset_ + sizeof(T) * idx; }

    uintdevptr_t End() const { return offset_ + sizeof(T) * size_; }

    void AssignOffsetSize(uintdevptr_t offset, size_t size) {
        offset_ = offset;
        size_ = size;
    }
    void AssignRangeOffsetSize(const DevLocalVector<T> &base, uintdevptr_t offset, size_t size) {
        AssignOffsetSize(base.offset_ + sizeof(T) * offset, size);
    }

    void HostInitDataSizeOffset(uintdevptr_t &offset, size_t size) {
        offset = ALIGN_UP(offset, alignof(T));
        offset_ = offset;
        size_ = size;
        offset = offset_ + size_ * sizeof(T);
    }

    size_t ByteSize() const { return size_ * sizeof(T); }

private:
    uintdevptr_t offset_{0};
    size_t size_{0};
};

struct DevCceBinary {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
    DevRelocVector<uint8_t> binary;
};

static_assert(sizeof(DynFuncBin) == sizeof(DevCceBinary));

struct PrefetchInfo {
    uint64_t tensorSize;
    uint64_t tensorIdx;
};

enum class DevIOProperty : uint32_t {
    NONE,
    ROOT_INCAST,
    ROOT_OUTCAST,
};

static inline const BiMap<DevIOProperty> &GetDevIOPropertyDict() {
    static BiMap<DevIOProperty> propertyDict = {
        {        DevIOProperty::NONE,         "NONE"},
        { DevIOProperty::ROOT_INCAST,  "ROOT_INCAST"},
        {DevIOProperty::ROOT_OUTCAST, "ROOT_OUTCAST"},
    };
    return propertyDict;
}

static inline std::string DevIOProperty2String(DevIOProperty property) {
    return GetDevIOPropertyDict().Find(property);
}

struct DevSymShape {
    int dimSize{0};
    SymInt dim[DEV_SHAPE_DIM_MAX];

    void SetShape(const std::vector<SymInt> &shape) {
        for (std::size_t i = 0; i < shape.size(); i++) {
            dim[i] = shape[i];
        }
        dimSize = shape.size();
    }

    uint64_t At(size_t idx, uint64_t *exprTbl) const {
        if (dim[idx].IsExpression())
            return exprTbl[idx];
        else
            return dim[idx].Value();
    }

    void ToStride(uint64_t *stides, uint64_t *exprTbl) const {
        stides[dimSize - 1] = 1;
        for (int i = dimSize - 1; i > 0; i--) {
            stides[i - 1] = stides[i] * At(i, exprTbl);
        }
    }
};

struct DevAscendRawTensor {
    // Offset in DevAscendFunction (root outcasts & non i/o raw tensors, separately recorded)
    uint64_t addrOffset{UINT64_MAX};
    uint64_t memoryRequirement; // Only available for incast/outcast
                                // For workspace tensors, the memoryRequirement property is deprecated
    DataType dataType;
    DevSymShape shape;
    DevIOProperty ioProperty{DevIOProperty::NONE};
    int32_t ioIndex;

    int GetDim() const { return shape.dimSize; }

    std::string Dump() const {
        std::ostringstream oss;

        oss << "<";
        for (int i = 0; i < shape.dimSize; i++) {
            if (shape.dim[i].IsExpression())
                oss << "?x";
            else
                oss << shape.dim[i].Value() << "x";
        }
        oss << DataType2String(dataType);
        oss << ",#iokind:" << DevIOProperty2String(ioProperty);
        oss << ",#ioindex:" << ioIndex;
        oss << ",#memory:" << memoryRequirement;
        oss << ",#baseoffset:" << addrOffset;
        oss << ">";
        return oss.str();
    }
};

struct DevAscendTensor {
    uint64_t rawIndex;
};

/* please modify macros in aicore.cpp at the same time !!! */
constexpr uint32_t TASKID_FUNC_BITS = 11;
#define TASKID_FUNC_MASK ((1 << TASKID_FUNC_BITS) - 1)
constexpr uint32_t TASKID_TASK_BITS = 20;
#define TASKID_TASK_MASK ((1 << TASKID_TASK_BITS) - 1)

inline uint32_t MakeTaskID(uint32_t funcId, uint32_t taskId) {
    return (funcId << TASKID_TASK_BITS) | taskId;
}

inline uint32_t MakeBatchTaskID(uint32_t batchNum) {
    return MakeTaskID(FUNC_ID_BATCH, batchNum);
}

inline uint32_t FuncID(uint32_t id) {
    return id >> TASKID_TASK_BITS;
}

inline uint32_t FuncNum(uint32_t id) {
    return id & TASKID_TASK_MASK;
}

inline uint32_t TaskID(uint32_t id) {
    return id & TASKID_TASK_MASK;
}

inline bool IsInitTaskID(uint32_t id) {
    return id == AICORE_TASK_INIT;
}

inline bool IsTaskFinish(uint32_t id, uint32_t finValue) {
    return (id | AICORE_FIN_MASK) == finValue;
}

using DevStitch = Vector<uint32_t, WsMemCategory::VECTOR_DEV_STITCH>;

struct DevAscendOperationDynamicField {
    predcount_t currPredCount{0};
    DevStitch stitch;
};

struct DevAscendOperationOperandInfo {
    int tensorIndex{0};
    int staticOffsetAttrBeginIndex{0};
    int staticShapeAttrBeginIndex{0};

    DevAscendOperationOperandInfo() {}
    DevAscendOperationOperandInfo(int tTensorIndex, int tStaticAttrBeginIndex, int tStaticDim)
        : tensorIndex(tTensorIndex),
          staticOffsetAttrBeginIndex(tStaticAttrBeginIndex),
          staticShapeAttrBeginIndex(tStaticAttrBeginIndex + tStaticDim) {}
    int GetDim() const { return staticShapeAttrBeginIndex - staticOffsetAttrBeginIndex; }
};

struct DevAscendOperation {
    DevLocalVector<DevAscendOperationOperandInfo> ioperandList;
    DevLocalVector<DevAscendOperationOperandInfo> ooperandList;
    DevLocalVector<SymInt> attrList; // opattr[0] -> hash
    int32_t outcastStitchIndex;
    uint32_t predCount;
    uint64_t opmagic;
    DevLocalVector<int> succList;
};

struct DevAscendFunctionIncast {
    int tensorIndex;
    DevLocalVector<int> fromSlotList;

    int dim;
    int fastStitchEnable;
    DevLocalVector<int> shapeAttrIdx;
    DevLocalVector<int> offsetAttrIdx;
    DevLocalVector<int> consumer;
    DevLocalVector<int> operandIdx;
    DevLocalVector<int> fastStitchTileIdx;
};

struct DevAscendFunctionOutcast {
    int tensorIndex;
    DevLocalVector<int> toSlotList;

    int dim;
    int fastStitchEnable;
    DevLocalVector<int> shapeAttrIdx;
    DevLocalVector<int> offsetAttrIdx;
    DevLocalVector<int> producer;
    DevLocalVector<int> operandIdx;
    DevLocalVector<int> minimalShape;
    DevLocalVector<int> minimalTileIdx;
    DevLocalVector<int> fastStitchTileIdx;
};

struct InoutOperationAttr {
    int dim;
    int minimalTileListSize;
    std::vector<int> tileEachDim;
    std::vector<int> minimalShape;
    std::vector<int> offsetAttrIdx;
    std::vector<int> shapeAttrIdx;
    std::vector<int> operandIdx;
    std::vector<int> ops;
};

struct AddressDescriptor {
    union {
        struct {
            uint64_t outcastIdx : 32;
            uint64_t dupIdx : 31; // in stitch window
            uint64_t : 1;
        };
        struct {
            uint64_t addr : 63;
            uint64_t isAddress : 1;
        };
    };

    std::string ToString() {
        std::stringstream ss;
        if (isAddress)
            ss  << std::hex << addr;
        else
            ss  << '(' << dupIdx << ", " << outcastIdx << ')';
        return ss.str();
    }

    bool IsAddress() const { return isAddress; }
    uint64_t GetAddress() const { DEV_ASSERT(isAddress); return addr; }
    bool IsNullAddress() const { return IsAddress() && addr == 0; }

    explicit AddressDescriptor(uint64_t address = 0): addr(address) { isAddress = true; }
    AddressDescriptor(int tdupIdx, int toutcastIdx): outcastIdx(toutcastIdx) , dupIdx(tdupIdx) { isAddress = false; }
};

constexpr int INVALID_INDEX = -1;

struct DevAscendFunctionDuppedData;

struct DevAscendFunctionPredInfo {
    uint64_t totalZeroPred;
    uint64_t totalZeroPredAIV;
    uint64_t totalZeroPredAIC;
    uint64_t totalZeroPredHub;
};

struct DevAscendFunction {
    uint64_t funcKey;
    // source root function after duplication
    DevAscendFunction *sourceFunc{nullptr};

    // Fill base address after stitch
    uintdevptr_t runtimeWorkspace;
    // Base address of invoke entries
    uintdevptr_t opAttrs;

    int funcidx;

    int stackWorkSpaceSize;

    DevLocalVector<AddressDescriptor> incastAddressList;
    DevLocalVector<AddressDescriptor> outcastAddressList;

    DevLocalVector<DevAscendOperationDynamicField> opDynamicFieldList;
#define duplicateLastField opDynamicFieldList
    DevLocalVector<uint64_t> expressionList;
#define allocateLastField expressionList

    DevAscendFunctionPredInfo predInfo_;
    uint64_t duppedDataAllocSize_;
    uint64_t duppedDataCopySize_;
    DevLocalVector<uint8_t> duppedData_;

public:
    // total memory requirement of non-root-incast/outcast raw tensors
    uint64_t rawTensorWsMemoryRequirement{0};
    uint64_t outcastWsMemoryRequirement{0};

private:
    DevLocalVector<DevAscendRawTensor> rawTensorList_;
    DevLocalVector<DevRawTensorDesc> rawTensorDescList_;
    DevLocalVector<DevAscendTensor> tensorList_;
    DevLocalVector<int> noPredOpList_;
    DevLocalVector<int> noSuccOpList_;
    DevLocalVector<DevAscendOperation> operationList_;
    DevLocalVector<DevAscendOperationOperandInfo> operationOperandInfoList_;
    DevLocalVector<SymInt> operationAttrList_;
    DevLocalVector<int> opAttrOffsetList_;
    DevLocalVector<int> opCalleeList_;
    DevLocalVector<int> operationSuccList_;
    DevLocalVector<DevAscendFunctionIncast> incastList;
    DevLocalVector<DevAscendFunctionOutcast> outcastList;
    DevLocalVector<int> slotList;
    DevLocalVector<int> minimalShapeList;
    DevLocalVector<int> offsetIdxList;
    DevLocalVector<int> shapeIdxList;
    DevLocalVector<int> producerConsumerList;
    DevLocalVector<int> inoutOperandIdxList;
    DevLocalVector<int> minimalTileIdxList;
    DevLocalVector<int> outcastMinimalTileIdxList;
    DevLocalVector<int> incastMinimalTileIdxList;
    DevLocalVector<char> rawName_;
#define sharedLastField rawName_
public:
    uint8_t data[0];
    /*
     *  Duplicated:
     *      AddressDescriptor                                   incastAddressListData;
     *      AddressDescriptor                                   outcastAddressListData;
     *      DevAscendOperationDynamicField                      opDynamicFieldListData[];
     *  Allocated:
     *      uint64_t                                            expressionListData[];
     *
     *  Shared:
     *      DevRawTensorDesc                              rawTensorDescListData[];
     *      DevAscendRawTensor                                  rawTensorListData[];
     *      DevAscendTensor                                     tensorListData[];
     *      int                                                 noPredOpListData[];
     *      int                                                 noSuccOpListData[];
     *      DevAscendOperation                                  operationListData[];
     *      DevAscendOperationOperandInfo                       operationOperandListData[];
     *      SymInt                                              operationAttrListData[];
     *      int                                                 operationSuccListData[];
     *      DevAscendFunctionIncast                             incastListData[];
     *      DevAscendFunctionOutcast                            outcastListData[];
     *      int                                                 slotListData[];
     *      int                                                 offsetIdxListData[];
     *      int                                                 shapeIdxListData[];
     *      int                                                 producerConsumerListData[];
     *      char                                                rawNameData[];
     *      uint8_t                                             duppedData[];
     */

    void Verify(uintptr_t /*base*/) { DEV_ASSERT(opDynamicFieldList.size() == operationList_.size()); }

    template <typename T>
    const T &At(const DevLocalVector<T> &localvec, int index) const {
        return *reinterpret_cast<T *>((reinterpret_cast<uint64_t>(this) + localvec.Offset(index)));
    }
    template <typename T>
    T &At(const DevLocalVector<T> &localvec, int index) {
        return *reinterpret_cast<T *>((reinterpret_cast<uint64_t>(this) + localvec.Offset(index)));
    }
    template <typename T>
    const T &At(const DevRelocVector<T> &localvec, int index) const {
        return localvec[index];
    }
    template <typename T>
    T &At(DevRelocVector<T> &localvec, int index) {
        return localvec[index];
    }

private:
    std::string DumpTensor(int tensorIndex) const {
        std::ostringstream oss;
        oss << "%" << tensorIndex << "@" << GetTensor(tensorIndex)->rawIndex;
        return oss.str();
    }

public:
    static std::string DumpAddress(uintdevptr_t addr, int width = 0) {
        char bufData[0x20];
        (void)sprintf_s(bufData, sizeof(bufData), "%lx", addr);
        std::string buf = bufData;
        if (buf.size() < static_cast<size_t>(width)) {
            buf = std::string(width - buf.size(), '0') + buf;
        }
        return "&0x" + buf;
    }
    static std::string DumpByte(uint8_t byte) {
        char buf[0x10];
        (void)sprintf_s(buf, sizeof(buf), "0x%02x", byte);
        return buf;
    }

    std::string DumpOperation(int operationIndex, int &totalAttrStartIdx, const std::vector<uintdevptr_t> &ooperandAddrList = {},
        const std::vector<uintdevptr_t> &ioperandAddrList = {}, const std::vector<uint64_t> &exprList = {}) const {
        std::ostringstream oss;
        for (size_t j = 0; j < GetOperationOOperandSize(operationIndex); j++) {
            if (j != 0) {
                oss << ", ";
            }
            oss << DumpTensor(GetOperationOOperandInfo(operationIndex, j).tensorIndex);
            if (j < ooperandAddrList.size()) {
                oss << DumpAddress(ooperandAddrList[j]);
            }
        }
        oss << " = "
            << "!" << operationIndex << " ";
        oss << "[";
        for (size_t j = 0; j < GetOperationAttrSize(operationIndex); j++) {
            const SymInt &s = GetOperationAttr(operationIndex, j);
            if (j != 0) {
                oss << ",";
            }
            oss << "[" << j << "]=";
            if (s.IsExpression()) {
                if (s.Value() < exprList.size()) {
                    oss << exprList[s.Value()];
                } else {
                    oss << "?" << s.Value();
                }
            } else {
                oss << s.Value();
            }
        }
        totalAttrStartIdx += GetOperationAttrSize(operationIndex);
        oss << "] ";
        for (size_t j = 0; j < GetOperationIOperandSize(operationIndex); j++) {
            if (j != 0) {
                oss << ", ";
            }
            oss << DumpTensor(GetOperationIOperandInfo(operationIndex, j).tensorIndex);
            if (j < ioperandAddrList.size()) {
                oss << DumpAddress(ioperandAddrList[j]);
            }
        }

        oss << " #pred:" << GetOperationPredCount(operationIndex);
        oss << " #succ:[";
        const DevLocalVector<int> &succList = GetOperationSuccList(operationIndex);
        for (size_t j = 0; j < succList.size(); j++) {
            if (j != 0) {
                oss << ",";
            }
            oss << "!" << At(succList, j);
        }
        oss << "]";
        return oss.str();
    }

    std::string DumpRawTensor(int rawIndex, uintdevptr_t addr = 0) const {
        std::ostringstream oss;
        oss << "@" << rawIndex << " = " << GetRawTensor(rawIndex)->Dump();
        if (addr != 0) {
            oss << DumpAddress(addr);
        }
        return oss.str();
    }

    std::string DumpIncast(int incastIndex, const std::string &indent, const std::vector<uintdevptr_t> &slotAddrList = {}) const {
        std::ostringstream oss;
        const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
        oss << "#incast:" << incastIndex << " = " << DumpTensor(incast.tensorIndex);
        for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
            int slot = At(incast.fromSlotList, j);
            oss << " <- #slot:" << slot;
            if (slot < static_cast<int>(slotAddrList.size())) {
                oss << DumpAddress(slotAddrList[slot]);
            }
        }
        oss << "\n";
        for (size_t j = 0; j < incast.consumer.size(); j++) {
            int consumer = At(incast.consumer, j);
            int operandIdx = At(incast.operandIdx, j);
            int offsetAttrIdx = At(incast.offsetAttrIdx, j);
            int shapeAttrIdx = At(incast.shapeAttrIdx, j);
            oss << indent;
            oss << " | #consumer:!" << consumer;
            oss << " | #operandIdx:" << operandIdx;
            oss << " | #offsetAttrIdx:" << offsetAttrIdx;
            oss << " | #shapeAttrIdx:" << shapeAttrIdx;
            oss << "\n";
        }
        return oss.str();
    }

    std::string DumpOutcast(int outcastIndex, const std::string &indent, const std::vector<uintdevptr_t> &slotAddrList = {}) const {
        std::ostringstream oss;
        const DevAscendFunctionOutcast &outcast = GetOutcast(outcastIndex);
        oss << "#outcast:" << outcastIndex << " = " << DumpTensor(outcast.tensorIndex);
        for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
            int slot = At(outcast.toSlotList, j);
            oss << " -> #slot:" << slot;
            if (slot < static_cast<int>(slotAddrList.size())) {
                oss << DumpAddress(slotAddrList[slot]);
            }
        }
        oss << " #minimalTileIdx size:" << outcast.minimalTileIdx.size();
        oss << "\n";
        for (size_t j = 0; j < outcast.producer.size(); j++) {
            int producer = At(outcast.producer, j);
            int operandIdx = At(outcast.operandIdx, j);
            int offsetAttrIdx = At(outcast.offsetAttrIdx, j);
            int shapeAttrIdx = At(outcast.shapeAttrIdx, j);
            oss << indent;
            oss << " | #producer:!" << producer;
            oss << " | #operandIdx:" << operandIdx;
            oss << " | #offsetAttrIdx:" << offsetAttrIdx;
            oss << " | #shapeAttrIdx:" << shapeAttrIdx;
            oss << " | #minimalShape:";
            for (size_t k = 0; k < static_cast<size_t>(outcast.dim); k++) {
                int minimalShape = At(outcast.minimalShape,  k);
                oss << minimalShape << ",";
            }
            oss << "\n";
        }
        return oss.str();
    }

public:
    std::string Dump(int indent = 0) const {
        std::string INDENT(indent, ' ');
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');
        std::ostringstream oss;

        oss << INDENT << "DevFunction " << funcKey << " {\n";
        oss << INDENTINNER << "#name:" << GetRawName() << "\n";
        for (size_t i = 0; i < GetRawTensorSize(); i++) {
            oss << INDENTINNER << DumpRawTensor(i) << "\n";
        }
        for (size_t i = 0; i < GetIncastSize(); i++) {
            oss << INDENTINNER << DumpIncast(i, INDENTINNER) << "\n";
        }
        for (size_t i = 0; i < GetOutcastSize(); i++) {
            oss << INDENTINNER << DumpOutcast(i, INDENTINNER) << "\n";
        }

        oss << INDENTINNER << "#internal-byte:" << rawTensorWsMemoryRequirement << "\n";
        oss << INDENTINNER << "#outcast-byte:" << outcastWsMemoryRequirement << "\n";

        oss << INDENTINNER << "#zeropred:" << predInfo_.totalZeroPred << "\n";
        oss << INDENTINNER << "#zeropred-aiv:" << predInfo_.totalZeroPredAIV << "\n";
        oss << INDENTINNER << "#zeropred-aic:" << predInfo_.totalZeroPredAIC << "\n";
        int totalAttrStartIdx = 0;
        for (size_t i = 0; i < GetOperationSize(); i++) {
            oss << INDENTINNER << DumpOperation(i, totalAttrStartIdx) << "\n";
        }
        oss << INDENT << "}";
        return oss.str();
    }

    template <typename T>
    uint64_t GetEndOffset(const DevLocalVector<T> &localvec) const {
        return localvec.End();
    }

    void Reloc(intptr_t /* shift */, bool /* relocShared */) {}

    void DuplicateTo(DevAscendFunction *func) {
        auto size = GetDuplicateSize();
        memcpy_s(func, size, this, size);
        func->sourceFunc = this;
        func->Reloc(reinterpret_cast<intptr_t>(this) - reinterpret_cast<intptr_t>(func), false);
    }

    const DevAscendFunction *GetSource() const { return sourceFunc; }
    DevAscendFunction *GetSource() { return sourceFunc; }

    const int &GetFuncidx() const { return funcidx; }
    int &GetFuncidx() { return funcidx; }

    const DevAscendFunctionPredInfo &GetPredInfo() const { return predInfo_; }
    uint64_t GetDuppedDataAllocSize() const { return duppedDataAllocSize_; }
    uint64_t GetDuppedDataCopySize() const { return duppedDataCopySize_; }
    DevAscendFunctionDuppedData *GetDuppedData() const { return reinterpret_cast<DevAscendFunctionDuppedData *>(const_cast<uint8_t*>(&At(duppedData_, 0))); }

    uint64_t GetDuplicateSize() const { return GetEndOffset(duplicateLastField); }

    int32_t *GetOpAttrOffsetAddr() { return &At(opAttrOffsetList_, 0); }
    int *GetCalleeIndexAddr() { return &At(opCalleeList_, 0); }
    uint64_t *GetExpressionAddr() { return &At(expressionList, 0); }
    uint64_t GetAllocateSize() const { return GetEndOffset(allocateLastField); }

    uint64_t GetSize() const { return GetEndOffset(sharedLastField); }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace; }
    uintdevptr_t &GetRuntimeWorkspace() { return runtimeWorkspace; }

    inline AddressDescriptor GetIncastAddress(int index) const { return At(incastAddressList, index); }
    inline AddressDescriptor &GetIncastAddress(int index) { return At(incastAddressList, index); }

    inline AddressDescriptor GetOutcastAddress(int index) const { return At(outcastAddressList, index); }
    inline AddressDescriptor &GetOutcastAddress(int index) { return At(outcastAddressList, index); }

    inline uint64_t GetExpression(int tableIndex) const { return At(expressionList, tableIndex); }
    inline uint64_t &GetExpression(int tableIndex) { return At(expressionList, tableIndex); }
    inline uint64_t GetRawTensorSize() const { return rawTensorList_.size(); }
    inline const DevAscendRawTensor *GetRawTensor(const DevAscendTensor *tensor) const {
        int rawTensorIndex = tensor->rawIndex;
        return &At(rawTensorList_, rawTensorIndex);
    }
    inline const DevAscendRawTensor *GetRawTensor(int rawIndex) const { return &At(rawTensorList_, rawIndex); }
    inline DevAscendRawTensor *GetRawTensor(int rawIndex) { return &At(rawTensorList_, rawIndex); }
    inline const DevRawTensorDesc *GetRawTensorDesc(int rawIndex) const { return &At(rawTensorDescList_, rawIndex); }
    inline DevRawTensorDesc *GetRawTensorDesc(int rawIndex) { return &At(rawTensorDescList_, rawIndex); }
    inline size_t GetRawTensorDescSize() { return rawTensorDescList_.size(); }

    inline uint64_t GetTensorSize() const { return tensorList_.size(); }
    inline const DevAscendTensor *GetTensor(int index) const { return &At(tensorList_, index); }
    inline DevAscendTensor *GetTensor(int index) { return &At(tensorList_, index); }

    inline size_t GetNoPredOpSize() const { return noPredOpList_.size(); }
    inline int GetNoPredOpIdx(size_t idx) const { return At(noPredOpList_, idx); }

    inline size_t GetNoSuccOpSize() const { return noSuccOpList_.size(); }
    inline int GetNoSuccOpIdx(size_t idx) const { return At(noSuccOpList_, idx); }

    inline size_t GetOperationSize() const { return operationList_.size(); }
    inline uint32_t GetOperationOutcastStitchIndex(int operationIndex) const {
        return At(operationList_, operationIndex).outcastStitchIndex;
    }
    inline uint32_t GetOperationOpmagic(int operationIndex) const {
        return At(operationList_, operationIndex).opmagic;
    }
    inline size_t GetOperationIOperandSize(int operationIndex) const {
        return At(operationList_, operationIndex).ioperandList.size();
    }
    inline size_t GetOperationOOperandSize(int operationIndex) const {
        return At(operationList_, operationIndex).ooperandList.size();
    }

    inline const DevAscendOperationOperandInfo &GetOperationIOperandInfo(int operationIndex, int operandIndex) const {
        return At(At(operationList_, operationIndex).ioperandList, operandIndex);
    }
    inline const DevAscendTensor *GetOperationIOperand(int operationIndex, int operandIndex) const {
        int tensorIndex = GetOperationIOperandInfo(operationIndex, operandIndex).tensorIndex;
        return GetTensor(tensorIndex);
    }

    inline const DevAscendOperationOperandInfo &GetOperationOOperandInfo(int operationIndex, int operandIndex) const {
        return At(At(operationList_, operationIndex).ooperandList, operandIndex);
    }
    inline const DevAscendTensor *GetOperationOOperand(int operationIndex, int operandIndex) const {
        int tensorIndex = GetOperationOOperandInfo(operationIndex, operandIndex).tensorIndex;
        return GetTensor(tensorIndex);
    }

    inline const DevAscendOperationOperandInfo &GetOperationOperandInfo(
        int operationIndex, int operandIndex, bool isIOperand = true) const {
        if (isIOperand) {
            return GetOperationIOperandInfo(operationIndex, operandIndex);
        } else {
            return GetOperationOOperandInfo(operationIndex, operandIndex);
        }
    }

    inline size_t GetOperationAttrSize(int operationIndex) const {
        return At(operationList_, operationIndex).attrList.size();
    }
    inline const SymInt &GetOperationAttr(int operationIndex, int attrIndex) const {
        return At(At(operationList_, operationIndex).attrList, attrIndex);
    }
    inline int GetOperationAttrCalleeIndex(int operationIndex) const {
        return GetOperationAttr(operationIndex, 0).Value();
    }

    inline int GetOpAttrSize() { return operationAttrList_.size(); }

    inline void FillOpAttrs(DevCceBinary *cceInfo) {
        (void)cceInfo;
    }

    inline const uint32_t &GetOperationPredCount(int operationIndex) const { return At(operationList_, operationIndex).predCount; }
    inline uint32_t &GetOperationPredCount(int operationIndex) { return At(operationList_, operationIndex).predCount; }

    inline const DevLocalVector<int> &GetOperationSuccList(int operationIndex) const {
        return At(operationList_, operationIndex).succList;
    }

    inline const int *GetOperationSuccAddr(int operationIndex, size_t &size) const {
        auto &succList = At(operationList_, operationIndex).succList;
        size = succList.size();
        return &At(succList, 0);
    }

    inline size_t GetIncastSize() const { return incastList.size(); }
    inline const struct DevAscendFunctionIncast &GetIncast(int index) const { return At(incastList, index); }
    inline struct DevAscendFunctionIncast &GetIncast(int index) { return At(incastList, index); }
    inline const DevAscendRawTensor *GetIncastRawTensor(int index) const {
        int tensorIndex = GetIncast(index).tensorIndex;
        return GetRawTensor(GetTensor(tensorIndex));
    }

    inline size_t GetOutcastSize() const { return outcastList.size(); }
    inline const struct DevAscendFunctionOutcast &GetOutcast(int index) const { return At(outcastList, index); }
    inline struct DevAscendFunctionOutcast &GetOutcast(int index) { return At(outcastList, index); }

    int LookupIncastBySlotIndex(int slotIndex) const {
        for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
            const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
            for (size_t fromIndex = 0; fromIndex < incast.fromSlotList.size(); fromIndex++) {
                int slot = At(incast.fromSlotList, fromIndex);
                if (slot == slotIndex) {
                    return static_cast<int>(incastIndex);
                }
            }
        }
        return INVALID_INDEX;
    }
    std::vector<int> LookupIncastBySlotIndexList(const std::vector<int> &slotIndexList) const {
        std::vector<int> resultList(slotIndexList.size());
        for (size_t i = 0; i < slotIndexList.size(); i++) {
            resultList[i] = LookupIncastBySlotIndex(slotIndexList[i]);
        }
        return resultList;
    }

    int LookupOutcastBySlotIndex(int slotIndex) const {
        for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
            const DevAscendFunctionOutcast &outcast = GetOutcast(outcastIndex);
            for (size_t toIndex = 0; toIndex < outcast.toSlotList.size(); toIndex++) {
                int slot = At(outcast.toSlotList, toIndex);
                if (slot == slotIndex) {
                    return static_cast<int>(outcastIndex);
                }
            }
        }
        return INVALID_INDEX;
    }
    std::vector<int> LookupOutcastBySlotIndexList(const std::vector<int> &slotIndexList) const {
        std::vector<int> resultList(slotIndexList.size());
        for (size_t i = 0; i < slotIndexList.size(); i++) {
            resultList[i] = LookupOutcastBySlotIndex(slotIndexList[i]);
        }
        return resultList;
    }

    std::vector<std::tuple<int, int, int>> LookupConnectionSlotIndexFrom(const DevAscendFunction *func) const {
        std::vector<std::tuple<int, int, int>> connectionList;
        for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
            const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
            for (size_t fromIndex = 0; fromIndex < incast.fromSlotList.size(); fromIndex++) {
                int fromSlot = At(incast.fromSlotList, fromIndex);

                for (size_t outcastIndex = 0; outcastIndex < func->GetOutcastSize(); outcastIndex++) {
                    const DevAscendFunctionOutcast &outcast = func->GetOutcast(outcastIndex);
                    for (size_t toIndex = 0; toIndex < outcast.toSlotList.size(); toIndex++) {
                        int toSlot = func->At(outcast.toSlotList, toIndex);
                        if (fromSlot == toSlot) {
                            connectionList.push_back(std::tuple(outcastIndex, incastIndex, fromSlot));
                        }
                    }
                }
            }
        }
        return connectionList;
    }

    inline const DevAscendRawTensor *GetOutcastRawTensor(int index) const {
        int tensorIndex = GetOutcast(index).tensorIndex;
        return GetRawTensor(GetTensor(tensorIndex));
    }

    inline void GetTensorOffset(uint64_t offset[DEV_SHAPE_DIM_MAX], const DevAscendRawTensor *rawTensor,
        const DevAscendOperationOperandInfo &operandInfo) const {
        const SymInt *offsetSymList = &At(operationAttrList_, operandInfo.staticOffsetAttrBeginIndex);
        for (int i = 0; i < rawTensor->GetDim(); i++) {
            offset[i] = offsetSymList[i].IsExpression() ? At(expressionList, offsetSymList[i].Value()) :
                                                          offsetSymList[i].Value();
        }
    }

    inline const SymInt *GetSymoffset(int offset) const { return &At(operationAttrList_, offset); }

    inline const char *GetRawName() const { return &At(rawName_, 0); }

    std::string DumpIncastOutcast() const {
        std::ostringstream oss;
        for (size_t i = 0; i < GetIncastSize(); i++) {
            auto &incast = GetIncast(i);
            for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
                int slot = At(incast.fromSlotList, j);
                oss << "INCAST:" << i << " <- slot: " << slot << "\n";
            }
        }
        for (size_t i = 0; i < GetOutcastSize(); i++) {
            auto &outcast = GetOutcast(i);
            for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
                int slot = At(outcast.toSlotList, j);
                oss << "OUTCAST:" << i << " <- slot: " << slot << "\n";
            }
        }
        return oss.str();
    }

private:
    friend struct EncodeDevAscendFunctionInfo;

    void InitIncastOutcastAttr(
            uintdevptr_t &initOffset,
            const std::vector<std::shared_ptr<LogicalTensor>> &iList,
            const std::vector<std::shared_ptr<LogicalTensor>> &oList, bool fillContent);
    void InitOperationDynamicField(
            uintdevptr_t &initOffset,
            DevAscendFunctionPredInfo predInfo,
            uint32_t outcastStitchCount,
            const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
            const SymbolicExpressionTable *expressionTable,
            const OrderedSet<Operation *> &callList,
            const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
            const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
            const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict, bool fillContent);

    void InitRawTensorAndMemoryRequirement(
            uintdevptr_t &initOffset,
            const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
            const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
            const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
            const std::vector<EncodeRawTensorAttr> &rawAttrs,
            const IncastOutcastLink *inoutLink, const IncastOutcastSlot *slot,
            const SymbolicExpressionTable *expressionTable,
            bool fillContent);

    void UpdateRawTensorDesc(const std::shared_ptr<RawTensor> &rawTensor, size_t i, size_t incastRawListSize,
        DevAscendRawTensor &encoded);

    void InitTensor(
            uintdevptr_t &initOffset,
            const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList, bool fillContent);

    void InitOperation(
            uintdevptr_t &initOffset,
            const SymbolicExpressionTable *expressionTable,
            const OrderedSet<Operation *> &callList,
            const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
            const std::unordered_map<Operation *, uint64_t> &callOpPredDict,
            const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict,
            const std::unordered_map<uint64_t, int> &calleeHashIndexDict,
            const std::vector<int32_t> &outcastStitchIndexList,
            bool fillContent);

    void InitIncastOutcast(uintdevptr_t &initOffset, const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &inoutOpAttrs,
        const IncastOutcastSlot *slot, const std::string &initRawName, bool fillContent);
};

struct DevAscendFunctionDuppedOperation {
    uint16_t size;
    uint16_t predCountBase;
    uint16_t stitchBase;
};
struct DevAscendFunctionDuppedVector {
    uint16_t size;
    uint16_t base;
};

constexpr uint32_t DUPPED_STITCH_SIZE  = 13;
struct DevAscendFunctionDuppedStitch {
    void InitWithNext(DevAscendFunctionDuppedStitch *next) {
        next_ = next;
        size_ = 0;
    }

    void PushBack(uint32_t taskId) {
        DEV_DEBUG_ASSERT(size_ < DUPPED_STITCH_SIZE);
        taskList_[size_++] = taskId;
    }

    uint32_t Size() const { return size_; }
    DevAscendFunctionDuppedStitch *Next() const { return next_; }

    uint32_t At(uint32_t idx) const {
        DEV_DEBUG_ASSERT(idx < size_);
        return taskList_[idx];
    }

    void ForEach(const std::function<void(uint32_t id)> &callback) const {
        for (uint32_t i = 0; i < size_; i++) {
            callback(taskList_[i]);
        }
    }

private:
    DevAscendFunctionDuppedStitch *next_;
    uint32_t size_;
    uint32_t taskList_[DUPPED_STITCH_SIZE];
};

struct DevAscendFunctionDuppedStitchList {
    DevAscendFunctionDuppedStitchList() = default;

    bool IsNull() const { return head_ == nullptr; }

    DevAscendFunctionDuppedStitch *Head() const { return head_; }

    // Low performance, only used in debug
    void ForEach(const std::function<void(uint32_t id)> &callback) const {
        for (auto *p = head_; p != nullptr; p = p->Next()) {
            p->ForEach(callback);
        }
    }

    void PushBack(uint32_t taskId, std::function<DevAscendFunctionDuppedStitch *()> allocate) {
        if (head_ == nullptr || head_->Size() == DUPPED_STITCH_SIZE) {
            auto *newNode = allocate();
            newNode->InitWithNext(head_);
            head_ = newNode;
        }
        head_->PushBack(taskId);
    }

    friend std::ostream &operator<<(std::ostream &os, DevAscendFunctionDuppedStitchList ptr) {
        os << "[";
        bool isFirstElem = true;
        ptr.ForEach([&isFirstElem, &os](uint32_t id) {
            if (isFirstElem) {
                isFirstElem = false;
            } else {
                os << ", ";
            }
            os << id;
        });
        os << "]";
        return os;
    }

private:
    DevAscendFunctionDuppedStitch *head_{nullptr};
};
static_assert(sizeof(DevAscendFunctionDuppedStitchList) == sizeof(void *));

struct RuntimeReuseInfo {
    uint32_t blockIdx;
    uint32_t poolResetTimes;
};

struct DevAscendFunctionDuppedData {
    DevAscendFunction *source_;
    DevAscendFunctionDuppedOperation operationList_;
    DevAscendFunctionDuppedVector incastList_;
    DevAscendFunctionDuppedVector outcastList_;
    DevAscendFunctionDuppedVector expressionList_;
    uintdevptr_t runtimeWorkspace_;
    RuntimeReuseInfo runtimeWsReuseInfo_;
    uintdevptr_t runtimeOutcastWorkspace_;
    uint8_t data_[0];
    /*
     *  Duplicated:
     *      predcount_t                                         predCountListData[];
     *  Allocated (& zero-ed):
     *      AddressDescriptor                                   incastAddressListData[];
     *      AddressDescriptor                                   outcastAddressListData[];
     *      uint64_t                                            expressionListData[];
     *      DevAscendFunctionDuppedStitch                       *stitchListData[];
     */
#define GET_DATA(type, data, base, index) ((reinterpret_cast<type *>(const_cast<uint8_t *>((data) + (base)))[index]))
    uint16_t GetOperationSize() const { return operationList_.size; }
    predcount_t &GetOperationCurrPredCount(int index) {
        return GET_DATA(predcount_t, data_, operationList_.predCountBase, index);
    }

    uint64_t GetExpressionSize() const { return expressionList_.size; }
    uint64_t GetExpression(int index) const { return GET_DATA(uint64_t, data_, expressionList_.base, index); }
    uint64_t &GetExpression(int index) { return GET_DATA(uint64_t, data_, expressionList_.base, index); }

    uint64_t *GetExpressionAddr() {
        return &GetExpression(0);
    }

    uint64_t GetIncastSize() const { return incastList_.size; }
    AddressDescriptor GetIncastAddress(int index) const { return GET_DATA(AddressDescriptor, data_, incastList_.base, index); }
    AddressDescriptor &GetIncastAddress(int index) { return GET_DATA(AddressDescriptor, data_, incastList_.base, index); }

    uint64_t GetOutcastSize() const { return outcastList_.size; }
    AddressDescriptor GetOutcastAddress(int index) const { return GET_DATA(AddressDescriptor, data_, outcastList_.base, index); }
    AddressDescriptor &GetOutcastAddress(int index) { return GET_DATA(AddressDescriptor, data_, outcastList_.base, index); }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return runtimeWsReuseInfo_; }
    RuntimeReuseInfo &GetRuntimeReuseInfo() { return runtimeWsReuseInfo_; }

    uintdevptr_t GetRuntimeWorkspace() const { return runtimeWorkspace_; }
    uintdevptr_t &GetRuntimeWorkspace() { return runtimeWorkspace_; }

    uintdevptr_t GetRuntimeOutcastWorkspace() const { return runtimeOutcastWorkspace_; }
    uintdevptr_t &GetRuntimeOutcastWorkspace() { return runtimeOutcastWorkspace_; }

    DevAscendFunction *GetSource() const { return source_; }
    DevAscendFunction *&GetSource() { return source_; }

    DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex) {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }
};
const uint32_t RAW_TENSOR_OFFSET_SIZE = 63;
const uint32_t RAW_TENSOR_DESC_PRE_SIZE = 8;

struct DevAscendFunctionDupped {
    DevAscendFunctionDupped() = default;
    explicit DevAscendFunctionDupped(WsAllocation tinyAlloc) : dupTiny_(tinyAlloc) {}

    static DevAscendFunctionDupped DuplicateRoot(DevAscendFunction *func, WsAicpuCoherentAllocator &allocator) {
        WsAllocation tinyAlloc = allocator.Malloc(func->GetDuppedDataAllocSize(), WsMemCategory::DUP_FUNC);
        DevAscendFunctionDuppedData *dupData = tinyAlloc.As<DevAscendFunctionDuppedData>();
        DevAscendFunctionDuppedData *sourceData = func->GetDuppedData();
        memcpy_s(reinterpret_cast<uint8_t *>(dupData),
            func->GetDuppedDataCopySize(),
            sourceData,
            func->GetDuppedDataCopySize());
        memset_s(reinterpret_cast<uint8_t *>(dupData) + func->GetDuppedDataCopySize(),
            func->GetDuppedDataAllocSize() - func->GetDuppedDataCopySize(),
            0,
            func->GetDuppedDataAllocSize() - func->GetDuppedDataCopySize());
        dupData->GetSource() = func;

        DevAscendFunctionDupped dup(tinyAlloc);
        return dup;
    }

    void ReleaseDuppedMemory(WsAicpuCoherentAllocator &allocator) {
        (void)allocator;
    }

    void LogAicpuAlloc(WsAicpuCoherentAllocator &allocator) {
#ifdef DEBUG_SWITCH
        allocator.GetCounter()->LogMalloc(dupTiny_);
#endif // DEBUG_SWITCH
        (void)allocator;
    }

    RuntimeReuseInfo GetRuntimeReuseInfo() const { return DupData()->GetRuntimeReuseInfo(); }
    RuntimeReuseInfo &GetRuntimeReuseInfo() { return DupData()->GetRuntimeReuseInfo(); }

    uintdevptr_t RuntimeWorkspace() const { return DupData()->GetRuntimeWorkspace(); }
    uintdevptr_t &RuntimeWorkspace() { return DupData()->GetRuntimeWorkspace(); }

    uintdevptr_t RuntimeOutcastBase() const { return DupData()->GetRuntimeOutcastWorkspace(); }
    uintdevptr_t &RuntimeOutcastBase() { return DupData()->GetRuntimeOutcastWorkspace(); }

    const DevAscendFunction *GetSource() const { return DupData()->GetSource(); }
    DevAscendFunction *GetSource() { return DupData()->GetSource(); }

    inline uint64_t GetExpression(int arg) const { return DupData()->GetExpression(arg); };
    inline uint64_t &GetExpression(int arg) { return DupData()->GetExpression(arg); };
    inline uint64_t *GetExpressionAddr() { return DupData()->GetExpressionAddr(); }

    inline predcount_t &GetOperationCurrPredCount(int arg) { return DupData()->GetOperationCurrPredCount(arg); };
    inline auto &GetOperationStitch(int arg) { return DupData()->GetOperationStitch(arg); };

    inline AddressDescriptor GetIncastAddress(int arg) const { return DupData()->GetIncastAddress(arg); };
    inline AddressDescriptor &GetIncastAddress(int arg) { return DupData()->GetIncastAddress(arg); };

    inline AddressDescriptor GetOutcastAddress(int arg) const { return DupData()->GetOutcastAddress(arg); };
    inline AddressDescriptor &GetOutcastAddress(int arg) { return DupData()->GetOutcastAddress(arg); };

    inline uintdevptr_t GetRawTensorAddr(int rawIndex) const {
        uintdevptr_t addr = 0ULL;
        const DevAscendRawTensor *rawTensor = GetSource()->GetRawTensor(rawIndex);
        if (rawTensor->ioProperty == DevIOProperty::ROOT_INCAST) {
            AddressDescriptor incast = GetIncastAddress(rawTensor->ioIndex);
            DEV_DEBUG_ASSERT(!incast.IsNullAddress());
            addr = incast.addr;
        } else if (rawTensor->ioProperty == DevIOProperty::ROOT_OUTCAST) {
            AddressDescriptor outcast = GetOutcastAddress(rawTensor->ioIndex);
            DEV_DEBUG_ASSERT(!outcast.IsNullAddress());
            addr = outcast.addr;
        } else {
            uintdevptr_t runtimeWorkspace = RuntimeWorkspace();
            DEV_DEBUG_ASSERT(runtimeWorkspace != 0);
            addr = runtimeWorkspace + rawTensor->addrOffset;
        }
        return addr;
    }

    // for stitch
    inline void GetInTensorOffset(int32v8 &offset, int operationIndex, int operandIndex) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationIOperandInfo(operationIndex, operandIndex);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) : offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) : offsetSymList[1].Value();
    }

    inline void GetOutTensorOffset(int32v8 &offset, int operationIndex, int operandIndex) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationOOperandInfo(operationIndex, operandIndex);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        offset[0] = offsetSymList[0].IsExpression() ? GetExpression(offsetSymList[0].Value()) : offsetSymList[0].Value();
        offset[1] = offsetSymList[1].IsExpression() ? GetExpression(offsetSymList[1].Value()) : offsetSymList[1].Value();
    }

    inline void GetTensorOffsetAndShape(uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t shape[DEV_SHAPE_DIM_MAX], int dims,
        int operationIndex, int operandIndex, bool isIOperand = true) const {
        auto func = GetSource();
        auto &operandInfo = func->GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);

        const SymInt *offsetSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        for (int i = 0; i < dims; i++) {
            offset[i] =
                offsetSymList[i].IsExpression() ? GetExpression(offsetSymList[i].Value()) : offsetSymList[i].Value();
        }

        const SymInt *shapeSymList = &func->GetOperationAttr(operationIndex, operandInfo.staticShapeAttrBeginIndex);
        for (int i = 0; i < dims; i++) {
            shape[i] =
                shapeSymList[i].IsExpression() ? GetExpression(shapeSymList[i].Value()) : shapeSymList[i].Value();
        }
    }

    std::string Dump(const std::vector<uintdevptr_t> &slotAddrList, int indent = 0) {
        std::string INDENT(indent, ' ');
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');
        std::ostringstream oss;

        oss << INDENT << "DevFunction " << GetSource()->funcKey << " #Dup():" << GetSource()->GetFuncidx() << " {\n";
        for (size_t i = 0; i < GetSource()->GetRawTensorSize(); i++) {
            oss << INDENTINNER << GetSource()->DumpRawTensor(i, GetRawTensorAddr(i)) << "\n";
        }
        for (size_t i = 0; i < GetSource()->GetIncastSize(); i++) {
            oss << INDENTINNER << GetSource()->DumpIncast(i, INDENTINNER, slotAddrList) << "\n";
        }
        for (size_t i = 0; i < GetSource()->GetOutcastSize(); i++) {
            oss << INDENTINNER << GetSource()->DumpOutcast(i, INDENTINNER, slotAddrList) << "\n";
        }
        oss << INDENT << "}";
        return oss.str();
    }

    inline int64_t GetValue(const SymInt *attrs, int idx) {
        return attrs[idx].IsExpression() ? funcData->exprTbl[attrs[idx].Value()] : attrs[idx].Value();
    }

    inline uint64_t GetRawTensorAddrEx(int idx) const {
        auto desc = funcData->rawTensorDesc[idx];
        if (desc.location == RAW_TENSOR_LOCATION_LOCAL)
            return funcData->workspaceAddr + desc.offsetOrIndex;
        else
            return funcData->rawTensorAddr[desc.offsetOrIndex] & ((1UL << RAW_TENSOR_OFFSET_SIZE) - 1);
    }

    std::string DumpDyn(int funcIdx, int operIdx, const DevCceBinary *cceBinary) {
        std::stringstream oss;
        auto func = GetSource();

        auto attrBase = reinterpret_cast<SymInt *>(&funcData->opAttrs[funcData->opAtrrOffsets[operIdx]]);
        auto funcIndex = attrBase[0].Value();
        oss << std::hex << " #funcKey " << func->funcKey << " #operIndex " << operIdx
            << " #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
            << " #coreType: " << cceBinary[funcIndex].coreType
            << " #taskID:" << MakeTaskID(funcIdx, operIdx) << "\n";

        auto dumpAttr = [this, &oss](auto attrs, auto &info) {
            int attrIndex = info.staticOffsetAttrBeginIndex;
            auto rawIndex = attrs[attrIndex - 1].Value();
            oss << rawIndex << "@" << GetRawTensorAddrEx(rawIndex) << ", ";
            int dim = info.GetDim();
            for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
                oss << GetValue(attrs, attrIndex + i) << ", ";
            }
        };

        int offset = 0;
        for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
            auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            dumpAttr(attrBase, opInfo);
        }
        for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
            auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
            offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            dumpAttr(attrBase, opInfo);
        }
        for (size_t idx = offset; idx < func->GetOperationAttrSize(operIdx); idx++) {
            oss << GetValue(attrBase, idx) << ", ";
        }
        return oss.str();
    }

    void DumpTopo(std::ofstream &os, int seqNo, int funcIdx, const DevCceBinary *cceBinary) {
        auto func = GetSource();
        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," << func->funcKey << "," << func->GetOperationAttrCalleeIndex(opIdx) << ","
               << func->GetOperationOpmagic(opIdx) << ",";
            auto &cceInfo = cceBinary[func->GetOperationAttrCalleeIndex(opIdx)];
            os << cceInfo.coreType << "," << cceInfo.psgId << "," << cceInfo.funcHash;
            auto &succList = func->GetOperationSuccList(opIdx);
            for (size_t j = 0; j < succList.size(); j++) {
                os << "," << MakeTaskID(funcIdx, func->At(succList, j));
            }
            auto &stitch = GetOperationStitch(opIdx);
            stitch.ForEach([&os](uint32_t id) {
                os << "," << id;
            });
            os << "\n";
        }
    }

    std::string DumpDyn(int funcIdx, const DevCceBinary *cceBinary) {
        std::stringstream oss;
        auto func = GetSource();

        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            oss << std::hex << "[" << opIdx << "] #predCnt:" << GetOperationCurrPredCount(opIdx);
            auto &succList = func->GetOperationSuccList(opIdx);
            oss << " #succList: [";
            for (size_t j = 0; j < succList.size(); j++) {
                if (j != 0)
                    oss << ", ";
                oss << func->At(succList, j);
            }
            oss << ']';
            auto &stitch = GetOperationStitch(opIdx);
            if (!stitch.IsNull())
                oss << std::hex << " #stitch:" << stitch;
            oss << "\n";
        }

        auto dumpAttr = [this, &oss, func](const SymInt *attrs, const auto &info) {
            int attrOffset = info.staticOffsetAttrBeginIndex;
            int attrIndex = attrOffset;
            auto rawIndex = attrs[attrIndex - 1].Value();
            oss << "@" << rawIndex << ", ";

            int dim = info.GetDim();
            auto rawTensor = func->GetRawTensor(rawIndex);
            DEV_ASSERT(rawIndex < func->GetRawTensorSize());
            DEV_ASSERT(dim == rawTensor->GetDim());

            for (int d = 0; d < rawTensor->GetDim(); d++) {
                auto shapeIdx = attrOffset + d + rawTensor->GetDim() * 2;
                auto shape = static_cast<int64_t>(rawTensor->shape.At(d, funcData->exprTbl));
                DEV_ASSERT(GetValue(attrs, shapeIdx) == shape);
            }
            DEV_ASSERT(dim == rawTensor->GetDim());
            for (int i = 0; i < dim * ARG_ATTR_TYPE; i++) {
                oss << GetValue(attrs, attrOffset + i) << ", ";
            }
        };

        oss << " #funcKey: " << func->funcKey << " #gmStackBase: " << funcData->stackWorkSpaceAddr
            << " #stackSize: " << funcData->stackWorkSpaceSize << " #workspace: " << funcData->workspaceAddr << "\n";
        oss << "#funcData: [\n" << std::dec;
        for (size_t operIdx = 0; operIdx < func->GetOperationSize(); operIdx++) {
            auto attrBase = &func->GetOperationAttr(operIdx, 0);
            auto funcIndex = attrBase[0].Value();
            oss << "  [" << operIdx << "]  #funcHash: " << std::to_string(cceBinary[funcIndex].funcHash)
                << " #funcIndex: " << funcIndex
                << " #funcAddr: " << reinterpret_cast<uint64_t>(cceBinary[funcIndex].binary.Data())
                << " #taskID:" << MakeTaskID(funcIdx, operIdx) << " #opMagic: " << func->GetOperationOpmagic(operIdx)
                << "\n";
            oss << "  #invokeAttrs : ";

            int offset = 0;
            for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
                auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
                dumpAttr(attrBase, opInfo);
            }
            for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
                auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
                dumpAttr(attrBase, opInfo);
            }
            for (size_t idx = offset; idx < func->GetOperationAttrSize(operIdx); idx++) {
                oss << GetValue(attrBase, idx) << ", ";
            }
            oss << "\n";
        }
        oss << std::hex << "  #rawTensorAddrs: ";
        for (uint64_t i = 0; i < func->GetRawTensorDescSize(); i++) {
            if (i % RAW_TENSOR_DESC_PRE_SIZE == 0)
                oss << "\n   ";
            DEV_ASSERT(GetRawTensorAddrEx(i) == GetRawTensorAddr(i));
            oss << GetRawTensorAddrEx(i) << ", ";
        }
        oss << "\n]";
        return oss.str();
    }

    bool IsNull() const { return !dupTiny_; }
    void ResetNull() { dupTiny_.Invalidate(); }
    DynFuncData *GetFuncData() { return funcData; }
    void SetFuncData(DynFuncData *data) { funcData = data; }

private:
    const DevAscendFunctionDuppedData *DupData() const { return dupTiny_.As<DevAscendFunctionDuppedData>(); }
    DevAscendFunctionDuppedData *DupData() { return dupTiny_.As<DevAscendFunctionDuppedData>(); }

private:
    DynFuncData *funcData{nullptr}; // used by aicore
    WsAllocation dupTiny_;
};

struct EncodeDevAscendFunctionParam {
    std::unordered_map<uint64_t, int> calleeHashIndexDict;
    std::vector<CceCodeInfo> cceCodeInfoList;
    const SymbolicSymbolTable *symbolTable;
    const SymbolicExpressionTable *expressionTable;
    const IncastOutcastLink *inoutLink;
    const IncastOutcastSlot *slot;
    Function *devRoot;
};

void EncodeDevAscendFunction(const EncodeDevAscendFunctionParam &param, uint64_t &offset, DevAscendFunction *base);

struct DevAscendProgramSymbol {
    DevRelocVector<char> name;
    uint64_t index;
};

#ifndef PAGE_SIZE
#define PAGE_SIZE       4096
#endif
struct DevAscendProgram {
    DeviceArgs devArgs;
    uint64_t hashKey;
    uint64_t slotSize;
    uint64_t aicoreLocalWorkspaceSize;
    uint64_t rootFuncStandardMemReq;
    uint64_t aicpuCoherentWorkspaceSize;
    uint64_t slotStandardMemReq; // max memory requirement of a single slot
    uint64_t slotPoolSize;
    uint64_t standardStackWorkspacePerCore;
    uint64_t stitchPoolSize;
    uint64_t globalTensorMem;
    uint32_t workspaceRecyclePeriod;
    DevRelocVector<DevAscendProgramSymbol> symbolTable;
    DevRelocVector<char> symbolTableNameList;
    uint64_t expressionTableSize;
    DevRelocVector<uint64_t> expressionTableOffsetList;
    DevRelocVector<uint8_t> preGuardPage;
    DevRelocVector<uint8_t> expressionTableBinary;
    DevRelocVector<uint8_t> hostControlFlowBinary;  // compiled by system gcc (host arch)
    DevRelocVector<uint8_t> devControlFlowBinary;   // compiled by CANN gcc (ARM arch)
    DevRelocVector<uint8_t> postGuardPage;
    DevRelocVector<DevRelocVector<uint8_t>> devEncodeList;
    DevRelocVector<uint8_t> devEncodeDataList;
    DevRelocVector<DevCceBinary> cceCodeList;
    DevRelocVector<uint8_t> cceCodeDataList;
    DevRelocVector<int> startArgsInputTensorSlotIndexList;
    DevRelocVector<int> startArgsOutputTensorSlotIndexList;
    DevRelocVector<int> startArgsInputSymbolIndexList;
    DevRelocVector<SymbolHandler> startArgsSymbolHandlerList;
    DevRelocVector<int> assembleSlotIndexList;
    DevRelocVector<int> inplaceSlotList;
    DevRelocVector<PrefetchInfo> prefetchInfoList;
#define programLastField                              prefetchInfoList
    uint8_t data[0];

    /*
     *      DevAscendProgramSymbol symbolTableData[]
     *      char symbolTableNameListData[]
     *      uint64_t expressionTableOffsetListData[]
     *      uint8_t preGuardPageData[PAGE_SIZE]
     *      uint8_t expressionTableBinaryData[]
     *      uint8_t hostControlFlowBinaryData[]
     *      uint8_t devControlFlowBinaryData[]
     *      DevRelocVector<uint8_t> devEncodeList[]
     *      uint8_t devEncodeDataList[]
     *      DevRelocVector<uint8_t> cceCodeList[]
     *      uint8_t cceCodeDataList[]
     *      int startArgsInputTensorSlotIndexListData[]
     *      int startArgsOutputTensorSlotIndexListData[]
     *      int startArgsInputSymbolIndexListData[]
     *      SymbolHandler startArgsSymbolHandlerListData[]
     *      int assembleSlotIndexList[]
     */

    template <typename T>
    const T &At(const DevRelocVector<T> &localvec, int index) const {
        return localvec[index];
    }
    template <typename T>
    T &At(DevRelocVector<T> &localvec, int index) {
        return localvec[index];
    }

    std::string Dump(int indent = 0, bool dumpAddr = false) const {
        const int WIDTH = 16;
        const int ADDRESS_MIN_WIDTH = 6;
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');
        std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
        std::ostringstream oss;
        oss << "DevProgram {\n";
        oss << INDENTINNER << "#aicoreLocalWorkspaceSize:" << aicoreLocalWorkspaceSize << "\n";
        oss << INDENTINNER << "#aicpuCoherentWorkspaceSize:" << aicpuCoherentWorkspaceSize << "\n";
        oss << INDENTINNER << "#slot:" << slotSize << "\n";
        oss << INDENTINNER << "#symbolCount:" << symbolTable.size() << "\n";
        for (size_t i = 0; i < symbolTable.size(); i++) {
            const DevAscendProgramSymbol &symbol = At(symbolTable, i);
            oss << INDENTINNER << "#symbol:" << symbol.index << " = " << &At(symbol.name, 0) << "\n";
        }
        for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
            oss << INDENTINNER << "#input:" << i << " -> #slot:" << At(startArgsInputTensorSlotIndexList, i) << "\n";
        }
        for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
            oss << INDENTINNER << "#output:" << i << " <- #slot:" << At(startArgsOutputTensorSlotIndexList, i)
                << "\n";
        }
        for (auto slotIndex : assembleSlotIndexList) {
            oss << INDENTINNER << "#slot-assemble: " << slotIndex << "\n";
        }
        for (size_t i = 0; i < inplaceSlotList.size(); i++) {
            oss << INDENTINNER << "#inplace:" << i << " <- #slot:" << At(inplaceSlotList, i) << "\n";
        }
        for (size_t i = 0; i < startArgsInputSymbolIndexList.size(); i++) {
            oss << INDENTINNER << "#symbol:" << i << " -> #symbolTable:" << At(startArgsInputSymbolIndexList, i) << "\n";
        }
        oss << INDENTINNER << "#ExprCount:" << expressionTableSize << "\n";

        oss << INDENTINNER << "#ExprCodeSize:" << expressionTableBinary.size();
        if (dumpAddr) {
            if (expressionTableBinary.size() != 0) {
                oss << " #ExprCodeAddr:" << DevAscendFunction::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(expressionTableBinary, 0)));
            }
        }
        oss << "\n";

        for (size_t i = 0; i < expressionTableBinary.size(); i += WIDTH) {
            oss << INDENTINNERINNER << DevAscendFunction::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
            for (size_t off = i; off < std::min(i + WIDTH, expressionTableBinary.size()); off++) {
                oss << " " << DevAscendFunction::DumpByte(At(expressionTableBinary, off));
            }
            oss << "\n";
        }

        oss << "====\n"; // Dump control flow code (begin)

        oss << INDENTINNER << "#HostControlCodeSize:" << hostControlFlowBinary.size();
        if (dumpAddr) {
            oss << " #HostControlCodeAddr:" <<
                DevAscendFunction::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(hostControlFlowBinary, 0)));
        }
        oss << "\n";

        for (size_t i = 0; i < hostControlFlowBinary.size(); i += WIDTH) {
            oss << INDENTINNERINNER << DevAscendFunction::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
            for (size_t off = i; off < std::min(i + WIDTH, hostControlFlowBinary.size()); off++) {
                oss << " " << DevAscendFunction::DumpByte(At(hostControlFlowBinary, off));
            }
            oss << "\n";
        }

        oss << "====\n"; // Dump control flow code: ^^^ Host / Dev vvv

        oss << INDENTINNER << "#DevControlCodeSize:" << devControlFlowBinary.size();
        if (dumpAddr) {
            oss << " #DevControlCodeAddr:" <<
                DevAscendFunction::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(devControlFlowBinary, 0)));
        }
        oss << "\n";

        for (size_t i = 0; i < devControlFlowBinary.size(); i += WIDTH) {
            oss << INDENTINNERINNER << DevAscendFunction::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
            for (size_t off = i; off < std::min(i + WIDTH, devControlFlowBinary.size()); off++) {
                oss << " " << DevAscendFunction::DumpByte(At(devControlFlowBinary, off));
            }
            oss << "\n";
        }

        oss << "====\n"; // Dump control flow code (ends)

        oss << INDENTINNER << "#func:" << devEncodeList.size() << "\n";
        for (size_t i = 0; i < devEncodeList.size(); i++) {
            const DevAscendFunction *func = reinterpret_cast<const DevAscendFunction *>(&At(At(devEncodeList, i), 0));
            oss << func->Dump(IDENT_SIZE) << "\n";
        }
        oss << INDENTINNER << "#cce:" << cceCodeList.size() << "\n";
        for (size_t i = 1; i < cceCodeList.size(); i++) {
            const DevCceBinary &cceCode = At(cceCodeList, i);
            oss << INDENTINNER << "#cce-" << i << " #CoreType:" << cceCode.coreType
                << " #FuncHash:" << cceCode.funcHash;
            if (dumpAddr) {
                std::string address = DevAscendFunction::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(cceCode.binary, 0)));
                oss << " #CoreAddr:" << address;
            }
            oss << "\n";

            for (size_t j = 0; j < cceCode.binary.size(); j += WIDTH) {
                oss << INDENTINNERINNER << DevAscendFunction::DumpAddress(j, ADDRESS_MIN_WIDTH) << ":";
                for (size_t off = j; off < std::min(j + WIDTH, cceCode.binary.size()); off++) {
                    oss << " " << DevAscendFunction::DumpByte(At(cceCode.binary, off));
                }
                oss << "\n";
            }
        }
        oss << "}";
        return oss.str();
    }

    std::vector<int> GetInputTensorSlotIndexList() const {
        std::vector<int> indexList;
        for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
            indexList.push_back(At(startArgsInputTensorSlotIndexList, i));
        }
        return indexList;
    }
    std::vector<int> GetOutputTensorSlotIndexList() const {
        std::vector<int> indexList;
        for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
            indexList.push_back(At(startArgsOutputTensorSlotIndexList, i));
        }
        return indexList;
    }

    std::vector<int> GetAssembleTensorSlotIndexList() const {
        std::vector<int> indexList;
        for (size_t i = 0; i < assembleSlotIndexList.size(); i++) {
            indexList.push_back(At(assembleSlotIndexList, i));
        }
        return indexList;
    }

    std::tuple<const void *, uint64_t> GetDevControlFlowBinary() const {
        return std::make_tuple(
            reinterpret_cast<const void *>(devControlFlowBinary.Data()),
            (uint64_t)devControlFlowBinary.size());
    }

    std::tuple<const void *, uint64_t> GetHostControlFlowBinary() const {
        return std::make_tuple(
            reinterpret_cast<const void *>(hostControlFlowBinary.Data()),
            (uint64_t)hostControlFlowBinary.size());
    }

    std::tuple<const void *, uint64_t, const uint64_t *, uint64_t> GetExpressionTableBinary() const {
        return std::make_tuple(
            reinterpret_cast<const void *>(expressionTableBinary.Data()),
            static_cast<uint64_t>(expressionTableBinary.size()),
            expressionTableOffsetList.Data(),
            static_cast<uint64_t>(expressionTableOffsetList.size()));
    }

    uint64_t GetSymbolTableSize() const { return symbolTable.size(); }

    uint64_t GetExpressionTableSize() const { return expressionTableSize; }

    uint64_t GetFunctionSize() const { return devEncodeList.size(); }

    DevAscendFunction *GetFunction(int index) const {
        return reinterpret_cast<DevAscendFunction *>(const_cast<uint8_t *>(devEncodeList[index].Data()));
    }

    DevAscendFunction *GetFunctionByRawName(const std::string &rawName) const {
        for (size_t i = 0; i < GetFunctionSize(); i++) {
            DevAscendFunction *func = GetFunction(static_cast<int>(i));
            if (func->GetRawName() == rawName) {
                return func;
            }
        }
        return nullptr;
    }

    const DevCceBinary *GetCceBinary(int index) const { return &cceCodeList[index]; }

    void Reloc(intptr_t shift, bool relocFunc = false) {
        auto symbolTablePtr = AddOffset<DevAscendProgramSymbol>(data, 0);
        symbolTable.DeviceRelocData(shift);
        for (size_t i = 0; i < symbolTable.size(); i++) {
            symbolTablePtr[i].name.DeviceRelocData(shift);
        }

        auto symbolTableNameListPtr = AddOffset<uint8_t>(symbolTablePtr, symbolTable.size());
        symbolTableNameList.DeviceRelocData(shift);

        auto expressionTableOffsetListPtr = AddOffset<uint64_t>(symbolTableNameListPtr, symbolTableNameList.size());
        expressionTableOffsetList.DeviceRelocData(shift);

        auto preGuardPagePtr = AddOffset<uint8_t>(expressionTableOffsetListPtr, expressionTableOffsetList.size());
        preGuardPage.DeviceRelocData(shift);

        auto expressionTableBinaryPtr = AddOffset<uint8_t>(preGuardPagePtr, preGuardPage.size());
        expressionTableBinary.DeviceRelocData(shift);

        auto hostControlFlowBinaryPtr = AddOffset<uint8_t>(expressionTableBinaryPtr, expressionTableBinary.size());
        hostControlFlowBinary.DeviceRelocData(shift);

        auto devControlFlowBinaryPtr = AddOffset<uint8_t>(hostControlFlowBinaryPtr, hostControlFlowBinary.size());
        devControlFlowBinary.DeviceRelocData(shift);

        auto devEncodeListPtr = AddOffset<DevRelocVector<uint8_t>>(devControlFlowBinaryPtr, devControlFlowBinary.size());
        devEncodeList.DeviceRelocData(shift);
        for (size_t i = 0; i < devEncodeList.size(); i++) {
            devEncodeListPtr[i].DeviceRelocData(shift);
        }

        auto devEncodeDataListPtr = AddOffset<uint8_t>(devEncodeListPtr, devEncodeList.size());
        devEncodeDataList.DeviceRelocData(shift);

        auto cceCodeListPtr = AddOffset<DevCceBinary>(devEncodeDataListPtr, devEncodeDataList.size());
        shift = ALIGN_UP(shift, alignof(DevCceBinary));
        cceCodeList.DeviceRelocData(shift);
        for (size_t i = 0; i < cceCodeList.size(); i++) {
            cceCodeListPtr[i].binary.DeviceRelocData(shift);
        }
        cceCodeDataList.DeviceRelocData(shift);

        startArgsInputTensorSlotIndexList.DeviceRelocData(shift);
        startArgsOutputTensorSlotIndexList.DeviceRelocData(shift);
        startArgsSymbolHandlerList.DeviceRelocData(shift);
        startArgsInputSymbolIndexList.DeviceRelocData(shift);
        assembleSlotIndexList.DeviceRelocData(shift);
        inplaceSlotList.DeviceRelocData(shift);
        prefetchInfoList.DeviceRelocData(shift);
        if (relocFunc) {
            for (int i = 0; i < static_cast<int>(GetFunctionSize()); i++) {
                DevAscendFunction *func = GetFunction(i);
                func->Reloc(reinterpret_cast<uint64_t>(func), true);
                func->Verify(reinterpret_cast<uint64_t>(func));
            }
        }
    }

    uint64_t GetSize() const { return reinterpret_cast<uintptr_t>(programLastField.End()) - reinterpret_cast<uintptr_t>(this); }

private:
    friend struct EncodeDevAscendProgramInfo;

    void InitSymbolTable(
            uintdevptr_t &initOffset, SymbolicSymbolTable *symbolTableInput, bool fillContent);
    void InitExpressionTableBinary(
            uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &expressionTableBinaryListInput, bool fillContent);
    void InitControlFlowBinary(
            uintdevptr_t &initOffset,
            const std::vector<uint8_t> &hostControlFlowBinaryInput,
            const std::vector<uint8_t> &devControlFlowBinaryInput,
            bool fillContent);
    void InitDevEncodeList(
            uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &devEncodeListInput, bool fillContent);
    void InitCceCodeList(
            uintdevptr_t &initOffset, const std::vector<std::vector<uint8_t>> &cceCodeListInput,
            const std::vector<CceCodeInfo> &cceInfo, bool fillContent);
    void InitPrefetchInfoList(
            uintdevptr_t &initOffset, const std::vector<L2Info> &l2InfoList, bool fillContent);
    void InitStartArgsABIParamList(uintdevptr_t &initOffset, const std::vector<int> &tStartArgsInputTensorSlotIndexList,
        const std::vector<int> &tStartArgsOutputTensorSlotIndexList,
        const std::vector<int> &tStartArgsInputSymbolIndexList,
        const std::vector<SymbolHandler> &tStartArgsSymbolHandlerList,
        const std::vector<int> &tAsembleSlotIndexList,
        const std::vector<int> &tInplaceSlotIndexList, bool fillContent);
};

void EncodeDevAscendProgram(Function *func, uint64_t &offset, DevAscendProgram *base);

struct InputsHeader {
    uint32_t dim;
    uint32_t cnt;
    int64_t dimVal[0];

    uint32_t size() { return sizeof(InputsHeader) + dim * sizeof(uint64_t); }
    InputsHeader *next() { return reinterpret_cast<InputsHeader *>(reinterpret_cast<uint64_t>(this) + size()); }
};

struct DevAscendTensorDataCreator {
    template<typename T>
    static DevAscendTensorData Create(uintdevptr_t tensorAddress, const std::vector<T> &tensorShape) {
        DevAscendTensorData tensorData;
        Init(&tensorData, tensorAddress, tensorShape.data(), tensorShape.size());
        return tensorData;
    }

    template<typename T>
    static void Init(DevAscendTensorData *tensorData, uintdevptr_t tensorAddress, const T *dims, int n) {
        DEV_ASSERT(n <= DEV_SHAPE_DIM_MAX);

        tensorData->address = tensorAddress;
        tensorData->shape.dimSize = n;
        for (int i = 0; i < n; i++) {
            tensorData->shape.dim[i] = dims[i];
        }
    }

    static int Decode(int64_t *inputs, DevAscendTensorData *tensorData) {
        int64_t addrOffset = *inputs;
        int64_t *ptrBase = reinterpret_cast<int64_t *>(reinterpret_cast<uint64_t>(inputs) + addrOffset);

        int n = 0;
        InputsHeader *h = reinterpret_cast<InputsHeader *>(inputs + 1);
        int64_t *ptr = ptrBase;
        while (reinterpret_cast<int64_t *>(h) < ptrBase) {
            Init(&tensorData[n], *ptr, h->dimVal, h->dim);
            n++;
            ptr++;
            h = h->next();
        }
        return n;
    }

    /*
     *                  |    8 bytes  |
     *  start -->       |  ptr_offset |
     *  input0 -->      |  dim | cnt  |
     *                  | dim * int64 |
     *  input1 -->      |  dim | cnt  |
     *                  | dim * int64 |
     *                  |     ...     |
     *   ptrstart -->   |    ptr1     |
     *                  |    ptr2     |
     *                  |     ...     |
     */
    static std::vector<int64_t> Encode(const std::vector<DevAscendTensorData> &tensors) {
        size_t size = tensors.size() * 0x2 + 1;
        for (auto &t : tensors) {
            size += t.shape.dimSize;
        }

        std::vector<int64_t> data(size);
        int64_t *ptr = data.data() + (data.size() - tensors.size());
        data[0] = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(data.data()); // ptroffset
        auto h = reinterpret_cast<InputsHeader *>(&data[1]);
        for (auto &t : tensors) {
            h->dim = t.shape.dimSize;
            h->cnt = 1;
            for (int i = 0; i < static_cast<int>(h->dim); i++) {
                h->dimVal[i] = t.shape.dim[i];
            }
            *ptr++ = t.address;
            h = h->next();
        }
        DEV_ASSERT(ptr == data.data() + data.size());

        return data;
    }
};

struct DevInputSymbol {
    uint64_t value;
};
const uint32_t DUMP_INDEX_SIZE_2 = 2;
const uint32_t DUMP_INDEX_SIZE_4 = 4;
struct DevStartArgs : DevStartArgsBase {
    uint64_t workspaceAddr;
    DevAscendProgram *devProg;

    uint64_t aicoreLocalWorkspaceSize;
    uint64_t aicpuCoherentWorkspaceSize;
    DevInputSymbol *inputSymbolList;
    uint64_t inputSymbolSize;
    const void *controlFlowEntry;

public:
    void InitWorkspace(DevAscendProgram *tDevProg, void *workspace) {
        workspaceAddr = reinterpret_cast<uint64_t>(workspace);
        devProg = tDevProg;
        aicoreLocalWorkspaceSize = tDevProg->aicoreLocalWorkspaceSize;
        aicpuCoherentWorkspaceSize = tDevProg->aicpuCoherentWorkspaceSize;
        inputSymbolList = nullptr;
        inputSymbolSize = 0;
    }

public:
    template<typename T>
    const T &At(const DevLocalVector<T> &localvec, int index) const {
        return *reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(this) + localvec.Offset(index));
    }
    template<typename T>
    T &At(const DevLocalVector<T> &localvec, int index) {
        return *reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(this) + localvec.Offset(index));
    }

    int GetInputTensorSize() const { return inputTensorSize; }
    const DevAscendTensorData &GetInputTensor(int index) const { return inputTensorList[index]; }
    DevAscendTensorData &GetInputTensor(int index) { return inputTensorList[index]; }

    int GetOutputTensorSize() const { return outputTensorSize; }
    const DevAscendTensorData &GetOutputTensor(int index) const { return outputTensorList[index]; }
    DevAscendTensorData &GetOutputTensor(int index) { return outputTensorList[index]; }

    int GetInputSymbolSize() const { return inputSymbolSize; }
    const DevInputSymbol &GetInputSymbol(int index) const { return inputSymbolList[index]; }
    DevInputSymbol &GetInputSymbol(int index) { return inputSymbolList[index]; }

    std::string Dump(int indent = 0) const {
        std::string INDENTINNER(indent + DUMP_INDEX_SIZE_2, ' ');
        std::string INDENTINNERINNER(indent + DUMP_INDEX_SIZE_4, ' ');
        std::ostringstream oss;
        oss << "DevStartArgs {" << "\n";
        for (int i = 0; i < GetInputTensorSize(); i++) {
            const DevAscendTensorData &input = GetInputTensor(i);
            oss << INDENTINNER << "#input-" << i << ": #address:" << DevAscendFunction::DumpAddress(input.address);
            oss << " #shape:[";
            for (int j = 0; j < input.shape.dimSize; j++) {
                if (j != 0) {
                    oss << ",";
                }
                oss << input.shape.dim[j];
            }
            oss << "]\n";
        }
        for (int i = 0; i < GetOutputTensorSize(); i++) {
            const DevAscendTensorData &output = GetOutputTensor(i);
            oss << INDENTINNER << "#output-" << i << ": #address:" << DevAscendFunction::DumpAddress(output.address);
            oss << " #shape:[";
            for (int j = 0; j < output.shape.dimSize; j++) {
                if (j != 0) {
                    oss << ",";
                }
                oss << output.shape.dim[j];
            }
            oss << "]\n";
        }
        oss << INDENTINNER << "#workspaceAddr:" << DevAscendFunction::DumpAddress(workspaceAddr) << "\n";
        oss << INDENTINNER << "#aicoreLocalWorkspaceSize:" << aicoreLocalWorkspaceSize << "\n";
        oss << INDENTINNER << "#aicpuCoherentWorkspaceSize:" << aicpuCoherentWorkspaceSize << "\n";
        oss << INDENTINNER << "#devProg:" << DevAscendFunction::DumpAddress(reinterpret_cast<uintdevptr_t>(devProg)) << "\n";
        oss << "}";
        return oss.str();
    }
    static std::unordered_map<std::string, SymbolHandlerId> symbolIndexDict;
};

constexpr uint32_t MAX_SYM_NUM = 16;

constexpr uint32_t MAX_DEV_FUNCTION_NUM = 16;

using CfgFuncCallback = void(*)(void *ctx, uint64_t rootKey);
using ExprFunc = uint64_t(*)(uint64_t *symbolTable, uint64_t exprIndex);
using CfgFunc = uint64_t(*)(uint64_t *symbolTable, CfgFuncCallback callback, void *ctx);

struct DevFunctionMain {
    uint64_t symNum;
    uint64_t symTbl[MAX_SYM_NUM];
    uint64_t workspaceSize;
    uint64_t workspaceAddr;
    uint64_t exprSize;
    ExprFunc exprFunc;
    uint64_t funcNum;
    DevAscendFunction *devFuncList[MAX_DEV_FUNCTION_NUM];
    uint64_t cfgSize;
    CfgFunc cfgFunc;
};

struct DevIncastOutcastLink {
    DevAscendFunction *prev;
    uint64_t prevOutIndex;
    DevAscendFunction *succ;
    uint64_t succInIndex;
};

// Runtime
struct DevSymbolInitializer {
    uint64_t symbolIndex;
    int64_t symbolValue;
};
} // namespace dynamic
} // namespace npu::tile_fwk
