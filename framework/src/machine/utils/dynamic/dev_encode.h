/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "machine/utils/machine_ws_intf.h"
#include "tilefwk/core_func_data.h"
#include "interface/schema/schema.h"
#include "securec.h"
#include "interface/utils/common.h"
#include "tilefwk/data_type.h"
#include "tilefwk/aicpu_runtime.h"
#include "interface/tensor/symbol_handler.h"
#include "machine/kernel/aicore.h"
#include "machine/utils/dynamic/allocator/allocators.h"
#include "machine/utils/dynamic/vector.h"
#include "machine/utils/dynamic/item_pool.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk {
class Function;
class DyndevFunctionAttribute;
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
constexpr uint32_t FRIENDLY_CACHE_ALIGN_U64_SIZE = 2; // 友好的cache对齐是2个u64

namespace dynamic {
struct EncodeRawTensorAttr;

using  int32v8 = int32_t __attribute__((vector_size(32)));
using  int32v4 = int32_t __attribute__((vector_size(16)));
using  uint32v8 = uint32_t __attribute__((vector_size(32)));
using  uint32v4 = uint32_t __attribute__((vector_size(16)));
using  uint16v4 = uint16_t __attribute__((vector_size(8)));
using  uint16v8 = uint16_t __attribute__((vector_size(16)));

#define ALIGN_UP(val, align)            (((val) + (align) - 1) & ~((align) - 1))

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
inline void DeviceRelocMaybeNull(T *&ptr, intdevptr_t shift) {
    if (ptr != nullptr) {
        ptr = reinterpret_cast<T *>(reinterpret_cast<uintdevptr_t>(ptr) + shift);
    }
}

struct RelocRange {
    RelocRange(uintdevptr_t src, uintdevptr_t dst) : src_(src), dst_(dst) {}

    template<typename T>
    inline void RelocNullable(T *&ptr) const {
        if (ptr != nullptr) {
            Reloc(ptr);
        }
    }
    template<typename T>
    inline void Reloc(T *&ptr) const {
        DeviceReloc(ptr, dst_ - src_);
    }
    inline void Reloc(uint64_t &addr) const {
        addr += dst_ - src_;
    }
    inline void RelocNullable(uint64_t &addr) const {
        if (addr != 0) {
            Reloc(addr);
        }
    }

    uintdevptr_t GetDst() const { return dst_; }
private:
    uintdevptr_t src_;
    uintdevptr_t dst_;
};

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

    size_t DataSize() const { return size_ * sizeof(T); }

    void HostAssignDataSize(uintdevptr_t offset, size_t size) {
        HostAssign(data_, offset);
        size_ = size;
    }
    void HostAssignRangeOffsetSize(const DevRelocVector<T> &base, uintdevptr_t offset, size_t size) {
        HostAssignDataSize(reinterpret_cast<uintdevptr_t>((base.Data() + offset)), size);
    }
    void HostInitDataSizeOffset(uintdevptr_t &offset, size_t size) {
        ASSERT(offset % alignof(T) == 0); // Ensure offset is aligned
        HostAssign(data_, offset);
        size_ = size;
        offset = reinterpret_cast<uintdevptr_t>(data_ + size);
    }
    void DeviceRelocData(intdevptr_t shift) { DeviceReloc(data_, ALIGN_UP(shift, alignof(T))); }
    void DeviceRelocDataMaybeNull(intdevptr_t shift) { DeviceRelocMaybeNull(data_, ALIGN_UP(shift, alignof(T))); }
    uintdevptr_t End() const { return reinterpret_cast<uintdevptr_t>(data_ + size_); }

    static uint64_t ElementSize() { return sizeof(T); }
    using ElementType = T;

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

struct DevAscendStride {
    int64_t dimSize;
    uint64_t dimStride[DEV_SHAPE_DIM_MAX];

    const uint64_t &operator[](int index) const { return dimStride[index]; }
    uint64_t &operator[](int index) { return dimStride[index]; }

    int GetShape(int dim) const {
        if (dim == dimSize - 1) {
            return dimStride[dim];
        } else {
            return dimStride[dim] / dimStride[dim + 1];
        }
    }

    void SetShape(const int *shape, int dim) {
        /* For shape [d0, d1, d2], the stride is [d0 * d1 * d2, d1 * d2, d2] */
        dimSize = dim;
        for (int i = DEV_SHAPE_DIM_MAX - 1; i >= 0; i--) {
            if (i > dimSize - 1) {
                dimStride[i] = 0;
            } else if (i == dimSize - 1) {
                dimStride[i] = shape[i];
            } else {
                dimStride[i] = shape[i] * dimStride[i + 1];
            }
        }
    }
    void SetShape(const std::vector<int> &shape) {
        SetShape(shape.data(), (int)shape.size());
    }
    void SetShape(const DevShape &shape) {
        SetShape(shape.dim, shape.dimSize);
    }
};

struct DevCellMatchTableDesc {
    DevShape cellShape;
    DevAscendStride stride;

    int GetDimensionSize() const { return cellShape.dimSize; }

    const int &GetCellShape(int index) const { return cellShape.dim[index]; }

    const uint64_t &GetStride(int index) const { return stride.dimStride[index]; }
    int GetStrideShape(int index) const { return stride.GetShape(index); }

    void SetCellShape(const std::vector<int> &shape) {
        cellShape.dimSize = shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            cellShape.dim[i] = shape[i];
        }
    }
    void SetStrideShape(const std::vector<int> &shape) {
        stride.SetShape(shape);
    }
};

struct DevCceBinary {
    uint32_t coreType;
    uint32_t psgId;
    uint64_t funcHash;
#ifdef SUPPORT_WRAP
    int32_t wrapVecId {-1};
    uint32_t mixResourceType {0};
#endif
};

struct DevAicpuLeafBinary {
    DevRelocVector<int32_t> aicpuLeafCode;
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

static inline std::string Delim(bool cond, const std::string &delim) {
    return cond ? delim : "";
}

struct DevSymShape {
    int dimSize{0};
    SymInt dim[DEV_SHAPE_DIM_MAX];

    void SetShape(const std::vector<SymInt> &shape) {
        for (std::size_t i = 0; i < shape.size(); i++) {
            dim[i] = shape[i];
        }
        dimSize = static_cast<int>(shape.size());
    }

    uint64_t At(size_t idx, const uint64_t *exprTbl) const {
        if (dim[idx].IsExpression())
            return exprTbl[dim[idx].Value()];
        else
            return dim[idx].Value();
    }

    void ToStride(uint64_t *stides, const uint64_t *exprTbl) const {
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
    uint64_t maxStaticMemReq; // 0 if cannot find a non-symbolic raw shape
    DataType dataType;
    DevSymShape shape;
    DevIOProperty ioProperty{DevIOProperty::NONE};
    int32_t ioIndex;
    int32_t linkedIncastId; //outcast shared same addr with incast
    int rawMagic;

    int GetDim() const { return shape.dimSize; }

    uint64_t GetMemoryRequirement(const uint64_t *exprTbl) const {
        if (memoryRequirement != 0)
            return memoryRequirement;
        uint64_t memReq = BytesOf(dataType);
        for (int i = 0; i < GetDim(); i++) {
            memReq *= shape.At(i, exprTbl);
        }
        return memReq;
    }

    std::string DumpType() const {
        std::ostringstream oss;
        oss << "<";
        for (int i = 0; i < shape.dimSize; i++) {
            if (shape.dim[i].IsExpression()) {
                oss << "? x ";
            } else {
                oss << shape.dim[i].Value() << " x ";
            }
        }
        oss << DataType2String(dataType);
        oss << ">";
        return oss.str();
    }

    std::string DumpAttr() const {
        std::ostringstream oss;
        if (ioProperty == DevIOProperty::ROOT_INCAST) {
            oss << schema::incast(ioIndex).Dump() << " ";
        } else if (ioProperty == DevIOProperty::ROOT_OUTCAST) {
            oss << schema::outcast(ioIndex).Dump() << " ";
        }
        oss << schema::mem(memoryRequirement).Dump() << " ";
        oss << schema::off(addrOffset).Dump();
        return oss.str();
    }

    static std::string DumpAttrDesc(const DevRawTensorDesc *desc) {
        std::ostringstream oss;
        if (desc->location == RAW_TENSOR_LOCATION_INCAST) {
            oss << schema::incast(desc->offsetOrIndex).Dump();
        } else if (desc->location == RAW_TENSOR_LOCATION_OUTCAST) {
            oss << schema::outcast(desc->offsetOrIndex).Dump();
        } else {
            oss << schema::off(desc->offsetOrIndex).Dump();
        }
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
constexpr uint32_t TASKID_SHIFT32 = 32;

inline uint32_t MakeTaskID(uint32_t funcId, uint32_t taskId) {
    return (funcId << TASKID_TASK_BITS) | taskId;
}

#ifdef SUPPORT_WRAP
inline uint32_t MakeWrapID(uint32_t funcId, uint32_t wrapId) {
    return (funcId << TASKID_TASK_BITS) | wrapId;
}
#endif

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

struct DevAscendOperationOperandInfo {
    int tensorIndex{0};
    int staticOffsetAttrBeginIndex{0};
    int staticShapeAttrBeginIndex{0};
    int staticRawShapeAttrBeginIndex{0};

    DevAscendOperationOperandInfo() {}
    DevAscendOperationOperandInfo(int tTensorIndex, int tStaticAttrBeginIndex, int tStaticDim)
        : tensorIndex(tTensorIndex),
          staticOffsetAttrBeginIndex(tStaticAttrBeginIndex),
          staticShapeAttrBeginIndex(tStaticAttrBeginIndex + tStaticDim),
          staticRawShapeAttrBeginIndex(staticShapeAttrBeginIndex + tStaticDim) {}
    int GetDim() const { return staticShapeAttrBeginIndex - staticOffsetAttrBeginIndex; }
};

struct DevAscendOperation {
    DevLocalVector<DevAscendOperationOperandInfo> ioperandList;
    DevLocalVector<DevAscendOperationOperandInfo> ooperandList;
    DevLocalVector<SymInt> attrList; // opattr[0] -> hash
    int32_t outcastStitchIndex;
    uint32_t depGraphPredCount;
    DevLocalVector<int> depGraphSuccList;
    DevLocalVector<int> depGraphCopyOutResolveSuccIndexList;
    uint64_t debugOpmagic; // DEBUG_ONLY
};

struct DevAscendFunctionCallOperandUse {
    int operationIdx{-1};
    int operandIdx{-1};
    int offsetAttrIdx{-1};
    int shapeAttrIdx{-1};

    DevAscendFunctionCallOperandUse() = default;
    DevAscendFunctionCallOperandUse(int operationIdx_, int operandIdx_, int offsetAttrIdx_, int shapeAttrIdx_)
        : operationIdx(operationIdx_), operandIdx(operandIdx_), offsetAttrIdx(offsetAttrIdx_), shapeAttrIdx(shapeAttrIdx_) {}
};

struct DevAscendFunctionIncast {
    int tensorIndex;
    DevLocalVector<int> fromSlotList;

    int dim;
    int stitchByAllFullMatch;
    DevLocalVector<DevAscendFunctionCallOperandUse> consumerList;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevLocalVector<uint32_t> cellMatchStaticIncastTable;

    DevLocalVector<uint32_t> stitchPolicyFullCoverConsumerAllOpIdxList;
};

struct DevAscendFunctionOutcast {
    int tensorIndex;
    DevLocalVector<int> toSlotList;

    int dim;
    int stitchByAllFullMatch;

    DevLocalVector<DevAscendFunctionCallOperandUse> producerList;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevLocalVector<uint32_t> cellMatchStaticOutcastTable;
    DevLocalVector<uint32_t> cellMatchRuntimeFullUpdateTable;

    int stitchPolicyFullCoverProducerHubOpIdx;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList;
    DevLocalVector<uint32_t> stitchPolicyFullCoverProducerAllOpIdxList;
    int exprListIndex;
};

#define ADDRESS_CACHE_KIND_WORKSPACE         0
#define ADDRESS_CACHE_KIND_INPUT             1
#define ADDRESS_CACHE_KIND_OUTPUT            2
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
        struct {
            uint64_t cacheValue : 60;
            uint64_t cacheKind : 4;
        };
    };

    static AddressDescriptor MakeAddress(uint64_t addr) {
        AddressDescriptor desc;
        desc.addr = addr;
        desc.isAddress = 1;
        return desc;
    }

    static AddressDescriptor MakeCache(uint64_t kind, uint64_t value) {
        AddressDescriptor desc;
        desc.cacheValue = value;
        desc.cacheKind = kind;
        return desc;
    }

    bool IsAddress() const { return isAddress; }
    uint64_t GetAddress() const { DEV_ASSERT(isAddress); return addr; }
    uint64_t GetAddressValue() const { return addr; }
    bool IsNullAddress() const { return IsAddress() && addr == 0; }

    explicit AddressDescriptor(uint64_t address = 0): addr(address) { isAddress = true; }
    AddressDescriptor(int tdupIdx, int toutcastIdx): outcastIdx(toutcastIdx) , dupIdx(tdupIdx) { isAddress = false; }

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
    std::string Dump() const {
        std::stringstream ss;
        if (isAddress) {
            ss << DumpAddress(addr);
        } else {
            ss << "&&" << dupIdx << ":" << outcastIdx;
        }
        return ss.str();
    }
};

constexpr int INVALID_INDEX = -1;

struct DevAscendFunctionDuppedData;

struct DevAscendFunctionPredInfo {
    uint64_t totalZeroPred;
    uint64_t totalZeroPredAIV;
    uint64_t totalZeroPredAIC;
    uint64_t totalZeroPredHub;
    uint64_t totalZeroPredAicpu;
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

struct InoutOperationAttr {
    int dim;
    std::vector<DevAscendFunctionCallOperandUse> useList;
    int cellMatchSize;

    std::vector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList;
    int stitchPolicyFullCoverProducerHubOpIdx;

    DevCellMatchTableDesc cellMatchTableDesc;

    std::vector<uint32_t> useOpList;
    int bindTensorExprIndex{-1};
};

struct DevAscendFunction {
    uint64_t rootHash;
    uint64_t funcKey;
    // source root function after duplication
    DevAscendFunction *sourceFunc{nullptr};

    // Fill base address after stitch
    uintdevptr_t runtimeWorkspace;
    // Base address of invoke entries
    uintdevptr_t opAttrs;

    int funcidx;

    int stackWorkSpaceSize;

    uint32_t getInputDataCount;
    uint32_t getTensorDataCount;

    DevLocalVector<AddressDescriptor> incastAddressList;
    DevLocalVector<AddressDescriptor> outcastAddressList;

    DevLocalVector<uint64_t> expressionList;
#define allocateLastField expressionList

    DevAscendFunctionPredInfo predInfo_;
    uint64_t duppedDataAllocSize_;
    uint64_t duppedDataCopySize_;
    DevLocalVector<uint8_t> duppedData_;
#ifdef SUPPORT_WRAP
    uint64_t wrapIdNum_;
#endif
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
#ifdef SUPPORT_WRAP
    DevLocalVector<int> opWrapList_;
    DevLocalVector<int> opWrapTaskNumList_;
#endif
    DevLocalVector<int> operationSuccList_;
    DevLocalVector<int> operationCopyOutResolveSuccIndexList_;

    DevLocalVector<DevAscendFunctionIncast> incastList;
    DevLocalVector<DevAscendFunctionOutcast> outcastList;
    DevLocalVector<int> slotList;

    DevLocalVector<DevAscendFunctionCallOperandUse> useList;
    DevLocalVector<DevAscendFunctionCallOperandUse> stitchPolicyFullCoverProducerList_;
    DevLocalVector<uint32_t> stitchPolicyFullCoverOpList_;

    DevLocalVector<uint32_t> cellMatchRuntimeFullUpdateTableList;
    DevLocalVector<uint32_t> cellMatchStaticOutcastTableList;
    DevLocalVector<uint32_t> cellMatchStaticIncastTableList;
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
     *      int                                                 operationCopyOutResolveSuccIndexData[];
     *      DevAscendFunctionIncast                             incastListData[];
     *      DevAscendFunctionOutcast                            outcastListData[];
     *      int                                                 slotListData[];
     *      int                                                 offsetIdxListData[];
     *      int                                                 shapeIdxListData[];
     *      int                                                 producerConsumerListData[];
     *      char                                                rawNameData[];
     *      uint8_t                                             duppedData[];
     */

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
    bool HasValueDepend() const {
        return getInputDataCount + getTensorDataCount;
    }
    static std::string DumpByte(uint8_t byte) {
        char buf[0x10];
        (void)sprintf_s(buf, sizeof(buf), "0x%02x", byte);
        return buf;
    }

    schema::coa SchemaGetCoa(int operationIndex, uint64_t *runtimeExpressionList = nullptr, bool dumpIndex = false) const {
        std::vector<schema::TextType> coaDataList;
        for (size_t j = 0; j < GetOperationAttrSize(operationIndex); j++) {
            const SymInt &s = GetOperationAttr(operationIndex, j);
            std::string textData;
            if (s.IsExpression()) {
                if (runtimeExpressionList != nullptr) {
                    textData = std::to_string(runtimeExpressionList[s.Value()]);
                } else {
                    textData = "?" + std::to_string(s.Value());
                }
            } else {
                textData = std::to_string(s.Value());
            }
            coaDataList.push_back(schema::TextType(textData));
        }
        return schema::coa(schema::coaType(coaDataList, dumpIndex));
    }

    static std::string DumpShape(const DevShape &shape) {
        std::ostringstream oss;
        oss << "<";
        for (int k = 0; k < shape.dimSize; k++) {
            oss << Delim(k != 0, ",") << shape.dim[k];
        }
        oss << ">";
        return oss.str();
    }

    static std::string DumpStride(const DevAscendStride &stride) {
        std::ostringstream oss;
        oss << "<";
        for (int k = 0; k < stride.dimSize; k++) {
            oss << Delim(k != 0, ",") << stride.dimStride[k];
        }
        oss << ">";
        return oss.str();
    }

    static std::string DumpCellMatchTableDesc(const DevCellMatchTableDesc &desc) {
        return DumpShape(desc.cellShape) + " x " + DumpStride(desc.stride);
    }

    static std::string DumpSymInt(const SymInt &s, const uint64_t *runtimeExpressionList) {
        std::ostringstream oss;
        if (s.IsExpression()) {
            if (runtimeExpressionList == nullptr) {
                oss << "?" << s.Value();
            } else {
                oss << runtimeExpressionList[s.Value()];
            }
        } else {
            oss << s.Value();
        }
         return oss.str();
    }

    static std::string DumpSymIntList(const SymInt *s, int count, uint64_t *runtimeExpressionList) {
        std::ostringstream oss;
        oss << "<";
        for (int i = 0; i < count; i++) {
            oss << Delim(i != 0, ",") << DumpSymInt(s[i], runtimeExpressionList);
        }
        oss << ">";
        return oss.str();
    }

    std::string DumpOperationAttr(int operationIndex, uint64_t *runtimeExpressionList = nullptr, bool dumpIndex=false) const {
        std::ostringstream oss;
        oss << SchemaGetCoa(operationIndex, runtimeExpressionList, dumpIndex).Dump();
        oss << " " << schema::pred(GetOperationDepGraphPredCount(operationIndex)).Dump();
        const DevLocalVector<int> &succList = GetOperationDepGraphSuccList(operationIndex);
        std::vector<schema::operation> succDataList;
        for (size_t j = 0; j < succList.size(); j++) {
            succDataList.push_back(At(succList, j));
        }
        oss << " " << schema::succ(succDataList).Dump();

        const DevLocalVector<int> &copyOutResolveCounterIndexList = GetOperationDepGraphCopyOutResolveSuccIndexList(operationIndex);
        std::vector<int64_t> outSuccIndexDataList;
        for (size_t j = 0; j < copyOutResolveCounterIndexList.size(); j++) {
            outSuccIndexDataList.push_back(At(copyOutResolveCounterIndexList, j));
        }
        oss << " " << schema::outSuccIndex(outSuccIndexDataList).Dump();
        oss << " " << "#outcastStitch{" << GetOperationOutcastStitchIndex(operationIndex) << "}";
        return oss.str();
    }

    std::string DumpOperation(int operationIndex, int &totalAttrStartIdx, const std::vector<uintdevptr_t> &ooperandAddrList = {},
        const std::vector<uintdevptr_t> &ioperandAddrList = {}, uint64_t *runtimeExpressionList = nullptr) const {
        std::ostringstream oss;
        for (size_t j = 0; j < GetOperationOOperandSize(operationIndex); j++) {
            oss << Delim(j != 0, ",") << DumpTensor(GetOperationOOperandInfo(operationIndex, j).tensorIndex);
            if (j < ooperandAddrList.size()) {
                oss << AddressDescriptor::DumpAddress(ooperandAddrList[j]);
            }
        }
        oss << " = "
            << "!" << operationIndex << " ";
        for (size_t j = 0; j < GetOperationIOperandSize(operationIndex); j++) {
            oss << Delim(j != 0, ",") << DumpTensor(GetOperationIOperandInfo(operationIndex, j).tensorIndex);
            if (j < ioperandAddrList.size()) {
                oss << AddressDescriptor::DumpAddress(ioperandAddrList[j]);
            }
        }

        oss << " " << DumpOperationAttr(operationIndex, runtimeExpressionList, true);
        totalAttrStartIdx += static_cast<int>(GetOperationAttrSize(operationIndex));
        return oss.str();
    }

    std::string DumpRawTensor(int rawIndex, uintdevptr_t addr = 0) const {
        std::ostringstream oss;
        auto rawTensor = GetRawTensor(rawIndex);
        auto rawTensorDesc = GetRawTensorDesc(rawIndex);
        oss << rawTensor->DumpType() << " @" << rawIndex << " = ";
        oss << rawTensor->DumpAttr() << " ";
        oss << DevAscendRawTensor::DumpAttrDesc(rawTensorDesc);
        if (addr != 0) {
            oss << AddressDescriptor::DumpAddress(addr);
        }
        return oss.str();
    }

    std::string DumpIncast(int incastIndex, const std::string &indent, uint64_t *runtimeExpressionList = nullptr, const std::vector<uintdevptr_t> &slotAddrList = {}) const {
        std::ostringstream oss;
        const DevAscendFunctionIncast &incast = GetIncast(incastIndex);
        oss << "#incast:" << incastIndex << " = " << DumpTensor(incast.tensorIndex);
        for (size_t j = 0; j < incast.fromSlotList.size(); j++) {
            int slot = At(incast.fromSlotList, j);
            oss << " <- #slot:" << slot;
            if (slot < static_cast<int>(slotAddrList.size())) {
                oss << AddressDescriptor::DumpAddress(slotAddrList[slot]);
            }
        }
        oss << "\n";
        oss << indent;
        oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(incast.cellMatchTableDesc);
        oss << " | #cellMatchStaticTable:" << incast.cellMatchStaticIncastTable.size();
        oss << "\n";

        oss << indent << " | #stitchPolicyFullCoverConsumerAllOpIdxList:[";
        for (size_t j = 0; j < incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); j++) {
            oss << Delim(j != 0, ",") << At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, j);
        }
        oss << "]\n";

        for (size_t j = 0; j < incast.consumerList.size(); j++) {
            auto &consumer = At(incast.consumerList, j);
            int consumerIdx = consumer.operationIdx;
            int operandIdx = consumer.operandIdx;
            int offsetAttrIdx = consumer.offsetAttrIdx;
            int shapeAttrIdx = consumer.shapeAttrIdx;
            oss << indent;
            oss << " | #consumerIdx:!" << consumerIdx;
            oss << " | #operandIdx:" << operandIdx;
            oss << " | #offsetAttrIdx:" << offsetAttrIdx;
            oss << " | #shapeAttrIdx:" << shapeAttrIdx;
            oss << " | #offsetAttr:" << DumpSymIntList(&GetOperationAttr(consumerIdx, offsetAttrIdx), incast.dim, runtimeExpressionList);
            oss << " | #shapeAttr:" << DumpSymIntList(&GetOperationAttr(consumerIdx, shapeAttrIdx), incast.dim, runtimeExpressionList);
            oss << "\n";
        }
        return oss.str();
    }

    std::string DumpOutcast(int outcastIndex, const std::string &indent, uint64_t *runtimeExpressionList = nullptr, const std::vector<uintdevptr_t> &slotAddrList = {}) const {
        std::ostringstream oss;
        const DevAscendFunctionOutcast &outcast = GetOutcast(outcastIndex);
        auto dumpProducer = [this, &oss, &indent, &outcast, &runtimeExpressionList](const DevLocalVector<DevAscendFunctionCallOperandUse>& producerList) -> void {
            for (size_t j = 0; j < producerList.size(); j++) {
                auto &producer = At(producerList, j);
                int producerIdx = producer.operationIdx;
                int operandIdx = producer.operandIdx;
                int offsetAttrIdx = producer.offsetAttrIdx;
                int shapeAttrIdx = producer.shapeAttrIdx;
                oss << indent;
                oss << " | #producerIdx:!" << producerIdx;
                oss << " | #operandIdx:" << operandIdx;
                oss << " | #offsetAttrIdx:" << offsetAttrIdx;
                oss << " | #shapeAttrIdx:" << shapeAttrIdx;
                oss << " | #offsetAttr:" << DumpSymIntList(&GetOperationAttr(producerIdx, offsetAttrIdx), outcast.dim, runtimeExpressionList);
                oss << " | #shapeAttr:" << DumpSymIntList(&GetOperationAttr(producerIdx, shapeAttrIdx), outcast.dim, runtimeExpressionList);
                oss << "\n";
            }
        };

        oss << "#outcast:" << outcastIndex << " = " << DumpTensor(outcast.tensorIndex);
        for (size_t j = 0; j < outcast.toSlotList.size(); j++) {
            int slot = At(outcast.toSlotList, j);
            oss << " -> #slot:" << slot;
            if (slot < static_cast<int>(slotAddrList.size())) {
                oss << AddressDescriptor::DumpAddress(slotAddrList[slot]);
            }
        }
        oss << "\n";
        oss << indent;
        oss << " | #cellMatchTableDesc:" << DumpCellMatchTableDesc(outcast.cellMatchTableDesc);
        oss << " | #cellMatchStaticTable:" << outcast.cellMatchStaticOutcastTable.size();
        oss << " | #cellMatchFullUpdateTable:" << outcast.cellMatchRuntimeFullUpdateTable.size();
        oss << "\n";

        oss << indent << " | #stitchPolicyFullCoverProducerList:[";
        for (size_t j = 0; j < outcast.stitchPolicyFullCoverProducerList.size(); j++) {
            oss << Delim(j != 0, ",") << At(outcast.stitchPolicyFullCoverProducerList, j).operationIdx;
        }
        oss << "]\n";
        dumpProducer(outcast.stitchPolicyFullCoverProducerList);

        oss << indent << " | #stitchPolicyFullCoverProducerHubOpIdx:" << outcast.stitchPolicyFullCoverProducerHubOpIdx << "\n";
        oss << indent << " | #stitchPolicyFullCoverProducerAllOpIdxList:[";
        for (size_t j = 0; j < outcast.stitchPolicyFullCoverProducerAllOpIdxList.size(); j++) {
            oss << Delim(j != 0, ",") << At(outcast.stitchPolicyFullCoverProducerAllOpIdxList, j);
        }
        oss << "]\n";
        dumpProducer(outcast.producerList);
        return oss.str();
    }

public:
    std::string Dump(int indent = 0) const {
        std::string INDENT(indent, ' ');
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');
        std::ostringstream oss;

        oss << INDENT << "DevFunction " << funcKey;
        oss << " " << schema::name(GetRawName()).Dump();
        oss << " " << schema::mem(rawTensorWsMemoryRequirement).Dump();
        oss << " " << schema::memOut(outcastWsMemoryRequirement).Dump();
        oss << " {\n";
        for (size_t i = 0; i < GetRawTensorSize(); i++) {
            oss << INDENTINNER << DumpRawTensor(i) << "\n";
        }
        for (size_t i = 0; i < GetIncastSize(); i++) {
            oss << INDENTINNER << DumpIncast(i, INDENTINNER) << "\n";
        }
        for (size_t i = 0; i < GetOutcastSize(); i++) {
            oss << INDENTINNER << DumpOutcast(i, INDENTINNER) << "\n";
        }

        oss << INDENTINNER << "#zeropred:" << predInfo_.totalZeroPred << "\n";
        oss << INDENTINNER << "#zeropred-aiv:" << predInfo_.totalZeroPredAIV << "\n";
        oss << INDENTINNER << "#zeropred-aic:" << predInfo_.totalZeroPredAIC << "\n";
        oss << INDENTINNER << "#zeropred-aicpu:" << predInfo_.totalZeroPredAicpu << "\n";
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

    int GetFuncKey() const { return funcKey; }

    const DevAscendFunction *GetSource() const { return sourceFunc; }
    DevAscendFunction *GetSource() { return sourceFunc; }

    int GetRootIndex() const { return funcKey; }

    const int &GetFuncidx() const { return funcidx; }
    int &GetFuncidx() { return funcidx; }

    const DevAscendFunctionPredInfo &GetPredInfo() const { return predInfo_; }
    uint64_t GetDuppedDataAllocSize() const { return duppedDataAllocSize_; }
    uint64_t GetDuppedDataCopySize() const { return duppedDataCopySize_; }
    DevAscendFunctionDuppedData *GetDuppedData() const { return reinterpret_cast<DevAscendFunctionDuppedData *>(const_cast<uint8_t*>(&At(duppedData_, 0))); }

    int32_t *GetOpAttrOffsetAddr() { return &At(opAttrOffsetList_, 0); }
    inline int32_t GetOpAttrOffsetSize() { return opAttrOffsetList_.size(); }
    int *GetCalleeIndexAddr() { return &At(opCalleeList_, 0); }
#ifdef SUPPORT_WRAP
    int *GetOpWrapListAddr() { return &At(opWrapList_, 0); }
    int *GetOpWrapTaskNumListAddr() { return &At(opWrapTaskNumList_, 0); }
#endif
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
    inline uint64_t GetExpressionSize() const { return expressionList.size(); }
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
    inline uint32_t GetOperationDebugOpmagic(int operationIndex) const {
        return At(operationList_, operationIndex).debugOpmagic;
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

    inline const uint32_t &GetOperationDepGraphPredCount(int operationIndex) const {
        return At(operationList_, operationIndex).depGraphPredCount;
    }
    inline uint32_t &GetOperationDepGraphPredCount(int operationIndex) {
        return At(operationList_, operationIndex).depGraphPredCount;
    }

    inline const DevLocalVector<int> &GetOperationDepGraphSuccList(int operationIndex) const {
        return At(operationList_, operationIndex).depGraphSuccList;
    }

    inline const DevLocalVector<int> &GetOperationDepGraphCopyOutResolveSuccIndexList(int operationIndex) const {
        return At(operationList_, operationIndex).depGraphCopyOutResolveSuccIndexList;
    }

    inline const int *GetOperationDepGraphSuccAddr(int operationIndex, size_t &size) const {
        auto &succList = At(operationList_, operationIndex).depGraphSuccList;
        size = succList.size();
        return &At(succList, 0);
    }

    inline const int *GetOperationDepGraphCopyOutResolveSuccIndexAddr(int operationIndex, size_t &size) const {
        auto &succIndexList = At(operationList_, operationIndex).depGraphCopyOutResolveSuccIndexList;
        size = succIndexList.size();
        return &At(succIndexList, 0);
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

    struct SymIntPair {
        const SymInt *offsetSymList;
        const SymInt *shapeSymList;
    };
    inline SymIntPair GetTensorOffsetShapeSymList(
            int operationIndex, int operandIndex, bool isIOperand = true) const {
        auto &operandInfo = GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
        const SymInt *offsetSymList = &GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex);
        const SymInt *shapeSymList = &GetOperationAttr(operationIndex, operandInfo.staticShapeAttrBeginIndex);
        return SymIntPair{offsetSymList, shapeSymList};
    }

    template<bool skipExpression>
    inline bool GetTensorOffsetAndShape(
            uint64_t offset[DEV_SHAPE_DIM_MAX],
            uint64_t shape[DEV_SHAPE_DIM_MAX],
            const uint64_t *runtimeExpressionList, int dims, int operationIndex, int operandIndex,
            bool isIOperand = true) const {
        auto [offsetSymList, shapeSymList] = GetTensorOffsetShapeSymList(operationIndex, operandIndex, isIOperand);

        bool paramConcrete = true;
        for (int i = 0; i < dims; i++) {
            auto value = offsetSymList[i].Value();
            if (offsetSymList[i].IsExpression()) {
                if (skipExpression) {
                    paramConcrete = false;
                } else {
                    offset[i] = runtimeExpressionList[value];
                }
            } else {
                offset[i] = value;
            }
        }
        for (int i = 0; i < dims; i++) {
            auto value = shapeSymList[i].Value();
            if (shapeSymList[i].IsExpression()) {
                if (skipExpression) {
                    paramConcrete = false;
                } else {
                    shape[i] = runtimeExpressionList[value];
                }
            } else {
                shape[i] = value;
            }
        }
        return paramConcrete;
    }

    template<bool skipExpression>
    inline bool GetTensorRawShape(
            uint64_t rawShape[DEV_SHAPE_DIM_MAX],
            const uint64_t *runtimeExpressionList, int dims, int operationIndex, int operandIndex,
            bool isIOperand = true) const {
        auto &operandInfo = GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
        const SymInt *rawShapeSymList = &GetOperationAttr(operationIndex, operandInfo.staticRawShapeAttrBeginIndex);
        bool paramConcrete = true;
        for (int i = 0; i < dims; i++) {
            auto value = rawShapeSymList[i].Value();
            if (rawShapeSymList[i].IsExpression()) {
                if (skipExpression) {
                    paramConcrete = false;
                } else {
                    rawShape[i] = runtimeExpressionList[value];
                }
            } else {
                rawShape[i] = value;
            }
        }
        return paramConcrete;
    }

    static void CellMatchGetIndexRange(
            const uint64_t offset[DEV_SHAPE_DIM_MAX],
            const uint64_t shape[DEV_SHAPE_DIM_MAX],
            const DevCellMatchTableDesc &cellMatchTableDesc,
            uint64_t rangeBegin[DEV_SHAPE_DIM_MAX],
            uint64_t rangeEnd[DEV_SHAPE_DIM_MAX]) {
        for (int i = 0; i < cellMatchTableDesc.GetDimensionSize(); ++i) {
            auto cellMatchShapeDim = cellMatchTableDesc.GetCellShape(i);
            if(cellMatchShapeDim != 0) {
                rangeBegin[i] = offset[i] / cellMatchShapeDim;
                rangeEnd[i] = (offset[i] + shape[i] - 1) / cellMatchShapeDim;
            } else {
                DEV_ASSERT(0);
            }
        }
    }

    template<typename HandleType, typename ...TyArgs>
    static void CellMatch5Dimension(const DevCellMatchTableDesc &cellMatchTableDesc, uint64_t* rangeBegin, uint64_t* rangeEnd, TyArgs ... args) {
        int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
        int s2 = cellMatchTableDesc.GetStride(3), s3 = cellMatchTableDesc.GetStride(4), s4 = 1;
        for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1)
        for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2)
        for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3)
        for (int d4 = d3 + rangeBegin[4] * s4, e4 = d3 + rangeEnd[4] * s4; d4 <= e4; d4 += s4) {
            HandleType::Process(d4, args...);
        }
    }

    template<typename HandleType, typename ...TyArgs>
    static void CellMatch4Dimension(const DevCellMatchTableDesc &cellMatchTableDesc, uint64_t* rangeBegin, uint64_t* rangeEnd, TyArgs ... args) {
        int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
        int s2 = cellMatchTableDesc.GetStride(3), s3 = 1;
        for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1)
        for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2)
        for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
            HandleType::Process(d3, args...);
        }
    }

    template<typename HandleType, typename ...TyArgs>
    static void CellMatchHandle(
            const uint64_t offset[DEV_SHAPE_DIM_MAX],
            const uint64_t shape[DEV_SHAPE_DIM_MAX],
            const DevCellMatchTableDesc &cellMatchTableDesc,
            TyArgs ... args) {
        uint64_t rangeBegin[DEV_SHAPE_DIM_MAX];
        uint64_t rangeEnd[DEV_SHAPE_DIM_MAX];
        CellMatchGetIndexRange(offset, shape, cellMatchTableDesc, rangeBegin, rangeEnd);
        switch (cellMatchTableDesc.cellShape.dimSize) {
        case 1:
            {
                int s0 = 1;
                for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
                    HandleType::Process(d0, args...);
                }
            }
            break;
        case DEV_SHAPE_DIM_NUM_2:
            {
                int s0 = cellMatchTableDesc.GetStride(1), s1 = 1;
                for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
                for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
                    HandleType::Process(d1, args...);
                }
            }
            break;
        case DEV_SHAPE_DIM_NUM_3:
            {
                int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2), s2 = 1;
                for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
                for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1)
                for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                    HandleType::Process(d2, args...);
                }
            }
            break;
        case DEV_SHAPE_DIM_NUM_4:
            {
                CellMatch4Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
            }
            break;
        case DEV_SHAPE_DIM_NUM_5:
            {
                CellMatch5Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
            }
            break;
        default:
            DEV_ERROR("[Stitch] Too many dimension: %d\n", (int)cellMatchTableDesc.GetDimensionSize());
            break;
        }
    }

    static void CellMatchFill(
            const uint64_t offset[DEV_SHAPE_DIM_MAX],
            const uint64_t shape[DEV_SHAPE_DIM_MAX],
            uint32_t operationIdx,
            const DevCellMatchTableDesc &cellMatchTableDesc,
            uint32_t *cellMatchTableData) {
        struct HandleFill {
            static inline void Process(int index, uint32_t *cellMatchTableData, uint32_t operationIdx) {
                cellMatchTableData[index] = operationIdx;
                DEV_VERBOSE_DEBUG("cell match fill, operation %u , cellindex[%d] = operationindex(%u)",
                        operationIdx, index, operationIdx);
            }
        };
        CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, operationIdx);
    }

    static void CellMatchFill(
            const uint64_t offset[DEV_SHAPE_DIM_MAX],
            const uint64_t shape[DEV_SHAPE_DIM_MAX],
            uint32_t operationIdx,
            const DevCellMatchTableDesc &cellMatchTableDesc,
            uint64_t *cellMatchTableData, uint32_t devTaskId,
            uint32_t funcIdx) {
        struct HandleFill {
            static inline void Process(int index, uint64_t *cellMatchTableData, uint32_t devTaskId, uint32_t funcIdx, uint32_t operationIdx) {
                cellMatchTableData[index] = (static_cast<uint64_t>(devTaskId) << TASKID_SHIFT32) | MakeTaskID(funcIdx, operationIdx);
                DEV_VERBOSE_DEBUG("cell match fill, devtaskid:%u funcIdx %u operation %u , cellindex[%d] = taskid(%lx)",
                        devTaskId, funcIdx, operationIdx, index, cellMatchTableData[index]);
            }
        };
        CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, devTaskId, funcIdx, operationIdx);
    }

    template<bool skipExpression, typename ... TyArgs>
    bool CellMatchFillIncastOutcast(
            DevAscendFunctionCallOperandUse *operandUseList,
            size_t useSize,
            const uint64_t *runtimeExpressionList,
            bool isIOperand,
            const DevCellMatchTableDesc &cellMatchTableDesc,
            TyArgs... args) {
        bool allConcrete = true;
        auto validateAndRefreshOffsetShape =
        [this, &runtimeExpressionList, &cellMatchTableDesc, &isIOperand](
            const uint64_t offset[DEV_SHAPE_DIM_MAX],
            uint64_t shape[DEV_SHAPE_DIM_MAX],
            int operationIndex, int operandIndex) {
            uint64_t rawShape[DEV_SHAPE_DIM_MAX];
            bool paramConcrete = GetTensorRawShape<skipExpression>(rawShape, runtimeExpressionList,
                cellMatchTableDesc.GetDimensionSize(), operationIndex, operandIndex, isIOperand);
            if (paramConcrete) {
                for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                    DEV_VERBOSE_DEBUG("cell match fill, operation[%d] -> dimension[%d] = (offset:%lu ,shape:%lu, rawshape:%lu, cellshape:%d)",
                            operationIndex, j, offset[j], shape[j], rawShape[j], cellMatchTableDesc.cellShape.dim[j]);
                    if (offset[j] >= rawShape[j]) {
                        DEV_VERBOSE_DEBUG("cell match fill failed, exceed invalid cell");
                        return false;
                    } else if (offset[j] + shape[j] > rawShape[j]){
                        shape[j] = rawShape[j] - offset[j];
                    }
                }
            }
            return true;
        };

        for (size_t i = 0; i < useSize; i++) {
            auto &use = operandUseList[i];
            uint64_t offset[DEV_SHAPE_DIM_MAX];
            uint64_t shape[DEV_SHAPE_DIM_MAX];
            bool paramConcrete = GetTensorOffsetAndShape<skipExpression>(offset, shape, runtimeExpressionList,
                cellMatchTableDesc.GetDimensionSize(), use.operationIdx, use.operandIdx, isIOperand);
            if (paramConcrete) {
                if (!validateAndRefreshOffsetShape(offset, shape, use.operationIdx, use.operandIdx)) {
                    continue; // dassemble offset of outoperand maybe exceed the rawshape dimension
                }
                CellMatchFill(offset, shape, use.operationIdx, cellMatchTableDesc, args...);
            }
            allConcrete &= paramConcrete;
        }
        return allConcrete;
    }

    inline const char *GetRawName() const { return &At(rawName_, 0); }

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
            const std::unordered_map<Operation *, OrderedSet<Operation *>> &callOpSuccDict,bool fillContent);
    void FillOutputSlotMark(const IncastOutcastLink *inoutLink, std::vector<bool>& isOutputSlotMarks);
    void InitRawTensorAndMemoryRequirement(
            uintdevptr_t &initOffset,
            const OrderedSet<std::shared_ptr<RawTensor>> &incastRawList,
            const OrderedSet<std::shared_ptr<RawTensor>> &outcastRawList,
            const OrderedSet<std::shared_ptr<RawTensor>> &rawList,
            const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawMagicToRawTensor,
            const std::vector<EncodeRawTensorAttr> &rawAttrs,
            const EncodeDevAscendFunctionParam &param,
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
            const std::vector<int> &noPredOpList,
            const std::vector<int> &noSuccOpList,
            const std::unordered_map<Operation *, std::vector<int>> &copyOutResolveSuccIndexListDict,
            bool fillContent);

    void InitIncastOutcast(uintdevptr_t &initOffset, const std::vector<std::shared_ptr<LogicalTensor>> &incastTensorList,
        const std::vector<std::shared_ptr<LogicalTensor>> &outcastTensorList,
        const OrderedSet<std::shared_ptr<LogicalTensor>> &tlist,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &incastOpAttrDict,
        const std::unordered_map<std::shared_ptr<LogicalTensor>, InoutOperationAttr> &outcastOpAttrDict,
        const IncastOutcastSlot *slot, const std::string &initRawName, bool fillContent);
};

struct DevAscendFunctionDuppedOperation {
    uint32_t size;
    uint32_t predCountBase;
    uint32_t stitchBase;
    uint32_t stitchCount;
};
struct DevAscendFunctionDuppedVector {
    uint32_t size;
    uint32_t base;
};

constexpr uint32_t DUPPED_STITCH_SIZE  = 0x10 - (sizeof(void *) / sizeof(uint32_t)) - 0x1;
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
    DevAscendFunctionDuppedStitch * const &Next() const { return next_; }
    DevAscendFunctionDuppedStitch *&Next() { return next_; }

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

    DevAscendFunctionDuppedStitch * const &Head() const { return head_; }
    DevAscendFunctionDuppedStitch * &Head() { return head_; }

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

    template<typename T = uint32_t>
    static std::string DumpTask(T id) {
        std::ostringstream oss;
        if constexpr (std::is_same<T, uint64_t>::value) {
            oss << (id >> TASKID_SHIFT32) << "!"; // devicetaskid
        }
        oss << FuncID(static_cast<uint32_t>(id)) << "!" << TaskID(static_cast<uint32_t>(id));
        return oss.str();
    }

    template<typename T = uint32_t>
    static std::string DumpTask(T *idx, int size) {
        std::ostringstream oss;
        oss << "{";
        oss << "size = " << size << " -> ";
        for (int i = 0; i < size; i++) {
            if (idx[i] != AICORE_TASK_INIT) {
                oss << Delim(i != 0, ",");
                oss << "[" << std::dec << i << "]=" << DumpTask<T>(idx[i]);
            }
        }
        oss << "}";
        return oss.str();
    }

    std::string Dump() const {
        std::ostringstream oss;

        uint32_t index = 0;
        oss << "[";
        for (auto p = head_; p != nullptr; p = p->Next()) {
            oss << Delim(p != head_, ";");
            for (uint32_t i = 0; i < p->Size(); i++) {
                oss << Delim(i != 0, ",");
                oss << "[" << index++ << "]=" << DumpTask(p->At(i));
            }
        }
        oss << "]";
        return oss.str();
    }

private:
    DevAscendFunctionDuppedStitch *head_{nullptr};
};
static_assert(sizeof(DevAscendFunctionDuppedStitchList) == sizeof(void *));

struct RuntimeReuseInfo {
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
     *      DevAscendFunctionDuppedStitchList                   stitchListData[];
     */
#define GET_DATA(type, data, base, index) ((reinterpret_cast<type *>(const_cast<uint8_t *>((data) + (base)))[index]))
    uint32_t GetOperationSize() const { return operationList_.size; }
    const predcount_t &GetOperationCurrPredCount(int index) const { return GET_DATA(predcount_t, data_, operationList_.predCountBase, index); }
    predcount_t &GetOperationCurrPredCount(int index) { return GET_DATA(predcount_t, data_, operationList_.predCountBase, index); }

    uint32_t GetStitchSize() const { return operationList_.stitchCount; }
    const DevAscendFunctionDuppedStitchList &GetStitch(int index) const { return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index); }
    DevAscendFunctionDuppedStitchList &GetStitch(int index) { return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, index); }

    uint64_t GetExpressionSize() const { return expressionList_.size; }
    const uint64_t &GetExpression(int index) const { return GET_DATA(uint64_t, data_, expressionList_.base, index); }
    uint64_t &GetExpression(int index) { return GET_DATA(uint64_t, data_, expressionList_.base, index); }

    uint64_t *GetExpressionAddr() const {
        return &GET_DATA(uint64_t, data_, expressionList_.base, 0);
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

    const DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex, bool maybeNull = true) const {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        DEV_IF_NONDEVICE {
            DEV_ASSERT(maybeNull || outcastStitchIndex != 0);
        }
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }
    DevAscendFunctionDuppedStitchList &GetOperationStitch(int operationIndex, bool maybeNull = true) {
        int outcastStitchIndex = GetSource()->GetOperationOutcastStitchIndex(operationIndex);
        DEV_IF_NONDEVICE {
            DEV_ASSERT(maybeNull || outcastStitchIndex != 0);
        }
        return GET_DATA(DevAscendFunctionDuppedStitchList, data_, operationList_.stitchBase, outcastStitchIndex);
    }

    inline uint64_t GetIncastDataSize(int incastIndex) const {
        auto rawTensor = GetSource()->GetIncastRawTensor(incastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    inline uint64_t GetOutcastDataSize(int outcastIndex) const {
        auto rawTensor = GetSource()->GetOutcastRawTensor(outcastIndex);
        auto size = rawTensor->GetMemoryRequirement(GetExpressionAddr());
        return size;
    }

    schema::range SchemaGetIncastRange(int arg) const {
        auto base = GetIncastAddress(arg).GetAddress();
        auto size = GetIncastDataSize(arg);
        return schema::Range(base, base + size);
    }
    schema::range SchemaGetOutcastRange(int arg) const {
        auto base = GetOutcastAddress(arg).GetAddress();
        auto size = GetOutcastDataSize(arg);
        return schema::Range(base, base + size);
    }

    schema::RActWorkspace SchemaGetWorkspace() const {
        auto workspaceBegin = GetRuntimeWorkspace();
        auto workspaceEnd = GetRuntimeWorkspace() + GetSource()->rawTensorWsMemoryRequirement;
        return schema::RActWorkspace(schema::Range(workspaceBegin, workspaceEnd));
    }

    std::string Dump(int indent = 0) const {
        DEV_ASSERT(GetSource()->GetOperationSize() == GetOperationSize());
        std::string INDENT(indent, ' ');
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');

        std::ostringstream oss;
        oss << INDENT << "DevFunctionDupped " << GetSource()->GetFuncKey() << " {\n";
        for (size_t incastIndex = 0; incastIndex < GetIncastSize(); incastIndex++) {
            oss << INDENTINNER << "#incast:" << incastIndex << " = " << GetIncastAddress(incastIndex).Dump() << "\n";
        }
        for (size_t outcastIndex = 0; outcastIndex < GetOutcastSize(); outcastIndex++) {
            oss << INDENTINNER << "#outcast:" << outcastIndex << " = " << GetOutcastAddress(outcastIndex).Dump() << "\n";
        }
        for (size_t operationIndex = 0; operationIndex < GetOperationSize(); operationIndex++) {
            oss << INDENTINNER << "!" << operationIndex;
            oss << " #pred:" << GetSource()->GetOperationDepGraphPredCount(operationIndex);
            oss << " #succ:[";
            size_t succSize;
            auto succList = GetSource()->GetOperationDepGraphSuccAddr(operationIndex, succSize);
            for (size_t j = 0; j < succSize; j++) {
                oss << Delim(j != 0, ",") << "[" << j << "]=!" << succList[j];
            }
            oss << "]";
            oss << " #dynpred:" << GetOperationCurrPredCount(operationIndex);
            oss << " #dynsucc:" << GetOperationStitch(operationIndex).Dump();
            oss << "\n";
        }
        oss << INDENTINNER << "#expr:[";
        for (size_t exprIndex = 0; exprIndex < GetExpressionSize(); exprIndex++) {
            oss << Delim(exprIndex != 0, ",") << "[" << exprIndex << "]=" << GetExpression(exprIndex);
        }
        oss << "]";
        oss << INDENT << "}\n";
        return oss.str();
    }
};
const uint32_t RAW_TENSOR_OFFSET_SIZE = 63;
const uint32_t RAW_TENSOR_DESC_PRE_SIZE = 8;

struct DevAscendFunctionDupped {
    DevAscendFunctionDupped() = default;
    explicit DevAscendFunctionDupped(WsAllocation tinyAlloc) : dupTiny_(tinyAlloc) {}

    static DevAscendFunctionDupped DuplicateRoot(DevAscendFunction *func, WsAllocation tinyAlloc) {
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

    void ReleaseDuppedMemory(WsMetadataAllocator &allocator) {
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

    inline const uint64_t &GetExpression(int arg) const { return DupData()->GetExpression(arg); };
    inline uint64_t &GetExpression(int arg) { return DupData()->GetExpression(arg); };
    inline uint64_t GetExpressionSize() const { return DupData()->GetExpressionSize(); }
    inline uint64_t *GetExpressionAddr() const { return DupData()->GetExpressionAddr(); }

    inline auto GetOperationSize() const { return DupData()->GetOperationSize(); }
    inline const predcount_t &GetOperationCurrPredCount(int arg) const { return DupData()->GetOperationCurrPredCount(arg); };
    inline predcount_t &GetOperationCurrPredCount(int arg) { return DupData()->GetOperationCurrPredCount(arg); };
    inline const auto &GetOperationStitch(int arg, bool maybeNull = true) const { return DupData()->GetOperationStitch(arg, maybeNull); };
    inline auto &GetOperationStitch(int arg, bool maybeNull = true) { return DupData()->GetOperationStitch(arg, maybeNull); };

    inline AddressDescriptor GetIncastAddress(int arg) const { return DupData()->GetIncastAddress(arg); };
    inline AddressDescriptor &GetIncastAddress(int arg) { return DupData()->GetIncastAddress(arg); };

    inline AddressDescriptor GetOutcastAddress(int arg) const { return DupData()->GetOutcastAddress(arg); };
    inline AddressDescriptor &GetOutcastAddress(int arg) { return DupData()->GetOutcastAddress(arg); };

    schema::expr SchemaGetExpressionTable() const {
        std::vector<schema::Int64Type> exprTable;
        uint64_t *exprAddr = GetExpressionAddr();
        uint64_t exprSize = GetExpressionSize();
        for (uint64_t i = 0; i < exprSize; i++) {
            exprTable.push_back(exprAddr[i]);
        }
        return schema::expr(exprTable);
    }

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
        func->GetTensorOffsetAndShape<false>(offset, shape, &GetExpression(0), dims, operationIndex, operandIndex, isIOperand);
    }

    std::string Dump(int indent = 0) const {
        return DupData()->Dump(indent);
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
        for (size_t idx = static_cast<size_t>(offset); idx < func->GetOperationAttrSize(operIdx); idx++) {
            oss << GetValue(attrBase, idx) << ", ";
        }
        return oss.str();
    }

    void DumpTopo(std::ofstream &os, int seqNo, int funcIdx, const DevCceBinary *cceBinary) {
        auto func = GetSource();
        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            auto &cceInfo = cceBinary[func->GetOperationAttrCalleeIndex(opIdx)];
            os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," << func->funcKey << "," << func->rootHash << ","
               <<func->GetOperationDebugOpmagic(opIdx) << "," << func->GetOperationAttrCalleeIndex(opIdx) << ","
               << cceInfo.funcHash << "," << cceInfo.coreType << "," << cceInfo.psgId << ",";
            auto &succList = func->GetOperationDepGraphSuccList(opIdx);
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

#if DEBUG_INFINITE_LIFETIME
    void DumpTensorAddrInfo(std::vector<std::string> &infos, uint32_t seqNo, uint32_t funcIdx) {
        // seqNo,taskId,rawMagic,address,dtype,bytesOfDtype,(shapes,)
        auto *srcFunc = GetSource();

        auto dumpOperand = [&](const DevAscendOperationOperandInfo &operandInfo, size_t opIdx) {
            std::stringstream os;
            uint64_t rawIdx = srcFunc->GetTensor(operandInfo.tensorIndex)->rawIndex;
            auto *rawTensor = srcFunc->GetRawTensor(rawIdx);
            os << seqNo << "," << MakeTaskID(funcIdx, opIdx) << "," <<
                rawTensor->rawMagic << "," <<
                GetRawTensorAddrEx(rawIdx) << "," <<
                BriefDataType2String(rawTensor->dataType) << "," <<
                BytesOf(rawTensor->dataType);

            uint32_t dimSize = rawTensor->GetDim();
            os << ",(";
            bool isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    os << ",";
                }
                os << rawTensor->shape.At(i, GetExpressionAddr());
            }
            os << ")";

            os << "\n";
            infos.emplace_back(std::move(os).str());
        };

        for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
            for (size_t iopIdx = 0; iopIdx < srcFunc->GetOperationIOperandSize(opIdx); iopIdx++) {
                auto &iopInfo = srcFunc->GetOperationIOperandInfo(opIdx, iopIdx);
                dumpOperand(iopInfo, opIdx);
            }
            for (size_t oopIdx = 0; oopIdx < srcFunc->GetOperationOOperandSize(opIdx); oopIdx++) {
                auto &oopInfo = srcFunc->GetOperationOOperandInfo(opIdx, oopIdx);
                dumpOperand(oopInfo, opIdx);
            }
        }
    }
#endif // DEBUG_INFINITE_LIFETIME

    // Return result lines
    std::vector<std::string> DumpLeafs(uint32_t seqNo, uint32_t funcIdx) {
        std::vector<std::string> lines;
        std::stringstream oss;
        auto flushStream = [&] {
            lines.push_back(std::move(oss).str());
            oss.clear();
            oss.str("");
        };

        auto *srcFunc = GetSource();

        oss << "seqNo=" << seqNo << ", rootHash=" << srcFunc->rootHash;
        flushStream();

        auto dumpRawShape = [&](DevAscendRawTensor *rawTensor, uint32_t dimSize) {
            oss << "        rawShape=[";
            bool isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    oss << ", ";
                }
                oss << rawTensor->shape.At(i, GetExpressionAddr());
            }
            oss << "]";
            flushStream();
        };

        auto dumpOperandShape = [&](uint32_t dimSize, size_t opIdx, size_t operandIdx, bool isIn) {
            uint64_t offset[DEV_SHAPE_DIM_MAX];
            uint64_t shape[DEV_SHAPE_DIM_MAX];
            GetTensorOffsetAndShape(offset, shape, dimSize, opIdx, operandIdx, isIn);

            oss << "          offset=[";
            bool isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    oss << ", ";
                }
                oss << offset[i];
            }
            oss << "]";
            flushStream();

            oss << "           shape=[";
            isFirstDim = true;
            for (uint32_t i = 0; i < dimSize; i++) {
                if (isFirstDim) {
                    isFirstDim = false;
                } else {
                    oss << ", ";
                }
                oss << shape[i];
            }
            oss << "]";
            flushStream();
        };

        for (size_t opIdx = 0; opIdx < srcFunc->GetOperationSize(); opIdx++) {
            size_t iopNum = srcFunc->GetOperationIOperandSize(opIdx);
            size_t oopNum = srcFunc->GetOperationOOperandSize(opIdx);
            oss << "> taskId = " << MakeTaskID(funcIdx, opIdx) << ", opIdx=" << opIdx << ", #iop=" << iopNum << ", #oop=" << oopNum;
            flushStream();

            for (size_t iopIdx = 0; iopIdx < iopNum; iopIdx++) {
                uint64_t rawIdx = srcFunc->GetOperationIOperand(opIdx, iopIdx)->rawIndex;
                auto *rawTensor = srcFunc->GetRawTensor(rawIdx);

                oss << "    iop [" << std::setw(3) << iopIdx << "]: rawMagic=" << rawTensor->rawMagic
                    << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
                flushStream();

                uint32_t dimSize = rawTensor->GetDim();
                dumpOperandShape(dimSize, opIdx, iopIdx, true);
                dumpRawShape(rawTensor, dimSize);
            }

            for (size_t oopIdx = 0; oopIdx < oopNum; oopIdx++) {
                uint64_t rawIdx = srcFunc->GetOperationOOperand(opIdx, oopIdx)->rawIndex;
                auto *rawTensor = srcFunc->GetRawTensor(rawIdx);

                oss << "    oop [" << std::setw(3) << oopIdx << "]: rawMagic=" << rawTensor->rawMagic
                    << ", addr=0x" << std::hex << GetRawTensorAddrEx(rawIdx) << std::dec;
                flushStream();

                uint32_t dimSize = rawTensor->GetDim();
                dumpOperandShape(dimSize, opIdx, oopIdx, false);
                dumpRawShape(rawTensor, dimSize);
            }
        }

        return lines;
    }

    std::string DumpDyn(int funcIdx, const DevCceBinary *cceBinary) {
        std::stringstream oss;
        auto func = GetSource();

        for (size_t opIdx = 0; opIdx < DupData()->GetSource()->GetOperationSize(); opIdx++) {
            oss << std::hex << "[" << opIdx << "] #predCnt:" << GetOperationCurrPredCount(opIdx);
            auto &succList = func->GetOperationDepGraphSuccList(opIdx);
            oss << " #succList: [";
            for (size_t j = 0; j < succList.size(); j++) {
                if (j != 0)
                    oss << ", ";
                oss << func->At(succList, j);
            }
            oss << ']';
            auto &stitch = GetOperationStitch(opIdx);
            if (!stitch.IsNull())
                oss << std::hex << " #stitch:" << stitch.Dump();
            oss << "\n";
        }

        auto dumpAttr = [this, &oss, func](const SymInt *attrs, const auto &info) {
            int attrOffset = info.staticOffsetAttrBeginIndex;
            auto rawIndex = attrs[attrOffset - 1].Value();
            oss << "@(rawidx:" << rawIndex << " attridx:" << (attrOffset - 1) << ")" << ", ";

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
                << " #taskID:" << MakeTaskID(funcIdx, operIdx) << " #opMagic: " << func->GetOperationDebugOpmagic(operIdx)
                << "\n";
            oss << "  #invokeAttrs : ";
            int offset = 0;
            for (size_t idx = 0; idx < func->GetOperationIOperandSize(operIdx); idx++) {
                auto &opInfo = func->GetOperationIOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            oss << " in:";
            dumpAttr(attrBase, opInfo);
            }
            for (size_t idx = 0; idx < func->GetOperationOOperandSize(operIdx); idx++) {
                auto &opInfo = func->GetOperationOOperandInfo(operIdx, idx);
                offset = std::max(offset, opInfo.staticOffsetAttrBeginIndex + ARG_ATTR_TYPE * opInfo.GetDim());
            oss << " out:";
            dumpAttr(attrBase, opInfo);
            }
            oss << "\n other attr:";
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
            auto desc = funcData->rawTensorDesc[i];
            oss << GetRawTensorAddrEx(i) << "(location:" << desc.location << " offsetOrIdex: " << desc.offsetOrIndex << ")" << ", ";
        }
        oss << "\n]";
        return oss.str();
    }

    bool IsNull() const { return !dupTiny_; }
    void ResetNull() { dupTiny_.Invalidate(); }
    DynFuncData *GetFuncData() { return funcData; }
    void SetFuncData(DynFuncData *data) { funcData = data; }

    DevAscendFunctionDuppedData *DupDataForDynFuncData() { return DupData(); }
private:
    const DevAscendFunctionDuppedData *DupData() const { return dupTiny_.As<DevAscendFunctionDuppedData>(); }
    DevAscendFunctionDuppedData *DupData() { return dupTiny_.As<DevAscendFunctionDuppedData>(); }

private:
    DynFuncData *funcData{nullptr}; // used by aicore
    WsAllocation dupTiny_;
};

void EncodeDevAscendFunction(Function *dyndev, const EncodeDevAscendFunctionParam &param, uint64_t &offset, DevAscendFunction *base);

struct DevAscendProgramSymbol {
    DevRelocVector<char> name;
    uint64_t index;
};

struct DevAscendProgramPartialUpdate {
    int slotIndex;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTable; // devtaskid | taskid

    bool Empty() const {
        return cellMatchRuntimePartialUpdateTable.size() == 0;
    }
};

constexpr size_t READY_QUEUE_SIZE = 3UL;

struct ReadyQueueCache {
    uint32_t coreFunctionCnt;
    struct Queue {
        uint32_t head;
        uint32_t tail;
        uint32_t capacity;
        uint32_t *elem;
    } queueList[READY_QUEUE_SIZE];
    uint32_t readyTaskNum;
};

inline constexpr size_t MAX_CACHED_FUNC_NUM = 128;
struct DynFuncDataCache {
    DevAscendFunction *devFunc;
    predcount_t *predCount;
    int *calleeList;
#ifdef SUPPORT_WRAP
    int *opWrapList;
    int *opWrapTaskNumList;
#endif
    DevAscendFunctionDuppedData *duppedData;

    const DynFuncDataCache &At(size_t index) const { return this[index]; }
    DynFuncDataCache &At(size_t index) { return this[index]; }
};

struct DynFuncDataWorkspaceAddressBackup {
    uint64_t runtimeWorkspace;
    uint64_t runtimeOutcastWorkspace;
    uint64_t workspaceAddr;
    uint64_t stackWorkspaceAddr;
};

struct DynFuncDataBackup {
    predcount_t *predCountBackup;
    uint64_t *rawTensorAddrBackup;

    DynFuncDataWorkspaceAddressBackup workspaceAddressBackup;

    const DynFuncDataBackup &At(size_t index) const { return this[index]; }
    DynFuncDataBackup &At(size_t index) { return this[index]; }
};

struct DynDeviceTaskBase {
    DeviceTask devTask;
    DynFuncHeader* dynFuncDataList{nullptr};

    ReadyCoreFunctionQueue *readyQueue[READY_QUEUE_SIZE];
    DynFuncDataCache dynFuncDataCacheList[MAX_CACHED_FUNC_NUM];
    uint64_t dynFuncDataCacheListSize;

    const DevCceBinary *cceBinary;
    const DevAicpuLeafBinary *aicpuLeafBinary;

    ReadyQueueCache *readyQueueBackup;
    DynFuncDataBackup dynFuncDataBackupList[MAX_CACHED_FUNC_NUM];
    bool isLastTask{false};

    DynFuncHeader *GetDynFuncDataList() const { return dynFuncDataList; }
    DynFuncHeader *GetDynFuncDataList() { return dynFuncDataList; }
    const DynFuncDataCache *GetDynFuncDataCacheList() const { return dynFuncDataCacheList; }
    DynFuncDataCache *GetDynFuncDataCacheList() { return dynFuncDataCacheList; }

    uint64_t GetIndex() { return GetDynFuncDataList()->GetIndex(); }
    inline bool IsLastTask() const { return isLastTask;}
    void SetLastTask(bool b) { isLastTask = b;}
};
#define DYN_DEVICE_TASK_EXT_SIZE 0x300


struct DeviceTaskCache {
    DynDeviceTaskBase *dynTaskBase;
};

#define INVALID_STITCH_IDX      (static_cast<uint32_t>(-1))

struct DeviceExecuteSlot {
    AddressDescriptor desc;
    bool isOutputSlot{false};
    bool isAssembleSlot{false};
    bool isPartialUpdateStitch{false};
    bool isPartialUpdateDirty{false};
    int64_t refCntIndex{itemPoolInvalidIndex}; // refCnt to stored tensor
    uint32_t stitchDupIdx{INVALID_STITCH_IDX};
    uint32_t stitchOutcastIdx;

    DevAscendProgramPartialUpdate *partialUpdate{nullptr};
    bool IsFixedAddress() const {
        return isOutputSlot || isAssembleSlot;
    }

    bool RefCntIsNull() {
        return refCntIndex == itemPoolInvalidIndex;
    }

    void RefCntReset() {
        refCntIndex = itemPoolInvalidIndex;
    }

    template<typename T>
    void RefCntCopyFrom(T &info) {
        refCntIndex = info.refCntIndex;
    }

    template <WsMemCategory category>
    bool RefCntDec(ItemPool<uint32_t, category> &pool) {
        DEV_DEBUG_ASSERT(refCntIndex != itemPoolInvalidIndex);
        --pool.At(refCntIndex);
        if (pool.At(refCntIndex) == 0) {
            pool.DestroyAt(refCntIndex);
            RefCntReset();
            return true;
        }
        return false;
    }

    template <WsMemCategory category>
    void RefCntInc(ItemPool<uint32_t, category> &pool, uint32_t count) {
        pool.At(refCntIndex) += count;
    }
};

struct DevProgramControlFlowCacheRuntime {
    struct DeviceWorkspaceAllocator {
        struct {
            SeqWsAllocator dassembleDests;
            SeqWsAllocator rootInner;
            SeqWsAllocator devTaskInnerOutcasts;
            WsSlotAllocator slottedOutcasts;
            DevRelocVector<WsSlotAllocator::BlockHeader> slottedOutcastsBlockList;
        } tensorAllocators;
    } workspace;
    struct DeviceSlotContext {
        DevRelocVector<DeviceExecuteSlot> slotList;
        DevRelocVector<uint32_t> slotRefCntList;
    } slotContext;
};

template<typename T>
inline T *RelocControlFlowCachePointer(T *&ptrRef, const RelocRange &relocProgram) {
    T *result = nullptr;
    if (relocProgram.GetDst() == 0) {
        result = ptrRef;
        relocProgram.Reloc(ptrRef);
    } else {
        relocProgram.Reloc(ptrRef);
        result = ptrRef;
    }
    return result;
}

struct DevProgramControlFlowCache {
    /* Filled by user, true means try to allocate in cache. */
    bool isRecording;
    /* Filled by user, true means activate in cache. */
    bool isActivated;

    /* Filled in caching */
    DevRelocVector<DevTensorData> inputTensorDataList;
    /* Filled in caching */
    DevRelocVector<DevTensorData> outputTensorDataList;
    /* Filled in caching for runtime */
    DevProgramControlFlowCacheRuntime runtimeBackup;

    /* Filled in caching, true means some metadata is not cached. */
    bool isRecordingStopped;
    /* Filled in caching */
    uint64_t deviceTaskCount;
    /* Filled in caching */
    uint64_t rootTaskCount;
    /* Filled in caching */
    uint64_t cacheDataOffset;
    /* Filled in caching */
    uint64_t deviceTaskSkippedCount;
    /* Filled in caching */
    uint64_t contextWorkspaceAddr;
    /* Filled in caching */
    DevRelocVector<DeviceTaskCache> deviceTaskCacheList;
    /* Filled in caching */
    DevRelocVector<uint8_t> cacheData;

    bool inline IsRecording() const {
        if (IsDeviceMode()) {
            return false;
        }
        return isRecording;
    }

    bool inline IsRecordingStopped() const {
        return isRecordingStopped;
    }

    void inline StopRecording() {
        isRecordingStopped = true;
    }

#define CFGCACHE_ALIGN      8
    void *AllocateCache(uint64_t size) {
        void *result = nullptr;
        if (cacheDataOffset + size < cacheData.size()) {
            result = &cacheData[cacheDataOffset];
            /* make cache 8 byte aligned */
            cacheDataOffset += (size + CFGCACHE_ALIGN - 1) / CFGCACHE_ALIGN * CFGCACHE_ALIGN;
        } else {
            isRecordingStopped = true;
        }
        return result;
    }

    bool AppendDeviceTask(DynDeviceTaskBase *base) {
        if (!isRecordingStopped && (deviceTaskCount < deviceTaskCacheList.size())) {
            deviceTaskCacheList[deviceTaskCount].dynTaskBase = base;
            deviceTaskCount += 1;
            rootTaskCount += base->dynFuncDataList->Size();
            return true;
        } else {
            deviceTaskSkippedCount += 1;
            return false;
        }
    }

    void InitInputOutput(DevStartArgsBase *startArgs) {
        for (size_t i = 0; i < inputTensorDataList.size(); i++) {
            inputTensorDataList[i] = startArgs->GetInputTensor(i);
        }
        for (size_t i = 0; i < outputTensorDataList.size(); i++) {
            outputTensorDataList[i] = startArgs->GetOutputTensor(i);
        }
    }

    void MatchInputOutputDump(DevStartArgsBase *startArgs) const {
        DEV_VERBOSE_DEBUG("matchio cache input size: %d", (int)inputTensorDataList.size());
        for (size_t k = 0; k < inputTensorDataList.size(); k++) {
            DEV_VERBOSE_DEBUG("matchio cache input %d: %s", (int)k, DevAscendFunction::DumpShape(inputTensorDataList[k].shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio cache output size: %d", (int)outputTensorDataList.size());
        for (size_t k = 0; k < outputTensorDataList.size(); k++) {
            DEV_VERBOSE_DEBUG("matchio cache output %d: %s", (int)k, DevAscendFunction::DumpShape(outputTensorDataList[k].shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio real input size: %d", (int)startArgs->inputTensorSize);
        for (size_t k = 0; k < startArgs->inputTensorSize; k++) {
            DEV_VERBOSE_DEBUG("matchio real input %d: %s", (int)k, DevAscendFunction::DumpShape(startArgs->GetInputTensor(k).shape).c_str());
        }

        DEV_VERBOSE_DEBUG("matchio real output size: %d", (int)startArgs->outputTensorSize);
        for (size_t k = 0; k < startArgs->outputTensorSize; k++) {
            DEV_VERBOSE_DEBUG("matchio real output %d: %s", (int)k, DevAscendFunction::DumpShape(startArgs->GetOutputTensor(k).shape).c_str());
        }
    }

    inline bool MatchInputOutput(DevStartArgsBase *startArgs) const {
        MatchInputOutputDump(startArgs);

        if (inputTensorDataList.size() != startArgs->inputTensorSize) {
            return false;
        }
        if (outputTensorDataList.size() != startArgs->outputTensorSize) {
            return false;
        }
        for (size_t k = 0; k < inputTensorDataList.size(); k++) {
            if (!inputTensorDataList[k].shape.Equal(startArgs->GetInputTensor(k).shape)) {
                return false;
            }
        }
        for (size_t k = 0; k < outputTensorDataList.size(); k++) {
            if (!outputTensorDataList[k].shape.Equal(startArgs->GetOutputTensor(k).shape)) {
                return false;
            }
        }
        return true;
    }

    inline bool IsActivatedFullCache(DevStartArgsBase *startArgs) const {
        if (!isActivated) {
            return false;
        }
        if (deviceTaskSkippedCount != 0) {
            return false;
        }
        if (!MatchInputOutput(startArgs)) {
            return false;
        }
        return true;
    }

    inline bool IsActivatedPartialCache(DevStartArgsBase *startArgs) const {
        if (!isActivated) {
            return false;
        }
        if (deviceTaskCount == 0) {
            return false;
        }
        if (!MatchInputOutput(startArgs)) {
            return false;
        }
        return true;
    }

    void PredCountDataBackup(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

            predcount_t *predCountBackup = reinterpret_cast<predcount_t *>(AllocateCache(backupSize));
            if (predCountBackup == nullptr) {
                return;
            }
            dynDataBackup->predCountBackup = predCountBackup;

            memcpy_s(dynDataBackup->predCountBackup, backupSize, &duppedData->GetOperationCurrPredCount(0), backupSize);
        }
    }

    void PredCountDataRestore(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(predcount_t) * duppedData->GetOperationSize();

            memcpy_s(&duppedData->GetOperationCurrPredCount(0), backupSize, dynDataBackup->predCountBackup, backupSize);
        }
    }

    void ReadyQueueDataBackup(DynDeviceTaskBase *base) {
        ReadyQueueCache *readyQueueBackup = reinterpret_cast<ReadyQueueCache *>(AllocateCache(sizeof(ReadyQueueCache)));
        if (readyQueueBackup == nullptr) {
            return;
        }
        readyQueueBackup->coreFunctionCnt = base->devTask.coreFunctionCnt;
        uint32_t readyTaskNum = 0;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->capacity;
            uint32_t *readyQueueBackupElem = reinterpret_cast<uint32_t *>(AllocateCache(backupSize));
            if (readyQueueBackupElem == nullptr) {
                return;
            }

            readyQueueBackup->queueList[i].head = base->readyQueue[i]->head;
            readyQueueBackup->queueList[i].tail = base->readyQueue[i]->tail;
            readyQueueBackup->queueList[i].capacity = base->readyQueue[i]->capacity;
            readyQueueBackup->queueList[i].elem = readyQueueBackupElem;
            memcpy_s(readyQueueBackup->queueList[i].elem, backupSize, base->readyQueue[i]->elem, backupSize);

            readyTaskNum += base->readyQueue[i]->tail - base->readyQueue[i]->head;
        }
        readyQueueBackup->readyTaskNum = readyTaskNum;
        base->readyQueueBackup = readyQueueBackup;
    }

    void ReadyQueueDataRestore(DynDeviceTaskBase *base) {
        ReadyQueueCache *readyQueueBackup = base->readyQueueBackup;
        base->devTask.coreFunctionCnt = readyQueueBackup->coreFunctionCnt;
        for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
            size_t backupSize = sizeof(uint32_t) * base->readyQueue[i]->capacity;

            base->readyQueue[i]->head = readyQueueBackup->queueList[i].head;
            base->readyQueue[i]->tail = readyQueueBackup->queueList[i].tail;
            memcpy_s(base->readyQueue[i]->elem, backupSize, readyQueueBackup->queueList[i].elem, backupSize);
        }
    }

    static void RelocBuildInputOutputDesc(
            std::unordered_map<uint64_t, AddressDescriptor> &cacheInputOutputDict,
            DevStartArgsBase *devStartArgs) {
        for (uint64_t i = 0; i < devStartArgs->inputTensorSize; i++) {
            uint64_t addr = devStartArgs->GetInputTensor(i).address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_INPUT, i);
        }
        for (uint64_t i = 0; i < devStartArgs->outputTensorSize; i++) {
            uint64_t addr = devStartArgs->GetOutputTensor(i).address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_OUTPUT, i);
        }
    }

    static void RelocBuildInputOutputDesc(
            std::unordered_map<uint64_t, AddressDescriptor> &cacheInputOutputDict,
            DevRelocVector<DevTensorData> inputTensorDataList,
            DevRelocVector<DevTensorData> outputTensorDataList) {
        for (uint64_t i = 0; i < inputTensorDataList.size(); i++) {
            uint64_t addr = inputTensorDataList[i].address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_INPUT, i);
        }
        for (uint64_t i = 0; i < outputTensorDataList.size(); i++) {
            uint64_t addr = outputTensorDataList[i].address;
            cacheInputOutputDict[addr] = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_OUTPUT, i);
        }
    }

    static void RelocDescToCache(
            AddressDescriptor &desc,
            const RelocRange &relocWorkspace,
            std::unordered_map<uint64_t, AddressDescriptor> &cacheInputOutputDict) {
        AddressDescriptor resultDesc;
        uint64_t addr = desc.GetAddressValue();
        if (cacheInputOutputDict.count(addr)) {
            resultDesc = cacheInputOutputDict[addr];
        } else {
            relocWorkspace.Reloc(addr);
            resultDesc = AddressDescriptor::MakeCache(ADDRESS_CACHE_KIND_WORKSPACE, addr);
        }
        desc = resultDesc;
    }

    static void RelocDescFromCache(
            AddressDescriptor &desc,
            const RelocRange &relocWorkspace,
            DevStartArgsBase *devStartArgs) {
        uint64_t resultAddr = 0;
        switch (desc.cacheKind) {
            case ADDRESS_CACHE_KIND_WORKSPACE:
                resultAddr = desc.cacheValue;
                relocWorkspace.Reloc(resultAddr);
                break;
            case ADDRESS_CACHE_KIND_INPUT:
                resultAddr = devStartArgs->GetInputTensor(desc.cacheValue).address;
                break;
            case ADDRESS_CACHE_KIND_OUTPUT:
                resultAddr = devStartArgs->GetOutputTensor(desc.cacheValue).address;
                break;
            default:
                DEV_ERROR("[RelocDescFromCache] Invalid kind: %lu\n", (unsigned long)desc.cacheKind);
                break;
        }
        AddressDescriptor resultDesc = AddressDescriptor::MakeAddress(resultAddr);
        desc = resultDesc;
    }

    void IncastOutcastAddrBackup(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

            uint64_t *rawTensorAddrBackup = reinterpret_cast<uint64_t *>(AllocateCache(backupSize));
            if (rawTensorAddrBackup == nullptr) {
                return;
            }
            dynDataBackup->rawTensorAddrBackup = rawTensorAddrBackup;
            memcpy_s(dynDataBackup->rawTensorAddrBackup, backupSize, dynData->rawTensorAddr, backupSize);
        }
    }

    void IncastOutcastAddrRestore(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
            size_t backupSize = sizeof(uint64_t) * (duppedData->GetIncastSize() + duppedData->GetOutcastSize());

            memcpy_s(dynData->rawTensorAddr, backupSize, dynDataBackup->rawTensorAddrBackup, backupSize);
        }
    }

    void IncastOutcastAddrRestore() {
        for (size_t i = 0; i < deviceTaskCount; i++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
            IncastOutcastAddrRestore(dynTaskBase);
        }
    }

    void TaskAddrBackupWorkspace(DynDeviceTaskBase * base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;

            dynDataBackup->workspaceAddressBackup.runtimeWorkspace = duppedData->runtimeWorkspace_;
            dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace = duppedData->runtimeOutcastWorkspace_;
            dynDataBackup->workspaceAddressBackup.workspaceAddr = dynData->workspaceAddr;
            dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr = dynData->stackWorkSpaceAddr;
        }
    }

    void TaskAddrRestoreWorkspace(DynDeviceTaskBase *base) {
        DynFuncHeader *dynFuncDataList = base->GetDynFuncDataList();
        DynFuncDataCache *dynFuncDataCacheList = base->dynFuncDataCacheList;
        DynFuncDataBackup *dynFuncDataBackupList = base->dynFuncDataBackupList;
        for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); ++dupIndex) {
            DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
            DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
            DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);
            DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;

            duppedData->runtimeWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeWorkspace;
            duppedData->runtimeOutcastWorkspace_ = dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace;
            dynData->workspaceAddr = dynDataBackup->workspaceAddressBackup.workspaceAddr;
            dynData->stackWorkSpaceAddr = dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr;
        }
    }

    void TaskAddrRestoreWorkspace() {
        for (size_t i = 0; i < deviceTaskCount; i++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[i].dynTaskBase;
            TaskAddrRestoreWorkspace(dynTaskBase);
        }
    }

    void TaskAddrRelocWorkspace(
            uint64_t srcWorkspace, uint64_t dstWorkspace,
            DevStartArgsBase *devStartArgs) {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;

            DynFuncHeader *dynFuncDataList = dynTaskBase->dynFuncDataList;
            DynFuncDataCache *dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup *dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
                DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                if (devStartArgs == nullptr) {
                    // Host: addr uses backup

                    // Reloc Dupped
                    relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeWorkspace);
                    relocWorkspace.RelocNullable(dynDataBackup->workspaceAddressBackup.runtimeOutcastWorkspace);

                    // Reloc DynFuncData
                    relocWorkspace.Reloc(dynDataBackup->workspaceAddressBackup.workspaceAddr);
                    relocWorkspace.Reloc(dynDataBackup->workspaceAddressBackup.stackWorkspaceAddr);
                } else {
                    // Device: addr uses actual

                    // Reloc Dupped
                    relocWorkspace.RelocNullable(duppedData->runtimeWorkspace_);
                    relocWorkspace.RelocNullable(duppedData->runtimeOutcastWorkspace_);

                    // Reloc DynFuncData
                    relocWorkspace.Reloc(dynData->workspaceAddr);
                    relocWorkspace.Reloc(dynData->stackWorkSpaceAddr);
                }
            }
        }
    }

    void IncastOutcastAddrReloc(
            uint64_t srcWorkspace, uint64_t dstWorkspace,
            DevStartArgsBase *devStartArgs) {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        /* empty constructor's overhead should be negligible */
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        if (devStartArgs == nullptr) {
            /* only run on host */
            RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
        }
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            DynDeviceTaskBase *dynTaskBase = deviceTaskCacheList[deviceIndex].dynTaskBase;
            DynFuncHeader *dynFuncDataList = dynTaskBase->dynFuncDataList;
            DynFuncDataCache *dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup *dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;
            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                DevAscendFunctionDuppedData *duppedData = dynDataCache->duppedData;
                if (devStartArgs == nullptr) {
                    // Host: addr uses backup
                    for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                        AddressDescriptor *addr = reinterpret_cast<AddressDescriptor *>(dynDataBackup->rawTensorAddrBackup + i);
                        RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                    }
                    for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                        AddressDescriptor *addr = reinterpret_cast<AddressDescriptor *>(dynDataBackup->rawTensorAddrBackup + duppedData->GetIncastSize() + i);
                        RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                    }
                } else {
                    // Device: addr uses actual
                    for (uint64_t i = 0; i < duppedData->GetIncastSize(); i++) {
                        AddressDescriptor *addr = &duppedData->GetIncastAddress(i);
                        RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                    }
                    for (uint64_t i = 0; i < duppedData->GetOutcastSize(); i++) {
                        AddressDescriptor *addr = &duppedData->GetOutcastAddress(i);
                        RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                    }
                }

                dynData->startArgs = devStartArgs;
            }
        }
    }

    void RuntimeAddrBackup(DeviceExecuteSlot *runtimeSlotList, uint32_t *runtimeSlotRefCntList, uint32_t slotSize, TensorAllocator &allocator) {
        uint32_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
        uint32_t slotRefCntDataSize = sizeof(uint32_t) * slotSize;
        memcpy_s(runtimeBackup.slotContext.slotList.Data(), slotDataSize, runtimeSlotList, slotDataSize);
        memcpy_s(runtimeBackup.slotContext.slotRefCntList.Data(), slotRefCntDataSize, runtimeSlotRefCntList, slotRefCntDataSize);

        struct Backup {
            static void BackupBlockHeader(WsSlotAllocator::BlockHeader *&ptr, WsSlotAllocator::BlockHeader *base) {
                ptr = reinterpret_cast<WsSlotAllocator::BlockHeader *>(static_cast<uintptr_t>(ptr - base));
            }
        };
        runtimeBackup.workspace.tensorAllocators.dassembleDests = allocator.dassembleDests;
        runtimeBackup.workspace.tensorAllocators.rootInner = allocator.rootInner;
        runtimeBackup.workspace.tensorAllocators.devTaskInnerOutcasts = allocator.devTaskInnerOutcasts;
        runtimeBackup.workspace.tensorAllocators.slottedOutcasts = allocator.slottedOutcasts;

        uint64_t backupSize = sizeof(WsSlotAllocator::BlockHeader) * allocator.slottedOutcasts.slotNum_;
        memcpy_s(runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.Data(), backupSize, allocator.slottedOutcasts.GetBlockHeaderBase(), backupSize);

        WsSlotAllocator::BlockHeader *base = allocator.slottedOutcasts.GetBlockHeaderBase();
        Backup::BackupBlockHeader(runtimeBackup.workspace.tensorAllocators.slottedOutcasts.freeListHeader_, base);
        Backup::BackupBlockHeader(runtimeBackup.workspace.tensorAllocators.slottedOutcasts.notInUseHeaders_, base);
        WsSlotAllocator::BlockHeader *checkpointBase = runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.Data();
        for (uint64_t k = 0; k < allocator.slottedOutcasts.slotNum_; k++) {
            Backup::BackupBlockHeader(checkpointBase[k].listNext, base);
        }
    }

    void RuntimeAddrRestore(DeviceExecuteSlot *runtimeSlotList, uint32_t *runtimeSlotRefCntList, uint32_t slotSize, TensorAllocator &allocator) {
        uint32_t slotDataSize = sizeof(DeviceExecuteSlot) * slotSize;
        uint32_t slotRefCntDataSize = sizeof(uint32_t) * slotSize;
        memcpy_s(runtimeSlotList, slotDataSize, runtimeBackup.slotContext.slotList.Data(), slotDataSize);
        memcpy_s(runtimeSlotRefCntList, slotRefCntDataSize, runtimeBackup.slotContext.slotRefCntList.Data(), slotRefCntDataSize);

        struct Restore {
            static void RestoreBlockHeader(WsSlotAllocator::BlockHeader *&ptr, WsSlotAllocator::BlockHeader *base, WsSlotAllocator::BlockHeader *index) {
                ptr = base + (uintptr_t)index;
            }
            static void RestoreSeqAllocator(SeqWsAllocator &dst, SeqWsAllocator &src) {
                dst.allocated_ = src.allocated_;
                dst.resetTimes_ = src.resetTimes_;
            }
        };
        Restore::RestoreSeqAllocator(allocator.dassembleDests, runtimeBackup.workspace.tensorAllocators.dassembleDests);
        Restore::RestoreSeqAllocator(allocator.rootInner, runtimeBackup.workspace.tensorAllocators.rootInner);
        Restore::RestoreSeqAllocator(allocator.devTaskInnerOutcasts, runtimeBackup.workspace.tensorAllocators.devTaskInnerOutcasts);
        allocator.slottedOutcasts.availableSlots_ = runtimeBackup.workspace.tensorAllocators.slottedOutcasts.availableSlots_;

        WsSlotAllocator::BlockHeader *base = allocator.slottedOutcasts.GetBlockHeaderBase();
        Restore::RestoreBlockHeader(allocator.slottedOutcasts.freeListHeader_, base, runtimeBackup.workspace.tensorAllocators.slottedOutcasts.freeListHeader_);
        Restore::RestoreBlockHeader(allocator.slottedOutcasts.notInUseHeaders_, base, runtimeBackup.workspace.tensorAllocators.slottedOutcasts.notInUseHeaders_);
        WsSlotAllocator::BlockHeader *checkpointBase = runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList.Data();
        for (uint64_t k = 0; k < allocator.slottedOutcasts.slotNum_; k++) {
            Restore::RestoreBlockHeader(base[k].listNext, base, checkpointBase[k].listNext);
        }
    }

    void RuntimeAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram) {
        RelocRange relocProgram(srcProgram, dstProgram);
        {
            auto &slotList = runtimeBackup.slotContext.slotList;
            DeviceExecuteSlot *base = slotList.Data();
            uint64_t size = slotList.size();
            for (uint64_t k = 0; k < size; k++) {
                relocProgram.RelocNullable(base[k].partialUpdate);
            }
        }
    }

    void RuntimeAddrRelocWorkspace(
            uint64_t srcWorkspace, uint64_t dstWorkspace,
            DevStartArgsBase *devStartArgs, DeviceExecuteSlot *runtimeSlotList) {
        RelocRange relocWorkspace(srcWorkspace, dstWorkspace);
        /* empty constructor's overhead should be negligible */
        std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
        if (devStartArgs == nullptr) {
            /* only run on host */
            RelocBuildInputOutputDesc(cacheInputOutputDict, inputTensorDataList, outputTensorDataList);
        }
        {
            auto &slottedOutcastsBlockList = runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList;
            WsSlotAllocator::BlockHeader *base = slottedOutcastsBlockList.Data();
            uint64_t size = slottedOutcastsBlockList.size();
            for (uint64_t k = 0; k < size; k++) {
                relocWorkspace.RelocNullable(base[k].ptr);
            }
        }
        {
            auto &slotList = runtimeBackup.slotContext.slotList;
            DeviceExecuteSlot *base = slotList.Data();
            uint64_t size = slotList.size();
            for (uint64_t k = 0; k < size; k++) {
                if (devStartArgs == nullptr) {
                    // Host: addr uses backup
                    AddressDescriptor *addr = &base[k].desc;
                    RelocDescToCache(*addr, relocWorkspace, cacheInputOutputDict);
                } else {
                    // Device: addr uses actual
                    AddressDescriptor *addr = &runtimeSlotList[k].desc;
                    RelocDescFromCache(*addr, relocWorkspace, devStartArgs);
                }
            }
        }
    }

    /* Host-to-cache: devStartArgs should be nullptr. Cache-to-Device: devStartArgs should be filled */
    void TaskAddrRelocProgram(uint64_t srcProgram, uint64_t dstProgram) {
        RelocRange relocProgram(srcProgram, dstProgram);
        for (uint64_t deviceIndex = 0; deviceIndex < deviceTaskCount; deviceIndex++) {
            /* When cached, the pointer is always legal */
            DynDeviceTaskBase *&dynTaskBaseRef = deviceTaskCacheList[deviceIndex].dynTaskBase;
            DynDeviceTaskBase *dynTaskBase = RelocControlFlowCachePointer(dynTaskBaseRef, relocProgram);
            relocProgram.Reloc(dynTaskBase->devTask.readyAivCoreFunctionQue);
            relocProgram.Reloc(dynTaskBase->devTask.readyAicCoreFunctionQue);
            relocProgram.Reloc(dynTaskBase->devTask.readyAicpuFunctionQue);
            for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
                ReadyCoreFunctionQueue *&readyQueueRef = dynTaskBase->readyQueue[i];
                ReadyCoreFunctionQueue *readyQueue = RelocControlFlowCachePointer(readyQueueRef, relocProgram);
                relocProgram.Reloc(readyQueue->elem);
            }
            relocProgram.Reloc(dynTaskBase->cceBinary);
            relocProgram.Reloc(dynTaskBase->aicpuLeafBinary);

            ReadyQueueCache *&readyQueueBackupRef = dynTaskBase->readyQueueBackup;
            ReadyQueueCache *readyQueueBackup = RelocControlFlowCachePointer(readyQueueBackupRef, relocProgram);
            for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
                relocProgram.Reloc(readyQueueBackup->queueList[i].elem);
            }

            DynFuncHeader *&dynFuncDataListRef = dynTaskBase->dynFuncDataList;
            DynFuncHeader *dynFuncDataList = RelocControlFlowCachePointer(dynFuncDataListRef, relocProgram);
            DynFuncDataCache *dynFuncDataCacheList = dynTaskBase->dynFuncDataCacheList;
            DynFuncDataBackup *dynFuncDataBackupList = dynTaskBase->dynFuncDataBackupList;

            for (uint32_t dupIndex = 0; dupIndex < dynFuncDataList->funcNum; dupIndex++) {
                DynFuncData *dynData = &dynFuncDataList->At(dupIndex);
                DynFuncDataCache *dynDataCache = &dynFuncDataCacheList->At(dupIndex);
                DynFuncDataBackup *dynDataBackup = &dynFuncDataBackupList->At(dupIndex);

                DevAscendFunctionDuppedData *&duppedDataRef = dynDataCache->duppedData;
                DevAscendFunctionDuppedData *duppedData = RelocControlFlowCachePointer(duppedDataRef, relocProgram);

                // Reloc Stitch
                for (uint32_t i = 0; i < duppedData->GetStitchSize(); i++) {
                    DevAscendFunctionDuppedStitchList &stitchList = duppedData->GetStitch(i);
                    DevAscendFunctionDuppedStitch *&stitchRef = stitchList.Head();
                    for (DevAscendFunctionDuppedStitch **nodePtr = &stitchRef; *nodePtr != nullptr; ) {
                        DevAscendFunctionDuppedStitch *node = RelocControlFlowCachePointer(*nodePtr, relocProgram);
                        nodePtr = &node->Next();
                    }
                }

                // Reloc Dupped
                relocProgram.Reloc(duppedData->source_);

                // Reloc DynFuncData
                relocProgram.Reloc(dynData->opAttrs);
                relocProgram.Reloc(dynData->opAtrrOffsets);
                relocProgram.Reloc(dynData->exprTbl);
                relocProgram.Reloc(dynData->rawTensorDesc);
                relocProgram.Reloc(dynData->rawTensorAddr);

                relocProgram.Reloc(dynDataCache->devFunc);
                relocProgram.Reloc(dynDataCache->predCount);
                relocProgram.Reloc(dynDataCache->calleeList);
                relocProgram.RelocNullable(dynDataBackup->predCountBackup);
                relocProgram.RelocNullable(dynDataBackup->rawTensorAddrBackup);
            }
        }
    }
};

#define ControlFlowAllocateSlab(devProg, size, expr) \
    ({ \
        WsAllocation ws; \
        DevProgramControlFlowCache *c = (devProg)->GetControlFlowCache(); \
        if (c->IsRecording()) { \
            void *ptr = c->AllocateCache(size); \
            if (ptr != nullptr) { \
                ws.ptr = reinterpret_cast<uintdevptr_t>(ptr); \
            } else { \
                ws = (expr); \
            } \
        } else { \
            ws = (expr); \
        } \
        ws; \
    })

#ifndef PAGE_SIZE
#define PAGE_SIZE       4096
#endif
struct DevAscendProgram {
    DeviceArgs devArgs;
    uint64_t workspaceSize;
    uint64_t l2CacheOffset;
    uint64_t configKey;
    uint64_t hashKey;
    uint64_t slotSize;
    struct {
        struct {
            // root func inner tensors
            uint64_t rootInner;
            // root func outcasts & dassemble-dst, automatically upgraded to DeviceTask boundary outcasts
            uint64_t dassembleDests;
            // root func outcasts & non-dassemble-dst & DeviceTask inner tensors
            uint64_t devTaskInnerOutcasts;
            // root func outcasts & non-dassemble-dst & DeviceTask boundary outcasts: singleSlotMem * pooledSlotNum
            uint64_t singleSlotMem;
            uint64_t pooledSlotNum;

            uint64_t Total() const {
                uint64_t total = rootInner +       // root func inner tensors
                    dassembleDests +               // root func outcasts & dassemble-dst, automatically upgraded to DeviceTask boundary outcasts
                    devTaskInnerOutcasts +         // root func outcasts & non-dassemble-dst & DeviceTask inner tensors
                    singleSlotMem * pooledSlotNum; // root func outcasts & non-dassemble-dst & DeviceTask boundary outcasts
                static constexpr uint64_t ALIGNMENT_32K = 32 * 1024;
                return AlignUp(total, ALIGNMENT_32K);
            }
        } tensor;
        uint64_t aicoreSpilled;
        struct {
            uint64_t general;
            uint64_t stitchPool;

            uint64_t Total() const {
                return general + stitchPool;
            }
            uint64_t ContextTotal() const {
                return Total();
            }
            uint64_t ContextGeneral() const {
                return general;
            }
        } metadata;
        struct {
            uint64_t dumpTensor;
        } debug;
    } memBudget;
    const void *controlFlowBinaryAddr{nullptr};
    uint64_t hcclContext[HCCL_GROUP_NUM];
    uint64_t commGroupNum;
    uint16_t firstStitchTaskLoopNum;
    uint16_t stitchTaskIncrLoopNum;
    uint32_t stitchCallopMaxNum;
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
    DevRelocVector<DevAicpuLeafBinary> aicpuLeafCodeList;
    DevRelocVector<int32_t> aicpuLeafCodeDataList;
    DevRelocVector<uint64_t> startArgsInputTensorSlotIndexList;
    DevRelocVector<uint64_t> startArgsOutputTensorSlotIndexList;
    DevRelocVector<uint64_t> startArgsInputSymbolIndexList;
    DevRelocVector<SymbolHandler> startArgsSymbolHandlerList;
    DevRelocVector<uint64_t> assembleSlotIndexList;
    DevRelocVector<uint64_t> outputInplaceSlotList;
    DevRelocVector<DevAscendProgramPartialUpdate> partialUpdateList;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTableList;
    DevRelocVector<PrefetchInfo> prefetchInfoList;
    DevRelocVector<uint8_t> disableL2List;
    DevProgramControlFlowCache controlFlowCache;
#define programLastField                              controlFlowCache.cacheData
    uint64_t dataSize;
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
     *      uint64_t startArgsInputTensorSlotIndexListData[]
     *      uint64_t startArgsOutputTensorSlotIndexListData[]
     *      uint64_t startArgsInputSymbolIndexListData[]
     *      SymbolHandler startArgsSymbolHandlerListData[]
     *      uint64_t assembleSlotIndexList[]
	 *      uint64_t outputInplaceSlotList[];
     *      DevAscendProgramPartialUpdate partialUpdateList[]
     *      DevAscendProgramSlot slotList[]
     */

    template <typename T>
    const T &At(const DevRelocVector<T> &localvec, int index) const {
        return localvec[index];
    }
    template <typename T>
    T &At(DevRelocVector<T> &localvec, int index) {
        return localvec[index];
    }

    void DumpCce(std::ostringstream& oss, int indent) const {
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');
        std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
        oss << INDENTINNER << "#cce:" << cceCodeList.size() << "\n";
        for (size_t i = 1; i < cceCodeList.size(); i++) {
            const DevCceBinary &cceCode = At(cceCodeList, i);
            oss << INDENTINNER << "#cce-" << i << " #CoreType:" << cceCode.coreType
                << " #FuncHash:" << cceCode.funcHash;
            oss << "\n";
        }
    }

    std::string Dump(int indent = 0, bool dumpAddr = false) const {
        const int WIDTH = 16;
        const int ADDRESS_MIN_WIDTH = 6;
        std::string INDENTINNER(indent + IDENT_SIZE, ' ');
        std::string INDENTINNERINNER(indent + IDENT2_SIZE, ' ');
        std::ostringstream oss;
        oss << "DevProgram {\n";
        oss << INDENTINNER << "#tensorMemBudget:" << memBudget.tensor.Total() << "\n";
        oss << INDENTINNER << "#metadataMemBudget:" << memBudget.metadata.ContextTotal() << "\n";
        oss << INDENTINNER << "#machineSchMode:" << devArgs.machineConfig << "\n";
        oss << INDENTINNER << "#firstStitchTaskLoopNum:" << firstStitchTaskLoopNum << "\n";
        oss << INDENTINNER << "#stitchTaskIncrLoopNum:" << stitchTaskIncrLoopNum << "\n";
        oss << INDENTINNER << "#stitchCallopMaxNum:" << stitchCallopMaxNum << "\n";
        oss << INDENTINNER << "#slot:" << slotSize << "\n";
        oss << INDENTINNER << "#symbolCount:" << symbolTable.size() << "\n";
        for (size_t i = 0; i < symbolTable.size(); i++) {
            const DevAscendProgramSymbol &symbol = At(symbolTable, i);
            oss << INDENTINNER << "#symbol:" << symbol.index << " = " << &At(symbol.name, 0) << "\n";
        }
        oss << INDENTINNER << "#inputCount:" << startArgsInputTensorSlotIndexList.size() << "\n";
        for (size_t i = 0; i < startArgsInputTensorSlotIndexList.size(); i++) {
            oss << INDENTINNER << "#input:" << i << " -> #slot:" << At(startArgsInputTensorSlotIndexList, i) << "\n";
        }
        oss << INDENTINNER << "#outputCount:" << startArgsOutputTensorSlotIndexList.size() << "\n";
        for (size_t i = 0; i < startArgsOutputTensorSlotIndexList.size(); i++) {
            oss << INDENTINNER << "#output:" << i << " <- #slot:" << At(startArgsOutputTensorSlotIndexList, i) << "\n";
        }
        oss << INDENTINNER << "#assembleSlotCount:" << assembleSlotIndexList.size() << "\n";
        for (size_t i = 0; i < assembleSlotIndexList.size(); i++) {
            oss << INDENTINNER << "#assembleSlot:" << i << " -> #slot:" << At(assembleSlotIndexList, i) << "\n";
        }
        oss << INDENTINNER << "#outputInplaceSlotCount:" << outputInplaceSlotList.size() << "\n";
        for (size_t i = 0; i < outputInplaceSlotList.size(); i++) {
            oss << INDENTINNER << "#outputInplaceSlot:" << i << " -> #slot:" << At(outputInplaceSlotList, i) << "\n";
        }
        for (size_t i = 0; i < partialUpdateList.size(); i++) {
            auto &partialUpdate = At(partialUpdateList, i);
            oss << INDENTINNER << "#slot-partial-update-" << i << ":" << !partialUpdate.Empty();
            if (!partialUpdate.Empty()) {
                oss << " | #cellMatchTableDesc:" << DevAscendFunction::DumpCellMatchTableDesc(partialUpdate.cellMatchTableDesc)
                    << " | #cellMatchStaticTable:" << partialUpdate.cellMatchRuntimePartialUpdateTable.size();
            }
            oss << "\n";
        }
        for (size_t i = 0; i < startArgsInputSymbolIndexList.size(); i++) {
            oss << INDENTINNER << "#symbol:" << i << " -> #symbolTable:" << At(startArgsInputSymbolIndexList, i) << "\n";
        }
        oss << INDENTINNER << "#ExprCount:" << expressionTableSize << "\n";

        oss << INDENTINNER << "#ExprCodeSize:" << expressionTableBinary.size();
        if (dumpAddr) {
            if (expressionTableBinary.size() != 0) {
                oss << " #ExprCodeAddr:" << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(expressionTableBinary, 0)));
            }
        }
        oss << "\n";

        for (size_t i = 0; i < expressionTableBinary.size(); i += WIDTH) {
            oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
            for (size_t off = i; off < std::min(i + WIDTH, expressionTableBinary.size()); off++) {
                oss << " " << DevAscendFunction::DumpByte(At(expressionTableBinary, off));
            }
            oss << "\n";
        }

        oss << INDENTINNER << "#func:" << devEncodeList.size() << "\n";
        for (size_t i = 0; i < devEncodeList.size(); i++) {
            const DevAscendFunction *func = reinterpret_cast<const DevAscendFunction *>(&At(At(devEncodeList, i), 0));
            oss << func->Dump(IDENT_SIZE) << "\n";
        }

        oss << "====\n"; // Dump control flow code (begin)

        oss << INDENTINNER << "#HostControlCodeSize:" << hostControlFlowBinary.size();
        if (dumpAddr) {
            oss << " #HostControlCodeAddr:" <<
                AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(hostControlFlowBinary, 0)));
        }
        oss << "\n";

        for (size_t i = 0; i < hostControlFlowBinary.size(); i += WIDTH) {
            oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
            for (size_t off = i; off < std::min(i + WIDTH, hostControlFlowBinary.size()); off++) {
                oss << " " << DevAscendFunction::DumpByte(At(hostControlFlowBinary, off));
            }
            oss << "\n";
        }

        oss << "====\n"; // Dump control flow code: ^^^ Host / Dev vvv

        oss << INDENTINNER << "#DevControlCodeSize:" << devControlFlowBinary.size();
        if (dumpAddr) {
            oss << " #DevControlCodeAddr:" <<
                AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(&At(devControlFlowBinary, 0)));
        }
        oss << "\n";

        for (size_t i = 0; i < devControlFlowBinary.size(); i += WIDTH) {
            oss << INDENTINNERINNER << AddressDescriptor::DumpAddress(i, ADDRESS_MIN_WIDTH) << ":";
            for (size_t off = i; off < std::min(i + WIDTH, devControlFlowBinary.size()); off++) {
                oss << " " << DevAscendFunction::DumpByte(At(devControlFlowBinary, off));
            }
            oss << "\n";
        }

        oss << "====\n"; // Dump control flow code (ends)

        DumpCce(oss, indent);
        oss << "}";
        return oss.str();
    }

    void DumpFile(const std::string &filePath) const {
        std::ofstream ofs(filePath);
        ofs << Dump();
        ofs.close();
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

    std::vector<int> GetPartialUpdateTensorSlotIndexList() const {
        const int &front = At(assembleSlotIndexList, 0);
        const int &back = At(assembleSlotIndexList, assembleSlotIndexList.size() - 1);
        std::vector<int> slotIndexList(&front, &back + 1);
        return slotIndexList;
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
    const DevAicpuLeafBinary *GetAicpuLeafBinary(int index) const { return &aicpuLeafCodeList[index]; }

    DevProgramControlFlowCache *GetControlFlowCache() { return &controlFlowCache; }

    uint64_t GetWorkspaceSize() const {
        return memBudget.metadata.Total() +
               memBudget.tensor.Total() +
               memBudget.aicoreSpilled +
               memBudget.debug.dumpTensor;
    }

    template<typename Ty>
    typename Ty::ElementType *RelocOffset(intptr_t shift, void *&offset, Ty &list) {
        typename Ty::ElementType *ptr = reinterpret_cast<typename Ty::ElementType *>(offset);
        offset = (void *)((uintptr_t)(offset) + list.ElementSize() * list.size());
        list.DeviceRelocData(shift);
        return ptr;
    }

    void RelocProgram(uint64_t srcProgram, uint64_t dstProgram, bool relocFunc = false) {
        intptr_t shift = static_cast<int64_t>(dstProgram) - static_cast<int64_t>(srcProgram);
        void *offset = data;

        auto symbolTablePtr = RelocOffset(shift, offset, symbolTable);
        for (size_t i = 0; i < symbolTable.size(); i++) {
            symbolTablePtr[i].name.DeviceRelocData(shift);
        }

        RelocOffset(shift, offset, symbolTableNameList);
        RelocOffset(shift, offset, expressionTableOffsetList);
        RelocOffset(shift, offset, preGuardPage);
        RelocOffset(shift, offset, expressionTableBinary);
        RelocOffset(shift, offset, hostControlFlowBinary);
        RelocOffset(shift, offset, devControlFlowBinary);

        auto devEncodeListPtr = RelocOffset(shift, offset, devEncodeList);
        for (size_t i = 0; i < devEncodeList.size(); i++) {
            devEncodeListPtr[i].DeviceRelocData(shift);
        }
        RelocOffset(shift, offset, devEncodeDataList);
        RelocOffset(shift, offset, cceCodeList);
        auto aicpuLeafCodeListPtr = RelocOffset(shift, offset, aicpuLeafCodeList);
        for (size_t i = 0; i < aicpuLeafCodeList.size(); i++) {
            aicpuLeafCodeListPtr[i].aicpuLeafCode.DeviceRelocData(shift);
        }
        RelocOffset(shift, offset, aicpuLeafCodeDataList);

        RelocOffset(shift, offset, startArgsInputTensorSlotIndexList);
        RelocOffset(shift, offset, startArgsOutputTensorSlotIndexList);
        RelocOffset(shift, offset, startArgsSymbolHandlerList);
        RelocOffset(shift, offset, startArgsInputSymbolIndexList);
        RelocOffset(shift, offset, assembleSlotIndexList);
        RelocOffset(shift, offset, outputInplaceSlotList);
        auto partialUpdateListPtr = RelocOffset(shift, offset, partialUpdateList);
        for (size_t i = 0; i < partialUpdateList.size(); i++) {
            partialUpdateListPtr[i].cellMatchRuntimePartialUpdateTable.DeviceRelocDataMaybeNull(shift);
        }
        RelocOffset(shift, offset, cellMatchRuntimePartialUpdateTableList);

        RelocOffset(shift, offset, prefetchInfoList);
        RelocOffset(shift, offset, disableL2List);
        if (relocFunc) {
            for (int i = 0; i < static_cast<int>(GetFunctionSize()); i++) {
                DevAscendFunction *func = GetFunction(i);
                func->Reloc(reinterpret_cast<uint64_t>(func), true);
            }
        }

        RelocOffset(shift, offset, controlFlowCache.inputTensorDataList);
        RelocOffset(shift, offset, controlFlowCache.outputTensorDataList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.slotContext.slotList);
        RelocOffset(shift, offset, controlFlowCache.runtimeBackup.slotContext.slotRefCntList);
        RelocOffset(shift, offset, controlFlowCache.deviceTaskCacheList);
        RelocOffset(shift, offset, controlFlowCache.cacheData);
    }

    void ResetFromLaunch() {
        memset_s(&devArgs, sizeof(devArgs), 0, sizeof(devArgs));
        controlFlowBinaryAddr = nullptr;
        workspaceSize = 0;
        RelocProgram(reinterpret_cast<int64_t>(this), 0);
    }

    void ResetRerun() {
        uint64_t *RuntimePartialUpdateTable = cellMatchRuntimePartialUpdateTableList.Data();
        uint64_t RuntimePartialUpdateTableSize = cellMatchRuntimePartialUpdateTableList.DataSize();
        memset_s(RuntimePartialUpdateTable, RuntimePartialUpdateTableSize, 0, RuntimePartialUpdateTableSize);
    }

    struct DevRelocRange {
        template<typename T>
        DevRelocRange(const DevRelocVector<T> &v) : begin(reinterpret_cast<uintptr_t>(v.begin())), end(reinterpret_cast<uintptr_t>(v.end())) {}

        uintptr_t begin;
        uintptr_t end;
    };

    void RuntimeVerify(uintptr_t workspaceBegin, uintptr_t workspaceEnd) const {
        (void)workspaceBegin, (void)workspaceEnd;
        DEV_IF_VERBOSE_DEBUG {
        } else {
            return;
        }
        std::vector<DevRelocRange> rangeList = {
            symbolTable, // 0
            symbolTableNameList,
            expressionTableOffsetList,
            hostControlFlowBinary,
            devControlFlowBinary,
            devEncodeList, // 5
            devEncodeDataList,
            cceCodeList,
            aicpuLeafCodeList,
            aicpuLeafCodeDataList,
            startArgsInputTensorSlotIndexList, // 10
            startArgsOutputTensorSlotIndexList,
            assembleSlotIndexList,
            outputInplaceSlotList,
            partialUpdateList,
            cellMatchRuntimePartialUpdateTableList, // 15
            prefetchInfoList,
            disableL2List,
            controlFlowCache.inputTensorDataList,
            controlFlowCache.outputTensorDataList,
            controlFlowCache.runtimeBackup.workspace.tensorAllocators.slottedOutcastsBlockList, // 20
            controlFlowCache.runtimeBackup.slotContext.slotList,
            controlFlowCache.runtimeBackup.slotContext.slotRefCntList,
            controlFlowCache.deviceTaskCacheList,
            controlFlowCache.cacheData,
        };
        DEV_ASSERT((uintptr_t)data == rangeList[0].begin);
        DEV_ASSERT(rangeList[0].begin <= rangeList[0].end);
        for (size_t k = 1; k < rangeList.size(); k++) {
            DEV_ASSERT_MSG(rangeList[k - 1].end <= rangeList[k].begin, "range:%d->%d", (int)(k - 1), (int)(k));
            DEV_ASSERT_MSG(rangeList[k].begin <= rangeList[k].end, "range:%d", (int)k);
        }
        DEV_ASSERT(rangeList.back().end == (uintptr_t)&data[dataSize]);
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
    void InitCceCodeList(uintdevptr_t &initOffset, const std::vector<CceCodeInfo> &cceInfo, bool fillContent);
    void InitPrefetchInfoList(
            uintdevptr_t &initOffset, const std::vector<L2Info> &l2InfoList, bool fillContent);
    void InitDisableL2List(uintdevptr_t &initOffset, const std::vector<uint8_t> &disableL2, bool fillContent);
    void InitStartArgsABIParamList(uintdevptr_t &initOffset, const std::vector<int> &tStartArgsInputTensorSlotIndexList,
        const std::vector<int> &tStartArgsOutputTensorSlotIndexList,
        const std::vector<int> &tStartArgsInputSymbolIndexList,
        const std::vector<SymbolHandler> &tStartArgsSymbolHandlerList,
        const std::vector<int> &tAsembleSlotIndexList,
        const std::vector<int> &tInplaceSlotIndexList, bool fillContent);
    void InitPartialUpdateSlot(
            uintdevptr_t &initOffset,
            const std::vector<std::vector<uint8_t>> &devEncodeListInput,
            const std::unordered_map<Function *, int> &rootFuncKeyDict,
            const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootIncastDict,
            const std::unordered_map<int, std::unordered_map<Function *, int>> &slotRootOutcastDict,
            const std::vector<int> &tPartialUpdateSlotIndexList,
            bool fillContent);
    void InitControlFlowCache(
            uintdevptr_t &initOffset,
            const std::shared_ptr<DyndevFunctionAttribute> &dyndevAttr,
            bool fillContent);
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
    static DevTensorData Create(uintdevptr_t tensorAddress, const std::vector<T> &tensorShape) {
        DevTensorData tensorData;
        Init(&tensorData, tensorAddress, tensorShape.data(), tensorShape.size());
        return tensorData;
    }

    template<typename T>
    static void Init(DevTensorData *tensorData, uintdevptr_t tensorAddress, const T *dims, int n) {
        DEV_ASSERT(n <= DEV_SHAPE_DIM_MAX);

        tensorData->address = tensorAddress;
        tensorData->shape.dimSize = n;
        for (int i = 0; i < n; i++) {
            tensorData->shape.dim[i] = dims[i];
        }
    }

    static int Decode(int64_t *inputs, DevAscendProgram* devProg, int idxOffset,
                      DevTensorData *tensorData) {
        int64_t addrOffset = *inputs;
        int64_t *ptrBase = reinterpret_cast<int64_t *>(reinterpret_cast<uint64_t>(inputs) + addrOffset);

        int n = 0;
        InputsHeader *h = reinterpret_cast<InputsHeader *>(inputs + 1);
        int64_t *ptr = ptrBase;
        while (reinterpret_cast<int64_t *>(h) < ptrBase) {
            int64_t addr = *ptr;
            if (devProg->disableL2List[idxOffset + n] == 1) {
              DEV_INFO("Tensor index %d disable l2.", idxOffset + n);
              addr += static_cast<int64_t>(devProg->l2CacheOffset);
            }
            Init(&tensorData[n], addr, h->dimVal, h->dim);
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
    static std::vector<int64_t> Encode(const std::vector<DevTensorData> &tensors) {
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
    uint64_t contextWorkspaceAddr;
    uint64_t contextWorkspaceSize;
    DevAscendProgram *devProg;

    DevInputSymbol *inputSymbolList;
    uint64_t inputSymbolSize;
    const void *controlFlowEntry;

public:
    void InitWorkspace(DevAscendProgram *tDevProg, void *workspace) {
        contextWorkspaceAddr = reinterpret_cast<uint64_t>(workspace);
        devProg = tDevProg;
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
    const DevTensorData &GetInputTensor(int index) const { return devTensorList[index]; }
    DevTensorData &GetInputTensor(int index) { return devTensorList[index]; }

    int GetOutputTensorSize() const { return outputTensorSize; }
    const DevTensorData &GetOutputTensor(int index) const { return devTensorList[index + inputTensorSize]; }
    DevTensorData &GetOutputTensor(int index) { return devTensorList[index + inputTensorSize]; }

    int GetInputSymbolSize() const { return inputSymbolSize; }
    const DevInputSymbol &GetInputSymbol(int index) const { return inputSymbolList[index]; }
    DevInputSymbol &GetInputSymbol(int index) { return inputSymbolList[index]; }

    std::string Dump(int indent = 0) const {
        std::string INDENTINNER(indent + DUMP_INDEX_SIZE_2, ' ');
        std::string INDENTINNERINNER(indent + DUMP_INDEX_SIZE_4, ' ');
        std::ostringstream oss;
        oss << "DevStartArgs {" << "\n";
        for (int i = 0; i < GetInputTensorSize(); i++) {
            const DevTensorData &input = GetInputTensor(i);
            oss << INDENTINNER << "#input-" << i << ": #address:" << AddressDescriptor::DumpAddress(input.address);
            oss << " #shape:[";
            for (int j = 0; j < input.shape.dimSize; j++) {
                oss << Delim(j != 0, ",");
                oss << input.shape.dim[j];
            }
            oss << "]\n";
        }
        for (int i = 0; i < GetOutputTensorSize(); i++) {
            const DevTensorData &output = GetOutputTensor(i);
            oss << INDENTINNER << "#output-" << i << ": #address:" << AddressDescriptor::DumpAddress(output.address);
            oss << " #shape:[";
            for (int j = 0; j < output.shape.dimSize; j++) {
                oss << Delim(j != 0, ",");
                oss << output.shape.dim[j];
            }
            oss << "]\n";
        }
        oss << INDENTINNER << "#workspaceAddr:" << AddressDescriptor::DumpAddress(contextWorkspaceAddr) << "\n";
        oss << INDENTINNER << "#tensorMemBudget:" << devProg->memBudget.tensor.Total() << "\n";
        oss << INDENTINNER << "#metadataMemBudget:" << devProg->memBudget.metadata.ContextTotal() << "\n";
        oss << INDENTINNER << "#devProg:" << AddressDescriptor::DumpAddress(reinterpret_cast<uintdevptr_t>(devProg)) << "\n";
        oss << "}";
        return oss.str();
    }
    static std::unordered_map<std::string, SymbolHandlerId> symbolIndexDict;
};

} // namespace dynamic
} // namespace npu::tile_fwk
