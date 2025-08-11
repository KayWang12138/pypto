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
 * \file memory_reuse.h
 * \brief
 */

#pragma once
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "passes/pass_utils/connection_matrix.h"

namespace npu::tile_fwk {

struct WorkspaceInfo {
    int64_t count = -1;
    size_t position = 0; // 在leaf的incast/outcast中的位置
    uint64_t size = 0;
    bool used = false;
    LogicalTensorPtr tensor = nullptr;
    WorkspaceInfo() {}
    WorkspaceInfo(int64_t countIn, size_t positionIn, uint64_t sizeIn, const LogicalTensorPtr &tensorIn) :
        count(countIn), position(positionIn), size(sizeIn), tensor(tensorIn) {}
};

struct TensorsDesc {
    bool isDummy = false;
    LargeBitmap connectionOpsBitmap; // 作为新tensor，判断是否可以复用bucket时使用的连接bitmap
    std::set<LogicalTensorPtr> tensors;
    std::unordered_set<uint64_t> consumerOpIdxs; // 放入bucket后，作为内存桶是否可以再次复用，需要判断的consumerOp集合
    TensorsDesc(): connectionOpsBitmap(0) {}
    TensorsDesc(Function *func): connectionOpsBitmap(func->Operations().size()) {}
};

class TensorBucket {
public:
    uint64_t GetSize() const { return size_; }

    void UpdateOffset(const uint64_t offset);

    void AddRef(const TensorsDesc &tensorsDesc);

    // 检查previous的所有consumer是否有一条到tensor的producer的通路
    // 保证tensor在写的时候，previous的所有consumer都已经读取完毕
    bool HasTopoDependency(const LargeBitmap &PreducersOp) const;
private:
    
    uint64_t offset_{0};
    uint64_t size_{0};
    std::vector<std::set<LogicalTensorPtr>> refs_; // 所有rawTensor相同的tensor构成了一个ref
    std::unordered_set<uint64_t> consumerOpIdxs_;  // 新tensor能否复用本bucket，需要判断的consumerOp集合
};

class Allocator {
public:
    explicit Allocator(Function *function) : connectionMatrix_(function), function_(function) {}
    uint64_t Allocate();
    void Init();
    static bool IsRawQualified(const WorkspaceInfo &outWspInfo, const WorkspaceInfo &inWspInfo);

    const std::vector<WorkspaceInfo>& GetOutReuseInCasts(Function* leafFunc) const { 
        auto it = outReuseInCasts_.find(leafFunc);
        if (it != outReuseInCasts_.end()) {
            return it->second;
        }
        static std::vector<WorkspaceInfo> empty;
        return empty;
    }
    std::vector<WorkspaceInfo>& GetOutReuseInCasts(Function* leafFunc) { 
        return outReuseInCasts_[leafFunc];
    }
    void SetOutReuseInCasts(Function* leafFunc, const std::vector<WorkspaceInfo>& outReuseInCasts) { 
        outReuseInCasts_[leafFunc] = outReuseInCasts;
    }
    
private:
    void storageNeedToAllocatePreProcess(TensorsDesc &tensorsDesc);
    void UpdateStorageId(TensorsDesc &tensorsDesc, std::unordered_map<int64_t, int> &idMap, int &storageId);
    void CheckConsumerNoOverLap();
    void InitInnerLeafReuse();
    void CheckOneLeaf(Function *leafFunc);

    bool CheckTopoDependancy(const LogicalTensorPtr &tensor, Operation &op) const;
    // 检查某个CallOp的输出是否可以复用输入
    bool CheckReuseInnerCall(Operation &callOp, size_t outputIdx, LogicalTensorPtr &previous, 
        uint64_t &storageOffset) const;
    void UpdateActualRaw(LogicalTensorPtr &input) const;
    TensorBucket &GetBestFitBucket(const TensorsDesc &tensorsDesc);

    std::vector<TensorBucket> buckets_;
    std::map<int64_t, std::vector<int64_t>> bucketsSizeToIdx_; // first为buckets的最新一个tensor的size，second是对应的bucket index集合

    TensorBucket dummyPackets_; // dummy tensor的bucket
    std::unordered_map<int, size_t> storageMap_;
    // 按照topo序排列
    std::vector<TensorsDesc> storageNeedToAllocate_;
    ConnectionMatrix connectionMatrix_; // 标注任意两个leafFunction之间是否有连接
    uint64_t size_{0};
    Function *function_;
    std::unordered_set<int> rootInCasts_;
    std::unordered_set<int> rootOutCasts_;
    
    // 使用 map 存储，key 为 function 指针
    // function内，outcast可以和哪个incast复用gm内存，-1表示不能复用
    std::unordered_map<Function*, std::vector<WorkspaceInfo>> outReuseInCasts_;
};

class MemoryReuse : public Pass {
public:
    MemoryReuse() : Pass("MemoryReuse") {}
    ~MemoryReuse() override {}
    Status RunOnFunction(Function &function) override;
};
} // namespace npu::tile_fwk
