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
 * \file connection_matrix.cpp
 * \brief
 */

#include "connection_matrix_ir.h"
#include "passes/pass_log/pass_log.h"
#include "ir/opcode.h"
#include "temp_utils_ir.h"

#define MODULE_NAME "GlobalMemoryReuse"

namespace npu::tile_fwk {
ConnectionMatrix::ConnectionMatrix(pto::Function *func)
    : impl_(std::make_shared<ConnectionMatrixImpl>(func)) {}

bool ConnectionMatrix::IsConnected(const pto::Operation &a, const pto::Operation &b) const {
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->IsConnected(a, b);
}

bool ConnectionMatrix::IsConnected(uint64_t indexA, uint64_t indexB) const {
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->IsConnected(indexA, indexB);
}

void ConnectionMatrix::SetConnectivity(const std::unordered_set<pto::Operation *> &producers,
    pto::Operation &op) {
    if (impl_ == nullptr) {
        return;
    }
    impl_->SetConnectivity(producers, op);
}

void ConnectionMatrix::Generate(pto::Function *func) {
    impl_->Generate(func);
}

uint64_t ConnectionMatrix::GetIndex(const pto::Operation &op) const {
    if (impl_ == nullptr) {
        APASS_LOG_WARN_F(Elements::Function, "Func ConnectionMatrix::GetIndex impl_ is nullptr.");
        return INVALID_INDEX;
    }
    return impl_->GetIndex(op);
}

const LargeBitmap& ConnectionMatrix::GetBitMap(const pto::Operation &op) const {
    const ConnectionMatrixImpl& const_impl = *impl_;
    return const_impl.GetBitMap(op); 
}

const LargeBitmap& ConnectionMatrix::GetBitMap(uint64_t index) const {
    const ConnectionMatrixImpl& const_impl = *impl_;
    return const_impl.GetBitMap(index); 
}

ConnectionMatrixImpl::ConnectionMatrixImpl(pto::Function *func) : func_(func) {
    operations_ = GetAllOperations(func);
    size_ = operations_.size();
    invalidBitmap_.ResizeBits(size_);
    bitMaps_.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
        bitMaps_.emplace_back(size_);
    }
    
    // Build operation index map
    for (uint64_t i = 0; i < operations_.size(); ++i) {
        if (operations_[i]) {
            opIndexMap_[operations_[i].get()] = i;
        }
    }
};

ConnectionMatrixImpl::~ConnectionMatrixImpl() {
    bitMaps_.clear();
}

void ConnectionMatrixImpl::Generate(pto::Function *func) {
    if (func == nullptr) {
        return;
    }
    func_ = func;

    // Rebuild operations list and index map
    operations_ = GetAllOperations(func);
    size_ = operations_.size();
    opIndexMap_.clear();
    for (uint64_t i = 0; i < operations_.size(); ++i) {
        if (operations_[i]) {
            opIndexMap_[operations_[i].get()] = i;
        }
    }
    
    // Resize bitmaps if needed
    if (bitMaps_.size() < size_) {
        invalidBitmap_.ResizeBits(size_);
        bitMaps_.reserve(size_);
        for (size_t i = bitMaps_.size(); i < size_; ++i) {
            bitMaps_.emplace_back(size_);
        }
    }

    for (auto &op : operations_) {
        if (!op) {
            continue;
        }
        std::unordered_set<pto::Operation *> producers = GetProducerOps(*op, operations_);
        SetConnectivity(producers, *op);
    }
    return;
}

void ConnectionMatrixImpl::SetConnectivity(const std::unordered_set<pto::Operation *> &producers,
    pto::Operation &op) {
    LargeBitmap &bitmap = GetBitMap(op);
    if (producers.count(&op) == 0) {
        bitmap.SetValues(0U);
    }

    bitmap.SetBit(static_cast<size_t>(GetIndex(op)));
    for (pto::Operation *producer : producers) {
        if (producer != &op) {
            bitmap.Or(GetBitMap(*producer));
            APASS_LOG_DEBUG_F(Elements::Function, "SetConnectivity for op %s %d and op %s %d.", 
                GetOpcodeName(producer->GetOpcode()).c_str(), producer->GetID(),
                GetOpcodeName(op.GetOpcode()).c_str(), op.GetID());
        }
    }
}

uint64_t ConnectionMatrixImpl::GetIndex(const pto::Operation &op) const {
    auto it = opIndexMap_.find(&op);
    if (it != opIndexMap_.end()) {
        return it->second;
    }
    APASS_LOG_WARN_F(Elements::Function, "Func ConnectionMatrixImpl::GetIndex operation not found in index map.");
    return ConnectionMatrix::INVALID_INDEX;
}

bool ConnectionMatrixImpl::IsConnected(const pto::Operation &a, const pto::Operation &b) const {
    return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

bool ConnectionMatrixImpl::IsConnected(uint64_t indexA, uint64_t indexB) const {
    if (indexA >= size_ || indexB >= size_) {
        APASS_LOG_WARN_F(Elements::Function, "Func ConnectionMatrixImpl::IsConnected invalid index: indexA %d, indexB, %d.", indexA, indexB);
        return false;
    }
    return GetBitMap(indexB).GetBit(static_cast<size_t>(indexA));
}

const LargeBitmap &ConnectionMatrixImpl::GetBitMap(const pto::Operation &op) const {
    return bitMaps_[static_cast<uint64_t>(GetIndex(op))];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(const pto::Operation &op) {
    return bitMaps_[static_cast<uint64_t>(GetIndex(op))];
}

const LargeBitmap &ConnectionMatrixImpl::GetBitMap(uint64_t index) const {
    if (index >= size_) {
        APASS_LOG_WARN_F(Elements::Function, "Func ConnectionMatrixImpl::GetBitMap invalid index: index %d.", index);
        return invalidBitmap_;
    }
    return bitMaps_[index];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(uint64_t index) {
    if (index >= size_) {
        APASS_LOG_WARN_F(Elements::Function, "Func ConnectionMatrixImpl::GetBitMap invalid index: index %d.", index);
        return invalidBitmap_;
    }
    return bitMaps_[index];
}
}
