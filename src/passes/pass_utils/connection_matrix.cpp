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
 * \file connection_matrix.cpp
 * \brief
 */

#include "connection_matrix.h"

namespace npu::tile_fwk {
ConnectionMatrix::ConnectionMatrix(Function *func)
    : impl_(std::make_shared<ConnectionMatrixImpl>(func)) {}

bool ConnectionMatrix::IsConnected(const Operation &a, const Operation &b) const {
    if (impl_ == nullptr) {
        return false;
    }
    return impl_->IsConnected(a, b);
}

void ConnectionMatrix::SetConnectivity(const std::unordered_set<Operation *> &producers,
    Operation &op) {
    if (impl_ == nullptr) {
        return;
    }
    impl_->SetConnectivity(producers, op);
}

int ConnectionMatrix::Generate(Function *func) {
    return impl_->Generate(func);
}

ConnectionMatrixImpl::ConnectionMatrixImpl(Function *func) : func_(func) {
    auto operations = func->Operations();
    size_ = operations.size();
    bitMaps_.reserve(size_);
    for (size_t i = 0; i < size_; ++i) {
        bitMaps_.emplace_back(size_);
    }
};

ConnectionMatrixImpl::~ConnectionMatrixImpl() {
    bitMaps_.clear();
}

int ConnectionMatrixImpl::Generate(Function *func) {
    if (func != nullptr) {
        func_ = func;
    } else {
        return 0;
    }

    for (auto &op : func->Operations()) {
        std::unordered_set<Operation *> producers = op.ProducerOps();
        SetConnectivity(producers, op);
    }
    return 0;
}

void ConnectionMatrixImpl::SetConnectivity(const std::unordered_set<Operation *> &producers,
    Operation &op) {
    LargeBitmap &bitmap = GetBitMap(op);
    if (producers.count(&op) == 0) {
        bitmap.SetValues(0U);
    }

    bitmap.SetBit(static_cast<size_t>(GetIndex(op)));
    for (Operation *prod : producers) {
        if (prod != &op) {
            bitmap.Or(GetBitMap(*prod));
            ALOG_DEBUG_F("222222SetConnectivity for op %s %d and op %s %d", prod->GetOpcodeStr().c_str(), prod->opmagic,
                op.GetOpcodeStr().c_str(), op.opmagic);
        }
    }
}

uint64_t ConnectionMatrixImpl::GetIndex(const Operation &op) const {
    return static_cast<uint64_t>(func_->Operations().GetOpPosition(op));
}

bool ConnectionMatrixImpl::IsConnected(const Operation &a, const Operation &b) const {
    return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

const LargeBitmap &ConnectionMatrixImpl::GetBitMap(const Operation &op) const {
    return bitMaps_[static_cast<uint64_t>(GetIndex(op))];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(const Operation &op) {
    return bitMaps_[static_cast<uint64_t>(GetIndex(op))];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(uint64_t index) {
    return bitMaps_[index];
}
}
