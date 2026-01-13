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
 * \file connection_matrix_impl.h
 * \brief
 */

#pragma once

#include <unordered_map>
#include "ir/function.h"
#include "large_bm_ir.h"

namespace npu::tile_fwk {
class ConnectionMatrixImpl {
public:
    explicit ConnectionMatrixImpl(pto::Function *func);

    ~ConnectionMatrixImpl();

    bool IsConnected(const pto::Operation &a, const pto::Operation &b) const;

    bool IsConnected(uint64_t indexA, uint64_t indexB) const;

    void SetConnectivity(const std::unordered_set<pto::Operation *> &producers, pto::Operation &op);

    void Generate(pto::Function *func);

    uint64_t GetIndex(const pto::Operation &op) const;

    const LargeBitmap &GetBitMap(const pto::Operation &op) const;

    const LargeBitmap &GetBitMap(uint64_t index) const;

private:
    ConnectionMatrixImpl() = delete;

    LargeBitmap &GetBitMap(const pto::Operation &op);

    LargeBitmap &GetBitMap(uint64_t index);

    size_t size_ = 0;

    LargeBitmap invalidBitmap_ = LargeBitmap(0);

    std::vector<LargeBitmap> bitMaps_;

    pto::Function *func_;
    
    // Map from operation pointer to its index
    std::unordered_map<const pto::Operation *, uint64_t> opIndexMap_;
    
    // Store all operations for quick access
    std::vector<pto::OperationPtr> operations_;
};
}