/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file communication.h
 * \brief
 */

#pragma once

#include <torch/torch.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <cstdint>
#include <string>

namespace npu::tile_fwk {

class CommManager {
public:
    static CommManager &getInstance() {
        static CommManager instance;
        return instance;
    }

    void InitControl(std::string name, int rank, int worldSize);

    void InitData(std::string name, size_t slotSize);

    void Put(const torch::Tensor &t, int dstRank);
    void Signal(int dstRank);
    void Wait(int srcRank);
    torch::Tensor Get(int srcRank, at::IntArrayRef shape, torch::ScalarType dtype);

    ~CommManager();

private:
    CommManager() = default;
    int rank_ = -1;
    int worldSize_ = -1;
    size_t slotSize_ = 0;

    sem_t *ctrlBase_ = nullptr;
    uint8_t *dataBase_ = nullptr;

    size_t ctrlShmSize = 0;
    size_t dataShmSize = 0l
    std::string ctrlName_, dataName_;
}

}