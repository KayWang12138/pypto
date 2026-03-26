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
 * \file communication.cpp
 * \brief
 */

#include "communication.h"
#include <cstdio>

namespace npu::tile_fwk {

void CommManager::InitControl(std::string name, int rank , int worldSize) {
    if (ctrlBase_) {
        return;
    }

    rank_ = rank;
    worldSize_ = worldSize;
    ctrlName_ = "/" + name + "_ctrl";
    // TODO: 需要根据 bindTensor 变化
    ctrlShmSize_ = sizeof(sem_t) * worldSize * worldSize;

    int fd = shm_open(ctrlName_.c_str(), O_CREAT | O_RDWR, 0666);
    if (ftruncate(fd, ctrlShmSize_) == -1) {
        perror("ftruncate error!");
        close(fd);
        return;
    };

    ctrlBase_ = (sem_t *) mmap(0, ctrlShmSize_, PORT_READ | PORT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (rank == 0) {
        for (int i = 0; i < worldSize * worldSize; i++) {
            sem_init(&ctrlBase_[i], 1, 0);
        }
    }
}


void CommManager::InitData(std::string name, size_t slotSize) {
    if (dataBase_) {
        return;
    }

    slotSize_ = slotSize;
    dataName_ = "/" + name + "_data";
    dataShmSize_ = slotSize * worldSize_ * worldSize_;

    int fd = shm_open(dataName_.c_str(), O_CREAT | O_RDWR, 0666);
    if (ftruncate(fd, dataShmSize_) == -1) {
        perror("ftruncate error!");
        close(fd);
        return;
    };

    dataBase_ = (uint8_t *) mmap(0, dataShmSize_, PORT_READ | PORT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
}

void CommManager::Put(const torch::Tensor &t, int dstRank) {
    auto cpu_t = t.contiguous().cpu();
    size_t offset = (rank_ * worldSize_ + dstRank) * slotSize_;
    memcpy(dataBase_ + offset, cpu_t.data_ptr(), cpu_t.nbytes());
}

void CommManager::Signal(int dstRank) {
    sem_post(&ctrlBase_[rank_ * worldSize_ + dstRank]);
}

void CommManager::Wait(int srcRank) {
    sem_wait(&ctrlBase_[srcRank * worldSize_ + rank_]);
}

torch::Tensor CommManager::Get(int srcRank, at::IntArrayRef shape, torch::ScalarType dtype) {
    size_t offset = (srcRank * worldSize_ + rank_) * slotSize_;
    void *srcPtr = dataBase_ + offset;
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(srcPtr, shape, options).clone()
}

CommManager::~CommManager() {
    if (ctrlBase_) {
        if (rank == 0) {
            for (int i = 0; i < worldSize * worldSize; i++) {
                sem_destroy(&ctrlBase_[i]);
            }
        }
        munmap(ctrlBase_, ctrlShmSize_);
        shm_unlink(ctrlName_.c_str());
    }

    if (dataBase_) {
        munmap(dataBase_, dataShmSize_);
        shm_unlink(dataName_.c_str());
    }
}

}