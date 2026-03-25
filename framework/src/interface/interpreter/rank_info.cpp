/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rank_info.cpp
 * \brief Rank information manager implementation for distributed precision tool
 */

#include "interface/interpreter/rank_info.h"
#include <unistd.h>
#include <sys/types.h>
#include <cstring>
#include <string>

namespace npu::tile_fwk {

RankInfo* RankInfo::GetInstance() {
    static RankInfo instance;
    return &instance;
}

void RankInfo::Initialize() {
    // 1. 尝试从OpenMPI环境变量获取rank和world size
    const char* rankEnv = nullptr;
    if ((rankEnv = std::getenv("OMPI_COMM_WORLD_RANK")) != nullptr) {
        rankId_ = std::atoi(rankEnv);
    } else if ((rankEnv = std::getenv("PMI_RANK")) != nullptr) {
        rankId_ = std::atoi(rankEnv);
    } else if ((rankEnv = std::getenv("MV2_COMM_WORLD_RANK")) != nullptr) {
        rankId_ = std::atoi(rankEnv);
    }
    
    const char* sizeEnv = nullptr;
    if ((sizeEnv = std::getenv("OMPI_COMM_WORLD_SIZE")) != nullptr) {
        worldSize_ = std::atoi(sizeEnv);
    } else if ((sizeEnv = std::getenv("PMI_SIZE")) != nullptr) {
        worldSize_ = std::atoi(sizeEnv);
    } else if ((sizeEnv = std::getenv("MV2_COMM_WORLD_SIZE")) != nullptr) {
        worldSize_ = std::atoi(sizeEnv);
    }

    // 2. 获取job ID（用于生成唯一的共享内存名称）
    const char* jobId = nullptr;
    if ((jobId = std::getenv("SLURM_JOB_ID")) != nullptr) {
        jobId_ = jobId;
    } else if ((jobId = std::getenv("OMPI_MCA_orte_jobid")) != nullptr) {
        jobId_ = jobId;
    } else {
        // 使用进程ID作为fallback
        jobId_ = "pypto_" + std::to_string(getpid());
    }
}

std::string RankInfo::GenerateShmemName(const std::string& groupName) const {
    return "/pypto_shmem_" + jobId_ + "_" + groupName;
}

} // namespace npu::tile_fwk
