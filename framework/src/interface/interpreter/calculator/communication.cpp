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
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include "tilefwk/comm_group_recorder.h"

namespace npu::tile_fwk {

int GetRankId(int hcclGroupIndex) {
    std::string groupName = Distributed::CommGroupRecorder::GetInstance().Output()[hcclGroupIndex];
    auto pg = c10d::resolve_process_group(groupName);
    if (pg) {
        return pg->getRank();
    }
    return -1;
}

int GetRankId(std::string groupName) {
    auto pg = c10d::resolve_process_group(groupName);
    if (pg) {
        return pg->getRank();
    }
    return -1;
}

int GetWorldSize(int hcclGroupIndex) {
    std::string groupName = Distributed::CommGroupRecorder::GetInstance().Output()[hcclGroupIndex];
    auto pg = c10d::resolve_process_group(groupName);
    if (pg) {
        return pg->getSize();
    }
    return -1;
}

int GetWorldSize(std::string groupName) {
    auto pg = c10d::resolve_process_group(groupName);
    if (pg) {
        return pg->getSize();
    }
    return -1;
}
}