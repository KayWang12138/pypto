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
 * \file mc2_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>

#include "securec.h"

namespace npu::tile_fwk::dynamic {

#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)

struct Mc2CommConfig {
    uint32_t version;
    uint32_t hcommCnt;
    struct Mc2ServerCfg serverCfg;
    struct Mc2HcommCfg hcommCfg;
};

constexpr uint32_t INIT_TILING_VERSION = 100U;
constexpr uint32_t MAX_CC_TILING_NUM = 8U;

#pragma pack(push, 8)
struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};
#pragma pack(pop)

constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

inline int32_t MakeMc2TilingStruct(Mc2CommConfig &commConfig, const std::string &groupName)
{
    (void)memset_s(&commConfig, sizeof(commConfig), 0, sizeof(commConfig));
    constexpr uint32_t version = 2;
    constexpr uint32_t hcommCnt = 1;
    constexpr uint32_t opTypeAllToAll = 6; // numeric representation of AlltoAll
    const char *algConfig = "AllGather=level0:ring";

    commConfig.version = version;
    commConfig.hcommCnt = hcommCnt;
    commConfig.hcommCfg.skipLocalRankCopy = 0;
    commConfig.hcommCfg.skipBufferWindowCopy = 0;
    commConfig.hcommCfg.stepSize = 0;
    commConfig.hcommCfg.opType = opTypeAllToAll;
    if (strcpy_s(commConfig.hcommCfg.groupName, sizeof(commConfig.hcommCfg.groupName), groupName.c_str()) != EOK) {
        return -1;
    }
    if (strcpy_s(commConfig.hcommCfg.algConfig, sizeof(commConfig.hcommCfg.algConfig), algConfig) != EOK) {
        return -1;
    }
    return 0;
}

inline int32_t MakeMc2TilingStructV2(Mc2CommConfigV2 &commConfig, const std::string &groupName)
{
    (void)memset_s(&commConfig, sizeof(commConfig), 0, sizeof(commConfig));
    const char *algConfig = "BatchWrite=level0:fullmesh";
    commConfig.init.version = INIT_TILING_VERSION;
    commConfig.init.mc2HcommCnt = 1;
    commConfig.init.queueNum = 0;
    commConfig.init.commBlockNum = 48U;
    commConfig.init.devType = 4U;
    commConfig.inner.skipLocalRankCopy = 0;
    commConfig.inner.skipBufferWindowCopy = 0;
    commConfig.inner.stepSize = 0;
    commConfig.inner.opType = 18U;
    commConfig.inner.version = 1;
    commConfig.init.offset[0] = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(&commConfig.inner) - reinterpret_cast<uint64_t>(&commConfig.init));
    if (strcpy_s(commConfig.inner.groupName, sizeof(commConfig.inner.groupName), groupName.c_str()) != EOK) {
        return -1;
    }
    if (strcpy_s(commConfig.inner.algConfig, sizeof(commConfig.inner.algConfig), algConfig) != EOK) {
        return -1;
    }
    return 0;
}

} // namespace npu::tile_fwk::dynamic
