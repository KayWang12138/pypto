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
 * \file runtime.cpp
 * \brief
 */

#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN

#include "runtime.h"

namespace {
const int32_t MODULE_TYPE_AI_CORE = 4;
const int32_t INFO_TYPE_OCCUPY = 8;
const uint8_t AICORE_MAP_BUFF_LEN = 2;
} // namespace
namespace npu::tile_fwk {
namespace machine {
bool RuntimeAgent::GetPgmsk(uint64_t &valid, int32_t &deviceId) const{
    rtGetDevice(&deviceId);
    rtSetDevice(deviceId);
    uint64_t aicore_bitmap[AICORE_MAP_BUFF_LEN] = {0};
    int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * AICORE_MAP_BUFF_LEN;
    auto halFuncDevInfo = (int (*)(uint32_t deviceId, int32_t moduleType, int32_t infoType,
                           void* buf, int32_t *size))dlsym(nullptr, "halGetDeviceInfoByBuff");
    if (halFuncDevInfo == nullptr) {
        ALOG_ERROR("halGetDeviceInfoByBuff function not found\n");
        return false;
    }
    auto ret = halFuncDevInfo(static_cast<uint32_t>(deviceId), MODULE_TYPE_AI_CORE, INFO_TYPE_OCCUPY,
                              reinterpret_cast<void *>(&aicore_bitmap[0]), &size_n);
    if (ret != 0) {
        return false;
    }
    valid = aicore_bitmap[0];
    return true;
}

int RuntimeAgent::GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType) const {
    int nrCore = 25;
    int nrSubCore = 3;
    int32_t deviceId = 0;
    uint64_t valid = 0;
    if (!GetPgmsk(valid, deviceId)) {
        ALOG_ERROR("Get Device Info failed or no valid core exists\n");
        return -1;
    }
    ASLOGI("The valid cores are: %ld", valid);
    uint64_t coreStride = 8 * 1024 * 1024; // 8M
    uint64_t subCoreStride = 0x100000ULL;

    auto isValid = [&valid](int id) {
        const uint64_t mask = (1ULL << 25) - 1;
        return ((static_cast<uint64_t>(valid) ^ mask) & (1ULL << id)) == 0;
    };
    auto halFunc = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue,
        size_t *outSizeRet))dlsym(nullptr, "halMemCtl");
    if (halFunc == nullptr) {
        ALOG_ERROR("halMemCtlSpeical function not found\n");
        return -1;
    }

    struct AddrMapInPara inMapPara;
    struct AddrMapOutPara outMapPara;
    inMapPara.devid = deviceId;
    inMapPara.addr_type = addrType;
    auto ret = halFunc(0, reinterpret_cast<void *>(&inMapPara), sizeof(struct AddrMapInPara),
        reinterpret_cast<void *>(&outMapPara), nullptr);
    if (ret != 0) {
        ALOG_ERROR("CTRL_TYPE_ADDR_MAP fail. (ret=%d)\n", ret);
        return ret;
    }
    for (int i = 0; i < nrCore; i++) {
        for (int j = 0; j < nrSubCore; j++) {
            uint64_t vaddr = 0UL;
            if (isValid(i)) {
                vaddr = outMapPara.ptr + (i * coreStride + j * subCoreStride);
            }
            if (j == 0) {
                aic.push_back(vaddr);
            } else {
                aiv.push_back(vaddr);
            }
        }
    }

    return 0;
}
} // namespace runtime
} // namespace npu::tile_fwk

#endif // AC_ENABLE_FRAMEWORK_WITHOUT_CANN
