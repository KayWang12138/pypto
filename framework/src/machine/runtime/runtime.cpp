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

#ifdef BUILD_WITH_CANN

#include "machine/runtime/runtime.h"

namespace {
const int32_t MODULE_TYPE_AI_CORE = 4;
const int32_t INFO_TYPE_OCCUPY = 8;
const uint8_t AICORE_MAP_BUFF_LEN = 2;
} // namespace
namespace npu::tile_fwk {
int RuntimeAgentMemory::GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType) const {
    int nrCore = 25;
    int nrSubCore = 3;
    uint64_t coreStride = 8 * 1024 * 1024; // 8M
    uint64_t subCoreStride = 0x100000ULL;
    auto halFunc = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue,
        size_t *outSizeRet))dlsym(nullptr, "halMemCtl");
    if (halFunc == nullptr) {
        ALOG_ERROR_F("halMemCtlSpeical function not found.");
        return -1;
    }
    struct AddrMapInPara inMapPara;
    struct AddrMapOutPara outMapPara;
    inMapPara.devid = GetLogDeviceId();
    inMapPara.addr_type = addrType;
    auto ret = halFunc(0, reinterpret_cast<void *>(&inMapPara), sizeof(struct AddrMapInPara),
        reinterpret_cast<void *>(&outMapPara), nullptr);
    if (ret != 0) {
        ALOG_ERROR_F("CTRL_TYPE_ADDR_MAP fail. (ret=%d).", ret);
        return ret;
    }
    for (int i = 0; i < nrCore; i++) {
        for (int j = 0; j < nrSubCore; j++) {
            uint64_t vaddr = outMapPara.ptr + (i * coreStride + j * subCoreStride);
            if (j == 0) {
                aic.push_back(vaddr);
            } else {
                aiv.push_back(vaddr);
            }
        }
    }

    return 0;
}

void *RuntimeAgentMemory::MapAiCoreReg() {
    std::vector<int64_t> aiv;
    std::vector<int64_t> aic;

    if (GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
        return nullptr;
    }

    std::vector<int64_t> regAddr;
    regAddr.insert(regAddr.end(), aic.begin(), aic.end());
    regAddr.insert(regAddr.end(), aiv.begin(), aiv.end());
    void *devAddr = nullptr;
    size_t regAddrSize = sizeof(void *) * regAddr.size();
    int rc = rtMalloc(&devAddr, regAddrSize, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        ASLOGE("rtMalloc failed. size: %zu", regAddrSize);
        return nullptr;
    }

    rc = rtMemcpy(devAddr, regAddrSize, regAddr.data(), regAddrSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        ASLOGE("rtMemcpy failed. size: %zu", regAddrSize);
        return nullptr;
    }

    ASLOGI("All AiCore Reg mapped: %p. size: %zu", devAddr, regAddrSize);
    allocatedDevAddr.emplace_back((uint8_t *)devAddr);
    return devAddr;
}

} // namespace npu::tile_fwk

#endif // BUILD_WITH_CANN
