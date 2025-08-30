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
 * \file runtime.h
 * \brief
 */

#ifndef AST2_RUNTIME_H
#define AST2_RUNTIME_H

#include <cstring>
#include <iostream>
#include <vector>
#include <iomanip>
#include <dlfcn.h>
#include <map>
#include <cassert>
#ifdef ENABLE_BUILD_WITH_CANN
#include "driver/ascend_hal_define.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include "runtime/rt_preload_task.h"
#endif
#include "interface/utils/log.h"
#include "interface/utils/common.h"
#include "interface/machine/host/stubs.h"
#include "tilefwk/data_type.h"

using HalHostRegisterFunc = int (*)(void *srcPtr, uint64_t size, uint32_t flag, uint32_t devid, void **dstPtr);

namespace npu::tile_fwk {
namespace machine {
#ifdef ENABLE_BUILD_WITH_CANN

struct HugePageDesc {
  uint8_t *baseAddr;
  size_t allSize;
  size_t current;
  HugePageDesc(uint8_t *addr, size_t size) : baseAddr(addr), allSize(size), current(0) {}
};

inline size_t MemSizeAlign(const size_t bytes, const uint32_t aligns = 512U) {
    const size_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

class RuntimeAgent {
public:
    RuntimeAgent(RuntimeAgent &other) = delete;

    void operator=(const RuntimeAgent &other) = delete;

    static RuntimeAgent *GetAgent() {
        static RuntimeAgent inst;
        return &inst;
    }

protected:
    RuntimeAgent() {
#ifdef RUN_WITH_ASCEND_CAMODEL
        // don't call aclInit, it will cause camodel running fail
#else
        aclInit(nullptr);
#endif
        Init();
    }

public:
    ~RuntimeAgent() { Finalize(); }

public:
    uint64_t GetL2Offset () {
        int32_t deviceId = 0;
        uint64_t offset = 0;
        rtGetDevice(&deviceId);
        rtGetL2CacheOffset(deviceId, &offset);
        ALOG_DEBUG_F("rtGetL2CacheOffset %lu", offset);
        return offset;
    }

    void AllocDevAddr(uint8_t **devAddr, uint64_t size) {
        auto alignSize = MemSizeAlign(size);
        ALOG_INFO_F("RuntimeAgent::Alloc size[%u] with align size[%lu].", size, alignSize);
        if (TryGetHugePageMem(devAddr, alignSize)) {
          return;
        }
        size_t allocSize = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        int res = rtMalloc((void **)devAddr, allocSize, ONG_GB_HUGE_PAGE_FLAGS, 0);
        if (res != 0) {
          ALOG_WARN_F("1G page mem alloc failed, turn to 2M page.\n");
          res = rtMalloc((void **)devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0);
          if (res != 0) {
            ALOG_ERROR_F("RuntimeAgent::AllocDevAddr failed for size %lu", size);
            return;
          }
          allocatedDevAddr.emplace_back(*devAddr);
          ALOG_INFO_F("AllocDevAddr %p size is %lu", *devAddr, size);
          return;
        }
        allocatedDevAddr.emplace_back(*devAddr);
        hugePageVec.emplace_back(HugePageDesc(*devAddr, allocSize));
        if (!TryGetHugePageMem(devAddr, alignSize)) {
          ALOG_ERROR_F("RuntimeAgent::AllocDevAddr failed for size %lu", size);
          return;
        }
        ALOG_INFO_F("Alloc 1G page mem %p size is %lu", *devAddr, allocSize);
        return;
    }

    uint8_t* AllocHostAddr(uint64_t size) {
      if (size == 0) {
        ALOG_ERROR_F("Malloc size is 0!");
        return nullptr;
      }
      auto hostPtr = (uint8_t *)malloc(size);
      allocatedHostAddr.emplace_back(hostPtr);
      return hostPtr;
    }

    bool IsHugePageMemory(uint8_t *devAddr) const {
        for (auto &hugepage : hugePageVec) {
            if (devAddr >= hugepage.baseAddr && devAddr < hugepage.baseAddr + hugepage.allSize)
                return true;
        }
        return false;
    }

    void CopyToDev(uint8_t *devAddr, uint8_t *hostSrcAddr, uint64_t size) {
        rtMemcpy(devAddr, size, hostSrcAddr, size, RT_MEMCPY_HOST_TO_DEVICE);
        ALOG_DEBUG_F("RuntimeAgent::CopyToDev for src %lx to dst %lx with size %u", reinterpret_cast<uint64_t>(hostSrcAddr),
            reinterpret_cast<uint64_t>(devAddr), size);
    }

    void CopyFromTensor(uint8_t *hostDstAddr, uint8_t *devSrcAddr, uint64_t size) {
#ifdef RUN_WITH_ASCEND_CAMODEL
        rtMemcpy(hostDstAddr, size, devSrcAddr, size, RT_MEMCPY_DEVICE_TO_HOST);
#else
        rtMemcpyAsync(hostDstAddr, size, devSrcAddr, size, RT_MEMCPY_DEVICE_TO_HOST, raStreamInstance);
        rtStreamSynchronize(raStreamInstance);
#endif
    }

    rtStream_t &GetStream() { return raStreamInstance; }

    aclrtStream &GetStreamAICPU() { return raStreamInstanceAicpu; }

    void FreeTensor(uint8_t *devAddr) const {
        ALOG_DEBUG_F("RuntimeAgent::FreeTensor");
        if (IsHugePageMemory(devAddr))
            return;
        rtFree(devAddr);
    }

    int GetAicoreRegInfo(std::vector<int64_t> &aic, std::vector<int64_t> &aiv, const int &addrType) const;

    void *MapAiCoreReg() {
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

private:
    void Init() {
        ALOG_INFO_F("RuntimeAgent: Init acl runtime!");

        rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
        ALOG_DEBUG_F("RuntimeAgent: Create a default stream!");
        rtStreamCreate(&raStreamInstance, RT_STREAM_PRIORITY_DEFAULT);
        rtStreamCreate(&raStreamInstanceAicpu, RT_STREAM_PRIORITY_DEFAULT);
    }
    bool GetPgmsk(uint64_t &valid, int32_t &deviceId) const;

    bool TryGetHugePageMem(uint8_t **devAddr, uint64_t alignSize) {
        for (size_t i = 0; i < hugePageVec.size(); ++i) {
          if (hugePageVec[i].current + alignSize <= hugePageVec[i].allSize) {
            *devAddr = hugePageVec[i].baseAddr + hugePageVec[i].current;
            hugePageVec[i].current += alignSize;
            ALOG_INFO_F("HugePage Mem get with size:%u addr:%p.", alignSize, *devAddr);
            return true;
          }
        }
        return false;
    }

    rtStream_t raStreamInstance;
    aclrtStream raStreamInstanceAicpu;
    std::vector<HugePageDesc> hugePageVec;
    std::vector<uint8_t *> allocatedDevAddr;
    std::vector<uint8_t *> allocatedHostAddr;

public:
    void Finalize() {
        for (uint8_t *addr : allocatedDevAddr) {
            rtFree(addr);
        }
        for (uint8_t *addr : allocatedHostAddr) {
            free(addr);
        }
        rtStreamDestroy(raStreamInstance);
        rtStreamDestroy(raStreamInstanceAicpu);
#ifndef RUN_WITH_ASCEND_CAMODEL
        aclFinalize();
#endif
        ALOG_DEBUG_F("RuntimeAgent: runtime quit");
    }
};

inline RuntimeAgent *GetRA() {
    return RuntimeAgent::GetAgent();
}
#endif

} // namespace machine
} // namespace npu::tile_fwk

#endif // AST2_RUNTIME_H
