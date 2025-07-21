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
 * \file kernel_runner.h
 * \brief
 */

#ifndef ASCENDTENSOR_KERNEL_RUNNER_H
#define ASCENDTENSOR_KERNEL_RUNNER_H
#include <cassert>
#include <cstdint>
#include <system_error>
#include "interface/utils/common.h"
#include "securec.h"
#ifdef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
#else
#include <iostream>
#include <ostream>
#include <vector>
#include <map>
#include <cstring>
#include "machine/rt.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
constexpr int64_t NUM_TEN = 10;
/**
 * @brief Kernel binary manager.
 *
 */
class KernelBinMgr {
public:
    ~KernelBinMgr() {
        for (const auto &ele : binHdl_) {
            int rc = rtDevBinaryUnRegister(ele.second);
            printf("INFO unload kernel binary. \n");
            if (rc != 0) {
                printf("ERROR rtBinaryUnLoad failed. %p, %d", ele.second, rc);
            }
        }
    }

    /**
     * @brief Lazy load Kernel binary to device.
     *        The actual loading to device is delayed till first kernel launch,
     *        so the memory holding the kernel binary must not be freed till first launch.
     *
     * @param bin.  pointer to Kernel binary.
     * @param binSize. Kernel binary size
     * @return rtBinHandle. Kernel binary handle.
     */
    rtBinHandle LoadKernelBin(void *bin, size_t binSize, CoreType coreType) {
        const auto &bh = binHdl_.find(bin);
        if (bh != binHdl_.end()) {
            return bh->second;
        }

        rtBinHandle hdl;
        uint32_t magic = RT_DEV_BINARY_MAGIC_ELF;
        if (coreType == CoreType::AIV) {
            magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
        } else if (coreType == CoreType::AIC) {
            magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
        }
        rtDevBinary_t binary{.magic = magic, .version = 0, .data = bin, .length = binSize};
        int rc = rtRegisterAllKernel(&binary, &hdl);
        if (rc != 0) {
            printf("ERROR rtRegisterAllKernel failed: %p, %zu", bin, binSize);
            return nullptr;
        }
        binHdl_.emplace(bin, hdl);
        return hdl;
    }

private:
    std::map<void *, rtBinHandle> binHdl_;
};

/**
 * @brief Get the Kernel binary manager singleton instance
 *
 * @return KernelBinMgr&
 */
inline KernelBinMgr &GetKernelBinMgr() {
    static KernelBinMgr mgr;
    return mgr;
}

/**
 * @brief Kernel launcher
 *
 */
class KernelRunner {
public:
    KernelRunner(void *bin, size_t binSize, CoreType coretype = CoreType::MIX)
        : bin_(bin), binSize_(binSize), coreType_(coretype) {}

    KernelRunner() = default;

    ~KernelRunner() {
        if (rtArgHdl_ != nullptr) {
            rtDestroyLaunchArgs(rtArgHdl_);
        }
    }

    void ResetRtArg() {
        memset_s(&rtArg_, sizeof(rtArg_), 0, sizeof(rtArg_));
        rtArg_.args = nullptr;
        rtArg_.hostInputInfoPtr = nullptr;
        rtArg_.hasTiling = false;
        rtArg_.isNoNeedH2DCopy = false;
    }

    /**
     * @brief Launch Kernel on device. Only support device arguments currently.
     *
     * @param stream
     * @param blockDim
     * @param arg. kernel args.
     * @param sync. if sync stream after launch.
     * @return int
     */
    int Run(rtStream_t stream, uint32_t blockDim, const std::vector<void *> &arg, bool sync = false) {
        rtBinHandle binHdl = GetKernelBinMgr().LoadKernelBin(bin_, binSize_, coreType_);
        if (binHdl == nullptr) {
            return 1;
        }

        ResetRtArg(); // must reset RTS args before use.
        argPlaceholder_.resize(arg.size() + NUM_TEN, nullptr);
        rtArg_.args = &argPlaceholder_[0];
        rtArg_.argsSize = arg.size() * sizeof(void *);

        for (size_t i = 0; i < arg.size(); i++) {
            argPlaceholder_[i] = arg[i];
        }

        int rc = rtKernelLaunchWithHandleV2(binHdl, 0, blockDim, &rtArg_, nullptr, stream, nullptr);
        if (rc != 0) {
            printf("ERROR rtKernelLaunchWithHandleV2 failed. stream:%p, blockDim: %u, binHdl: %p. %d \n", stream,
                blockDim, binHdl, rc);
            return rc;
        }
        if (sync) {
            rc = rtStreamSynchronize(stream);
            if (rc != 0) {
                printf("ERROR rtStreamSynchronize failed. stream:%p. %d \n", stream, rc);
            }
        }
        stream_ = stream;
        return rc;
    }

    int RunAicpu(rtStream_t stream, uint32_t blockDim, const void *args, size_t size, bool sync = false) {
        ASSERT(size <= sizeof(apiParam_.arg)) << "too many aicpu arg";
        if (memcpy_s(apiParam_.arg, sizeof(apiParam_.arg) - 1, args, size) != EOK) {
            ASLOGE("memcpy_s faild with destsize %zu & srcsize %zu", sizeof(apiParam_.arg) - 1, size);
            return -1;
        }

        rtAicpuArgsEx_t argsInfoExt;
        argsInfoExt.args = reinterpret_cast<void *>(&apiParam_);
        argsInfoExt.hostInputInfoPtr = nullptr;
        argsInfoExt.argsSize = sizeof(apiParam_);
        argsInfoExt.hostInputInfoNum = 0;
        argsInfoExt.isNoNeedH2DCopy = false;
        ASLOGI("===TEST=== %zu.", sizeof(apiParam_.arg));
        argsInfoExt.kernelNameAddrOffset = sizeof(apiParam_.arg);
        argsInfoExt.soNameAddrOffset = argsInfoExt.kernelNameAddrOffset + sizeof(apiParam_.kernelName);
        int rc = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_AICPU", blockDim, &argsInfoExt, nullptr, stream, 0);
        if (rc != 0) {
            ASLOGE("rtAicpuKernelLaunchWithArgs failed. stream:%p. %d", stream, rc);
            return rc;
        }

        if (sync) {
            rc = rtStreamSynchronize(stream);
            if (rc != 0) {
                ASLOGE("RunAicpu  rtStreamSynchronize failed. stream:%p. %d \n", stream, rc);
            } else {
                ASLOGI("INFO rtStreamSynchronize success. stream:%p. %d \n", stream, rc);
            }
        }
        aicpuStream_ = stream;
        return rc;
    }

    // int RunAicpu(rtStream_t stream, uint32_t blockDim, void *arg, size_t size, bool sync = false)
    int RunAicpu(rtStream_t stream, uint32_t blockDim, const std::vector<void *> &arg, bool sync = false) {
        return RunAicpu(stream, blockDim, arg.data(), arg.size() * sizeof(void *), sync);
    }

    int StreamSync(bool aiCpuStream = false) const {
        int rc = 0;
        rtStream_t stream = stream_;
        if (aiCpuStream) {
            stream = aicpuStream_;
        }
        if (stream != nullptr) {
            rc = rtStreamSynchronize(stream);
            if (rc != 0) {
                ASLOGE("rtStreamSynchronize failed. stream:%p. %d, isAiCpu Stream: %d\n", stream, rc,
                    static_cast<int>(aiCpuStream));
            } else {
                ASLOGI("INFO rtStreamSynchronize success. stream:%p. %d, isAiCpu Stream: %d\n", stream, rc,
                    static_cast<int>(aiCpuStream));
            }
        }
        return rc;
    }

private:
    struct ApiParamDef {
        void *arg[16];
        const char kernelName[32] = {"AscendTensorRuntimeServer"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    };

    void ConstructAicpuArgs(rtAicpuArgsEx_t &argsInfo, const std::vector<void *> &arg) {
        ASSERT(arg.size() <= ARRAY_SIZE(apiParam_.arg)) << "too many aicpu arg";
        for (size_t i = 0; i < arg.size(); i++) {
            apiParam_.arg[i] = arg[i];
        }
        argsInfo.args = &apiParam_;
        argsInfo.hostInputInfoPtr = nullptr;
        argsInfo.kernelOffsetInfoPtr = nullptr;
        argsInfo.argsSize = sizeof(apiParam_);
        argsInfo.hostInputInfoNum = 0;
        argsInfo.kernelOffsetInfoNum = 0;
        argsInfo.soNameAddrOffset = static_cast<uint16_t>(
            reinterpret_cast<const char *>(&apiParam_.soName) - reinterpret_cast<const char *>(&apiParam_));
        argsInfo.kernelNameAddrOffset = static_cast<uint16_t>(
            reinterpret_cast<const char *>(&apiParam_.kernelName) - reinterpret_cast<const char *>(&apiParam_));
        argsInfo.isNoNeedH2DCopy = false;
    }

    void *bin_{nullptr};
    size_t binSize_{0};
    CoreType coreType_{CoreType::MIX};
    rtLaunchArgsHandle rtArgHdl_{nullptr};
    std::vector<void *> argPlaceholder_;
    rtArgsEx_t rtArg_;

    rtStream_t stream_{nullptr};
    rtStream_t aicpuStream_{nullptr};
    ApiParamDef apiParam_;
};

} // namespace npu::tile_fwk

#endif
#endif // AC_ENABLE_FRAMEWORK_WITHOUT_CANN
