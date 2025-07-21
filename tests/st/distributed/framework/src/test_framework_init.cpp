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
 * \file test_framework_init.cpp
 * \brief
 */

#include <string>
#include <future>
#include <dlfcn.h>
#include "hccl/hccl.h"
#include "machine/runtime.h"
#include "distributed_test_framework.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace Distributed {
namespace {
// 定义MPI类型
using MPI_Comm = int;
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
using MPI_Datatype = int;
#define MPI_CHAR ((MPI_Datatype)0x4c000101)

// 定义MPI函数类型
using MpiInitFunc = int(*)(int*, char***);
using MpiCommSizeFunc = int(*)(MPI_Comm, int*);
using MpiCommRankFunc = int(*)(MPI_Comm, int*);
using MpiBcastFunc = int(*)(void*, int, MPI_Datatype, int, MPI_Comm);
using MpiBarrierFunc = int(*)(MPI_Comm);
using MpiAbortFunc = int (*)(MPI_Comm, int);
using MpiFinalizeFunc = int (*)();

const std::string MPI_LIB_PATH = "/usr/local/mpich/lib";
const std::string MPI_LIB_NAME = "libmpi.so";
 
void* GetLibHandle()
{
    static auto handle = []() {
        const auto libPath = MPI_LIB_PATH + "/" + MPI_LIB_NAME;
        auto handler = dlopen(libPath.c_str(), RTLD_NOW | RTLD_NOLOAD);
        return handler ? handler : dlopen(libPath.c_str(), RTLD_LAZY);
    }();
    return handle;
}

// 消除reinterpret_cast
template<typename FuncType>
struct FunctionConverter {
    static auto Convert(void* ptr) -> FuncType
    {
        union {
            void* from;
            FuncType to;
        } converter;
        
        converter.from = ptr;
        return converter.to;
    }
};

template<typename FuncType>
auto GetFunction(const std::string& funcName) -> FuncType
{
    auto handle = GetLibHandle();
    if (!handle) {
        ALOG_ERROR("Failed to load MPI library");
        return nullptr;
    }
    
    auto func = dlsym(handle, funcName.c_str());
    if (!func) {
        ALOG_ERROR("Failed to find function %s: %s", funcName.c_str(), dlerror());
        return nullptr;
    }
    return FunctionConverter<FuncType>::Convert(func);
}
} // namespace

void TestFrameworkInit(OpTestParam &testParam, HcomTestParam &hcomTestParam)
{
    // 获取MPI函数指针（类型安全）
    auto mpiInit = GetFunction<MpiInitFunc>("MPI_Init");
    ASSERT(mpiInit != nullptr);
    auto mpiCommSize = GetFunction<MpiCommSizeFunc>("MPI_Comm_size");
    ASSERT(mpiCommSize != nullptr);
    auto mpiCommRank = GetFunction<MpiCommRankFunc>("MPI_Comm_rank");
    ASSERT(mpiCommRank != nullptr);
    auto mpiBcast = GetFunction<MpiBcastFunc>("MPI_Bcast");
    ASSERT(mpiBcast != nullptr);
    auto mpiBarrier = GetFunction<MpiBarrierFunc>("MPI_Barrier");
    ASSERT(mpiBarrier != nullptr);
    
    mpiInit(NULL, NULL);

    // 获取当前进程在所属进程组的编号
    mpiCommSize(MPI_COMM_WORLD, &testParam.rankSize);
    mpiCommRank(MPI_COMM_WORLD, &testParam.rankId);

    setenv("TILE_FWK_STEST_DEVICE_ID", std::to_string(testParam.rankId).c_str(), 1);
    ALOG_INFO_F("device_%d set stub device id to %s", testParam.rankId, std::to_string(testParam.rankId).c_str());

    // 设备资源初始化
    ASSERT(aclInit(NULL) == 0);
    // 指定集合通信操作使用的设备
    ASSERT(aclrtSetDevice(testParam.rankId) == 0);

    // 在 rootRank 获取 rootInfo
    hcomTestParam.rootRank = 0;
    if (testParam.rankId == hcomTestParam.rootRank) {
        ASSERT(HcclGetRootInfo(&hcomTestParam.rootInfo) == 0);
    }
    // 将root_info广播到通信域内的其他rank, 初始化集合通信域
    mpiBcast(&hcomTestParam.rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, hcomTestParam.rootRank, MPI_COMM_WORLD);
    mpiBarrier(MPI_COMM_WORLD);
    ASSERT(HcclCommInitRootInfo(testParam.rankSize, &hcomTestParam.rootInfo, testParam.rankId,
        &hcomTestParam.hcclComm) == 0);

    // 获取 group name
    ASSERT(HcclGetCommName(hcomTestParam.hcclComm, testParam.group) == 0);

    ALOG_INFO_F("testParam.rankSize %d\n", testParam.rankSize);
    ALOG_INFO_F("testParam.rankId %d\n", testParam.rankId);
    ALOG_INFO_F("testParam.group %s\n", testParam.group);
    ALOG_INFO_F("rootInfo.internal %s\n", hcomTestParam.rootInfo.internal);

    return;
}

void TestFrameworkDestroy(int32_t timeout)
{
    auto mpiAbort = GetFunction<MpiAbortFunc>("MPI_Abort");
    ASSERT(mpiAbort != nullptr);
    std::future<void> finalizeTask = std::async(
        [] {
            auto mpiFinalize = GetFunction<MpiFinalizeFunc>("MPI_Finalize");
            ASSERT(mpiFinalize != nullptr);
            mpiFinalize();
        }
    );
    if (finalizeTask.wait_for(std::chrono::seconds(timeout)) == std::future_status::timeout) {
        ALOG_ERROR_F("MPI_Finalize timeout, forcing exit");
        mpiAbort(MPI_COMM_WORLD, 1);
    }
}

} // namespace Distributed
} // namespace npu::tile_fwk
