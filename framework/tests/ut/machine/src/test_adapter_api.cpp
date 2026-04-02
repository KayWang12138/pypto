/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_apapter_api.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "adapter/api/acl_api.h"
#include "adapter/api/hcomm_api.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/runtime_api.h"

namespace npu::tile_fwk {
class TestAdapterApi : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestAdapterApi, test_acl_api) {
    EXPECT_EQ(AclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclFinalize(), ACL_SUCCESS);
    EXPECT_EQ(AclRtMemcpy(nullptr, 0, nullptr, 0, ACL_MEMCPY_HOST_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(AclRtSetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(AclRtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(AclRtCreateEvent(nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclRtRecordEvent(nullptr, nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclRtCreateEventExWithFlag(nullptr, 0), ACL_SUCCESS);
    EXPECT_EQ(AclRtStreamWaitEvent(nullptr, nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclRtGetStreamResLimit(nullptr, ACL_RT_DEV_RES_CUBE_CORE, nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclRtGetStreamAttribute(nullptr, ACL_STREAM_ATTR_FAILURE_MODE, nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclRtCacheLastTaskOpInfo(nullptr, 0), ACL_SUCCESS);
    EXPECT_EQ(AclRtSetExceptionInfoCallback(nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclMdlRICaptureGetInfo(nullptr, nullptr, nullptr), ACL_SUCCESS);
    EXPECT_EQ(AclMdlRICaptureThreadExchangeMode(nullptr), ACL_SUCCESS);
}

TEST_F(TestAdapterApi, test_hccl_api) {
    EXPECT_EQ(HcommGetCommName(nullptr, nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommGetL0TopoTypeEx(nullptr, nullptr, 0), HCOMM_SUCCESS);
    EXPECT_EQ(HcommGetCommHandleByGroup(nullptr, nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommGetRootInfo(nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommCommDestroy(nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommCommInitRootInfo(0, nullptr, 0, nullptr), HCOMM_SUCCESS);
    EXPECT_EQ(HcommAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr), HCOMM_SUCCESS);
}

TEST_F(TestAdapterApi, test_msprof_api) {
    EXPECT_EQ(MspfSysCycleTime(), 0);
    EXPECT_EQ(MspfGetHashId(nullptr, 0), 0);
    EXPECT_EQ(MspfReportApi(0, nullptr), 0);
    EXPECT_EQ(MspfReportCompactInfo(0, nullptr, 0), 0);
    EXPECT_EQ(MspfReportAdditionalInfo(0, nullptr, 0), 0);
    EXPECT_EQ(MspfRegisterCallback(0, nullptr), 0);
}

TEST_F(TestAdapterApi, test_runtime_api) {
    EXPECT_EQ(RuntimeMalloc(nullptr, 0, 0, 0), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeMemset(nullptr, 0, 0, 0), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeMemcpy(nullptr, 0, nullptr, 0, RT_MEMCPY_HOST_TO_HOST), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeMemcpyAsync(nullptr, 0, nullptr, 0, RT_MEMCPY_HOST_TO_HOST, nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeFree(nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeSetDevice(0), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeGetDevice(nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeGetSocSpec(nullptr, nullptr, nullptr, 0), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeGetSocVersion(nullptr, 0), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeGetAiCpuCount(nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeGetL2CacheOffset(0, nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeGetLogicDevIdByUserDevId(0, nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeFuncGetByName(nullptr, nullptr, nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeBinaryLoadFromFile(nullptr, nullptr, nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeStreamCreate(nullptr, 0), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeStreamDestroy(nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeStreamAddToModel(nullptr, nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeStreamSynchronize(nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeDevBinaryUnRegister(nullptr), RT_ERROR_NONE);
    EXPECT_EQ(RuntimeRegisterAllKernel(nullptr, nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeKernelLaunchWithHandleV2(nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeLaunchCpuKernel(nullptr, 0, nullptr, nullptr, nullptr), RT_ERROR_NONE);

    EXPECT_EQ(RuntimeAicpuKernelLaunchExWithArgs(0, nullptr, 0, nullptr, nullptr, nullptr, 0), RT_ERROR_NONE);
}
}
