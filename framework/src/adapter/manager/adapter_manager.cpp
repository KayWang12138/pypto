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
 * \file adapter_manader.cpp
 * \brief
 */

#include "adapter/manager/adapter_manager.h"

namespace npu::tile_fwk {
namespace {
const std::string kAclLibName = "libascendcl.so";
const std::map<AclFunc, std::string> kAclFuncStrMap {
    {AclFunc::Init, "aclInit"},
    {AclFunc::Finalize, "aclFinalize"},
    {AclFunc::RtMemcpy, "aclrtMemcpy"},
    {AclFunc::RtSetDevice, "aclrtSetDevice"},
    {AclFunc::RtCreateEvent, "aclrtCreateEvent"},
    {AclFunc::RtRecordEvent, "aclrtRecordEvent"},
    {AclFunc::RtCreateEventExWithFlag, "aclrtCreateEventExWithFlag"},
    {AclFunc::RtStreamWaitEvent, "aclrtStreamWaitEvent"},
    {AclFunc::RtGetStreamResLimit, "aclrtGetStreamResLimit"},
    {AclFunc::RtGetStreamAttribute, "aclrtGetStreamAttribute"},
    {AclFunc::RtCacheLastTaskOpInfo, "aclrtCacheLastTaskOpInfo"},
    {AclFunc::RtSetExceptionInfoCallback, "aclrtSetExceptionInfoCallback"},
    {AclFunc::MdlRICaptureGetInfo, "aclmdlRICaptureGetInfo"},
    {AclFunc::MdlRICaptureThreadExchangeMode, "aclmdlRICaptureThreadExchangeMode"}
};

const std::string kHcclLibName = "libhccl.so";
const std::map<HcclFunc, std::string> kHcclFuncStrMap {
            {HcclFunc::GetCommName, "HcclGetCommName"},
            {HcclFunc::GetL0TopoTypeEx, "HcomGetL0TopoTypeEx"},
            {HcclFunc::GetCommHandleByGroup, "HcomGetCommHandleByGroup"},
            {HcclFunc::GetRootInfo, "HcclGetRootInfo"},
            {HcclFunc::CommInitRootInfo, "HcclCommInitRootInfo"},
            {HcclFunc::AllocComResourceByTiling, "HcclAllocComResourceByTiling"}
};

const std::string kRuntimeLibName = "libruntime.so";
const std::map<RuntimeFunc, std::string> kRuntimeFuncStrMap {
        {RuntimeFunc::Malloc, "rtMalloc"},
        {RuntimeFunc::Memset, "rtMemset"},
        {RuntimeFunc::Memcpy, "rtMemcpy"},
        {RuntimeFunc::MemcpyAsync, "rtMemcpyAsync"},
        {RuntimeFunc::Free, "rtFree"},
        {RuntimeFunc::SetDevice, "rtSetDevice"},
        {RuntimeFunc::GetDevice, "rtGetDevice"},
        {RuntimeFunc::GetSocSpec, "rtGetSocSpec"},
        {RuntimeFunc::GetSocVersion, "rtGetSocVersion"},
        {RuntimeFunc::GetAiCpuCount, "rtGetAiCpuCount"},
        {RuntimeFunc::GetL2CacheOffset, "rtGetL2CacheOffset"},
        {RuntimeFunc::GetLogicDevIdByUserDevId, "rtGetLogicDevIdByUserDevId"},
        {RuntimeFunc::FuncGetByName, "rtsFuncGetByName"},
        {RuntimeFunc::BinaryLoadFromFile, "rtsBinaryLoadFromFile"},
        {RuntimeFunc::StreamCreate, "rtStreamCreate"},
        {RuntimeFunc::StreamDestroy, "rtStreamDestroy"},
        {RuntimeFunc::StreamAddToModel, "rtStreamAddToModel"},
        {RuntimeFunc::StreamSynchronize, "rtStreamSynchronize"},
        {RuntimeFunc::DevBinaryUnRegister, "rtDevBinaryUnRegister"},
        {RuntimeFunc::RegisterAllKernel, "rtRegisterAllKernel"},
        {RuntimeFunc::LaunchCpuKernel, "rtsLaunchCpuKernel"},
        {RuntimeFunc::KernelLaunchWithHandleV2, "rtKernelLaunchWithHandleV2"},
        {RuntimeFunc::AicpuKernelLaunchExWithArgs, "rtAicpuKernelLaunchExWithArgs"}
};
}

AdapterManager& AdapterManager::Instance() {
    static AdapterManager adapterManager;
    return adapterManager;
}

AdapterManager::AdapterManager() {
    if (!aclAdapter_.Initialize(kAclLibName, kAclFuncStrMap)) {
        ADAPTER_LOGI("Acl adapter has not been initialized from library[%s].", kAclLibName.c_str());
    }
    if (!hcclAdapter_.Initialize(kHcclLibName, kHcclFuncStrMap)) {
        ADAPTER_LOGI("Hccl adapter has not been initialized from library[%s].", kHcclLibName.c_str());
    }
    if (!runtimeAdapter_.Initialize(kRuntimeLibName, kRuntimeFuncStrMap)) {
        ADAPTER_LOGI("Runtime adapter has not been initialized from library[%s].", kRuntimeLibName.c_str());
    }
}

AdapterManager::~AdapterManager() {}
}
