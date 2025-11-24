/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file load_aicpu_op.cpp
 * \brief
 */


#include <fstream>
#include <limits.h>
#include "load_aicpu_op.h"
#include "runtime/mem.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "runtime.h"
namespace {
    const std::string ControlFlowLaunchKernelName = "batchLoadsoFrombuf";
    const std::string ControlFlowKernelSoName = "libcontrol_flow.so";
    const std::string BuiltInKernelName = "DynTileFwkBackendKernelServer";
    const std::string BuiltInSoName = "libtilefwk_backend_server.so";
    constexpr int BuiltInOpNum = 3;
    std::string BuiltInFunName[BuiltInOpNum] = {"PyptoInit", "PyptoRun", "PyptoNull"};
}

namespace npu::tile_fwk {

void LoadAicpuOp::GenBuiltInOpInfo(const std::string &jsonPath) {
    std::ostringstream builtInInfo;
    builtInInfo << "{\n"
                << "  \"PyptoInit\": {\n"
                << "    \"opInfo\": {\n"
                << "      \"computeCost\": \"100\",\n"
                << "      \"engine\": \"DNN_VM_AICPU\",\n"
                << "      \"flagAsync\": \"False\",\n"
                << "      \"flagPartial\": \"False\",\n"
                << "      \"functionName\": \"DynPyptoKernelServerInit\",\n"
                << "      \"kernelSo\": \"libtilefwk_backend_server.so\",\n"
                << "      \"opKernelLib\": \"KFCKernel\",\n"
                << "      \"userDefined\": \"False\"\n"
                << "    }\n"  // closed opInfo
                << "  },\n"  // closed funcNameInit
                << "  \"PyptoRun\": {\n"
                << "    \"opInfo\": {\n"
                << "      \"computeCost\": \"100\",\n"
                << "      \"engine\": \"DNN_VM_AICPU\",\n"
                << "      \"flagAsync\": \"False\",\n"
                << "      \"flagPartial\": \"False\",\n"
                << "      \"functionName\": \"DynPyptoKernelServer\",\n"
                << "      \"kernelSo\": \"libtilefwk_backend_server.so\",\n"
                << "      \"opKernelLib\": \"KFCKernel\",\n"
                << "      \"userDefined\": \"False\"\n"
                << "    }\n"  // closed opInfo
                << "  },\n"   // closed funcNameRun
                << "  \"PyptoNull\": {\n"
                << "    \"opInfo\": {\n"
                << "      \"computeCost\": \"100\",\n"
                << "      \"engine\": \"DNN_VM_AICPU\",\n"
                << "      \"flagAsync\": \"False\",\n"
                << "      \"flagPartial\": \"False\",\n"
                << "      \"functionName\": \"DynPyptoKernelServerNull\",\n"
                << "      \"kernelSo\": \"libtilefwk_backend_server.so\",\n"
                << "      \"opKernelLib\": \"AICPUKernel\",\n"
                << "      \"userDefined\": \"False\"\n"
                << "    }\n"  // closed opInfo
                << "  },\n"  // closed funcNamenull
                << "  \"PyptoStatic\": {\n"
                << "    \"opInfo\": {\n"
                << "      \"computeCost\": \"100\",\n"
                << "      \"engine\": \"DNN_VM_AICPU\",\n"
                << "      \"flagAsync\": \"False\",\n"
                << "      \"flagPartial\": \"False\",\n"
                << "      \"functionName\": \"StaticPyptoKernelServer\",\n"
                << "      \"kernelSo\": \"libtilefwk_backend_server.so\",\n"
                << "      \"opKernelLib\": \"AICPUKernel\",\n"
                << "      \"userDefined\": \"False\"\n"
                << "    }\n"  // closed opInfo
                << "  }\n"  // closed funcNamestatic
                << "}";     // close all
    builtInOpJsonPath_ = jsonPath + "/pypto_op_info.json";
    if (!DumpFile(builtInInfo.str(), builtInOpJsonPath_)) {
        ALOG_ERROR_F("Contrust custom op json failed");
        return;
    }
}

void LoadAicpuOp::SetAiCpuKernel() {
    std::vector<char> buffer = OpInfoManager::GetInstance().GetControlBuffer();
    customKerBin_ = std::make_shared<OpKernelBin>("CONTROL_FLOW", buffer);
}

void LoadAicpuOp::CustomAiCpuSoLoad() {
    rtLoadBinaryConfig_t optionCfg;
    auto loadBinOptions = std::make_unique<rtLoadBinaryOption_t>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 1;
    optionCfg.numOpt = 1;
    std::string customOpJsonPath = OpInfoManager::GetInstance().GetCustomOpJsonPath();
    if (RealPath(customOpJsonPath).empty()) {
      ALOG_ERROR_F("Custom op json path is empty");
      return;
    }
    customBinHandle_ = OpInfoManager::GetInstance().GetControlBinHandle(customOpJsonPath);
    if (customBinHandle_ != nullptr) {
        return;
    }
    auto ret = rtsBinaryLoadFromFile(customOpJsonPath.c_str(), &optionCfg, reinterpret_cast<void**>(&customBinHandle_));
    if (ret != 0) {
        ALOG_ERROR_F("Load aicpu json failed ret is %d", ret);
    }
    OpInfoManager::GetInstance().SetControlBinHandle(customBinHandle_);
}

int LoadAicpuOp::LaunchCustomOp(rtStream_t stream, AstKernelArgs *kArgs, std::string &OpType) {
    ASSERT(customBinHandle_ != nullptr) << "customBinHandle cannot be null";
    rtFuncHandle custFuncHandle;
    auto ret = rtsFuncGetByName(customBinHandle_, OpType.c_str(), &custFuncHandle);
    if (ret != 0) {
        ALOG_ERROR_F("Get OpType[%s] funcHandle failed ret[%d]", OpType.c_str(), ret);
        return ret;
    }
    rtAicpuArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = kArgs;
    rtArgs.argsSize = sizeof(AstKernelArgs);

    rtCpuKernelArgs_t argInfo;
    memset_s(&argInfo, sizeof(argInfo), 0, sizeof(argInfo));
    argInfo.baseArgs = rtArgs;
    rtKernelLaunchCfg_t kernelLaunchCfg = {nullptr, 0U};
    auto launchKernelAttr = std::make_unique<rtLaunchKernelAttr_t>();
    kernelLaunchCfg.attrs = launchKernelAttr.get();
    return rtsLaunchCpuKernel(custFuncHandle, 1, stream, &kernelLaunchCfg, &argInfo);
}

int LoadAicpuOp::GetBuiltInOpBinHandle() {
  if (RealPath(builtInOpJsonPath_).empty()) {
    ALOG_ERROR_F("JsonPath is empty");
    return -1;
  }
  rtLoadBinaryConfig_t optionCfg;
  auto loadBinOptions = std::make_unique<rtLoadBinaryOption_t>();

  optionCfg.options = loadBinOptions.get();
  optionCfg.options->optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
  optionCfg.options->value.cpuKernelMode = 0;
  optionCfg.numOpt = 1;
  void *binHandle;
  auto ret = rtsBinaryLoadFromFile(builtInOpJsonPath_.c_str(), &optionCfg, reinterpret_cast<void**>(&binHandle));
  if (ret != 0) {
    ALOG_ERROR_F("Get built in bin handle failed");
    return -1;
  }

  for (int i = 0; i < BuiltInOpNum; i++) {
    rtFuncHandle funcHandle;
    ret = rtsFuncGetByName(binHandle, BuiltInFunName[i].c_str(), &funcHandle);
    if (ret != 0) {
        ALOG_ERROR_F("Get BuiltIn FuncName[%s] funcHandle failed ret[%d]", BuiltInFunName[i].c_str(), ret);
        return ret;
    }
    builtInFuncMap_[BuiltInFunName[i]] = funcHandle;
  }
  return 0;
}

int LoadAicpuOp::LaunchBuiltInOp(rtStream_t stream, AstKernelArgs *kArgs, const int &aicpuNum,
                                 const std::string &funcName) {
  rtFuncHandle funcHandle;
  auto it = builtInFuncMap_.find(funcName);
  if (it != builtInFuncMap_.end()) {
    funcHandle = it->second;
  } else {
    ALOG_ERROR_F("The func name[%s] is invalid", funcName.c_str());
    return -1;
  }
  rtAicpuArgsEx_t rtArgs;
  memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
  rtArgs.args = kArgs;
  rtArgs.argsSize = sizeof(AstKernelArgs);

  rtCpuKernelArgs_t argInfo;
  memset_s(&argInfo, sizeof(argInfo), 0, sizeof(argInfo));
  argInfo.baseArgs = rtArgs;
  rtKernelLaunchCfg_t kernelLaunchCfg = {nullptr, 0U};
  auto launchKernelAttr = std::make_unique<rtLaunchKernelAttr_t>();
  kernelLaunchCfg.attrs = launchKernelAttr.get();
  return rtsLaunchCpuKernel(funcHandle, aicpuNum, stream, &kernelLaunchCfg, &argInfo);
}
}// namespace
