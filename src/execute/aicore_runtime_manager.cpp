/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "aicore_runtime_manager.h"
#include <dlfcn.h>
#include "graph/buffer.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "register/hidden_inputs_func_registry.h"
#include "runtime/rt.h"
#include "driver/ascend_hal_define.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace {
const uint8_t AICORE_MAP_BUFF_LEN = 2;
const int32_t MODULE_TYPE_AI_CORE = 4;
const int32_t INFO_TYPE_OCCUPY = 8;
const uint64_t SHARE_BUFFER_SIZE = 512;
const uint64_t AICPU_COUNT = 5;
const std::string kAttrSubkernelOpBinaryStr = "_subkernel_op_binary";
}
AicoreRtManager::AicoreRtManager() {}

AicoreRtManager::~AicoreRtManager() {
    ALOG_DEBUG_F("DeInit with mem size %zu.", allocated_addrs_.size());
    for (uint8_t* addr : allocated_addrs_) {
        rtFree(addr);
    }
    allocated_addrs_.clear();
    op_to_hiddeninput_.clear();
}

bool AicoreRtManager::AllocDevAddr(uint8_t **dev_addr, size_t size) {
  ALOG_DEBUG_F("Alloc size is %zu.", size);
  int res = rtMalloc((void **)dev_addr, size, RT_MEMORY_HBM, 0);
  if (res != 0) {
    ALOG_ERROR_F("Failed to alloc mem with size %zu.");
    return false;
  }
  allocated_addrs_.emplace_back(*dev_addr);
  return true;
}

void AicoreRtManager::InsertHiddenInput(const int64_t &op_id, void *hidden_input) {
  op_to_hiddeninput_[op_id] = hidden_input;
}

void* AicoreRtManager::GetHiddenInput(const int64_t &op_id) {
  if (op_to_hiddeninput_.count(op_id) == 1) {
    ALOG_DEBUG_F("Op: %ld hit hidden input.", op_id);
    return op_to_hiddeninput_[op_id];
  }
  return nullptr;
}

bool GetPgmsk(uint64_t &valid, int32_t &deviceId) {
  rtGetDevice(&deviceId);
  uint64_t aicore_bitmap[AICORE_MAP_BUFF_LEN] = {0};
  int32_t size_n = static_cast<int32_t>(sizeof(uint64_t)) * AICORE_MAP_BUFF_LEN;
  auto halFuncDevInfo = (int (*)(uint32_t deviceId, int32_t moduleType, int32_t infoType,
                         void* buf, int32_t *size))dlsym(nullptr, "halGetDeviceInfoByBuff");
  if (halFuncDevInfo == nullptr) {
    ALOG_ERROR_F("Failed to find halGetDeviceInfoByBuff function.\n");
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

bool AicoreRtManager::GetAicoreRegInfo(const ge::OpDescPtr &op_desc, std::vector<int64_t> &aic,
                                       std::vector<int64_t> &aiv) {
  int nrCore = 25;
  int nrSubCore = 3;
  int32_t deviceId = 0;
  (void)rtGetDevice(&deviceId);
  uint64_t valid = 0;
  if (!GetPgmsk(valid, deviceId)) {
      ALOG_ERROR_F("Node[%s, %s]: failed to get device info or no valid core exists.",
              op_desc->GetNamePtr(), op_desc->GetTypePtr());
      return false;
  }
  ALOG_INFO_F("Node[%s, %s]: the valid cores are %ld", op_desc->GetNamePtr(), op_desc->GetTypePtr(), valid);
  uint64_t coreStride = 8 * 1024 * 1024; // 8M
  uint64_t subCoreStride = 0x100000ULL;  
  auto isValid = [&valid](int id) {
      const uint64_t mask = (1ULL << 25) - 1;
      return ((static_cast<uint64_t>(valid) ^ mask) & (1ULL << id)) == 0;
  };
  auto halFunc = (int (*)(int type, void *paramValue, size_t paramValueSize, void *outValue,
      size_t *outSizeRet))dlsym(nullptr, "halMemCtl");
  if (halFunc == nullptr) {
    ALOG_ERROR_F("Node[%s, %s]: failed to find halMemCtlSpeical function.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }  
  struct AddrMapInPara inMapPara;
  struct AddrMapOutPara outMapPara;
  inMapPara.devid = deviceId;
  inMapPara.addr_type = ADDR_MAP_TYPE_REG_AIC_CTRL;
  auto ret = halFunc(0, reinterpret_cast<void *>(&inMapPara), sizeof(struct AddrMapInPara),
      reinterpret_cast<void *>(&outMapPara), nullptr);
  if (ret != 0) {
    ALOG_ERROR_F("Node[%s, %s]: CTRL_TYPE_ADDR_MAP fail. (ret=%d).", op_desc->GetNamePtr(), op_desc->GetTypePtr(), ret);
    return false;
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
  return true;
}

bool AicoreRtManager::InitDyBinData(const ge::OpDescPtr &op_desc, std::vector<int64_t> &aic, std::vector<int64_t> &aiv,
                                    DevAscendProgram *host_args) {
  int64_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  ALOG_DEBUG_F("Node[%s, %s]: block dim is %ld.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), block_dim);
  std::vector<int64_t> regs;
  regs.insert(regs.end(), aic.begin(), aic.end());
  regs.insert(regs.end(), aiv.begin(), aiv.end());
  host_args->devArgs.nrAic = aic.size();
  host_args->devArgs.nrAiv = aiv.size();
  host_args->devArgs.nrAicpu = AICPU_COUNT;
  host_args->devArgs.nrValidAic = block_dim;
  host_args->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
  std::vector<int64_t> workspaces = op_desc->GetWorkspaceBytes();
  if (workspaces.empty()) {
    ALOG_ERROR_F("Node[%s, %s]: failed to get workspace.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  host_args->workspaceSize = static_cast<uint64_t>(workspaces[0]);
  (void)ge::AttrUtils::GetInt(op_desc, "_tile_fwk_op_config_key", host_args->configKey);
  ALOG_DEBUG_F("Node[%s, %s]: config key is %lu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), host_args->configKey);
  int nrCore = regs.size();
  size_t shared_size = nrCore * SHARE_BUFFER_SIZE;
  if (!AicoreRtManager::Instance().AllocDevAddr((uint8_t**)&host_args->devArgs.sharedBuffer, shared_size)) {
    ALOG_ERROR_F("Node[%s, %s]: failed to alloc shared buffer.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  if (rtMemset((void*)host_args->devArgs.sharedBuffer, shared_size, 0U, shared_size) != RT_ERROR_NONE) {
    ALOG_ERROR_F("Node[%s, %s]: failed to copy shared buffer to device.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  size_t core_reg_size = nrCore * sizeof(uint64_t);
  if (!AicoreRtManager::Instance().AllocDevAddr((uint8_t**)&host_args->devArgs.coreRegAddr, core_reg_size)) {
    ALOG_ERROR_F("Node[%s, %s]: failed to alloc core reg addr.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  if (rtMemcpy((void*)host_args->devArgs.coreRegAddr, core_reg_size, regs.data(), core_reg_size, 
      RT_MEMCPY_HOST_TO_DEVICE) != RT_ERROR_NONE) {
    ALOG_ERROR_F("Node[%s, %s]: failed to copy core reg addr to device.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return false;
  }
  ALOG_DEBUG_F("Node[%s, %s]: aic %d, aiv %d, block dim %d, sharedBuffer %lx, coreRegAddr %lx, workspace size %lu.",
          op_desc->GetNamePtr(), op_desc->GetTypePtr(), host_args->devArgs.nrAic, host_args->devArgs.nrAiv,
          host_args->devArgs.nrValidAic, host_args->devArgs.sharedBuffer, host_args->devArgs.coreRegAddr,
          host_args->workspaceSize);
  return true;
}

ge::graphStatus AicoreRtManager::TileFwkHiddenInput(const ge::OpDescPtr &op_desc, std::vector<void *> &contexts) {
  auto hit_ret = AicoreRtManager::Instance().GetHiddenInput(op_desc->GetId());
  if (hit_ret != nullptr) {
    contexts.emplace_back(hit_ret);
    return ge::GRAPH_SUCCESS;
  }
  ge::Buffer op_binary_buffer;
  ge::AttrUtils::GetBytes(op_desc, kAttrSubkernelOpBinaryStr, op_binary_buffer);
  size_t bin_size = op_binary_buffer.GetSize();
  if (op_binary_buffer.GetData() == nullptr || bin_size == 0) {
    ALOG_ERROR_F("Node[%s, %s]: failed to get subkernel binary data.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  ALOG_DEBUG_F("Node[%s, %s]: subkernel binary data size is %zu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), bin_size);
  auto bin_data = reinterpret_cast<uint8_t*>(op_binary_buffer.GetData());
  auto host_args = (DevAscendProgram*)bin_data;
  void *dev_args = nullptr;
  if (!AicoreRtManager::Instance().AllocDevAddr((uint8_t**)&dev_args, bin_size)) {
    ALOG_ERROR_F("Node[%s, %s]: failed to alloc dev args.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  std::vector<int64_t> aic;
  std::vector<int64_t> aiv;
  if (!AicoreRtManager::Instance().GetAicoreRegInfo(op_desc, aic, aiv)) {
    ALOG_ERROR_F("Node[%s, %s]: failed to get aicore reg info.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  if (!AicoreRtManager::Instance().InitDyBinData(op_desc, aic, aiv, host_args)) {
    ALOG_ERROR_F("Node[%s, %s]: failed to init bin data.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }  
  if (rtMemcpy(dev_args, bin_size, (void*)op_binary_buffer.GetData(), bin_size, RT_MEMCPY_HOST_TO_DEVICE) !=
      RT_ERROR_NONE) {
    ALOG_ERROR_F("Node[%s, %s]: failed to copy bin data to device.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  AicoreRtManager::Instance().InsertHiddenInput(op_desc->GetId(), dev_args);
  contexts.emplace_back(dev_args);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TileFwkHiddenInputsFunc(const ge::OpDescPtr &op_desc, std::vector<void *> &contexts) {
  return AicoreRtManager::Instance().TileFwkHiddenInput(op_desc, contexts);
}

REG_HIDDEN_INPUTS_FUNC(ge::HiddenInputsType::TILEFWK, TileFwkHiddenInputsFunc);
} // namespace fe
