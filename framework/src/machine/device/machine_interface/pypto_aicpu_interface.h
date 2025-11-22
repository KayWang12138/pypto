 /* Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TILE_FWK_AICPU_INTERFACE_H
#define TILE_FWK_AICPU_INTERFACE_H

#include <cstdint>
#include <fstream>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include "machine/utils/device_log.h"
using TileFwkKernelServelEnty = int (*)(void *);
namespace npu::tile_fwk {
  const std::string staticServerKernelkFun = "StaticTileFwkBackendKernelServer";
  const std::string dynServerKernelFun = "DynTileFwkBackendKernelServer";
  const std::string dynServerKernelInitFun = "DynTileFwkBackendKernelServerInit";
  const uint64_t staticFuncKey = 1;
  const uint64_t dyInitFuncKey = 2;
  const uint64_t dyExecFuncKey = 3;
  const std::string devicePath = "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libpypto_server.so";
  const uint64_t minSoLen = 1;
 struct AstKernelArgs {
    int64_t *syncaddr{nullptr}; // not used
    int64_t *inputs{nullptr};
    int64_t *outputs{nullptr};
    int64_t *workspace{nullptr};
    int64_t *tilingdata{nullptr};
    int64_t *cfgdata{nullptr};
    // following 4 paras need remove to binary
    void *costmodeldata{nullptr};
    void *aicoreModel{nullptr};
    uint64_t taskWastTime{0};
    uint8_t machineConfig;
  };

  struct DeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint64_t opaque{0};          // store device global data, must be init with zero
    uint64_t devQueueAddr;    // pcie/XLink mem, used between host and device, `DEVICE_QUEUE_SIZE`
    uint64_t sharedBuffer;    // SHARED_BUFFER_SIZE per core, aics first
    uint64_t coreRegAddr;     // core reg addr, uint64_t per core, aic first
    uint64_t corePmuRegAddr;  // pmu reg addr, uint64_t per core, aic first
    uint64_t corePmuAddr;     // pmu data addr, PAGE_SIZE per core, aic first
    uint64_t pmuEventAddr;    // pmu event addr
    uint64_t taskType : 4;    // initial task type
    uint64_t machineConfig : 8; // machine config
    uint64_t taskId   : 52;   // initial task id
    uint64_t taskData;        // initial task data
    uint64_t taskWastTime{0};
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
    uint64_t GetBlockNum() { return nrValidAic * (nrAiv / nrAic + 1); }
  };

class BackendServerHandleManager {
public:
  bool SaveSoFile(char *data, const uint64_t &len) {
    std::lock_guard<std::mutex> lock(funcLock_);
    if (len < minSoLen || firt_creat_so_) {
      DEV_WARN("Aicpu so len less than 1, don't to copy");
      return true;
    }
    std::ofstream file(devicePath, std::ios::out | std::ios::binary);
    DEV_DEBUG("Begin to create server.so");
    if (!file) {
        DEV_ERROR("Coundn't create file [%s]", devicePath.c_str());
        return false;
    }

    // write bin to file
    file.write(data, len);

    if (!file) {
        DEV_ERROR("Write to file [%s] not success", devicePath.c_str());
        return false;
    }
    DEV_DEBUG("create so success");
    file.close();
    firt_creat_so_ = true;
    return true;
  }

  BackendServerHandleManager() = default;

  void SetTileFwkKernelMap() {
    std::lock_guard<std::mutex> lock(funcLock_);
    if (firt_load_so_) {
      return;
    }
    (void)LoadTileFwkKernelFunc(staticServerKernelkFun);
    (void)LoadTileFwkKernelFunc(dynServerKernelInitFun);
    (void)LoadTileFwkKernelFunc(dynServerKernelFun);
    firt_load_so_ = true;
  }

  inline int32_t ExecuteFunc(void *args, const uint64_t funcKey) {
    auto func = GetTileFwkKernelFunc(funcKey);
    if (func == nullptr) {
      DEV_ERROR("kernel func[%lu] is invalid, cannot get from so %s", funcKey, devicePath.c_str()); 
      return -1;
    }
    return func(args);
  }

  ~BackendServerHandleManager() {
    if (soHandle_) {
      DEV_INFO("Close handle");
      (void)dlclose(soHandle_);
    }
  }
private:
  void LoadTileFwkKernelFunc(const std::string &kernelName) {
    if (soHandle_ == nullptr) {
      soHandle_ = dlopen(devicePath.c_str(), RTLD_LAZY);
    }
    if (!soHandle_) {
      DEV_ERROR("Cannot open so %s", devicePath.c_str());
      return;
    }
    uint64_t funcKey = staticFuncKey;
    if (kernelName == dynServerKernelInitFun) {
      funcKey = dyInitFuncKey;
    } else if (kernelName == dynServerKernelFun){
      funcKey = dyExecFuncKey;
    }
    DEV_DEBUG("Current to open kernel func %s funcKey %lu.", kernelName.c_str(), funcKey);
    auto iter = kernelKey2FuncHandle_.find(funcKey);
    if (iter != kernelKey2FuncHandle_.end()) {
      return;
    }

    TileFwkKernelServelEnty tileFwkServrFuncEnty = reinterpret_cast<TileFwkKernelServelEnty>(dlsym(soHandle_,
                                                                                             kernelName.c_str()));
    if (tileFwkServrFuncEnty == nullptr) {
      DEV_ERROR("Current KernelName [%s] is null", kernelName.c_str());
      (void)dlclose(soHandle_);
      return;
    }
    DEV_INFO("kernelName %s has been loaded", kernelName.c_str());

    kernelKey2FuncHandle_[funcKey] = tileFwkServrFuncEnty;
    return;
  }

  TileFwkKernelServelEnty GetTileFwkKernelFunc(const uint64_t funcKey) {
    auto iter = kernelKey2FuncHandle_.find(funcKey);
    if (iter != kernelKey2FuncHandle_.end()) {
      return iter->second;
    }
    DEV_ERROR("Function[%lu] is null.", funcKey);
    return nullptr;
  }

  std::unordered_map<uint64_t, TileFwkKernelServelEnty> kernelKey2FuncHandle_;
  std::mutex funcLock_;
  void *soHandle_;
  bool firt_creat_so_ = false;
  bool firt_load_so_ = false;
};

}// end name space
extern "C" {
__attribute__((visibility("default"))) uint32_t StaticPyptoKernelServer(void *args);
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServer(void *args);
__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerInit(void *args);
}


#endif // TILE_FWK_AICPU_INTERFACE_H