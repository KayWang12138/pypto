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

#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include "pypto_aicpu_interface.h"
#include "machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/machine_ws_intf.h"


namespace {
  npu::tile_fwk::BackendServerHandleManager g_handleManager;
}
using namespace npu::tile_fwk;
extern "C" {
__attribute__((visibility("default"))) uint32_t StaticPyptoKernelServer(void *args) {
    DEV_DEBUG("Start to exect static server");
    if (args == nullptr) {
        DEV_ERROR("Args is invalid");
        return 1;
    }
    auto devArgs = (DeviceArgs*)args;
    auto data = reinterpret_cast<char *>(devArgs->aicpuSoBin);
    if (!g_handleManager.SaveSoFile(data, devArgs->aicpuSoLen)) {
        DEV_ERROR("create so failed");
        return 1;
    }
    g_handleManager.SetTileFwkKernelMap();
    DEV_DEBUG("Begin to exe static tileFwk Server");
    auto ret = g_handleManager.ExecuteFunc(args, staticFuncKey);
    DEV_DEBUG("After Get kernel func [%s], with ret[%d]",
                        staticServerKernelkFun.c_str(), static_cast<int>(ret));
    if (ret != 0) {
        DEV_ERROR("TileFwk kernelFunc [%s] exec not Success", staticServerKernelkFun.c_str());
        return 1;
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerNull(void *args) {
  (void)args;
#if DEBUG_PLOG && defined(__DEVICE__)
    InitLogSwitch();
#endif
    if (args == nullptr) {
        DEV_ERROR("Server init input args is null");
        return 1;
    }
    auto kargs = (AstKernelArgs *)args;
    if (kargs == nullptr) {
        DEV_ERROR("Server init AstKernelArgs is null");
        return 1;
    }
    auto devArgs = reinterpret_cast<DeviceArgs*>(kargs->cfgdata);
    auto data = reinterpret_cast<char *>(devArgs->aicpuSoBin);
    if (!g_handleManager.SaveSoFile(data, devArgs->aicpuSoLen)) {
        DEV_ERROR("create so failed");
        return 1;
    }
    g_handleManager.SetTileFwkKernelMap();
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServer(void *args) {
    auto ret = g_handleManager.ExecuteFunc(args, dyExecFuncKey);
    if (ret != 0) {
        DEV_ERROR("TileFwk kernelFunc [%s] exec not Success", dynServerKernelFun.c_str());
        return 1;
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynPyptoKernelServerInit(void *args) {
    auto ret = g_handleManager.ExecuteFunc(args, dyInitFuncKey);
    if (ret != 0) {
        DEV_ERROR("TileFwk kernelFunc [%s] exec not Success", dynServerKernelInitFun.c_str());
        return 1;
    }
    return 0;
}
}