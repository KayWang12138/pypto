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
 * \file device_ctrl.cpp
 * \brief
 */

#include "device_ctrl.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {
    DeviceCtrlMachine g_ctrl_machine;
}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServerRegisterTaskInspector(
        DeviceTaskInspectorEntry inspectorEntry,
        void *inspector) {
    g_ctrl_machine.RegisterTaskInspector(inspectorEntry, inspector);
    return 0;
}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServerInit(void *targ) {
    (void)targ;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int PyptoKernelCtrlServer(void *targ) {
    auto kargs = (DeviceKernelArgs *)targ;
    PerfBegin(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
    if (kargs->inputs == nullptr || kargs->outputs == nullptr || kargs->cfgdata == nullptr) {
        DEV_ERROR("Args has null in inputs[%p] outputs[%p] or cfg[%p].\n", kargs->inputs,
                 kargs->outputs, kargs->cfgdata);
        return -1;
    }
    (void)g_ctrl_machine.InitDyn(kargs);
    PerfEnd(PERF_EVT_DEVICE_MACHINE_INIT_DYN);
    int rc = g_ctrl_machine.ExecDyn(kargs);
    if (rc == npu::tile_fwk::dynamic::DEVICE_MACHINE_OK) {
        auto devProg = PtrToPtr<int64_t, DevAscendProgram>(kargs->cfgdata);
 	    g_ctrl_machine.InitTaskPipeWithSched(devProg);
        DEV_INFO("All schedule exited, destroy the machine.\n");
        return 0;
    }
    return -1;
}
