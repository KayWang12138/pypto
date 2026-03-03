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
 * \file hccl_define.h
 * \brief
 */

#pragma once

#ifdef BUILD_WITH_CANN
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#else
#include <cstdint>

namespace npu::tile_fwk {
typedef void *HcclComm;
typedef enum {
    HCCL_SUCCESS = 0,               /**< success */
    HCCL_E_PARA = 1,                /**< parameter error */
    HCCL_E_PTR = 2,                 /**< empty pointer */
    HCCL_E_MEMORY = 3,              /**< memory error */
    HCCL_E_INTERNAL = 4,            /**< internal error */
    HCCL_E_NOT_SUPPORT = 5,         /**< not support feature */
    HCCL_E_NOT_FOUND = 6,           /**< not found specific resource */
    HCCL_E_UNAVAIL = 7,             /**< resource unavailable */
    HCCL_E_SYSCALL = 8,             /**< call system interface error */
    HCCL_E_TIMEOUT = 9,             /**< timeout */
    HCCL_E_OPEN_FILE_FAILURE = 10,  /**< open file fail */
    HCCL_E_TCP_CONNECT = 11,        /**< tcp connect fail */
    HCCL_E_ROCE_CONNECT = 12,       /**< roce connect fail */
    HCCL_E_TCP_TRANSFER = 13,       /**< tcp transfer fail */
    HCCL_E_ROCE_TRANSFER = 14,      /**< roce transfer fail */
    HCCL_E_RUNTIME = 15,            /**< call runtime api fail */
    HCCL_E_DRV = 16,                /**< call driver api fail */
    HCCL_E_PROFILING = 17,          /**< call profiling api fail */
    HCCL_E_CCE = 18,                /**< call cce api fail */
    HCCL_E_NETWORK = 19,            /**< call network api fail */
    HCCL_E_AGAIN = 20,              /**< try again */
    HCCL_E_REMOTE = 21,             /**< error cqe */
    HCCL_E_SUSPENDING = 22,         /**< error communicator suspending */
    HCCL_E_OPRETRY_FAIL = 23,       /**< retry constraint */
    HCCL_E_OOM = 24,                /**< out of memory */
    HCCL_E_IN_STATUS = 1041,        /**< The error information is in the status. */
    HCCL_E_RESERVED                 /**< reserved */
} HcclResult;

typedef enum {
    COMM_TOPO_RESERVED = -1,  ///< 保留拓扑
    COMM_TOPO_CLOS = 0,       ///< CLOS互联拓扑
    COMM_TOPO_1DMESH = 1,     ///< 1DMesh互联拓扑
    COMM_TOPO_910_93 = 2,     ///< 910_93互联拓扑(带SIO)
    COMM_TOPO_310P = 3,       ///< 310P互联拓扑
    COMM_TOPO_A2AXSERVER = 4, ///< A2_AX_SERVER
    COMM_TOPO_CUSTOM = 5      ///< 自定义
} CommTopo;

const uint32_t HCCL_ROOT_INFO_BYTES =  4108; // 4108: root info length

typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;
}
#endif