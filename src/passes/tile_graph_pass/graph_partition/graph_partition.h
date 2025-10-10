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
 * \file graph_partition.h
 * \brief
 */

#ifndef PASS_GRAPH_PARTITION_H
#define PASS_GRAPH_PARTITION_H
#include "passes/tile_graph_pass/graph_partition/iso_partitioner.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include "passes/tile_graph_pass/graph_partition/l1_copy_reuse.h"
#include "passes/tile_graph_pass/graph_partition/common_operation_eliminate.h"
#endif