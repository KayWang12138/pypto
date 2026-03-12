/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <string>
#include <vector>

#include "ir/expr.h"
#include "ir/op_registry.h"
#include "ir/pipe.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType([[maybe_unused]] const std::vector<ExprPtr> &args,
    [[maybe_unused]] const std::vector<std::pair<std::string, std::any>> &kwargs) {
    return GetUnknownType();
}

} // namespace

// ============================================================================
// Registration Function for Sync Operations
// ============================================================================

// Register system.sync_src (Set Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_src")
    .SetDescription("Send a synchronization signal (Set Flag)")
    .SetOpCategory("SyncOp")
    .SetPipe(PipeType::S)
    .NoArgument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .SetDeduceType(DeduceUnknownType);

// Register system.sync_dst (Wait Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_dst")
    .SetDescription("Wait for a synchronization signal (Wait Flag)")
    .SetOpCategory("SyncOp")
    .SetPipe(PipeType::S)
    .NoArgument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .SetDeduceType(DeduceUnknownType);

// Register system.bar_v (Vector Barrier)
// Attributes: None
REGISTER_OP("system.bar_v")
    .SetDescription("Vector unit barrier")
    .SetOpCategory("SyncOp")
    .SetPipe(PipeType::S)
    .NoArgument()
    .SetDeduceType(DeduceUnknownType);

// Register system.bar_m (Matrix Barrier)
// Attributes: None
REGISTER_OP("system.bar_m")
    .SetDescription("Matrix unit barrier")
    .SetOpCategory("SyncOp")
    .SetPipe(PipeType::S)
    .NoArgument()
    .SetDeduceType(DeduceUnknownType);

// Register system.bar_all (Global Barrier)
// Attributes: None
REGISTER_OP("system.bar_all")
    .SetDescription("Global barrier synchronization")
    .SetOpCategory("SyncOp")
    .SetPipe(PipeType::S)
    .NoArgument()
    .SetDeduceType(DeduceUnknownType);

} // namespace ir
} // namespace pypto
