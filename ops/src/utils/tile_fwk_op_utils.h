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
 * \file tile_fwk_op_utils.h
 * \brief
 */

#include <memory>
#include <vector>
#include "graph/ge_error_codes.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_kernel_registry.h"
#include "register/op_ct_impl_kernel_registry.h"
#include "proto/task.pb.h"
#include "graph/node.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "exe_graph/runtime/exe_res_generation_context.h"

using namespace ge;

namespace ops {
class TileFwkOpUtils {
 public:
  static ge::graphStatus CommonGenOpSpecificInfo(const gert::OpCheckContext *context, ge::AscendString &result);
  static ge::graphStatus CommonCalcOpParam(gert::ExeResGenerationContext *context, ge::AscendString &name,
                                           ge::AscendString &reuse_key);
  static ge::graphStatus CommonGenerateTask(const ge::Node &node, ge::RunContext &run_context,
                                            std::vector<domi::TaskDef> &task_defs);

 private:
  static ge::graphStatus InsertHiddenInput(const ge::Node &node, domi::KernelContext *kernel_context);
  static ge::graphStatus GenerateAicpuTask(const ge::Node &node, const int64_t &sub_stream_id,
                                           std::vector<domi::TaskDef> &task_defs);
};
} // namespace ops

