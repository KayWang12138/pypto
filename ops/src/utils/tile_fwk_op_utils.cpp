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
 * \file tile_fwk_op_utils.cpp
 * \brief
 */

#include "utils/tile_fwk_op_utils.h"
#include "nlohmann/json.hpp"
#include <cstdio>
#include "runtime/rt_model.h"
#include "runtime/mem.h"
#include "runtime/stream.h"
#include "framework/common/taskdown_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/args_format_desc_utils.h"
#include "common/checker.h"

using namespace ge;
using Json = nlohmann::json;

namespace ops {
namespace {
const std::string kTileFwkOpFlag = "tileFwkOp";
const std::string kAicCntKeyOp = "_op_aicore_num";
const std::string kAicpuInitTaskKernelName = "DynTileFwkKernelServerInit";
const std::string kAicpuMainTaskKernelName = "DynTileFwkKernelServer";
const std::string kAicpuSoName = "libaicpu_extend_kernels.so";
const uint8_t kAicpuInitTaskNum = 1;
const uint8_t kAicpuMainTaskNum = 5;
}

ge::graphStatus TileFwkOpUtils::CommonGenOpSpecificInfo(const gert::OpCheckContext *context, ge::AscendString &result) {
  Json op_specific_info;
  op_specific_info[kTileFwkOpFlag] = "true";
  result = op_specific_info.dump().c_str();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TileFwkOpUtils::CommonCalcOpParam(gert::ExeResGenerationContext *context, ge::AscendString &name,
                                                  ge::AscendString &reuse_key) {
  gert::StreamInfo stream_info;
  stream_info.name = name;
  stream_info.reuse_key = reuse_key;
  vector<int64_t> stream_depend_value_list(0);
  stream_info.depend_value_input_indices = stream_depend_value_list;
  stream_info.required = true;
  std::vector<gert::StreamInfo> stream_info_vec(0);
  stream_info_vec.push_back(stream_info);
  context->SetAttachedStreamInfos(stream_info_vec);

  gert::SyncResInfo sync_res_info;
  sync_res_info.type = gert::SyncResType::SYNC_RES_NOTIFY;
  sync_res_info.name = name;
  sync_res_info.reuse_key = reuse_key;
  sync_res_info.required = true;
  std::vector<gert::SyncResInfo> sync_info_vec(0);
  sync_info_vec.push_back(sync_res_info);
  context->SetSyncResInfos(sync_info_vec);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TileFwkOpUtils::InsertHiddenInput(const ge::Node &node, domi::KernelContext *kernel_context) {
  const auto ori_args_format = kernel_context->args_format();
  std::vector<ge::ArgDesc> arg_descs;
  auto ret = ge::ArgsFormatDescUtils::Parse(ori_args_format, arg_descs);
  if (ret != ge::GRAPH_SUCCESS || arg_descs.empty()) {
    printf("Node[%s, %s]: failed to parse args format.\n", node.GetNamePtr(), node.GetTypePtr());
    return GRAPH_FAILED;
  }
  ret = ge::ArgsFormatDescUtils::InsertHiddenInputs(arg_descs, arg_descs.size(), ge::HiddenInputsType::TILEFWK, 1);
  if (ret != ge::GRAPH_SUCCESS) {
    printf("Node[%s, %s]: failed to insert tilefwk hidden input.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  const auto new_args_format = ge::ArgsFormatDescUtils::Serialize(arg_descs);
  kernel_context->set_args_format(new_args_format);
  printf("Node[%s, %s]: args format is %s.\n", node.GetNamePtr(), node.GetTypePtr(), new_args_format.c_str());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TileFwkOpUtils::GenerateAicpuTask(const ge::Node &node, const int64_t &sub_stream_id,
                                                  std::vector<domi::TaskDef> &task_defs) {
  size_t i = 0;
  for (auto &task : task_defs) {
    if (task.type() == RT_MODEL_TASK_ALL_KERNEL) {
      break;
    }
    ++i;
  }
  if (i == task_defs.size()) {
    printf("Node[%s, %s]: failed to find aicore task.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  domi::TaskDef aicpu_task;
  aicpu_task.set_type(RT_MODEL_TASK_KERNEL);
  aicpu_task.set_stream_id(sub_stream_id);
  auto kernel_def = aicpu_task.mutable_kernel();
  auto aicore_kernel_def = task_defs[i].mutable_kernel_with_handle();
  std::string cust_block_dim_str;
  (void)ge::AttrUtils::GetStr(node.GetOpDesc(), kAicCntKeyOp, cust_block_dim_str);
  int32_t block_dim = 0;
  block_dim = std::atoi(cust_block_dim_str.c_str());
  if (block_dim > 0) {
    aicore_kernel_def->set_block_dim(block_dim);
  } else if (ge::AttrUtils::GetInt(node.GetOpDesc(), ge::TVM_ATTR_NAME_BLOCKDIM, block_dim) && block_dim > 0) {
    aicore_kernel_def->set_block_dim(block_dim);
  } else {
    printf("Node[%s, %s]: failed to get block dim.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  printf("Node[%s, %s]: block dim is %ld.", node.GetNamePtr(), node.GetTypePtr(), block_dim);
  auto aicore_context = aicore_kernel_def->mutable_context();
  if (TileFwkOpUtils::InsertHiddenInput(node, aicore_context) != ge::GRAPH_SUCCESS) {
    printf("Node[%s, %s]: failed to insert hidden input.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }

  kernel_def->set_args(aicore_kernel_def->args());
  kernel_def->set_args_size(aicore_kernel_def->args_size());
  kernel_def->set_so_name(kAicpuSoName);
  kernel_def->set_kernel_name(kAicpuMainTaskKernelName);
  kernel_def->set_block_dim(kAicpuMainTaskNum);

  auto context = kernel_def->mutable_context();
  context->set_kernel_type(static_cast<uint32_t>(ge::ccKernelType::AI_CPU_KFC));
  context->set_op_index(node.GetOpDesc()->GetId());
  context->set_args_format(aicore_context->args_format());

  domi::TaskDef aicpu_init_task = aicpu_task;
  auto init_kernel_def = aicpu_init_task.mutable_kernel();
  init_kernel_def->set_kernel_name(kAicpuInitTaskKernelName);
  init_kernel_def->set_block_dim(kAicpuInitTaskNum);
  task_defs.emplace_back(aicpu_init_task);
  task_defs.emplace_back(aicpu_task);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TileFwkOpUtils::CommonGenerateTask(const ge::Node &node, ge::RunContext &run_context,
                                                   std::vector<domi::TaskDef> &task_defs) {
  std::vector<ge::GeAttrValue::NAMED_ATTRS> stream_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node.GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         stream_info_attrs);
  if (stream_info_attrs.empty()) {
    printf("Node[%s, %s]: failed to get stream info.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  bool is_valid = false;
  (void)ge::AttrUtils::GetBool(stream_info_attrs[0], ge::ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, is_valid);
  if (!is_valid) {
    printf("Node[%s, %s]: stream info is not valid.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  int64_t sub_stream_id{-1};
  (void)ge::AttrUtils::GetInt(stream_info_attrs[0], ge::ATTR_NAME_ATTACHED_RESOURCE_ID, sub_stream_id);

  std::vector<ge::GeAttrValue::NAMED_ATTRS> sync_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node.GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         sync_info_attrs);
  if (sync_info_attrs.empty()) {
    printf("Node[%s, %s]: failed to get sync info.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  (void)ge::AttrUtils::GetBool(sync_info_attrs[0], ge::ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, is_valid);
  if (!is_valid) {
    printf("Node[%s, %s]: sync info is not valid.\n", node.GetNamePtr(), node.GetTypePtr());
    return ge::GRAPH_FAILED;
  }
  return TileFwkOpUtils::GenerateAicpuTask(node, sub_stream_id, task_defs);
}
} // namespace ops

