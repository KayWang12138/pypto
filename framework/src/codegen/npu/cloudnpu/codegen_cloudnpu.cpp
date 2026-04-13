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
 * \file codegen.cpp
 * \brief
 */
#include "codegen_cloudnpu.h"

#include <cstring>
#include <error.h>
#include <fstream>

#include "codegen/npu/codegen_npu.h"
#include "codegen/utils/parallel_execute.h"
#include "codegen_op_cloudnpu.h"
#include "codegen/stmt_mgr/codegen_for_block.h"
#include "interface/utils/file_utils.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "interface/operation/distributed/distributed_common.h"
#include "tilefwk/tilefwk.h"
#include "securec.h"

namespace npu::tile_fwk {} // namespace npu::tile_fwk
