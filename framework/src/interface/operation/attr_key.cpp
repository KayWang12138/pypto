
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "attr_key.h"

namespace npu::tile_fwk {
const std::string OpAttributeKey::aicpuCall = "AICPU_CALL";
const std::string OpAttributeKey::color = "COLOR";
const std::string OpAttributeKey::scalar = "SCALAR";
const std::string OpAttributeKey::dynScalar = "DYN_SCALAR";
const std::string OpAttributeKey::vectorScalar = "VECTORSCALAR";
const std::string OpAttributeKey::isGlobalInput = "IS_GLOBAL_INPUT";
const std::string OpAttributeKey::seqNo = "SEQ_NO";
const std::string OpAttributeKey::isCube = "IS_CUBE";
const std::string OpAttributeKey::blockPadding = "BLOCK_PADDING";
const std::string OpAttributeKey::tilePadding = "TILE_PADDING";
const std::string OpAttributeKey::reshapePadding = "RESHAPE_PADDING";
const std::string OpAttributeKey::shapePadded = "SHAPE_PADDED";
const std::string OpAttributeKey::needAlloc = "NEED_ALLOC";
const std::string OpAttributeKey::broadcastLastAxis = "BROADCAST_LAST_AXIS";
const std::string OpAttributeKey::dontTouch = "DONT_TOUCH";
const std::string OpAttributeKey::tag = "TAG";
const std::string OpAttributeKey::distTilingInfo = "DIST_TILING_INFO";
const std::string OpAttributeKey::sameInOut = "SAME_IN_OUT";
const std::string OpAttributeKey::inputCombineAxis = "op_attr_input_combine_axis";
const std::string OpAttributeKey::outputCombineAxis = "op_attr_output_combine_axis";
const std::string OpAttributeKey::inplaceIdx = "INPLACE_IDX";
const std::string OpAttributeKey::inplaceInfo = "INPLACE_INFO";
const std::string OpAttributeKey::cacheMode = "CACHE_MODE";
const std::string OpAttributeKey::panzBlockSize = "PA_NZ_BLOCK_SIZE";
const std::string OpAttributeKey::inputCombineAxisDone = "input_combine_axis_done";   // only for flow verify tool
const std::string OpAttributeKey::outputCombineAxisDone = "output_combine_axis_done"; // flow verify tool only
const std::string OpAttributeKey::requiresBoundaryCopy = "requires_boundary_copy";
const std::string OpAttributeKey::excludeBufferReuse = "exclude_buffer_reuse";
const std::string OpAttributeKey::bindTensor = "BIND_TENSOR";
const std::string OpAttributeKey::startOffset = "start_offset";
const std::string OpAttributeKey::distOpAttr = "DIST_OP_ATTR";
const std::string OpAttributeKey::subBlockIdx = "SUB_BLOCK_IDX";
const std::string OpAttributeKey::accumulate = "accumulate";
const std::string OpAttributeKey::indicesSize = "indicesSize";
const std::string OpAttributeKey::brcbIdx = "brcb_idx";
const std::string OpAttributeKey::quantFlag = "op_attr_vector_quant_flag";
const std::string OpAttributeKey::loopGroup = "LOOP_GROUP";
const std::string OpAttributeKey::loopAxes = "LOOP_AXES";
const std::string OpAttributeKey::loopGroupStart = "LOOP_GROUP_START";
const std::string OpAttributeKey::loopGroupEnd = "LOOP_GROUP_END";
const std::string OpAttributeKey::lastUse = "last_use";
const std::string OpAttributeKey::isUpper = "is_upper";
const std::string OpAttributeKey::blockSize = "block_size";

const std::string ConvOpAttributeKey::cin = "CIN";
const std::string ConvOpAttributeKey::cout = "COUT";
const std::string ConvOpAttributeKey::paddingLeft = "PADDING_LEFT";
const std::string ConvOpAttributeKey::paddingTop = "PADDING_TOP";
const std::string ConvOpAttributeKey::paddingRight = "PADDING_RIGHT";
const std::string ConvOpAttributeKey::paddingBottom = "PADDING_BOTTOM";
const std::string ConvOpAttributeKey::strideh = "STRIDEH";
const std::string ConvOpAttributeKey::stridew = "STRIDEW";
const std::string ConvOpAttributeKey::hposX = "HPOS_X";
const std::string ConvOpAttributeKey::hsteP = "HSTEP";
const std::string ConvOpAttributeKey::wposX = "WPOS_X";
const std::string ConvOpAttributeKey::wstep = "WSTEP";
const std::string ConvOpAttributeKey::hoffsetY = "HOFFSET_Y";
const std::string ConvOpAttributeKey::woffsetY = "WOFFSET_Y";
const std::string ConvOpAttributeKey::reluType = "RELU_TYPE";
const std::string ConvOpAttributeKey::reluAlpha = "RELU_ALPHA";
const std::string ConvOpAttributeKey::clearFlag = "CLEAR_FLAG";
const std::string ConvOpAttributeKey::hasAccFlag = "HAS_ACC_FLAG";
const std::string ConvOpAttributeKey::hasEltFlag = "HAS_ELT_FLAG";
const std::string ConvOpAttributeKey::hasBiasFlag = "HAS_BIAS_FLAG";
const std::string ConvOpAttributeKey::eltBrcbFlag = "ELT_BRCB_FLAG";
const std::string ConvOpAttributeKey::fmapSrcNum = "FMAP_SRC_NUM";
const std::string ConvOpAttributeKey::eltMode = "ELT_MODE";
const std::string ConvOpAttributeKey::fmapC0 = "FMAP_C0";

const std::string FixpOpAttributeKey::quantPreScalar = "QUANT_PRE_SCALAR";
const std::string FixpOpAttributeKey::quantPostScalar = "QUANT_POST_SCALAR";
const std::string FixpOpAttributeKey::antiqScalar = "ANTIQ_SCALAR";
const std::string FixpOpAttributeKey::hasQuantPreVector = "HAS_QUANT_PRE_VECTOR";
const std::string FixpOpAttributeKey::hasQuantPostVector = "HAS_QUANT_POST_VECTOR";
const std::string FixpOpAttributeKey::hasAntiqVector = "HAS_ANTIQ_VECTOR";
const std::string FixpOpAttributeKey::fbAddrSpace = "FIX_BUFFER_ADDR_SPACE";

const std::string PoolOpAttributeKey::poolh = "POOL_WIN_H";
const std::string PoolOpAttributeKey::poolw = "POOL_WIN_W";
} // namespace npu::tile_fwk
