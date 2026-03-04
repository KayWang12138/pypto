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
 * \file operation_impl.h
 * \brief
 */

#pragma once
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "interface/configs/config_manager.h"
#include "opcode.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tile_shape.h"

namespace npu::tile_fwk {
class Function;
class Operation;
using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;

void ExpandOperationInto(Function &function, const TileShape &tileShape, Opcode opCode,
                         const std::vector<std::shared_ptr<LogicalTensor>> &iOperand,
                         const std::vector<std::shared_ptr<LogicalTensor>> &oOperand, const Operation &op);

namespace Matrix {
const size_t M_INDEX = 0;
const size_t K_INDEX = 1;
const size_t N_INDEX = 2;
const int32_t MATRIX_MAXSIZE = 3;

const std::string ACC_A_MUL_B = OP_ATTR_PREFIX + "atomic_add";
const std::string MATMUL_NZ_ATTR = OP_ATTR_PREFIX + "matmul_nz_attr";
const std::string A_MUL_B_ACT_M = OP_ATTR_PREFIX + "act_m";
const std::string A_MUL_B_ACT_K = OP_ATTR_PREFIX + "act_k";
const std::string A_MUL_B_ACT_N = OP_ATTR_PREFIX + "act_n";
const std::string A_MUL_B_TRANS_A = OP_ATTR_PREFIX + "trans_a";
const std::string A_MUL_B_SCALE_A_COPY_IN_MODE = OP_ATTR_PREFIX + "scale_a_copy_in_mode";
const std::string A_MUL_B_TRANS_B = OP_ATTR_PREFIX + "trans_b";
const std::string A_MUL_B_SCALE_B_COPY_IN_MODE = OP_ATTR_PREFIX + "scale_b_copy_in_mode";
const std::string A_MUL_B_GM_ACC = OP_ATTR_PREFIX + "gm_acc";
const std::string A_MUL_B_MX_ATTR = OP_ATTR_PREFIX + "is_mx";
const std::string COPY_IN_L1_PADDING_MODE = OP_ATTR_PREFIX + "copy_in_l1_padding_mode";
const std::string L1_TO_L0_TRANSPOSE = OP_ATTR_PREFIX + "l1_to_l0_transpose";
const std::string A_MUL_B_BIAS_ATTR = OP_ATTR_PREFIX + "has_bias";
const std::string A_MUL_B_SCALE_ATTR = OP_ATTR_PREFIX + "scale_value";
// relu type 0: NoReLu, 1: ReLu
const std::string A_MUL_B_RELU_ATTR = OP_ATTR_PREFIX + "relu_type";
const std::string A_MUL_B_VECTOR_QUANT_FLAG = OP_ATTR_PREFIX + "vector_quant_flag";

const std::unordered_map<NPUArch, uint64_t> ubSizeMap = {
    {NPUArch::DAV_1001, 262144},
    {NPUArch::DAV_2201, 196608},
    {NPUArch::DAV_3510, 253952},
};

const std::unordered_map<std::string, NPUArch> socVersionToNpuArchMap = {
    // DAV_1001 架构
    {       "Ascend910A", NPUArch::DAV_1001},
    {       "Ascend910B", NPUArch::DAV_1001},
    {"Ascend910PremiumA", NPUArch::DAV_1001},
    {    "Ascend910ProA", NPUArch::DAV_1001},
    {    "Ascend910ProB", NPUArch::DAV_1001},

    // DAV_2201 架构
    {   "Ascend910_9362", NPUArch::DAV_2201},
    {   "Ascend910_9372", NPUArch::DAV_2201},
    {   "Ascend910_9381", NPUArch::DAV_2201},
    {   "Ascend910_9382", NPUArch::DAV_2201},
    {   "Ascend910_9391", NPUArch::DAV_2201},
    {   "Ascend910_9392", NPUArch::DAV_2201},
    {      "Ascend910B1", NPUArch::DAV_2201},
    {     "Ascend910B2C", NPUArch::DAV_2201},
    {      "Ascend910B2", NPUArch::DAV_2201},
    {      "Ascend910B3", NPUArch::DAV_2201},
    {    "Ascend910B4-1", NPUArch::DAV_2201},
    {      "Ascend910B4", NPUArch::DAV_2201},

    // DAV_3510 架构
    { "Ascend950DT_950x", NPUArch::DAV_3510},
    { "Ascend950DT_950y", NPUArch::DAV_3510},
    { "Ascend950DT_9571", NPUArch::DAV_3510},
    { "Ascend950DT_9572", NPUArch::DAV_3510},
    { "Ascend950DT_9573", NPUArch::DAV_3510},
    { "Ascend950DT_9574", NPUArch::DAV_3510},
    { "Ascend950DT_9575", NPUArch::DAV_3510},
    { "Ascend950DT_9576", NPUArch::DAV_3510},
    { "Ascend950DT_9577", NPUArch::DAV_3510},
    { "Ascend950DT_9578", NPUArch::DAV_3510},
    { "Ascend950DT_9581", NPUArch::DAV_3510},
    { "Ascend950DT_9582", NPUArch::DAV_3510},
    { "Ascend950DT_9583", NPUArch::DAV_3510},
    { "Ascend950DT_9584", NPUArch::DAV_3510},
    { "Ascend950DT_9585", NPUArch::DAV_3510},
    { "Ascend950DT_9586", NPUArch::DAV_3510},
    { "Ascend950DT_9587", NPUArch::DAV_3510},
    { "Ascend950DT_9588", NPUArch::DAV_3510},
    { "Ascend950DT_9591", NPUArch::DAV_3510},
    { "Ascend950DT_9592", NPUArch::DAV_3510},
    { "Ascend950DT_9595", NPUArch::DAV_3510},
    { "Ascend950DT_9596", NPUArch::DAV_3510},
    { "Ascend950DT_95A1", NPUArch::DAV_3510},
    { "Ascend950DT_95A2", NPUArch::DAV_3510},
    { "Ascend950PR_950z", NPUArch::DAV_3510},
    { "Ascend950PR_9579", NPUArch::DAV_3510},
    { "Ascend950PR_957b", NPUArch::DAV_3510},
    { "Ascend950PR_957c", NPUArch::DAV_3510},
    { "Ascend950PR_957d", NPUArch::DAV_3510},
    { "Ascend950PR_9589", NPUArch::DAV_3510},
    { "Ascend950PR_958b", NPUArch::DAV_3510},
    { "Ascend950PR_9599", NPUArch::DAV_3510},
};

struct MatmulTensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> offset;
    NodeType nodeType;
    TileOpFormat format;
    MemoryType memType;
    bool transFlag;

    MatmulTensorInfo(const std::string &nameIn, DataType dtypeIn, const std::vector<int64_t> &shapeIn,
                     const std::vector<int64_t> &offsetIn, NodeType nodeTypeIn, TileOpFormat formatIn,
                     MemoryType memTypeIn, bool transFlagIn = false)
        : name(nameIn),
          dtype(dtypeIn),
          shape(shapeIn),
          offset(offsetIn),
          nodeType(nodeTypeIn),
          format(formatIn),
          memType(memTypeIn),
          transFlag(transFlagIn)
    {
    }
};

struct MatmulTileInfo {
    int64_t mView = 0;
    int64_t kView = 0;
    int64_t nView = 0;
    int64_t tileML1 = 0;
    int64_t tileML0 = 0;
    int64_t tileNL1 = 0;
    int64_t tileNL0 = 0;
    int64_t tileKL0 = 0;
    int64_t tileKAL1 = 0;
    int64_t tileKBL1 = 0;
};

struct MatmulIterInfo {
    int64_t mOffset = 0;
    int64_t nOffset = 0;
    int64_t kOffset = 0;
    int64_t mL1Size = 0;
    int64_t mL0Size = 0;
    int64_t nL1Size = 0;
    int64_t nL0Size = 0;
    int64_t kAL1Size = 0;
    int64_t kBL1Size = 0;
    int64_t kL0Size = 0;
    bool isFirstK = false;
    bool isLastK = false;
};

struct MatmulGraphNodes {
    LogicalTensorPtr aTensorPtr = nullptr;
    LogicalTensorPtr aScaleTensorPtr = nullptr;
    LogicalTensorPtr bTensorPtr = nullptr;
    LogicalTensorPtr bScaleTensorPtr = nullptr;
    LogicalTensorPtr gmAccumulationTensorPtr = nullptr;
    LogicalTensorPtr biasTensorPtr = nullptr;
    LogicalTensorPtr scaleTensorPtr = nullptr;
    LogicalTensorPtr cL0PartialSumPtr = nullptr;
    LogicalTensorPtr outTensorPtr = nullptr;

    MatmulGraphNodes() = default;

    MatmulGraphNodes(LogicalTensorPtr aTensorIn, LogicalTensorPtr bTensorIn)
        : aTensorPtr(aTensorIn), bTensorPtr(bTensorIn){};

    MatmulGraphNodes(LogicalTensorPtr aTensorIn, LogicalTensorPtr bTensorIn, LogicalTensorPtr gmAccumulationTensorIn)
        : aTensorPtr(aTensorIn), bTensorPtr(bTensorIn), gmAccumulationTensorPtr(gmAccumulationTensorIn){};

    MatmulGraphNodes(LogicalTensorPtr aTensorIn, LogicalTensorPtr aScaleTensorIn, LogicalTensorPtr bTensorIn,
        LogicalTensorPtr bScaleTensorIn)
        : aTensorPtr(aTensorIn),
          aScaleTensorPtr(aScaleTensorIn),
          bTensorPtr(bTensorIn),
          bScaleTensorPtr(bScaleTensorIn){};
};

struct MatmulAttrParam {
    int64_t mValue = 0;
    int64_t kValue = 0;
    int64_t nValue = 0;
    int64_t reluType = 0;
    uint64_t scaleValue = 0;
    bool hasBias = false;
    bool hasScale = false;
    bool transA = false;
    bool transAScale = false;
    bool transB = false;
    bool transBScale = false;
    bool hasMXScale = false;
    bool gmAccumulationFlag = false;
    bool isCMatrixNZ = false;

    MatmulAttrParam() = default;

    MatmulAttrParam(bool isATrans, bool isBTrans, bool cMatrixNZ) {
        transA = isATrans;
        transB = isBTrans;
        isCMatrixNZ = cMatrixNZ;
    }

    MatmulAttrParam(bool isATrans, bool isAScaleTrans, bool isBTrans, bool isBScaleTrans, bool cMatrixNZ)
        : transA(isATrans),
          transAScale(isAScaleTrans),
          transB(isBTrans),
          transBScale(isBScaleTrans),
          isCMatrixNZ(cMatrixNZ) {
        hasMXScale = true;
    }
};

void ConstructTileGraph(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &operandVec,
                        const LogicalTensorPtr &cTensorPtr, const Operation &op);
}  // namespace Matrix
}  // namespace npu::tile_fwk
