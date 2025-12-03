/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.|Hisilicon Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_print_param_def.h
 * \brief
 */

#ifndef OP_PRINT_PARAM_DEF_H
#define OP_PRINT_PARAM_DEF_H

#include <string>
#include <vector>

namespace npu::tile_fwk {
struct PrintScatterElemParam {
    int axis;
    int scatterMode;
    const std::string &dVar;
    const std::string &s0Var;
    const std::string &s1Var;
    std::vector<int64_t> &dstRawShape;
    std::vector<int64_t> &src1RawShape;
    const std::string *dataTypeExpr;
};

struct PrintScatterParam {
    int axis;
    int scatterMode;
    const std::string &dVar;
    const std::string &s1Var;
    const std::string &s2Var;
    std::vector<int64_t> &dstRawShape;
    std::vector<int64_t> &src1RawShape;
    std::vector<int64_t> &src2RawShape;
    const std::string *dataTypeExpr;
};

struct PrintIndexAddParam {
    int axis;
    const std::string &dVar;
    const std::string &s0Var;
    const std::string &s1Var;
    const std::string &idxVar;
    std::vector<int64_t> &dstRawShape;
    std::vector<int64_t> &src0RawShape;
    std::vector<int64_t> &src1RawShape;
    const std::string *dataTypeExpr;
};

enum class WhereOpIdx : int {
    resIdx = 0,
    tempIdx,
    condIdx,
    src0Idx,
    src1Idx
};
struct WhereParam {
    std::vector<std::string> templateList;
    std::vector<std::string> paramList;
    std::vector<std::string> dynParamList;
    std::vector<std::string> varExpr;
    std::vector<std::string> dataTypeExpr;
};

struct PrintDupOpParam {
    const std::string &dVar;
    const std::string &dstDtypeStr;
    const std::string &dupV;
};

struct PrintUnaryParam {
    const std::string &s0Var;
    const std::string &dVar;
    const std::string &srcDtypeStr;
    const std::string &dstDtypeStr;
};

struct PrintUnaryTmpBuffParam {
    const std::string &s0Var;
    const std::string &tmpVar;
    const std::string &dVar;
    const std::string &srcDtypeStr;
    const std::string &tmpDtypeStr;
    const std::string &dstDtypeStr;
};

struct PrintMemCopyWithL0CParam {
    unsigned uf;
    unsigned gmIdx;
    unsigned localIdx;
    const std::string *addrTypeHead;
    const std::string *addrExpr;
    const std::vector<int64_t> &gmShape;
    const std::vector<int64_t> &tileShapeForMT;
    const std::string *dataTypeExpr;
};

struct PrintMemCopyWithL1Param {
    unsigned uf;
    unsigned gmIdx;
    unsigned localIdx;
    const std::string *addrTypeHead;
    const std::string *addrExpr;
    const std::vector<int64_t> &gmShape;
    const std::vector<int64_t> &tileShapeForMT;
    const std::string *dataTypeExpr;
};

struct PrintMemCopyWithUBParam {
    unsigned gmIdx;
    unsigned localIdx;
    const std::string *addrTypeHead;
    std::string *addrExpr;
    std::string *dataTypeExpr;
    bool isSpillIntoGM;
};

struct PrintGatherParam {
    const std::string &s0Var;
    const std::string &s1Var;
    const std::string &dVar;
    const std::string &src0DtypeStr;
    const std::string &src1DtypeStr;
    const std::string &dstDtypeStr;
    const int64_t axis;
};

struct PrintBinaryScalarParam {
    const std::string &s0Var;
    const std::string &dVar;
    const std::string &src0DtypeStr;
    const std::string &dstDtypeStr;
    const size_t dim;
};

struct PrintBinaryParam {
    const std::string &s0Var;
    const std::string &s1Var;
    const std::string &dVar;
    const std::string &src0DtypeStr;
    const std::string &src1DtypeStr;
    const std::string &dstDtypeStr;
};

struct PrintBinaryBrcParam {
    const std::string &s0Var;
    const std::string &s1Var;
    const std::string &dVar;
    const std::string &tmpVar;
    const std::string &src0DtypeStr;
    const std::string &src1DtypeStr;
    const std::string &dstDtypeStr;
    const std::string &tmpDtypeStr;
};

struct PrintTransposeDataMoveParam {
    const unsigned gmIdx;
    const unsigned localIdx;
    const std::string &localVar;
    const std::vector<int64_t> &gmShape;
    const std::string &localDtypeStr;
    const std::string &gmDtypeStr;
};

struct PrintGatherEleParam {
    int axis;
    const std::string &dVar;
    const std::string &s0Var;
    const std::string &s1Var;
    std::vector<int64_t> &dstOriginShape;
    std::vector<int64_t> &dstRawShape;
    std::vector<int64_t> &src0RawShape;
    std::vector<int64_t> &src1RawShape;
    const std::string *dataTypeExpr;
};

struct PrintIndexOutCastParam {
    const std::string &s0Var;
    const std::string &s1Var;
    const std::string *addrExpr;
    const std::vector<int64_t> &gmShape;
    std::vector<int64_t> &src0OriginShape;
    std::vector<int64_t> &src0RawShape;
    std::vector<int64_t> &src1OriginShape;
    std::vector<int64_t> &src1RawShape;
    const std::string *dataTypeExpr;
    const std::string &cacheMode;
    const std::string &blockSize;
};

struct DynamicParamPackMTE {
    std::vector<std::string> gmShapeExpr;
    std::vector<std::string> gmOffsetExpr;
    std::vector<std::string> paramList;
};

} // namespace npu::tile_fwk

#endif // OP_PRINT_PARAM_DEF_H
