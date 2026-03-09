/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>

namespace npu::tile_fwk {
struct OpAttributeKey {
    static const std::string aicpuCall;
    static const std::string scalar;
    static const std::string vectorScalar;
    static const std::string dynScalar;
    static const std::string isGlobalInput;
    static const std::string seqNo;
    static const std::string color;
    static const std::string isCube;
    static const std::string blockPadding;
    static const std::string broadcastLastAxis;
    static const std::string tilePadding;
    static const std::string reshapePadding;
    static const std::string shapePadded;
    static const std::string needAlloc;
    static const std::string dontTouch;
    static const std::string tag;
    static const std::string distTilingInfo;
    static const std::string sameInOut;
    static const std::string inputCombineAxis;
    static const std::string outputCombineAxis;
    static const std::string inputCombineAxisDone;
    static const std::string outputCombineAxisDone;
    static const std::string inplaceIdx;
    static const std::string inplaceInfo;
    static const std::string cacheMode;
    static const std::string panzBlockSize;
    static const std::string requiresBoundaryCopy;
    static const std::string excludeBufferReuse;
    static const std::string bindTensor;
    static const std::string startOffset;
    static const std::string distOpAttr;
    static const std::string subBlockIdx;
    static const std::string accumulate;
    static const std::string indicesSize;
    static const std::string brcbIdx;
    static const std::string quantFlag;
    static const std::string loopGroup;
    static const std::string loopAxes;
    static const std::string loopGroupStart;
    static const std::string loopGroupEnd;
    static const std::string lastUse;
    static const std::string isUpper;
    static const std::string blockSize;
};

struct ConvOpAttributeKey {
    static const std::string cin;
    static const std::string cout;
    static const std::string paddingLeft;
    static const std::string paddingTop;
    static const std::string paddingRight;
    static const std::string paddingBottom;
    static const std::string strideh;
    static const std::string stridew;
    static const std::string hposX;
    static const std::string hsteP;
    static const std::string wposX;
    static const std::string wstep;
    static const std::string hoffsetY;
    static const std::string woffsetY;
    static const std::string reluType;
    static const std::string reluAlpha;
    static const std::string clearFlag;
    static const std::string hasAccFlag;
    static const std::string hasEltFlag;
    static const std::string hasBiasFlag;
    static const std::string eltBrcbFlag;
    static const std::string fmapSrcNum;
    static const std::string eltMode;
    static const std::string fmapC0;
};

struct FixpOpAttributeKey {
    static const std::string hStart;
    static const std::string hEnd;
    static const std::string quantPreScalar;
    static const std::string quantPostScalar;
    static const std::string antiqScalar;
    static const std::string hasQuantPreVector;
    static const std::string hasQuantPostVector;
    static const std::string hasAntiqVector;
    static const std::string fbAddrSpace;
};

struct PoolOpAttributeKey {
    static const std::string poolh;
    static const std::string poolw;
};

struct L12L0ConvOpAttributeKey {
    static const std::string postK;
    static const std::string postM;
    static const std::string postN;
    static const std::string filterH;
    static const std::string filterW;
    static const std::string strideH;
    static const std::string strideW;
    static const std::string dilationH;
    static const std::string dilationW;
    static const std::string paddingLeft;
    static const std::string paddingRight;
    static const std::string paddingTop;
    static const std::string paddingBottom;
    static const std::string padValue;
};

struct LoadStoreConvOpAttributeKey {
    static const std::string copyInMode;
    static const std::string copyOutMode;
    static const std::string isFmap;
    static const std::string isConv3D;
};
} // namespace npu::tile_fwk
