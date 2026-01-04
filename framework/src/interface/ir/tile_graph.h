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
 * \file operation.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>
#include "value.h"
#include "operation.h"

namespace npu::tile_fwk {
namespace experimental {

class TileType : public Type {

};

class TileValue : public Value {
public:
    std::shared_ptr<ScalarType> GetType();
};

class TileOp : public Operation {
public:
    std::shared_ptr<TileValue> GetInOperand(size_t index) const;
    std::shared_ptr<TileValue> GetOutOperand(size_t index) const;
private:
};

class ElementWiseTileOp : public TileOp {
};

class ElementWiseUnaryTileOp : public TileOp {
};

class ElementWiseBinaryTileOp : public ElementWiseTileOp {
};

class ElementWiseScalarMixBinaryTileOp : public ElementWiseTileOp {
};

class ReduceTileOp : public TileOp {
};

class BroadcastTileOp : public TileOp {
};

class DataCopyTileOp : public TileOp {
};

class MatmulTileOp : public TileOp {
};

class SysOp : public ScalarOp {
public:
    std::string GetName() const;
private:
    std::string name_;
};

} // namespace experimental
} // namespace npu::tile_fwk
