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
 * \file codegen_symbol.h
 * \brief
 */

#pragma once

#include <map>
#include <tuple>
#include <cstdint>
#include <string>

#include "interface/utils/log.h"
#include "tilefwk/error.h"
#include "interface/utils/common.h"
#include "interface/tensor/logical_tensor.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
const std::string TILE_TENSOR = "TileTensor";
const std::string LAYOUT_DIM = "LayoutDim";
const std::string SCOPE_NAMESPACE = "Hardware";
using BufferType = enum OperandType;
using AllocKey = std::tuple<BufferType, int64_t /*RangeStart*/, int64_t /*RangeEnd*/>;

// e.g.
// UBTileTensorFP32Dim2 ubTile_0((__ubuf__ float*)UB_S0_E16384, DimLayout2(Shape<int, int>(sym_18_dim_0, sym_18_dim_1),
// Stride<int, int>(64, 1)));
struct TileTensor {
    int magic; // tensor magic numbuer
    int dim;
    DataType dtype;
    BufferType bufType;
    std::string bufVar;
    std::string usingType;
    std::string tensorName; // e.g. "ubTile_0"
    std::vector<std::string> shape;
    std::vector<std::string> stride;

    bool operator==(const TileTensor &other) const { return dim == other.dim && bufVar == other.bufVar; }

    /*  e.g.
        ((__ubuf__ float*)UB_S0_E16384,
        DimLayout2(Shape<int, int>(sym_18_dim_0, sym_18_dim_1), Stride<int, int>(64, 1)));
    */
    std::string GenInitParam() const {
        std::ostringstream oss;
        std::vector<std::string> params;
        // (__ubuf__ float*)UB_S0_E16384
        oss << "(" << OPERAND_TYPE_TO_ADDR_TYPE.at(bufType) << " " << DataType2CCEStr(dtype) << "*)" << bufVar;
        params.emplace_back(oss.str());
        oss.str("");
        // DimLayout2(Shape<int, int>(sym_18_dim_0, sym_18_dim_1), Stride<int, int>(64, 1)));
        oss << LAYOUT_DIM << dim << "(" << GenShapeParam() << ", " << GenStrideParam() << ")";
        params.emplace_back(oss.str());
        return PrintParams({"(", ")"}, params, ", ");
    }

    std::string ToString() const {
        std::ostringstream oss;
        oss << usingType << " " << tensorName << GenInitParam() << ";\n";
        return oss.str();
    }

private:
    std::string GenLayoutParam(const std::string &paramName, const std::vector<std::string>& paramValue) const {
        std::vector<std::string> templateParam(dim, "int");
        std::ostringstream oss;
        oss << paramName;
        oss << PrintParams({"<", ">"}, templateParam, ", ");
        oss << PrintParams({"(", ")"}, paramValue, ", ");
        return oss.str();
    }
    std::string GenShapeParam() const { return GenLayoutParam("Shape", shape); }
    std::string GenStrideParam() const { return GenLayoutParam("Stride", stride); }
};

struct TileTensorHash {
    std::size_t operator()(const TileTensor &t) const noexcept {
        std::size_t seed = 0;
        HashCombine(seed, t.dim);
        HashCombine(seed, t.bufVar);
        return seed;
    };
};

struct TileTensorUsing {
    DataType dtype;
    BufferType bufType;
    int dim;

    bool operator==(const TileTensorUsing &other) const {
        return dtype == other.dtype && bufType == other.bufType && dim == other.dim;
    }

    std::string GenName() const {
        std::ostringstream oss;
        // e.g. "UBTileTensorFP32Dim2"
        oss << BUFFER_TYPE_TO_PREFIX.at(bufType) << TILE_TENSOR << BriefDataType2String(dtype) << "Dim" << dim;
        return oss.str();
    }

    // e.g. "TileTensor<__ubuf__ float, LayoutDim2, Hardware::UB>"
    std::string ToString() const {
        std::ostringstream ss;
        ss << TILE_TENSOR << "<" << GetAddrTypeByOperandType(bufType) << " " << DataType2CCEStr(dtype) << ", "
           << LAYOUT_DIM << dim << ", " << SCOPE_NAMESPACE << "::" << BUFFER_TYPE_TO_PREFIX.at(bufType) << ">;\n";
        return ss.str();
    }
};

class SymbolManager {
public:
    SymbolManager() = default;
    virtual ~SymbolManager() = default;

    using AllocRecord = std::pair<uint64_t /*AllocaAddr*/, unsigned /*AllocaSize*/>;

    virtual std::string QueryVariableName(const AllocKey &key);
    virtual std::string QueryVarNameByTensorMagic(int magic);
    SymbolManager(SymbolManager &other) = delete;

    void operator=(const SymbolManager &other) = delete;

    bool BindAddrWithVariableName(const AllocKey &key, const std::string &varName);

    void AddToTensorMap(int magicNum, const std::shared_ptr<LogicalTensor> &tensor) {
        tensorMap_.insert({magicNum, tensor});
    }

    static std::string FormatAllocKey(const AllocKey &key);

    std::string AddTileTensorUsing(const TileTensorUsing &tileTensorUsing);
    void AddTileTensor(const TileTensor &tileTensor);
    std::string QueryTileTensorByMagic(int magic);

    std::string GenUsingList();
    std::string GenTileTensorDefList();

private:
    std::shared_ptr<LogicalTensor> GetTensorByMagic(int magicNum) const;
    AllocKey CreateAllocKey(const std::shared_ptr<LogicalTensor> &tensor) const;
    AllocKey CreateAllocKey(int tensorMagicNum) const;

    // <AllocKey, buffer variable name>
    std::map<AllocKey, std::string> key2VariableName_;
    //<tensor magic, LogicalTensor>
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorMap_;
    //<TileTensor, tensorName>
    std::unordered_map<TileTensor, std::string, TileTensorHash> tileTensor_;
    //<tensor magic, tensorName>
    std::unordered_map<int, std::string> tileTensorByMagic_;
    //<using type, TileTensorUsing>
    std::unordered_map<std::string, TileTensorUsing> tileTensorUsing_;
};
} // namespace npu::tile_fwk
