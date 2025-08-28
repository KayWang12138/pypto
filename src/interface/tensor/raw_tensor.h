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
 * \file raw_tensor.h
 * \brief
 */

#pragma once

#include <cstdlib>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "tilefwk/tilefwk.h"
#include "common/pre_def.h"
#include "tilefwk/data_type.h"
#include "interface/utils/log.h"

using Json = nlohmann::json;

namespace npu::tile_fwk {
class RawTensor {
public:
    struct TensorInfo {
        int tensorIndex{-1}; // 用户定义Tensor的唯一标识
        int subscript{-1}; // 该用户定义Tensor是第几个传参

        bool operator==(const TensorInfo &other) const {
            return tensorIndex == other.tensorIndex && subscript == other.subscript;
        }
    };

public:
    int rawmagic;
    int memoryId{-1};
    int actualRawmagic = -1;
    Shape rawshape;
    Shape oriRawshape;
    std::vector<SymbolicScalar> dynRawShape;
    DataType datatype;
    std::string symbol;
    uint64_t addrOffset = UINT64_MAX;
    RawTensor(DataType t, std::vector<int64_t> tshape, std::string tname = "", int trawmagic = -1);

    RawTensor(RawTensor &&) = delete;
    RawTensor(const RawTensor &other) = delete;
    RawTensor &operator=(RawTensor &&) = delete;
    RawTensor &operator=(const RawTensor &) = delete;

    int GetIndex(const std::vector<int> &indices) const;

    Element &operator()(const std::vector<int> &indices);

    const Element &operator()(const std::vector<int> &indices) const;

    Json DumpJson() const;
    static std::shared_ptr<RawTensor> LoadJson(const Json &rawTensorDump);

    std::string DumpType() const;
    std::string DumpSSA(bool showType = true, bool showSymbol = true) const;
    std::string DumpASM() const;

    std::string Dump() const;

    void InitData(const Element &value);
    void InitData(const std::vector<Element> &values);

    bool IsDummy() const;
    void SetIsDummy(bool dummy = true);

    void AddRefCount(int value);
    auto GetRefCount() const { return refCount_; }

    int GetRawMagic() const {
        if (actualRawmagic != -1) {
            return actualRawmagic;
        } else {
            return rawmagic;
        }
    }
    const std::string &GetSymbol() const { return symbol; }
    void SetSymbol(std::string s) { symbol = std::move(s); }
    DataType GetDataType() const { return datatype; }
    const std::vector<Element> &GetData() const { return data; }
    void SetData(const std::vector<Element> &srcData) { data = srcData; }
    const Shape &GetRawShape() const { return rawshape; }
    int64_t GetRawShapeSize() const;
    int64_t GetRawDataSize() const;
    const std::vector<SymbolicScalar> &GetDynRawShape() const { return dynRawShape; }
    SymbolicScalar GetDynRawShape(int axis) const { return dynRawShape[axis]; }
    void UpdateDynRawShape(const std::vector<SymbolicScalar> &dynShape) { dynRawShape = dynShape; }
    void UpdateRawShape(const std::vector<int64_t> &trawShape) {
        rawshape = trawShape;
        dynRawShape = SymbolicScalar::FromConcrete(trawShape);
    }
    /* rawData just used to identify the user value, RawTensor do not have the ownership */
    BinDataPtr GetRawDataPtr() const { return rawData; };
    void SetRawDataPtr(uint8_t *ptr) { rawData = ptr; }

    void SetTensorIndex(int tensorIndex) { tensorInfo_.tensorIndex = tensorIndex; }
    void SetTensorSubScript(int subscript) { tensorInfo_.subscript = subscript; }
    void SetTensorInfo(const TensorInfo &other) { tensorInfo_ = other; }

    const auto &GetTensorInfo() const { return tensorInfo_; }

    void SetCachePolicy(CachePolicy policy, bool value) {
      cachePolicy_[static_cast<int>(policy)] = value;
      if (value && (cachePolicy_[static_cast<int>(CachePolicy::PREFETCH)] ==
          cachePolicy_[static_cast<int>(CachePolicy::NONE_CACHEABLE)])) {
          ALOG_WARN_F("Prefetch and none cacheable can not apply at same time, use the first config policy.");
          cachePolicy_[static_cast<int>(policy)] = false;
      }
    }

    bool GetCachePolicy(CachePolicy policy) const {
      return cachePolicy_[static_cast<int>(policy)];
    }
private:
    std::vector<Element> data;
    BinDataPtr rawData{nullptr};
    bool isDummy_{false};
    int refCount_{0}; // 被 npu::tile_fwk::Tensor引用的次数，用于outcast自动推导
    TensorInfo tensorInfo_{};
    bool cachePolicy_[static_cast<int>(CachePolicy::MAX_NUM)] = {false};
};
} // namespace npu::tile_fwk
