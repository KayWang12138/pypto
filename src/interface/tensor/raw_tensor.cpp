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
 * \file raw_tensor.cpp
 * \brief
 */

#include "interface/configs/config_manager.h"
#include "interface/utils/id_gen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/utils/serialization.h"
#include "raw_tensor.h"
#include <string>

using namespace npu::tile_fwk;
RawTensor::RawTensor(DataType t, std::vector<int64_t> tshape, std::string tname, int trawmagic)
    : rawmagic((trawmagic == -1) ? IdGen<IdType::RAW_TENSOR>::Inst().NewId() : trawmagic),
      rawshape(std::move(tshape)),
      datatype(t),
      symbol(std::move(tname)) {
    dynRawShape = SymbolicScalar::FromConcrete(rawshape);
    memoryId = rawmagic;
}

void RawTensor::InitData(const Element &value) {
    data.clear();
    int totalSize = 1;
    for (int dim : rawshape) {
        totalSize *= dim;
    }
    data.resize(totalSize, value);
}

void RawTensor::InitData(const std::vector<Element> &values) {
    int64_t totalSize = 1;
    for (int dim : rawshape) {
        totalSize *= static_cast<int64_t>(dim);
    }
    assert(values.size() == static_cast<size_t>(totalSize));
    data = values;
}

int RawTensor::GetIndex(const std::vector<int> &indices) const {
    assert(indices.size() == rawshape.size());
    int index = 0;
    int stride = 1;
    for (int i = rawshape.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= rawshape[i];
    }
    assert(static_cast<size_t>(index) < data.size());
    return index;
}

Element &RawTensor::operator()(const std::vector<int> &indices) {
    assert(!data.empty() && "data not initialised yet");
    return data[GetIndex(indices)];
}

const Element &RawTensor::operator()(const std::vector<int> &indices) const {
    assert(!data.empty() && "data not initialised yet");
    return data[GetIndex(indices)];
}

std::string RawTensor::DumpASM() const {
    std::ostringstream oss;
    constexpr size_t width4 = 4;
    constexpr size_t width3 = 3;
    oss << std::setw(width4) << std::setfill(' ') << symbol << "(";
    oss << std::setw(width4) << std::setfill(' ') << rawmagic;
    oss << ")";
    oss << " rawshape:[";
    for (size_t i = 0; i < rawshape.size(); ++i) {
        oss << std::setw(width3) << std::setfill(' ') << rawshape[i];
        if (i != rawshape.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    oss << " oriRawshape:[";
    for (size_t i = 0; i < oriRawshape.size(); ++i) {
        oss << std::setw(width3) << std::setfill(' ') << oriRawshape[i];
        if (i != oriRawshape.size() - 1) {
            oss << ",";
        }
    }
    oss << "]\\n";
    return oss.str();
}

Json RawTensor::DumpJson() const {
    Json rawTensorDump;
    rawTensorDump[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_RAW_TENSOR);
    rawTensorDump["datatype"] = datatype;
    rawTensorDump["rawshape"] = rawshape;
    rawTensorDump["ori_rawshape"] = oriRawshape;
    rawTensorDump["rawmagic"] = rawmagic;
    rawTensorDump["tensorIndex"] = tensorInfo_.tensorIndex;
    rawTensorDump["tensorSubscript"] = tensorInfo_.subscript;
    if (actualRawmagic != -1) {
        rawTensorDump["actual_rawmagic"] = actualRawmagic;
    }

    if (symbol != "") {
        rawTensorDump["symbol"] = symbol;
    }

    if (rawData != nullptr) {
        rawTensorDump["raw_data_ptr"] = reinterpret_cast<uintptr_t>(rawData);
    }
    return rawTensorDump;
}

std::shared_ptr<RawTensor> RawTensor::LoadJson(const Json &rawTensorDump) {
    ASSERT(rawTensorDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_RAW_TENSOR));
    DataType dtype = static_cast<DataType>(rawTensorDump["datatype"].get<int>());
    std::vector<int64_t> rawshapeJson = rawTensorDump["rawshape"].get<std::vector<int64_t>>();
    int dumpRawmagic = rawTensorDump["rawmagic"].get<int>();
    std::string dumpSymbol;
    if (rawTensorDump.contains("symbol")) {
        dumpSymbol = rawTensorDump["symbol"].get<std::string>();
    }
    auto ret = std::make_shared<RawTensor>(dtype, rawshapeJson, dumpSymbol, dumpRawmagic);
    if (rawTensorDump.count("tensorIndex") > 0) {
        ret->tensorInfo_.tensorIndex = rawTensorDump["tensorIndex"].get<int>();
    }
    if (rawTensorDump.count("tensorSubscript") > 0) {
        ret->tensorInfo_.subscript = rawTensorDump["tensorSubscript"].get<int>();
    }
    if (rawTensorDump.count("actual_rawmagic") != 0) {
        ret->actualRawmagic = rawTensorDump["actual_rawmagic"].get<int>();
    }
    ret->oriRawshape = rawTensorDump["ori_rawshape"].get<std::vector<int64_t>>();
    if (rawTensorDump.count("raw_data_ptr") != 0) {
        ret->SetRawDataPtr(reinterpret_cast<uint8_t *>(rawTensorDump["raw_data_ptr"].get<uintptr_t>()));
    }
    return ret;
}

std::string RawTensor::DumpType() const {
    std::string result = "<";
    for (auto &value : rawshape) {
        result += std::to_string((value)) + " x ";
    }
    result += DataType2String(datatype);
    result += ">";
    return result;
}

std::string RawTensor::DumpSSA(bool showType, bool showSymbol) const {
    std::ostringstream oss;
    if (showType) {
        oss << DumpType() << " ";
    }
    oss << "@" << GetRawMagic();
    if (showSymbol) {
        if (GetSymbol().size() != 0) {
            oss << "\"" << GetSymbol() << "\"";
        }
    }
    return oss.str();
}

std::string RawTensor::Dump() const {
    if (config::GetPlatformConfig("USE_SSA", true)) {
        return DumpSSA();
    } else {
        return DumpASM();
    }
}

bool RawTensor::IsDummy() const {
    return isDummy_;
}

void RawTensor::SetIsDummy(bool dummy) {
    isDummy_ = dummy;
}

void RawTensor::AddRefCount(int value) {
    ASSERT(value == 1 || value == -1);
    refCount_ += value;
    if (refCount_ < 0) {
        ALOG_INFO("rawmagic = ", rawmagic, " refCount_ is negative: ", refCount_);
    }
}

int64_t RawTensor::GetRawDataSize() const {
    return GetRawShapeSize() * BytesOf(datatype);
}


int64_t RawTensor::GetRawShapeSize() const {
    return std::accumulate(rawshape.begin(), rawshape.end(), INT64_C(1), std::multiplies<int64_t>());
}