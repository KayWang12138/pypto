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
 * \file enum.cpp
 * \brief
 */
#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {

struct EnumItem {
    const char *name;
    int value;
    template <typename T>
    EnumItem(const char *tname, T tvalue) : name(tname), value((int)tvalue) {}
};

static void bind_enum(py::object &m, const char *name, const std::initializer_list<EnumItem> &input) {
    py::dict items;
    for (auto &item : input) {
        items[item.name] = item.value;
    }
    auto IntEnum = py::module::import("enum").attr("IntEnum");
    m.attr(name) = IntEnum(name, items);
}

void bind_enum(py::module &m){
    bind_enum(m, "DataType", {
        {"DT_INT4", DataType::DT_INT4},
        {"DT_INT8", DataType::DT_INT8},
        {"DT_INT16", DataType::DT_INT16},
        {"DT_INT32", DataType::DT_INT32},
        {"DT_INT64", DataType::DT_INT64},
        {"DT_FP8", DataType::DT_FP8},
        {"DT_FP16", DataType::DT_FP16},
        {"DT_FP32", DataType::DT_FP32},
        {"DT_BF16", DataType::DT_BF16},
        {"DT_HF4", DataType::DT_HF4},
        {"DT_HF8", DataType::DT_HF8},
        {"DT_FP8E4M3", DataType::DT_FP8E4M3},
        {"DT_FP8E5M2", DataType::DT_FP8E5M2},
        {"DT_UINT8", DataType::DT_UINT8},
        {"DT_UINT16", DataType::DT_UINT16},
        {"DT_UINT32", DataType::DT_UINT32},
        {"DT_UINT64", DataType::DT_UINT64},
        {"DT_BOOL", DataType::DT_BOOL},
        {"DT_DOUBLE", DataType::DT_DOUBLE},
        {"DT_BOTTOM", DataType::DT_BOTTOM},
    });

    bind_enum(m, "NodeType", {
        {"LOCAL", NodeType::LOCAL},
        {"INCAST", NodeType::INCAST},
        {"OUTCAST", NodeType::OUTCAST},
    });

    bind_enum(m, "TileOpFormat", {
        {"TILEOP_ND", TileOpFormat::TILEOP_ND},
        {"TILEOP_NZ", TileOpFormat::TILEOP_NZ},
        {"TILEOP_FORMAT_NUM", TileOpFormat::TILEOP_FORMAT_NUM},
    });

    bind_enum(m, "CachePolicy", {
        {"NONE_CACHEABLE", CachePolicy::NONE_CACHEABLE},
        {"MAX_NUM", CachePolicy::MAX_NUM},
    });

    bind_enum(m, "ReduceMode", {
        {"ATOMIC_ADD", ReduceMode::ATOMIC_ADD},
    });

    bind_enum(m, "ScatterMode", {
        {"NONE", ScatterMode::NONE},
        {"ADD", ScatterMode::ADD},
        {"MULTIPLY", ScatterMode::MULTIPLY},
    });

    bind_enum(m, "MemoryType", {
        {"MEM_UB", MemoryType::MEM_UB},
        {"MEM_L1", MemoryType::MEM_L1},
        {"MEM_L0A", MemoryType::MEM_L0A},
        {"MEM_L0B", MemoryType::MEM_L0B},
        {"MEM_L0C", MemoryType::MEM_L0C},
        {"MEM_L2", MemoryType::MEM_L2},
        {"MEM_L3", MemoryType::MEM_L3},
        {"MEM_DEVICE_DDR", MemoryType::MEM_DEVICE_DDR},
        {"MEM_HOST1", MemoryType::MEM_HOST1},
        {"MEM_FAR1", MemoryType::MEM_FAR1},
        {"MEM_FAR2", MemoryType::MEM_FAR2},
        {"MEM_UNKNOWN", MemoryType::MEM_UNKNOWN},
    });

    bind_enum(m, "FunctionType", {
        {"STATIC", FunctionType::STATIC},
        {"DYNAMIC", FunctionType::DYNAMIC},
        {"DYNAMIC_LOOP", FunctionType::DYNAMIC_LOOP},
        {"DYNAMIC_LOOP_PATH", FunctionType::DYNAMIC_LOOP_PATH},
    });

    bind_enum(m, "GraphType", {
        {"TENSOR_GRAPH", GraphType::TENSOR_GRAPH},
        {"TILE_GRAPH", GraphType::TILE_GRAPH},
        {"EXECUTE_GRAPH", GraphType::EXECUTE_GRAPH},
        {"BLOCK_GRAPH", GraphType::BLOCK_GRAPH},
        {"LEAF_VF_GRAPH", GraphType::LEAF_VF_GRAPH},
    });

    bind_enum(m, "CastMode", {
        {"CAST_NONE", CastMode::CAST_NONE},
        {"CAST_RINT", CastMode::CAST_RINT},
        {"CAST_ROUND", CastMode::CAST_ROUND},
        {"CAST_FLOOR", CastMode::CAST_FLOOR},
        {"CAST_CEIL", CastMode::CAST_CEIL},
        {"CAST_TRUNC", CastMode::CAST_TRUNC},
        {"CAST_ODD", CastMode::CAST_ODD},
    });

    bind_enum(m, "TileType", {
        {"VEC", TileType::VEC},
        {"CUBE", TileType::CUBE},
        {"DIST", TileType::DIST},
        {"MAX", TileType::MAX},
    });

    bind_enum(m, "OpType", {
        {"EQ", OpType::EQ},
        {"NE", OpType::NE},
        {"LT", OpType::LT},
        {"LE", OpType::LE},
        {"GT", OpType::GT},
        {"GE", OpType::GE},
    });

    bind_enum(m, "OutType", {
        {"BOOL", OutType::BOOL},
        {"BIT", OutType::BIT},
    });

    bind_enum(m, "ReLuType", {
        {"NO_RELU", Matrix::ReLuType::NoReLu},
        {"RELU", Matrix::ReLuType::ReLu},
    });

    bind_enum(m, "LogBaseType", {
        {"LOG_E", LogBaseType::LOG_E},
        {"LOG_2", LogBaseType::LOG_2},
        {"LOG_10", LogBaseType::LOG_10},
    });
}
}
