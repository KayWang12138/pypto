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

#include <nanobind/nanobind.h>
#include "nb_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindEnum(nb::module_ &m){
    nb::enum_<DataType>(m, "DataType", nb::is_arithmetic())
        .value("DT_INT4", DataType::DT_INT4)
        .value("DT_INT8", DataType::DT_INT8)
        .value("DT_INT16", DataType::DT_INT16)
        .value("DT_INT32", DataType::DT_INT32)
        .value("DT_INT64", DataType::DT_INT64)
        .value("DT_FP8", DataType::DT_FP8)
        .value("DT_FP16", DataType::DT_FP16)
        .value("DT_FP32", DataType::DT_FP32)
        .value("DT_BF16", DataType::DT_BF16)
        .value("DT_HF4", DataType::DT_HF4)
        .value("DT_HF8", DataType::DT_HF8)
        .value("DT_FP8E4M3", DataType::DT_FP8E4M3)
        .value("DT_FP8E5M2", DataType::DT_FP8E5M2)
        .value("DT_UINT8", DataType::DT_UINT8)
        .value("DT_UINT16", DataType::DT_UINT16)
        .value("DT_UINT32", DataType::DT_UINT32)
        .value("DT_UINT64", DataType::DT_UINT64)
        .value("DT_BOOL", DataType::DT_BOOL)
        .value("DT_DOUBLE", DataType::DT_DOUBLE)
        .value("DT_BOTTOM", DataType::DT_BOTTOM)
        .export_values();

    nb::enum_<NodeType>(m, "NodeType", nb::is_arithmetic())
        .value("LOCAL", NodeType::LOCAL)
        .value("INCAST", NodeType::INCAST)
        .value("OUTCAST", NodeType::OUTCAST)
        .export_values();

    nb::enum_<TileOpFormat>(m, "TileOpFormat", nb::is_arithmetic())
        .value("TILEOP_ND", TileOpFormat::TILEOP_ND)
        .value("TILEOP_NZ", TileOpFormat::TILEOP_NZ)
        .value("TILEOP_FORMAT_NUM", TileOpFormat::TILEOP_FORMAT_NUM)
        .export_values();

    nb::enum_<CachePolicy>(m, "CachePolicy", nb::is_arithmetic())
        .value("NONE_CACHEABLE", CachePolicy::NONE_CACHEABLE)
        .value("MAX_NUM", CachePolicy::MAX_NUM)
        .export_values();

    nb::enum_<ReduceMode>(m, "ReduceMode", nb::is_arithmetic())
        .value("ATOMIC_ADD", ReduceMode::ATOMIC_ADD)
        .export_values();

    nb::enum_<ScatterMode>(m, "ScatterMode", nb::is_arithmetic())
        .value("NONE", ScatterMode::NONE)
        .value("ADD", ScatterMode::ADD)
        .value("MULTIPLY", ScatterMode::MULTIPLY)
        .export_values();

    nb::enum_<MemoryType>(m, "MemoryType", nb::is_arithmetic())
        .value("MEM_UB", MemoryType::MEM_UB)
        .value("MEM_L1", MemoryType::MEM_L1)
        .value("MEM_L0A", MemoryType::MEM_L0A)
        .value("MEM_L0B", MemoryType::MEM_L0B)
        .value("MEM_L0C", MemoryType::MEM_L0C)
        .value("MEM_L2", MemoryType::MEM_L2)
        .value("MEM_L3", MemoryType::MEM_L3)
        .value("MEM_DEVICE_DDR", MemoryType::MEM_DEVICE_DDR)
        .value("MEM_HOST1", MemoryType::MEM_HOST1)
        .value("MEM_FAR1", MemoryType::MEM_FAR1)
        .value("MEM_FAR2", MemoryType::MEM_FAR2)
        .value("MEM_UNKNOWN", MemoryType::MEM_UNKNOWN)
        .export_values();

     nb::enum_<FunctionType>(m, "FunctionType", nb::is_arithmetic())
        .value("EAGER", FunctionType::EAGER)
        .value("STATIC", FunctionType::STATIC)
        .value("DYNAMIC", FunctionType::DYNAMIC)
        .value("DYNAMIC_LOOP", FunctionType::DYNAMIC_LOOP)
        .value("DYNAMIC_LOOP_PATH", FunctionType::DYNAMIC_LOOP_PATH)
        .value("INVALID", FunctionType::INVALID)
        .value("MAX", FunctionType::MAX)
        .export_values();

     nb::enum_<GraphType>(m, "GraphType", nb::is_arithmetic())
        .value("TENSOR_GRAPH", GraphType::TENSOR_GRAPH)
        .value("TILE_GRAPH", GraphType::TILE_GRAPH)
        .value("EXECUTE_GRAPH", GraphType::EXECUTE_GRAPH)
        .value("BLOCK_GRAPH", GraphType::BLOCK_GRAPH)
        .value("LEAF_VF_GRAPH", GraphType::LEAF_VF_GRAPH)
        .value("INVALID", GraphType::INVALID)
        .export_values();

     nb::enum_<CastMode>(m, "CastMode", nb::is_arithmetic())
        .value("CAST_NONE", CastMode::CAST_NONE)
        .value("CAST_RINT", CastMode::CAST_RINT)
        .value("CAST_ROUND", CastMode::CAST_ROUND)
        .value("CAST_FLOOR", CastMode::CAST_FLOOR)
        .value("CAST_CEIL", CastMode::CAST_CEIL)
        .value("CAST_TRUNC", CastMode::CAST_TRUNC)
        .value("CAST_ODD", CastMode::CAST_ODD)
        .export_values();

     nb::enum_<TileType>(m, "TileType", nb::is_arithmetic())
        .value("VEC", TileType::VEC)
        .value("CUBE", TileType::CUBE)
        .value("DIST", TileType::DIST)
        .value("MAX", TileType::MAX)
        .export_values();

     nb::enum_<OpType>(m, "OpType", nb::is_arithmetic())
        .value("EQ", OpType::EQ)
        .value("NE", OpType::NE)
        .value("LT", OpType::LT)
        .value("LE", OpType::LE)
        .value("GT", OpType::GT)
        .value("GE", OpType::GE)
        .export_values();

     nb::enum_<OutType>(m, "OutType", nb::is_arithmetic())
        .value("BOOL", OutType::BOOL)
        .value("BIT", OutType::BIT)
        .export_values();

     nb::enum_<Matrix::ReLuType>(m, "ReLuType", nb::is_arithmetic())
        .value("NO_RELU", Matrix::ReLuType::NoReLu)
        .value("RELU", Matrix::ReLuType::ReLu)
        .export_values();

     nb::enum_<LogBaseType>(m, "LogBaseType", nb::is_arithmetic())
        .value("LOG_E", LogBaseType::LOG_E)
        .value("LOG_2", LogBaseType::LOG_2)
        .value("LOG_10", LogBaseType::LOG_10)
        .export_values();
}
}
