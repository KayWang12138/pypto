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
 * \file enum.cpp
 * \brief
 */

#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void bind_enum(py::module &m){
    py::enum_<DataType>(m, "DataType")
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
        .value("DT_UINT8", DataType::DT_UINT8)
        .value("DT_UINT16", DataType::DT_UINT16)
        .value("DT_UINT32", DataType::DT_UINT32)
        .value("DT_UINT64", DataType::DT_UINT64)
        .value("DT_BOOL", DataType::DT_BOOL)
        .value("DT_DOUBLE", DataType::DT_DOUBLE)
        .value("DT_BOTTOM", DataType::DT_BOTTOM)
        .export_values();

    py::enum_<NodeType>(m, "NodeType")
        .value("LOCAL", NodeType::LOCAL)
        .value("INCAST", NodeType::INCAST)
        .value("OUTCAST", NodeType::OUTCAST)
        .export_values();

    py::enum_<TileOpFormat>(m, "tile_op_format")
        .value("TILEOP_ND", TileOpFormat::TILEOP_ND)
        .value("TILEOP_NZ", TileOpFormat::TILEOP_NZ)
        .value("TILEOP_FORMAT_NUM", TileOpFormat::TILEOP_FORMAT_NUM)
        .export_values();

    py::enum_<CachePolicy>(m, "cache_policy")
        .value("PREFETCH", CachePolicy::PREFETCH)
        .value("NONE_CACHEABLE", CachePolicy::NONE_CACHEABLE)
        .value("MAX_NUM", CachePolicy::MAX_NUM)
        .export_values();

    py::enum_<ReduceMode>(m, "reduce_mode")
        .value("ATOMIC_ADD", ReduceMode::ATOMIC_ADD)
        .export_values();

    py::enum_<MemoryType>(m, "MemoryType")
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

    py::enum_<FunctionType>(m, "function_type")
        .value("EAGER", FunctionType::EAGER)
        .value("STATIC", FunctionType::STATIC)
        .value("DYNAMIC", FunctionType::DYNAMIC)
        .value("DYNAMIC_LOOP", FunctionType::DYNAMIC_LOOP)
        .value("DYNAMIC_LOOP_PATH", FunctionType::DYNAMIC_LOOP_PATH)
        .value("INVALID", FunctionType::INVALID)
        .value("MAX", FunctionType::MAX)
        .export_values();

    py::enum_<GraphType>(m, "graph_type")
        .value("TENSOR_GRAPH", GraphType::TENSOR_GRAPH)
        .value("TILE_GRAPH", GraphType::TILE_GRAPH)
        .value("EXECUTE_GRAPH", GraphType::EXECUTE_GRAPH)
        .value("BLOCK_GRAPH", GraphType::BLOCK_GRAPH)
        .value("LEAF_VF_GRAPH", GraphType::LEAF_VF_GRAPH)
        .value("INVALID", GraphType::INVALID)
        .export_values();

    py::enum_<CastMode>(m, "cast_mode")
        .value("CAST_NONE", CastMode::CAST_NONE)
        .value("CAST_RINT", CastMode::CAST_RINT)
        .value("CAST_ROUND", CastMode::CAST_ROUND)
        .value("CAST_FLOOR", CastMode::CAST_FLOOR)
        .value("CAST_CEIL", CastMode::CAST_CEIL)
        .value("CAST_TRUNC", CastMode::CAST_TRUNC)
        .value("CAST_ODD", CastMode::CAST_ODD)
        .export_values();

    py::enum_<TileType>(m, "TileType")
        .value("VEC", TileType::VEC)
        .value("CUBE", TileType::CUBE)
        .value("DIST", TileType::DIST)
        .value("MAX", TileType::MAX)
        .export_values();
}
}
