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
 * \file schema_trace.h
 * \brief
 */

#pragma once
#ifndef SCHEMA_TRACE_H
#define SCHEMA_TRACE_H

#include "schema_trace_base.h"

namespace npu::tile_fwk::schema {

template<typename Ty0>
static inline std::string DumpAttr(const Ty0 &arg0) {
    return arg0.Dump();
}

template<typename Ty0, typename ...Tys>
static inline std::string DumpAttr(const Ty0 &arg0, const Tys&...args) {
    std::string tail = DumpAttr(args...);
    std::string head = arg0.Dump();
    return head + " " + tail;
}

#include "schema_trace_def.h"

#define DEV_TRACE_DEBUG(arg, args...) \
    do { \
        using namespace npu::tile_fwk::schema; \
        DEV_DEBUG("#trace: %s", DumpAttr(arg, ##args).c_str()); \
    } while(0)
#define DEV_TRACE_INFO(arg, args...) \
    do { \
        using namespace npu::tile_fwk::schema; \
        DEV_INFO("#trace: %s", DumpAttr(arg, ##args).c_str()); \
    } while(0)
#define DEV_TRACE_WARN(arg, args...) \
    do { \
        using namespace npu::tile_fwk::schema; \
        DEV_WARN("#trace: %s", DumpAttr(arg, ##args).c_str()); \
    } while(0)
#define DEV_TRACE_ERROR(arg, args...) \
    do { \
        using namespace npu::tile_fwk::schema; \
        DEV_ERROR("#trace: %s", DumpAttr(arg, ##args).c_str()); \
    } while(0)
}

#endif//SCHEMA_TRACE_H