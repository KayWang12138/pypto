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
 * \file test_schema_trace.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <thread>
#include "machine/utils/dynamic/schema_trace.h"
#include "interface/utils/log.h"

using namespace npu::tile_fwk;

namespace {

SCHEMA_DEF_TYPE_INT64(AIndex);
SCHEMA_DEF_TYPE_INT64(BIndex);

SCHEMA_DEF_ATTR(At0);
SCHEMA_DEF_ATTR(At1);
SCHEMA_DEF_TYPE_UNION(CIndex, At0, At1);
SCHEMA_DEF_ATTR(At2, AIndex, BIndex, CIndex);

SCHEMA_DEF_TYPE_INT32(TInt32);
SCHEMA_DEF_TYPE_INT64(TInt64);
SCHEMA_DEF_TYPE_UINT32(TUInt32);
SCHEMA_DEF_TYPE_UINT64(TUInt64);
SCHEMA_DEF_TYPE_ADDRESS(TAddress);
SCHEMA_DEF_TYPE_STRING(TString);
SCHEMA_DEF_TYPE_COORD(TCoord);
SCHEMA_DEF_ATTR(TInt, TInt32, TInt64, TUInt32, TUInt64);
SCHEMA_DEF_ATTR(TMeta, TAddress, TString, TCoord);
SCHEMA_DEF_TYPE_ARRAY(TCode, TInt64);

static_assert(SCHEMA_DEF_ATTR_NR(1, 2, 3) == 3, "Invalid nr");

TEST(SchemaTrace, Base) {
    At2 atv0(10, 20, At0());
    std::string s0 = atv0.Dump();
    At2 atv1(10, 20, At0());
    std::string s1 = atv1.Dump();
    EXPECT_EQ(s0, s1);
    EXPECT_EQ(s0, "#At2{10,20,At0}");
    At2 atv2(10, 20, At1());
    atv2 = atv0;
    std::string s2 = atv2.Dump();
    EXPECT_EQ(s0, s2);

    TInt a(1, 2, 3, 4);
    EXPECT_EQ(a.Dump(), "#TInt{1,2,3,4}");
    TMeta b(0x10, "bbb", TCoord(1, 2));
    EXPECT_EQ(b.Dump(), "#TMeta{0x10,\"bbb\",[1,2]}");

    auto data = []() {
        std::vector<TInt64> ts;
        for (int i = 0; i < 5; i++) {
            ts.push_back(TInt64(i));
        }
        return ts;
    }();
    EXPECT_EQ(TCode(data).Dump(), "[0,1,2,3,4]");
}

TEST(SchemaTrace, Log) {
    auto s0 = schema::LActStart(schema::none());
    EXPECT_EQ(s0.Dump(), "#LActStart{_}");
    auto s1 = schema::LActStart(1);
    EXPECT_EQ(s1.Dump(), "#LActStart{1}");
    schema::LEvent l0(schema::LUid(1, 2, 3, 4, 5), schema::LActStart(10));
    EXPECT_EQ(l0.Dump(), "#LEvent{LUid{1,2,3,4,5},LActStart{10}}");

    schema::RUid r0(1, 2, 3);
    std::string str = "#RUid{1,2,3} #RUid{1,2,3} #RUid{1,2,3}";
    EXPECT_EQ(str, schema::DumpAttr(r0, r0, r0));
}

}