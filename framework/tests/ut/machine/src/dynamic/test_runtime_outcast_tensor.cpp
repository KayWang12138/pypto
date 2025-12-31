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
 * \file test_runtime_outcast_tensor.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/utils/dynamic/runtime_outcast_tensor.h"
#include "machine/utils/dynamic/dev_workspace.h"

using namespace npu::tile_fwk::dynamic;

class RuntimeOutcastTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(RuntimeOutcastTensorTest, ConstructAndFields) {
    RuntimeOutcastTensor t(0xDEADBEEFull, RtMemProperty::EXTERNAL, 42u);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xDEADBEEFull));
    EXPECT_EQ(t.property, RtMemProperty::EXTERNAL);
    EXPECT_EQ(t.refCnt, 42u);
}

TEST_F(RuntimeOutcastTensorTest, DumpFormatWithNonZeroAddr) {
    RuntimeOutcastTensor t(0x1234ABCDull, RtMemProperty::EXTERNAL, 1u);
    std::string s = t.Dump();
    // std::hex outputs lowercase letters, and no leading zeros are added
    EXPECT_EQ(s, std::string("&0x1234abcd, EXTERNAL"));
}

TEST_F(RuntimeOutcastTensorTest, DumpFormatWithZeroAddr) {
    RuntimeOutcastTensor t(0x0ull, RtMemProperty::BOUNDARY_OUTCAST, 0u);
    std::string s = t.Dump();
    EXPECT_EQ(s, std::string("&0x0, BOUNDARY_OUTCAST"));
}

TEST_F(RuntimeOutcastTensorTest, GetRtMemPropertyNameMatchesEnum) {
    EXPECT_STREQ(GetRtMemPropertyName(RtMemProperty::EXTERNAL), "EXTERNAL");
    EXPECT_STREQ(GetRtMemPropertyName(RtMemProperty::DEVTASK_INNER_OUTCAST), "DEVTASK_INNER_OUTCAST");
    EXPECT_STREQ(GetRtMemPropertyName(RtMemProperty::BOUNDARY_OUTCAST), "BOUNDARY_OUTCAST");
    EXPECT_STREQ(GetRtMemPropertyName(RtMemProperty::DEPRECATED_ORIGINAL_DASSEMBLE_DST), "DEPRECATED_ORIGINAL_DASSEMBLE_DST");
}

// Helper to construct and initialize a DeviceWorkspaceAllocator with reasonable
// metadata budgets so `InitAicpuStitchSlabAllocator` won't assert.
static void InitDeviceWorkspaceAllocatorForTest(DeviceWorkspaceAllocator &d, DevAscendProgram &devProg,
                                                std::vector<uint8_t> &workspace) {
    // Ensure stitch pool and general metadata are non-zero and large enough
    devProg.memBudget.metadata.general = 1u << 18; // 256KB
    devProg.memBudget.metadata.stitchPool = 1u << 16; // 64KB

    devProg.devArgs.generalAddr = reinterpret_cast<uint64_t>(workspace.data());
    // Put stitch pool at an offset within the same workspace region
    devProg.devArgs.stitchPoolAddr = reinterpret_cast<uint64_t>(workspace.data()) +
            devProg.memBudget.metadata.general; // offset 256KB

    devProg.devArgs.nrAic = 1;
    devProg.devArgs.nrAiv = 1;
    devProg.devArgs.nrValidAic = 0;

    DevStartArgs args;
    args.InitWorkspace(&devProg, workspace.data());

    d.Init(&args);

    std::cerr << "InitDeviceWorkspaceAllocatorForTest finished" << std::endl;
}

TEST_F(RuntimeOutcastTensorTest, DeviceWorkspaceAllocatorBasicOps) {
    // Small workspace and dev program with minimal budgets to initialize allocators
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 8;
    // Keep tensor budgets small/zero but valid
    devProg.memBudget.tensor.rootInner = 0;
    devProg.memBudget.tensor.devTaskInnerExclusiveOutcasts = 0;
    devProg.memBudget.tensor.maxStaticOutcastMem = 0;
    devProg.memBudget.tensor.maxDynamicAssembleOutcastMem = 0;
    devProg.memBudget.tensor.devTaskBoundaryOutcastNum = 0;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // Now exercise runtime outcast APIs
    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0xAAull, RtMemProperty::EXTERNAL);
    EXPECT_NE(a, ITEM_POOL_INVALID_INDEX);

    auto &t = d.GetRuntimeOutcastTensor(a);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xAAull));
    EXPECT_EQ(t.property, RtMemProperty::EXTERNAL);
    EXPECT_EQ(t.refCnt, 1u);

    d.RuntimeOutcastTensorRef(a);
    EXPECT_EQ(t.refCnt, 2u);

    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(t.refCnt, 1u);

    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(a, 0xBBull, RtMemProperty::BOUNDARY_OUTCAST);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xBBull));
    EXPECT_EQ(t.property, RtMemProperty::BOUNDARY_OUTCAST);

    ItemPoolIter b = d.MakeRuntimeOutcastTensor(0xCCull, RtMemProperty::EXTERNAL);
    d.RuntimeOutcastTensorAssign(b, a);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(b).addr, static_cast<uintdevptr_t>(0xBBull));
    EXPECT_EQ(d.GetRuntimeOutcastTensor(b).refCnt, 2u);
}

TEST_F(RuntimeOutcastTensorTest, DeviceWorkspaceAllocatorSafeRefDerefNoCrash) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter invalid = ITEM_POOL_INVALID_INDEX;
    d.RuntimeOutcastTensorRefSafe(invalid);
    d.RuntimeOutcastTensorDerefSafe(invalid);
    SUCCEED();
}

TEST_F(RuntimeOutcastTensorTest, DerefToZeroReturnsItemToPool) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // initial free items should equal pool size
    size_t freeBefore = d.runtimeOutcastTensorPool_.FreeItemNum();
    EXPECT_EQ(freeBefore, devProg.runtimeOutcastPoolSize);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0xAAAull, RtMemProperty::EXTERNAL);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 1);

    // deref once (refCnt -> 0) should destroy and return to pool
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize);
}

TEST_F(RuntimeOutcastTensorTest, AssignReleasesPreviousDestination) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // allocate two items A and B
    ItemPoolIter A = d.MakeRuntimeOutcastTensor(0xA1ull, RtMemProperty::EXTERNAL);
    ItemPoolIter B = d.MakeRuntimeOutcastTensor(0xB2ull, RtMemProperty::EXTERNAL);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 2);

    // assign B = A; this should deref old B (destroy it and return to pool) and ref A
    d.RuntimeOutcastTensorAssign(B, A);

    // After assign: one item returned to pool
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 1);

    // The target iterator B now refers to A, so address equals A's address and refCnt is 2
    EXPECT_EQ(d.GetRuntimeOutcastTensor(B).addr, static_cast<uintdevptr_t>(0xA1ull));
    EXPECT_EQ(d.GetRuntimeOutcastTensor(B).refCnt, 2u);
}


