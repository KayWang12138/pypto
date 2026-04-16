/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_memref.cpp
 * \brief Unit tests for IR memory reference structures
 */

#include "gtest/gtest.h"

#include <memory>

#include "core/dtype.h"
#include "ir/memref.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

// ============================================================================
// MemorySpace Enum Tests
// ============================================================================

TEST(IRMemRefTest, TestMemorySpaceValues)
{
    // Test that MemorySpace enum values are distinct
    ASSERT_NE(static_cast<int>(MemorySpace::DDR), static_cast<int>(MemorySpace::UB));
    ASSERT_NE(static_cast<int>(MemorySpace::L1), static_cast<int>(MemorySpace::UB));
    ASSERT_NE(static_cast<int>(MemorySpace::L0A), static_cast<int>(MemorySpace::L0B));
    ASSERT_NE(static_cast<int>(MemorySpace::L0C), static_cast<int>(MemorySpace::L0A));
}

TEST(IRMemRefTest, TestMemorySpaceAssignment)
{
    // Test MemorySpace assignment
    MemorySpace space1 = MemorySpace::DDR;
    MemorySpace space2 = MemorySpace::UB;
    MemorySpace space3 = MemorySpace::L1;

    ASSERT_EQ(space1, MemorySpace::DDR);
    ASSERT_EQ(space2, MemorySpace::UB);
    ASSERT_EQ(space3, MemorySpace::L1);
}

// ============================================================================
// MemRef Constructor Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefBasicConstructor)
{
    // Test basic MemRef construction
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::DDR, addr, 1024, 0);

    ASSERT_EQ(memref.memorySpace_, MemorySpace::DDR);
    ASSERT_EQ(memref.addr_, addr);
    ASSERT_EQ(memref.size_, 1024);
}

TEST(IRMemRefTest, TestMemRefWithDifferentSpaces)
{
    // Test MemRef with different memory spaces
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());

    MemRef ddrRef(MemorySpace::DDR, addr, 1024, 0);
    MemRef ubRef(MemorySpace::UB, addr, 2048, 0);
    MemRef l1Ref(MemorySpace::L1, addr, 512, 0);
    MemRef l0aRef(MemorySpace::L0A, addr, 256, 0);

    ASSERT_EQ(ddrRef.memorySpace_, MemorySpace::DDR);
    ASSERT_EQ(ubRef.memorySpace_, MemorySpace::UB);
    ASSERT_EQ(l1Ref.memorySpace_, MemorySpace::L1);
    ASSERT_EQ(l0aRef.memorySpace_, MemorySpace::L0A);
}

TEST(IRMemRefTest, TestMemRefWithL0Spaces)
{
    // Test MemRef with L0 memory spaces
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());

    MemRef l0aRef(MemorySpace::L0A, addr, 128, 0);
    MemRef l0bRef(MemorySpace::L0B, addr, 128, 0);
    MemRef l0cRef(MemorySpace::L0C, addr, 128, 0);

    ASSERT_EQ(l0aRef.memorySpace_, MemorySpace::L0A);
    ASSERT_EQ(l0bRef.memorySpace_, MemorySpace::L0B);
    ASSERT_EQ(l0cRef.memorySpace_, MemorySpace::L0C);
}

TEST(IRMemRefTest, TestMemRefWithDDRSpace)
{
    // Test MemRef with DDR memory space
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::DDR, addr, 1024, 0);

    ASSERT_EQ(memref.memorySpace_, MemorySpace::DDR);
}

// ============================================================================
// MemRef Address Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefWithZeroAddress)
{
    // Test MemRef with zero address
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::DDR, addr, 1024, 0);

    ASSERT_NE(memref.addr_, nullptr);
    auto constAddr = std::dynamic_pointer_cast<const ConstInt>(memref.addr_);
    ASSERT_NE(constAddr, nullptr);
    ASSERT_EQ(constAddr->value_, 0);
}

TEST(IRMemRefTest, TestMemRefWithNonZeroAddress)
{
    // Test MemRef with non-zero address
    auto addr = std::make_shared<ConstInt>(0x1000, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::UB, addr, 2048, 0);

    ASSERT_NE(memref.addr_, nullptr);
    auto constAddr = std::dynamic_pointer_cast<const ConstInt>(memref.addr_);
    ASSERT_NE(constAddr, nullptr);
    ASSERT_EQ(constAddr->value_, 0x1000);
}

TEST(IRMemRefTest, TestMemRefWithVariableAddress)
{
    // Test MemRef with variable address expression
    auto base = std::make_shared<ConstInt>(0x1000, DataType::INT64, Span::Unknown());
    auto offset = std::make_shared<ConstInt>(256, DataType::INT64, Span::Unknown());
    auto addr = std::make_shared<Add>(base, offset, DataType::INT64, Span::Unknown());

    MemRef memref(MemorySpace::L1, addr, 512, 0);

    ASSERT_NE(memref.addr_, nullptr);
    ASSERT_EQ(memref.addr_->TypeName(), "Add");
}

// ============================================================================
// MemRef Size Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefWithZeroSize)
{
    // Test MemRef with zero size
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::DDR, addr, 0, 0);

    ASSERT_EQ(memref.size_, 0);
}

TEST(IRMemRefTest, TestMemRefWithSmallSize)
{
    // Test MemRef with small size
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::L0A, addr, 64, 0);

    ASSERT_EQ(memref.size_, 64);
}

TEST(IRMemRefTest, TestMemRefWithLargeSize)
{
    // Test MemRef with large size
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    MemRef memref(MemorySpace::DDR, addr, 1024 * 1024 * 1024, 0); // 1GB

    ASSERT_EQ(memref.size_, 1024 * 1024 * 1024);
}

TEST(IRMemRefTest, TestMemRefWithVariousSizes)
{
    // Test MemRef with various sizes
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());

    MemRef ref1(MemorySpace::DDR, addr, 128, 0);
    MemRef ref2(MemorySpace::UB, addr, 256, 0);
    MemRef ref3(MemorySpace::L1, addr, 512, 0);
    MemRef ref4(MemorySpace::UB, addr, 1024, 0);

    ASSERT_EQ(ref1.size_, 128);
    ASSERT_EQ(ref2.size_, 256);
    ASSERT_EQ(ref3.size_, 512);
    ASSERT_EQ(ref4.size_, 1024);
}

// ============================================================================
// MemRef Copy and Assignment Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefCopyConstructor)
{
    // Test MemRef shared_ptr usage (copy constructor is deleted since MemRef inherits from IRNode)
    auto addr = std::make_shared<ConstInt>(0x1000, DataType::INT64, Span::Unknown());
    auto original = std::make_shared<MemRef>(MemorySpace::DDR, addr, 1024, 0);

    // Verify fields via shared_ptr
    ASSERT_EQ(original->memorySpace_, MemorySpace::DDR);
    ASSERT_EQ(original->addr_, addr);
    ASSERT_EQ(original->size_, 1024);

    // Test that another MemRef with same params has same field values
    auto another = std::make_shared<MemRef>(MemorySpace::DDR, addr, 1024, 1);
    ASSERT_EQ(another->memorySpace_, original->memorySpace_);
    ASSERT_EQ(another->addr_, original->addr_);
    ASSERT_EQ(another->size_, original->size_);
}

TEST(IRMemRefTest, TestMemRefAssignment)
{
    // Test MemRef field comparison (assignment operator is deleted since MemRef inherits from IRNode)
    auto addr1 = std::make_shared<ConstInt>(0x1000, DataType::INT64, Span::Unknown());
    auto addr2 = std::make_shared<ConstInt>(0x2000, DataType::INT64, Span::Unknown());

    auto ref1 = std::make_shared<MemRef>(MemorySpace::DDR, addr1, 1024, 0);
    auto ref2 = std::make_shared<MemRef>(MemorySpace::UB, addr2, 2048, 1);

    // Verify they have different values
    ASSERT_NE(ref1->memorySpace_, ref2->memorySpace_);
    ASSERT_NE(ref1->addr_, ref2->addr_);
    ASSERT_NE(ref1->size_, ref2->size_);
}

// ============================================================================
// MemRef Practical Usage Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefForTensorAllocation)
{
    // Test MemRef for tensor allocation scenario
    // Allocate 10x20 float32 tensor in DDR (800 bytes)
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    size_t tensorSize = 10 * 20 * sizeof(float); // 800 bytes
    MemRef memref(MemorySpace::DDR, addr, tensorSize, 0);

    ASSERT_EQ(memref.memorySpace_, MemorySpace::DDR);
    ASSERT_EQ(memref.size_, tensorSize);
}

TEST(IRMemRefTest, TestMemRefForTileAllocation)
{
    // Test MemRef for tile allocation in L0A
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    size_t tileSize = 16 * 16 * sizeof(float); // 1024 bytes
    MemRef memref(MemorySpace::L0A, addr, tileSize, 0);

    ASSERT_EQ(memref.memorySpace_, MemorySpace::L0A);
    ASSERT_EQ(memref.size_, tileSize);
}

TEST(IRMemRefTest, TestMemRefForBufferAllocation)
{
    // Test MemRef for unified buffer allocation
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    size_t bufferSize = 4096; // 4KB buffer
    MemRef memref(MemorySpace::UB, addr, bufferSize, 0);

    ASSERT_EQ(memref.memorySpace_, MemorySpace::UB);
    ASSERT_EQ(memref.size_, bufferSize);
}

TEST(IRMemRefTest, TestMemRefWithOffsetAddress)
{
    // Test MemRef with offset address for sub-buffer
    auto baseAddr = std::make_shared<ConstInt>(0x10000, DataType::INT64, Span::Unknown());
    auto offset = std::make_shared<ConstInt>(1024, DataType::INT64, Span::Unknown());
    auto addr = std::make_shared<Add>(baseAddr, offset, DataType::INT64, Span::Unknown());

    MemRef memref(MemorySpace::L1, addr, 512, 0);

    ASSERT_EQ(memref.memorySpace_, MemorySpace::L1);
    ASSERT_EQ(memref.size_, 512);
    ASSERT_EQ(memref.addr_->TypeName(), "Add");
}

// ============================================================================
// MemRef Comparison Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefEquality)
{
    // Test MemRef equality comparison
    auto addr = std::make_shared<ConstInt>(0x1000, DataType::INT64, Span::Unknown());

    MemRef ref1(MemorySpace::DDR, addr, 1024, 0);
    MemRef ref2(MemorySpace::DDR, addr, 1024, 0);

    // Note: This tests structural equality, not pointer equality
    ASSERT_EQ(ref1.memorySpace_, ref2.memorySpace_);
    ASSERT_EQ(ref1.addr_, ref2.addr_);
    ASSERT_EQ(ref1.size_, ref2.size_);
}

TEST(IRMemRefTest, TestMemRefInequality)
{
    // Test MemRef inequality
    auto addr1 = std::make_shared<ConstInt>(0x1000, DataType::INT64, Span::Unknown());
    auto addr2 = std::make_shared<ConstInt>(0x2000, DataType::INT64, Span::Unknown());

    MemRef ref1(MemorySpace::DDR, addr1, 1024, 0);
    MemRef ref2(MemorySpace::UB, addr2, 2048, 0);

    ASSERT_NE(ref1.memorySpace_, ref2.memorySpace_);
    ASSERT_NE(ref1.addr_, ref2.addr_);
    ASSERT_NE(ref1.size_, ref2.size_);
}

// ============================================================================
// MemRef Edge Cases Tests
// ============================================================================

TEST(IRMemRefTest, TestMemRefWithMaxSize)
{
    // Test MemRef with maximum size_t value
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    size_t maxSize = std::numeric_limits<size_t>::max();
    MemRef memref(MemorySpace::DDR, addr, maxSize, 0);

    ASSERT_EQ(memref.size_, maxSize);
}

TEST(IRMemRefTest, TestMemRefWithComplexAddressExpression)
{
    // Test MemRef with complex address expression: base + (index * stride)
    auto base = std::make_shared<ConstInt>(0x10000, DataType::INT64, Span::Unknown());
    auto index = std::make_shared<ConstInt>(5, DataType::INT64, Span::Unknown());
    auto stride = std::make_shared<ConstInt>(256, DataType::INT64, Span::Unknown());

    auto offset = std::make_shared<Mul>(index, stride, DataType::INT64, Span::Unknown());
    auto addr = std::make_shared<Add>(base, offset, DataType::INT64, Span::Unknown());

    MemRef memref(MemorySpace::L1, addr, 256, 0);

    ASSERT_NE(memref.addr_, nullptr);
    ASSERT_EQ(memref.addr_->TypeName(), "Add");
}

} // namespace ir
} // namespace pypto
