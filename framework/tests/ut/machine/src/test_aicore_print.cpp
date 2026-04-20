/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <string>
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>

#include "interface/machine/device/tilefwk/aicore_print.h"

using namespace npu::tile_fwk;

namespace {

struct MockLogger {
    LogContext ctx{};
    std::string buffer;

    static void PrintInt(LogContext* c, __gm__ const char** /*fmt*/, int64_t val)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(val);
    }

    static void PrintFp32(LogContext* c, __gm__ const char** /*fmt*/, float val)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(val);
    }

    static void PrintBf16(LogContext* c, __gm__ const char** /*fmt*/, uint16_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(DecodeBf16(rawBits));
    }

    static void PrintFp16(LogContext* c, __gm__ const char** /*fmt*/, uint16_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(DecodeF16(rawBits));
    }

    static void PrintRaw(LogContext* c, __gm__ const char* fmt)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        if (fmt != nullptr) {
            self->buffer += fmt;
        }
    }

#if SUPPORT_FP8_HF8_PRINT
    static void PrintFp8E4M3(LogContext* c, __gm__ const char** /*fmt*/, uint8_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(DecodeFp8E4M3(rawBits));
    }

    static void PrintFp8E5M2(LogContext* c, __gm__ const char** /*fmt*/, uint8_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(DecodeFp8E5M2(rawBits));
    }

    static void PrintFp8E8M0(LogContext* c, __gm__ const char** /*fmt*/, uint8_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(DecodeFp8E8M0(rawBits));
    }

    static void PrintHf8(LogContext* c, __gm__ const char** /*fmt*/, uint8_t rawBits)
    {
        auto self = reinterpret_cast<MockLogger*>(c);
        self->buffer += std::to_string(DecodeHf8(rawBits));
    }
#endif

    MockLogger()
    {
        ctx.PrintInt64 = &MockLogger::PrintInt;
        ctx.PrintFp32  = &MockLogger::PrintFp32;
        ctx.PrintBf16  = &MockLogger::PrintBf16;
        ctx.PrintFp16  = &MockLogger::PrintFp16;
        ctx.PrintRaw   = &MockLogger::PrintRaw;
#if SUPPORT_FP8_HF8_PRINT
        ctx.PrintFp8E4M3 = &MockLogger::PrintFp8E4M3;
        ctx.PrintFp8E5M2 = &MockLogger::PrintFp8E5M2;
        ctx.PrintFp8E8M0 = &MockLogger::PrintFp8E8M0;
        ctx.PrintHf8     = &MockLogger::PrintHf8;
#endif
    }
};

}

class AiCorePrintUTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Section 1: DecodeF16 函数功能测试 - FP16 位模式解码为 float
// ============================================================================

// 测试 DecodeF16 对正零和负零的解码，验证符号位保持
TEST_F(AiCorePrintUTest, DecodeF16_ZeroValues)
{
    EXPECT_FLOAT_EQ(0.0f, DecodeF16(0x0000u));
    EXPECT_FLOAT_EQ(-0.0f, DecodeF16(0x8000u));
}

// 测试 DecodeF16 对正常正值范围(1.0~10.0)的解码正确性
TEST_F(AiCorePrintUTest, DecodeF16_NormalPositiveRange)
{
    EXPECT_NEAR(1.0f, DecodeF16(0x3C00u), 1e-6f);
    EXPECT_NEAR(2.0f, DecodeF16(0x4000u), 1e-6f);
    EXPECT_NEAR(0.5f, DecodeF16(0x3800u), 1e-6f);
    EXPECT_NEAR(0.25f, DecodeF16(0x3400u), 1e-6f);
    EXPECT_NEAR(1.5f, DecodeF16(0x3E00u), 1e-6f);
    EXPECT_NEAR(3.0f, DecodeF16(0x4200u), 1e-6f);
    EXPECT_NEAR(4.0f, DecodeF16(0x4400u), 1e-6f);
    EXPECT_NEAR(10.0f, DecodeF16(0x4900u), 1e-6f);
}

// 测试 DecodeF16 对正常负值范围的解码正确性，验证符号位处理
TEST_F(AiCorePrintUTest, DecodeF16_NormalNegativeRange)
{
    EXPECT_NEAR(-1.0f, DecodeF16(0xBC00u), 1e-6f);
    EXPECT_NEAR(-2.0f, DecodeF16(0xC000u), 1e-6f);
    EXPECT_NEAR(-0.5f, DecodeF16(0xB800u), 1e-6f);
    EXPECT_NEAR(-1.5f, DecodeF16(0xBE00u), 1e-6f);
    EXPECT_NEAR(-3.0f, DecodeF16(0xC200u), 1e-6f);
}

// 测试 DecodeF16 对次正规值(指数为0且尾数非0)的解码，验证极小值处理
TEST_F(AiCorePrintUTest, DecodeF16_SubnormalValues)
{
    float minSubnormal = DecodeF16(0x0001u);
    EXPECT_GT(minSubnormal, 0.0f);
    EXPECT_LT(minSubnormal, 6e-5f);

    EXPECT_NEAR(3.0517578125e-05f, DecodeF16(0x0200u), 1e-10f);
    EXPECT_NEAR(6.103515625e-05f, DecodeF16(0x0400u), 1e-10f);

    float maxSubnormal = DecodeF16(0x03FFu);
    EXPECT_GT(maxSubnormal, 0.0f);
    EXPECT_LT(maxSubnormal, 0.000061f);
}

// 测试 DecodeF16 对无穷大和 NaN 的解码，验证特殊浮点值识别
TEST_F(AiCorePrintUTest, DecodeF16_InfAndNan)
{
    float posInf = DecodeF16(0x7C00u);
    EXPECT_TRUE(std::isinf(posInf));
    EXPECT_GT(posInf, 0.0f);

    float negInf = DecodeF16(0xFC00u);
    EXPECT_TRUE(std::isinf(negInf));
    EXPECT_LT(negInf, 0.0f);

    float nan1 = DecodeF16(0x7E00u);
    EXPECT_TRUE(std::isnan(nan1));

    float nan2 = DecodeF16(0x7FFFu);
    EXPECT_TRUE(std::isnan(nan2));

    float nanNeg = DecodeF16(0xFE00u);
    EXPECT_TRUE(std::isnan(nanNeg));
}

// 测试 DecodeF16 对最大正常值(0x7BFF≈65504)的解码，验证边界值非溢出
TEST_F(AiCorePrintUTest, DecodeF16_MaxNormalValue)
{
    float maxNormal = DecodeF16(0x7BFFu);
    EXPECT_GT(maxNormal, 65000.0f);
    EXPECT_LT(maxNormal, 65600.0f);
    EXPECT_FALSE(std::isinf(maxNormal));
}

// ============================================================================
// Section 2: DecodeBf16 函数功能测试 - BF16 位模式解码为 float
// ============================================================================

// 测试 DecodeBf16 对正零和负零的解码
TEST_F(AiCorePrintUTest, DecodeBf16_ZeroValues)
{
    EXPECT_FLOAT_EQ(0.0f, DecodeBf16(0x0000u));
    EXPECT_FLOAT_EQ(-0.0f, DecodeBf16(0x8000u));
}

// 测试 DecodeBf16 对正常正值范围的解码正确性(1.0~128.0)
TEST_F(AiCorePrintUTest, DecodeBf16_NormalPositiveRange)
{
    EXPECT_NEAR(1.0f, DecodeBf16(0x3F80u), 1e-6f);
    EXPECT_NEAR(2.0f, DecodeBf16(0x4000u), 1e-6f);
    EXPECT_NEAR(0.5f, DecodeBf16(0x3F00u), 1e-6f);
    EXPECT_NEAR(1.5f, DecodeBf16(0x3FC0u), 1e-6f);
    EXPECT_NEAR(3.0f, DecodeBf16(0x4040u), 1e-6f);
    EXPECT_NEAR(4.0f, DecodeBf16(0x4080u), 1e-6f);
    EXPECT_NEAR(128.0f, DecodeBf16(0x4300u), 1e-6f);
}

// 测试 DecodeBf16 对正常负值范围的解码，验证符号位处理
TEST_F(AiCorePrintUTest, DecodeBf16_NormalNegativeRange)
{
    EXPECT_NEAR(-1.0f, DecodeBf16(0xBF80u), 1e-6f);
    EXPECT_NEAR(-2.0f, DecodeBf16(0xC000u), 1e-6f);
    EXPECT_NEAR(-0.5f, DecodeBf16(0xBF00u), 1e-6f);
    EXPECT_NEAR(-1.5f, DecodeBf16(0xBFC0u), 1e-6f);
}

// 测试 DecodeBf16 对无穷大和 NaN 的解码，验证特殊浮点值识别
TEST_F(AiCorePrintUTest, DecodeBf16_InfAndNan)
{
    float posInf = DecodeBf16(0x7F80u);
    EXPECT_TRUE(std::isinf(posInf));

    float negInf = DecodeBf16(0xFF80u);
    EXPECT_TRUE(std::isinf(negInf));

    float nan1 = DecodeBf16(0x7FC0u);
    EXPECT_TRUE(std::isnan(nan1));

    float nan2 = DecodeBf16(0x7FFFu);
    EXPECT_TRUE(std::isnan(nan2));
}

// ============================================================================
// Section 3: SafeBitCast 函数功能测试 - 类型间安全位转换
// ============================================================================

// 测试 SafeBitCast 在 float 和 uint32 之间往返转换，验证位模式保持
TEST_F(AiCorePrintUTest, SafeBitCast_FloatUintRoundTrip)
{
    float fVal = 3.14159f;
    uint32_t uBits = SafeBitCast<uint32_t>(fVal);
    float fRestored = SafeBitCast<float>(uBits);
    EXPECT_FLOAT_EQ(fVal, fRestored);

    float fNeg = -273.15f;
    uint32_t uNeg = SafeBitCast<uint32_t>(fNeg);
    float fNegRestored = SafeBitCast<float>(uNeg);
    EXPECT_FLOAT_EQ(fNeg, fNegRestored);
}

// 测试 SafeBitCast 对特殊浮点值(零、-零、Inf、-Inf)的位转换正确性
TEST_F(AiCorePrintUTest, SafeBitCast_SpecialFloatValues)
{
    float fZero = 0.0f;
    uint32_t uZero = SafeBitCast<uint32_t>(fZero);
    EXPECT_EQ(0u, uZero);

    float fNegZero = -0.0f;
    uint32_t uNegZero = SafeBitCast<uint32_t>(fNegZero);
    EXPECT_EQ(0x80000000u, uNegZero);

    float fInf = std::numeric_limits<float>::infinity();
    uint32_t uInf = SafeBitCast<uint32_t>(fInf);
    EXPECT_EQ(0x7F800000u, uInf);

    float fNegInf = -std::numeric_limits<float>::infinity();
    uint32_t uNegInf = SafeBitCast<uint32_t>(fNegInf);
    EXPECT_EQ(0xFF800000u, uNegInf);
}

// 测试 SafeBitCast 对整数类型扩展(uint8->uint32, uint16->uint32)时高位补零
TEST_F(AiCorePrintUTest, SafeBitCast_IntegerExtension)
{
    uint16_t u16 = 0xABCD;
    uint32_t u32 = SafeBitCast<uint32_t>(u16);
    EXPECT_EQ(0xABCDu, u32);

    uint8_t u8 = 0x42;
    uint32_t u32From8 = SafeBitCast<uint32_t>(u8);
    EXPECT_EQ(0x42u, u32From8);

    int64_t i64Neg = -1LL;
    uint64_t u64 = SafeBitCast<uint64_t>(i64Neg);
    EXPECT_EQ(0xFFFFFFFFFFFFFFFFULL, u64);
}

// ============================================================================
// Section 4: IndexedTypeInfo 模板功能测试 - 类型属性查询
// ============================================================================

// 测试 IndexedTypeInfo<float> 返回正确的 DataType
TEST_F(AiCorePrintUTest, IndexedTypeInfo_FloatProperties)
{
    EXPECT_EQ(AicorePrint::DataType::IndexedFp32, IndexedTypeInfo<float>::Type);
}

// 测试 IndexedTypeInfo<int64_t> 返回正确的 DataType
TEST_F(AiCorePrintUTest, IndexedTypeInfo_Int64Properties)
{
    EXPECT_EQ(AicorePrint::DataType::IndexedInt64, IndexedTypeInfo<int64_t>::Type);
}

// 测试 IndexedTypeInfo 对未知类型(char/double)返回默认值 End
TEST_F(AiCorePrintUTest, IndexedTypeInfo_UnknownTypeDefaults)
{
    EXPECT_EQ(AicorePrint::DataType::End, IndexedTypeInfo<char>::Type);
    EXPECT_EQ(AicorePrint::DataType::End, IndexedTypeInfo<double>::Type);
}

#if IS_AICORE

// 测试 IndexedTypeInfo<bfloat16_t> 在 AiCore 环境下的正确属性
TEST_F(AiCorePrintUTest, IndexedTypeInfo_Bf16Properties)
{
    EXPECT_EQ(AicorePrint::DataType::IndexedBf16, IndexedTypeInfo<bfloat16_t>::Type);
}

// 测试 IndexedTypeInfo<half> 在 AiCore 环境下的正确属性
TEST_F(AiCorePrintUTest, IndexedTypeInfo_Fp16Properties)
{
    EXPECT_EQ(AicorePrint::DataType::IndexedFp16, IndexedTypeInfo<half>::Type);
}

#endif

#if SUPPORT_FP8_HF8_PRINT

// 测试 IndexedTypeInfo 对 FP8 类型(E4M3/E5M2/E8M0/Hf8)的 DataType 返回值
TEST_F(AiCorePrintUTest, IndexedTypeInfo_Fp8Properties)
{
    EXPECT_EQ(AicorePrint::DataType::IndexedFp8E4M3, IndexedTypeInfo<float8_e4m3_t>::Type);
    EXPECT_EQ(AicorePrint::DataType::IndexedFp8E5M2, IndexedTypeInfo<float8_e5m2_t>::Type);
    EXPECT_EQ(AicorePrint::DataType::IndexedFp8E8M0, IndexedTypeInfo<float8_e8m0_t>::Type);
    EXPECT_EQ(AicorePrint::DataType::IndexedHf8, IndexedTypeInfo<hifloat8_t>::Type);
}

// ============================================================================
// Section 5: FP8/HF8 Decode 函数功能测试 - 8位浮点格式解码
// ============================================================================

class Fp8DecodeTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 测试 DecodeFp8E4M3 对零值的解码，验证正零和负零
TEST_F(Fp8DecodeTest, DecodeFp8E4M3_ZeroValues)
{
    EXPECT_FLOAT_EQ(0.0f, DecodeFp8E4M3(0x00));
    EXPECT_FLOAT_EQ(-0.0f, DecodeFp8E4M3(0x80));
}

// 测试 DecodeFp8E4M3 对正常正值(1.0~4.0)的解码正确性
TEST_F(Fp8DecodeTest, DecodeFp8E4M3_NormalPositive)
{
    EXPECT_NEAR(1.0f, DecodeFp8E4M3(0x38), 0.01f);
    EXPECT_NEAR(2.0f, DecodeFp8E4M3(0x40), 0.01f);
    EXPECT_NEAR(4.0f, DecodeFp8E4M3(0x50), 0.02f);
    EXPECT_NEAR(0.5f, DecodeFp8E4M3(0x30), 0.01f);
}

// 测试 DecodeFp8E4M3 对正常负值的解码，验证符号位处理
TEST_F(Fp8DecodeTest, DecodeFp8E4M3_NormalNegative)
{
    EXPECT_NEAR(-1.0f, DecodeFp8E4M3(0xB8), 0.01f);
    EXPECT_NEAR(-2.0f, DecodeFp8E4M3(0xC0), 0.01f);
    EXPECT_NEAR(-4.0f, DecodeFp8E4M3(0xD0), 0.02f);
}

// 测试 DecodeFp8E4M3 对最大值(0x7E≈448)的解码，验证无 Inf 溢出
TEST_F(Fp8DecodeTest, DecodeFp8E4M3_MaxValue)
{
    float maxVal = DecodeFp8E4M3(0x7E);
    EXPECT_NEAR(448.0f, maxVal, 1.0f);
    EXPECT_FALSE(std::isinf(maxVal));
}

// 测试 DecodeFp8E5M2 对零值和无穷大的解码，验证 Inf 支持
TEST_F(Fp8DecodeTest, DecodeFp8E5M2_ZeroAndInf)
{
    EXPECT_FLOAT_EQ(0.0f, DecodeFp8E5M2(0x00));
    EXPECT_FLOAT_EQ(-0.0f, DecodeFp8E5M2(0x80));

    float posInf = DecodeFp8E5M2(0x7C);
    EXPECT_TRUE(std::isinf(posInf));
    EXPECT_GT(posInf, 0.0f);

    float negInf = DecodeFp8E5M2(0xFC);
    EXPECT_TRUE(std::isinf(negInf));
    EXPECT_LT(negInf, 0.0f);
}

// 测试 DecodeFp8E5M2 对 NaN 值的解码，验证 NaN 位模式识别
TEST_F(Fp8DecodeTest, DecodeFp8E5M2_NanValues)
{
    EXPECT_TRUE(std::isnan(DecodeFp8E5M2(0x7E)));
    EXPECT_TRUE(std::isnan(DecodeFp8E5M2(0x7F)));
    EXPECT_TRUE(std::isnan(DecodeFp8E5M2(0xFE)));
}

// 测试 DecodeFp8E5M2 对正常正值(1.0~4.0)的解码正确性
TEST_F(Fp8DecodeTest, DecodeFp8E5M2_NormalPositive)
{
    EXPECT_NEAR(1.0f, DecodeFp8E5M2(0x3C), 0.01f);
    EXPECT_NEAR(2.0f, DecodeFp8E5M2(0x40), 0.01f);
    EXPECT_NEAR(4.0f, DecodeFp8E5M2(0x48), 0.02f);
    EXPECT_NEAR(0.5f, DecodeFp8E5M2(0x38), 0.01f);
}

// 测试 DecodeFp8E8M0 对 2 的幂次(0.5~4.0)的解码，验证纯指数格式
TEST_F(Fp8DecodeTest, DecodeFp8E8M0_PowersOfTwo)
{
    EXPECT_FLOAT_EQ(0.0f, DecodeFp8E8M0(0x00));
    EXPECT_NEAR(1.0f, DecodeFp8E8M0(0x80), 0.01f);
    EXPECT_NEAR(2.0f, DecodeFp8E8M0(0x81), 0.01f);
    EXPECT_NEAR(4.0f, DecodeFp8E8M0(0x82), 0.02f);
    EXPECT_NEAR(0.5f, DecodeFp8E8M0(0x7F), 0.01f);
}

// 测试 DecodeFp8E8M0 对负值的解码，验证符号位处理
TEST_F(Fp8DecodeTest, DecodeFp8E8M0_NegativePowersOfTwo)
{
    EXPECT_FLOAT_EQ(-0.0f, DecodeFp8E8M0(0x80));
    EXPECT_NEAR(-1.0f, DecodeFp8E8M0(0x00), 0.01f);
    EXPECT_NEAR(-2.0f, DecodeFp8E8M0(0x01), 0.01f);
    EXPECT_NEAR(-4.0f, DecodeFp8E8M0(0x02), 0.02f);
}

// 测试 DecodeHf8 对零值和小值(0.5~1.0)的解码，验证华为自定义格式
TEST_F(Fp8DecodeTest, DecodeHf8_ZeroAndSmallValues)
{
    EXPECT_FLOAT_EQ(0.0f, DecodeHf8(0x00));
    EXPECT_FLOAT_EQ(-0.0f, DecodeHf8(0x80));
    EXPECT_NEAR(0.5f, DecodeHf8(0x04), 0.01f);
    EXPECT_NEAR(1.0f, DecodeHf8(0x08), 0.01f);
}

// 测试 DecodeHf8 对所有位模式的符号保持，验证正值和负值对称性
TEST_F(Fp8DecodeTest, DecodeHf8_SignPreservation)
{
    for (int i = 1; i < 128; i++) {
        float posVal = DecodeHf8(static_cast<uint8_t>(i));
        float negVal = DecodeHf8(static_cast<uint8_t>(i | 0x80));

        EXPECT_GT(posVal, 0.0f);
        EXPECT_LT(negVal, 0.0f);
        EXPECT_NEAR(std::abs(posVal), std::abs(negVal), std::abs(posVal) * 0.01f);
    }
}

// 测试 DecodeHf8 对各分支覆盖(Tiny/Small/Medium/Large/Huge/Max)
TEST_F(Fp8DecodeTest, DecodeHf8_AllBranches)
{
    EXPECT_NEAR(0.0f, DecodeHf8(0x00), 1e-6f);
    EXPECT_NEAR(1.0f, DecodeHf8(0x08), 0.01f);
    EXPECT_NEAR(2.0f, DecodeHf8(0x12), 0.01f);
    EXPECT_NEAR(4.0f, DecodeHf8(0x24), 0.02f);
    EXPECT_NEAR(8.0f, DecodeHf8(0x48), 0.05f);
}

#endif

// ============================================================================
// Section 6: AicorePrintConst 常量功能测试 - 打印框架常量验证
// ============================================================================

// 测试 AicorePrintConst 各字段大小与对应类型 sizeof 一致
TEST_F(AiCorePrintUTest, AicorePrintConst_FieldSizes)
{
    EXPECT_EQ(AicorePrintConst::INDEXED_INDEX_SIZE, sizeof(int64_t));
    EXPECT_EQ(AicorePrintConst::TENSOR_RANGE_SIZE, sizeof(int64_t));
    EXPECT_EQ(AicorePrintConst::NAMELEN_FIELD_SIZE, sizeof(short));
    EXPECT_EQ(AicorePrintConst::TYPE_FIELD_SIZE, sizeof(uint8_t));
}

// 测试 AicorePrintConst 最小 buffer 总大小等于警告预留空间加 RemoteHeader
TEST_F(AiCorePrintUTest, AicorePrintConst_MinBufferRequirement)
{
    EXPECT_EQ(AicorePrintConst::MIN_BUFFER_TOTAL_SIZE,
              AicorePrintConst::WARNING_RESERVE_SPACE + AicorePrintConst::REMOTE_HEADER_SIZE);
    EXPECT_EQ(AicorePrintConst::MIN_BUFFER_TOTAL_SIZE, 26u);
}

// 测试 AicorePrintConst RemoteHeader 大小等于 2 个 int64_t(16字节)
TEST_F(AiCorePrintUTest, AicorePrintConst_RemoteHeaderSize)
{
    EXPECT_EQ(AicorePrintConst::REMOTE_HEADER_SIZE, 2 * sizeof(int64_t));
}

// 测试 AicorePrintConst MAX_SHAPE_DIMS 常量值为 6
TEST_F(AiCorePrintUTest, AicorePrintConst_MaxShapeDims)
{
    EXPECT_EQ(AicorePrintConst::MAX_SHAPE_DIMS, 6u);
}

// ============================================================================
// Section 7: DataType 枚举功能测试 - 数据类型标记值验证
// ============================================================================

// 测试 DataType 基础类型(End~Pointer)枚举值的顺序正确性
TEST_F(AiCorePrintUTest, DataType_BasicTypes)
{
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::End), 0);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Normal), 1);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Fp32), 2);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Int64), 3);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Char), 4);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::String), 5);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Pointer), 6);
}

// 测试 DataType Float16 类型(Bf16/Fp16)枚举值为 7 和 8
TEST_F(AiCorePrintUTest, DataType_Float16Types)
{
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Bf16), 7);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Fp16), 8);
}

// 测试 DataType Tensor 和 Indexed 类型枚举值为 9~14
TEST_F(AiCorePrintUTest, DataType_TensorAndIndexed)
{
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::TensorHeader), 9);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedFp32), 10);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedInt64), 11);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedBf16), 12);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedFp16), 13);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::OverflowWarning), 14);
}

#if SUPPORT_FP8_HF8_PRINT
// 测试 DataType FP8/HF8 类型枚举值为 15~22
TEST_F(AiCorePrintUTest, DataType_Fp8Hf8Types)
{
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Fp8E4M3), 15);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Fp8E5M2), 16);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Fp8E8M0), 17);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::Hf8), 18);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedFp8E4M3), 19);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedFp8E5M2), 20);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedFp8E8M0), 21);
    EXPECT_EQ(static_cast<uint8_t>(AicorePrint::DataType::IndexedHf8), 22);
}
#endif

// ============================================================================
// Section 8: 浮点格式常量功能测试 - FP32/FP16/BF16 位域常量验证
// ============================================================================

// 测试 Fp32Const 各常量(符号/指数/尾数 mask、bias、shift)的正确值
TEST_F(AiCorePrintUTest, Fp32Const_CorrectValues)
{
    EXPECT_EQ(Fp32Const::SIGN_MASK, 0x80000000u);
    EXPECT_EQ(Fp32Const::EXP_MASK, 0x7F800000u);
    EXPECT_EQ(Fp32Const::MANT_MASK, 0x007FFFFFu);
    EXPECT_EQ(Fp32Const::EXP_BIAS, 127u);
    EXPECT_EQ(Fp32Const::EXP_INF_NAN, 0xFFu);
    EXPECT_EQ(Fp32Const::SIGN_SHIFT, 31u);
    EXPECT_EQ(Fp32Const::EXP_SHIFT, 23u);
    EXPECT_EQ(Fp32Const::MANT_BITS, 23u);
}

// 测试 Fp16Const 各常量(符号/指数/尾数 mask、bias)的正确值
TEST_F(AiCorePrintUTest, Fp16Const_CorrectValues)
{
    EXPECT_EQ(Fp16Const::SIGN_MASK, 0x8000u);
    EXPECT_EQ(Fp16Const::EXP_MASK, 0x7C00u);
    EXPECT_EQ(Fp16Const::MANT_MASK, 0x03FFu);
    EXPECT_EQ(Fp16Const::EXP_BIAS, 15u);
    EXPECT_EQ(Fp16Const::EXP_INF_NAN, 0x1Fu);
}

// 测试 Bf16Const TO_FP32_SHIFT 常量为 16(BF16 是 FP32 高 16 位)
TEST_F(AiCorePrintUTest, Bf16Const_CorrectValue)
{
    EXPECT_EQ(Bf16Const::TO_FP32_SHIFT, 16u);
}

#if SUPPORT_FP8_HF8_PRINT

// 测试 Fp8E4M3Const 各常量(指数/尾数位数、bias、shift)的正确值
TEST_F(AiCorePrintUTest, Fp8E4M3Const_CorrectValues)
{
    EXPECT_EQ(Fp8E4M3Const::EXP_BITS, 4u);
    EXPECT_EQ(Fp8E4M3Const::MANT_BITS, 3u);
    EXPECT_EQ(Fp8E4M3Const::EXP_BIAS, 7u);
    EXPECT_EQ(Fp8E4M3Const::SIGN_SHIFT, 7u);
    EXPECT_EQ(Fp8E4M3Const::EXP_SHIFT, 3u);
}

// 测试 Fp8E5M2Const 各常量(指数/尾数位数、bias、shift)的正确值
TEST_F(AiCorePrintUTest, Fp8E5M2Const_CorrectValues)
{
    EXPECT_EQ(Fp8E5M2Const::EXP_BITS, 5u);
    EXPECT_EQ(Fp8E5M2Const::MANT_BITS, 2u);
    EXPECT_EQ(Fp8E5M2Const::EXP_BIAS, 15u);
    EXPECT_EQ(Fp8E5M2Const::SIGN_SHIFT, 7u);
    EXPECT_EQ(Fp8E5M2Const::EXP_SHIFT, 2u);
}

// 测试 Fp8E8M0Const 各常量(指数位数 7、尾数 0、bias 127)的正确值
TEST_F(AiCorePrintUTest, Fp8E8M0Const_CorrectValues)
{
    EXPECT_EQ(Fp8E8M0Const::EXP_BITS, 7u);
    EXPECT_EQ(Fp8E8M0Const::MANT_BITS, 0u);
    EXPECT_EQ(Fp8E8M0Const::EXP_BIAS, 127u);
    EXPECT_EQ(Fp8E8M0Const::SIGN_SHIFT, 7u);
}
#endif

// ============================================================================
// Section 9: 编译条件功能测试 - 宏定义值验证
// ============================================================================

// 测试 ENABLE_AICORE_PRINT 宏默认为 0(打印功能关闭)
TEST_F(AiCorePrintUTest, CompileConditions_EnableAicorePrint)
{
    EXPECT_EQ(ENABLE_AICORE_PRINT, 0);
}

// 测试 CACHE_LINE_SIZE 宏为 64 字节
TEST_F(AiCorePrintUTest, CompileConditions_CacheLineSize)
{
    EXPECT_EQ(CACHE_LINE_SIZE, 64);
}

#if SUPPORT_FP8_HF8_PRINT
// 测试 SUPPORT_FP8_HF8_PRINT 在 FP8 支持平台为 1
TEST_F(AiCorePrintUTest, CompileConditions_SupportFp8Hf8Enabled)
{
    EXPECT_EQ(SUPPORT_FP8_HF8_PRINT, 1);
}
#else
// 测试 SUPPORT_FP8_HF8_PRINT 在非 FP8 支持平台为 0
TEST_F(AiCorePrintUTest, CompileConditions_SupportFp8Hf8Disabled)
{
    EXPECT_EQ(SUPPORT_FP8_HF8_PRINT, 0);
}
#endif

// 测试 __TILE_FWK_HOST__ 宏在当前编译环境已定义(Host 测试可用)
TEST_F(AiCorePrintUTest, HostEnvironmentDefined)
{
#ifdef __TILE_FWK_HOST__
    EXPECT_TRUE(true) << "__TILE_FWK_HOST__ is defined";
#else
    EXPECT_TRUE(false) << "__TILE_FWK_HOST__ is NOT defined - Host tests won't run";
#endif
}

// 测试 IS_AICORE 宏在 Host 环境为 0(非 AiCore 设备侧)
TEST_F(AiCorePrintUTest, IsAicoreValue)
{
    EXPECT_EQ(IS_AICORE, 0) << "IS_AICORE should be 0 in Host environment";
}

// ============================================================================
// Section 10: MockLogger 打印功能测试 - LogContext 函数指针调用
// ============================================================================

// 测试通过 MockLogger 打印 FP16 解码后的浮点值(1.5)，验证 PrintFp16 路径
TEST_F(AiCorePrintUTest, PrintFp16DecodedValue)
{
    MockLogger logger;
    uint16_t bits = 0x3E00u;
    float v = DecodeF16(bits);
    AiCoreLogF(&logger.ctx, "%f", v);
    EXPECT_NE(std::string::npos, logger.buffer.find("1.5")) << "buffer: " << logger.buffer;
}

// 测试通过 MockLogger 打印 BF16 解码后的浮点值(2.0)，验证 PrintBf16 路径
TEST_F(AiCorePrintUTest, PrintBf16DecodedValue)
{
    MockLogger logger;
    uint16_t bits = 0x4000u;
    float v = DecodeBf16(bits);
    AiCoreLogF(&logger.ctx, "%f", v);
    EXPECT_NE(std::string::npos, logger.buffer.find("2")) << "buffer: " << logger.buffer;
}

// 测试通过 MockLogger 同时打印整数和浮点数，验证 PrintInt64 和 PrintFp32 协作
TEST_F(AiCorePrintUTest, PrintIntAndFloat)
{
    MockLogger logger;
    const char* dummyFmt = "%d";
    __gm__ const char* fmtPtr = dummyFmt;
    __gm__ const char** fmt = &fmtPtr;

    logger.ctx.PrintInt64(&logger.ctx, fmt, 42);
    logger.ctx.PrintFp32(&logger.ctx, fmt, 3.5f);

    EXPECT_NE(std::string::npos, logger.buffer.find("42"));
    EXPECT_NE(std::string::npos, logger.buffer.find("3.5"));
}

// 测试 AiCoreLogF 对空指针的安全处理，不崩溃且无输出
TEST_F(AiCorePrintUTest, AiCoreLogF_NullContextSafe)
{
    MockLogger logger;
    AiCoreLogF(nullptr, "test %d", 42);
    EXPECT_TRUE(logger.buffer.empty());
}

// 测试 AiCoreLogF 对空格式字符串的安全处理，不崩溃且无输出
TEST_F(AiCorePrintUTest, AiCoreLogF_NullFormatSafe)
{
    MockLogger logger;
    AiCoreLogF(&logger.ctx, nullptr, 42);
    EXPECT_TRUE(logger.buffer.empty());
}

// ============================================================================
// Section 11: AicoreLogger Host侧完整功能测试 - 编码解码全流程
// ============================================================================

#ifdef __TILE_FWK_HOST__

class AicoreLoggerHostFixture {
protected:
    static constexpr size_t BUFFER_SIZE = 4096;
    alignas(64) uint8_t buffer[BUFFER_SIZE];

    void InitBuffer()
    {
        std::fill(buffer, buffer + BUFFER_SIZE, 0);
    }

    std::string ReadAllOutput(AicoreLogger& logger)
    {
        char readBuf[512];
        std::string output;
        while (true) {
            int bytesRead = logger.Read(readBuf, sizeof(readBuf));
            if (bytesRead == 0) break;
            output.append(readBuf, bytesRead);
        }
        return output;
    }
};

// 测试 AicoreLogger 初始化成功后 head_/tail_/size_ 状态正确
TEST_F(AiCorePrintUTest, HostLogger_InitSuccessStateCorrect)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    LogContext* ctx = logger.Context();
    EXPECT_NE(ctx, nullptr);
    EXPECT_NE(ctx->PrintInt64, nullptr);
    EXPECT_NE(ctx->PrintFp32, nullptr);
}

// 测试 AicoreLogger 使用不足 buffer(10字节)初始化后，编码操作被拒绝无输出
TEST_F(AiCorePrintUTest, HostLogger_InitInsufficientBufferRejectsEncoding)
{
    alignas(64) uint8_t smallBuffer[10];
    std::fill(smallBuffer, smallBuffer + 10, 0);
    AicoreLogger logger;
    logger.Init(smallBuffer, 10);

    AicoreLoggerHostFixture fixture;
    logger.EncodeTensorHeader("test", 0, 100);
    logger.Sync();
    std::string output = fixture.ReadAllOutput(logger);
    EXPECT_TRUE(output.empty());
}

// 测试 EncodeTensorHeader 编码 tensor name 和 range，解码输出包含 name 和 range 信息
TEST_F(AiCorePrintUTest, HostLogger_TensorHeaderWithNameAndRange)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    const char* tensorName = "test_tensor";
    int64_t begin = 0;
    int64_t end = 100;
    logger.EncodeTensorHeader(tensorName, begin, end);
    logger.Sync();

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("test_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("range=[0, 100)") != std::string::npos);
}

// 测试 EncodeTensorHeader 对超长 name(>32767)的编码被拒绝，无输出
TEST_F(AiCorePrintUTest, HostLogger_TensorHeaderNameTooLongRejected)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    std::string longName(40000, 'x');
    logger.EncodeTensorHeader(longName.c_str(), 0, 100);
    logger.Sync();

    std::string output = fixture.ReadAllOutput(logger);
    EXPECT_TRUE(output.empty());
}

// 测试 EncodeTensorHeader 对多个不同 tensor name 的编码，解码输出包含所有 name
TEST_F(AiCorePrintUTest, HostLogger_TensorHeaderWithDifferentNames)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("input_data", 0, 50);
    logger.Sync();
    logger.EncodeTensorHeader("output_result", 0, 75);
    logger.Sync();
    logger.EncodeTensorHeader("weight_matrix", 100, 200);
    logger.Sync();

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("input_data") != std::string::npos);
    EXPECT_TRUE(output.find("output_result") != std::string::npos);
    EXPECT_TRUE(output.find("weight_matrix") != std::string::npos);
    EXPECT_TRUE(output.find("range=[0, 50)") != std::string::npos);
    EXPECT_TRUE(output.find("range=[100, 200)") != std::string::npos);
}

// 测试 EncodeIndexed 编码 FP32 值带索引，解码输出包含 tensor name、索引标记和值
TEST_F(AiCorePrintUTest, HostLogger_IndexedFp32WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("fp32_tensor", 0, 5);
    logger.Sync();

    float values[] = {1.5f, 2.5f, -3.14f, 0.0f, 100.5f};
    for (int64_t i = 0; i < 5; i++) {
        uint8_t* valBytes = reinterpret_cast<uint8_t*>(&values[i]);
        logger.EncodeIndexed(AicorePrint::DataType::IndexedFp32, i, valBytes, sizeof(float));
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("fp32_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
    EXPECT_TRUE(output.find("[1]") != std::string::npos);
    EXPECT_TRUE(output.find("[4]") != std::string::npos);
    EXPECT_TRUE(output.find("1.5") != std::string::npos);
}

// 测试 EncodeIndexed 编码 Int64 值带索引，解码输出包含索引标记和整数值
TEST_F(AiCorePrintUTest, HostLogger_IndexedInt64WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("int64_tensor", 0, 3);
    logger.Sync();

    int64_t values[] = {42, -100, 999999};
    for (int64_t i = 0; i < 3; i++) {
        uint8_t* valBytes = reinterpret_cast<uint8_t*>(&values[i]);
        logger.EncodeIndexed(AicorePrint::DataType::IndexedInt64, i, valBytes, sizeof(int64_t));
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("int64_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
    EXPECT_TRUE(output.find("[2]") != std::string::npos);
    EXPECT_TRUE(output.find("42") != std::string::npos);
    EXPECT_TRUE(output.find("-100") != std::string::npos);
}

// 测试 EncodeIndexed 编码 BF16 位模式带索引，解码输出包含解码后的浮点值
TEST_F(AiCorePrintUTest, HostLogger_IndexedBf16WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("bf16_tensor", 0, 2);
    logger.Sync();

    uint16_t bf16Bits[] = {0x3F80u, 0x4000u};
    for (int64_t i = 0; i < 2; i++) {
        uint8_t* valBytes = reinterpret_cast<uint8_t*>(&bf16Bits[i]);
        logger.EncodeIndexed(AicorePrint::DataType::IndexedBf16, i, valBytes, sizeof(uint16_t));
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("bf16_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
    EXPECT_TRUE(output.find("1") != std::string::npos);
}

// 测试 EncodeIndexed 编码 FP16 位模式带索引，解码输出包含 tensor name 和索引标记
TEST_F(AiCorePrintUTest, HostLogger_IndexedFp16WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("fp16_tensor", 0, 2);
    logger.Sync();

    uint16_t fp16Bits[] = {0x3C00u, 0x4000u};
    for (int64_t i = 0; i < 2; i++) {
        uint8_t* valBytes = reinterpret_cast<uint8_t*>(&fp16Bits[i]);
        logger.EncodeIndexed(AicorePrint::DataType::IndexedFp16, i, valBytes, sizeof(uint16_t));
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("fp16_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
}

// 测试 buffer 满时 EncodeOverflowWarning 触发，解码输出包含 WARNING 和 buffer 满提示
TEST_F(AiCorePrintUTest, HostLogger_OverflowWarningWhenBufferFull)
{
    size_t smallBufferSize = 100;
    alignas(64) uint8_t smallBuffer[100];
    std::fill(smallBuffer, smallBuffer + smallBufferSize, 0);

    AicoreLogger logger;
    logger.Init(smallBuffer, smallBufferSize);

    for (int i = 0; i < 20; i++) {
        logger.EncodeTensorHeader("overflow_test", 0, 100);
        logger.Sync();
    }

    AicoreLoggerHostFixture fixture;
    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("WARNING") != std::string::npos);
    EXPECT_TRUE(output.find("PRINT_BUFFER_SIZE") != std::string::npos);
    EXPECT_TRUE(output.find("full") != std::string::npos);
}

// 测试 OverflowWarning 解码输出包含 buffer size 推荐信息(建议加倍 buffer)
TEST_F(AiCorePrintUTest, HostLogger_OverflowWarningContainsRecommendation)
{
    size_t smallBufferSize = 50;
    alignas(64) uint8_t smallBuffer[50];
    std::fill(smallBuffer, smallBuffer + smallBufferSize, 0);

    AicoreLogger logger;
    logger.Init(smallBuffer, smallBufferSize);

    logger.EncodeTensorHeader("test_overflow", 0, 1000);
    logger.Sync();
    logger.EncodeTensorHeader("another_overflow", 0, 1000);
    logger.Sync();

    AicoreLoggerHostFixture fixture;
    std::string output = fixture.ReadAllOutput(logger);

    if (output.find("WARNING") != std::string::npos) {
        EXPECT_TRUE(output.find("Recommend") != std::string::npos ||
                    output.find("double") != std::string::npos);
    }
}

// 测试连续索引序列(0~9)的打印，解码输出包含所有索引标记 [0]~[9]
TEST_F(AiCorePrintUTest, HostLogger_IndexSequencePrint)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("indexed_data", 0, 10);
    logger.Sync();

    for (int64_t i = 0; i < 10; i++) {
        float val = static_cast<float>(i * 10);
        uint8_t* valBytes = reinterpret_cast<uint8_t*>(&val);
        logger.EncodeIndexed(AicorePrint::DataType::IndexedFp32, i, valBytes, sizeof(float));
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("[0]") != std::string::npos);
    EXPECT_TRUE(output.find("[5]") != std::string::npos);
    EXPECT_TRUE(output.find("[9]") != std::string::npos);
}

// 测试 Context() 返回有效 LogContext 指针，且各 Print 函数指针非空
TEST_F(AiCorePrintUTest, HostLogger_ContextReturnsValidPointer)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    LogContext* ctx = logger.Context();
    EXPECT_NE(ctx, nullptr);
    EXPECT_NE(ctx->PrintInt64, nullptr);
    EXPECT_NE(ctx->PrintFp32, nullptr);
    EXPECT_NE(ctx->PrintBf16, nullptr);
    EXPECT_NE(ctx->PrintFp16, nullptr);
    EXPECT_NE(ctx->PrintRaw, nullptr);
}

// 测试 GetBuffer() 返回的指针与传入 buffer 地址一致
TEST_F(AiCorePrintUTest, HostLogger_GetBufferReturnsCorrectPointer)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    __gm__ uint8_t* bufPtr = logger.GetBuffer();
    EXPECT_EQ(bufPtr, fixture.buffer);
}

#if SUPPORT_FP8_HF8_PRINT

// 测试 EncodeIndexed 编码 FP8 E4M3 值带索引，解码输出包含 tensor name 和索引标记
TEST_F(AiCorePrintUTest, HostLogger_IndexedFp8E4M3WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("fp8_e4m3_tensor", 0, 3);
    logger.Sync();

    uint8_t fp8Bits[] = {0x38, 0x40, 0x50};
    for (int64_t i = 0; i < 3; i++) {
        logger.EncodeIndexed(AicorePrint::DataType::IndexedFp8E4M3, i, &fp8Bits[i], 1);
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("fp8_e4m3_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
}

// 测试 EncodeIndexed 编码 FP8 E5M2 值带索引，解码输出包含 tensor name 和索引标记
TEST_F(AiCorePrintUTest, HostLogger_IndexedFp8E5M2WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("fp8_e5m2_tensor", 0, 2);
    logger.Sync();

    uint8_t fp8Bits[] = {0x3C, 0x40};
    for (int64_t i = 0; i < 2; i++) {
        logger.EncodeIndexed(AicorePrint::DataType::IndexedFp8E5M2, i, &fp8Bits[i], 1);
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("fp8_e5m2_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
}

// 测试 EncodeIndexed 编码 HF8 值带索引，解码输出包含 tensor name 和索引标记
TEST_F(AiCorePrintUTest, HostLogger_IndexedHf8WithIndex)
{
    AicoreLoggerHostFixture fixture;
    fixture.InitBuffer();
    AicoreLogger logger;
    logger.Init(fixture.buffer, AicoreLoggerHostFixture::BUFFER_SIZE);

    logger.EncodeTensorHeader("hf8_tensor", 0, 2);
    logger.Sync();

    uint8_t hf8Bits[] = {0x08, 0x88};
    for (int64_t i = 0; i < 2; i++) {
        logger.EncodeIndexed(AicorePrint::DataType::IndexedHf8, i, &hf8Bits[i], 1);
        logger.Sync();
    }

    std::string output = fixture.ReadAllOutput(logger);

    EXPECT_TRUE(output.find("hf8_tensor") != std::string::npos);
    EXPECT_TRUE(output.find("[0]") != std::string::npos);
}

#endif

#endif

// ============================================================================
// Section 12: Shape 打印功能测试 - AiCore 设备侧 Shape 打印(条件编译)
// ============================================================================

#if defined(__TILE_FWK_AICORE__) && defined(TILEOP_UTILS_TUPLE_H)

// 测试 AiCorePrintShape 对 2D Shape 的打印，输出包含 "shape=[3,5]"
TEST_F(AiCorePrintUTest, AiCorePrintShape2D)
{
    MockLogger logger;
    TileOp::Shape<int64_t, int64_t> shape = {3, 5};
    AiCorePrintShape(&logger.ctx, shape);

    std::string& buf = logger.buffer;
    EXPECT_NE(std::string::npos, buf.find("shape=[3,5]"));
}
#endif