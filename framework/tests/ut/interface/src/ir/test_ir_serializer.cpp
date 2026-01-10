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
 * \file test_ir.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "ir/utils_defop.h"
#include "ir/opcode.h"
#include "ir/serializer.h"

#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"

using namespace pto;
using namespace pto::serializer;

class IRSerializerTest : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(IRSerializerTest, MemoryBuffer) {
    MemoryIRBuffer buf;
    buf.Write("abcd");
    std::vector<uint8_t> data = {'0', '1', '2', '3'};
    buf.Write(data);

    char ch = 0;
    EXPECT_EQ(buf.GetRawBuffer(), "abcd0123");
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(1, buf.Read(&ch, 1));
    EXPECT_EQ('a', ch);

    EXPECT_EQ(IRBuffer::ErrorCode::ErrorInvalidOffset, buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(8, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(IRBuffer::ErrorCode::ErrorInvalidOffset, buf.ReadSeek(10, IRBuffer::ReadSeekMode::Absolute));
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(0, IRBuffer::ReadSeekMode::Relative));
    EXPECT_EQ(IRBuffer::ErrorCode::ErrorInvalidOffset, buf.ReadSeek(1, IRBuffer::ReadSeekMode::Relative));
    EXPECT_EQ(0, buf.Read(&ch, 1));
    EXPECT_EQ(IRBuffer::ErrorCode::OK, buf.ReadSeek(-1, IRBuffer::ReadSeekMode::Relative));
    EXPECT_EQ(1, buf.Read(&ch, 1));
    EXPECT_EQ('3', ch);
    EXPECT_EQ(0, buf.Read(&ch, 1));
}

TEST_F(IRSerializerTest, BufferRead) {
    {
        MemoryIRBuffer buf;
        buf.Write(" \t\t abcd\n0123");

        buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute);
        EXPECT_EQ(" \t\t ", buf.ReadSpace());
        EXPECT_EQ("abcd\n", buf.ReadLine());
        EXPECT_EQ("0123", buf.ReadLine());
        EXPECT_EQ("", buf.ReadLine());
        EXPECT_EQ("", buf.ReadSpace());
    }
    {
        MemoryIRBuffer buf;
        buf.Write("a\r\n\t\t/*ccc\ncc*/12345");
        buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute);
        EXPECT_EQ("a\r\n", buf.ReadLine());
        EXPECT_EQ("\t\t", buf.ReadSpace());
        EXPECT_EQ("/*ccc\ncc*/", buf.ReadUntil("*/"));
        EXPECT_EQ("12345", buf.ReadUntil("*/"));
    }
    {
        MemoryIRBuffer buf;
        buf.Write("ab123 123 0x123 0Xabc cde");
        buf.ReadSeek(0, IRBuffer::ReadSeekMode::Absolute);
        EXPECT_EQ("ab123", SourceReadIdentifier(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("123", SourceReadNumber(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("0x123", SourceReadNumber(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("0Xabc", SourceReadNumber(buf));
        EXPECT_EQ(" ", buf.ReadSpace());
        EXPECT_EQ("cde", SourceReadIdentifier(buf));
    }
}

TEST_F(IRSerializerTest, SourceCppTokenization) {
    MemoryIRBuffer buf;
    buf.Write("r0(r1,r2){la0(la1,la2)}");

    SourceCppIRSerializer serializer;
    std::vector<SourceCppToken> tokenList;
    serializer.Tokenization(buf, tokenList);

    std::vector<std::string> tokenTextList;
    for (auto &token : tokenList) {
        tokenTextList.push_back(token.GetToken());
    }
    std::vector<std::string> resultList = {
        "r0", "(", "r1", ",", "r2", ")", "{", "la0", "(", "la1", ",", "la2", ")", "}"
    };
    EXPECT_EQ(resultList, tokenTextList);
}

TEST_F(IRSerializerTest, SourceCppASTNode) {
    auto root = std::make_shared<SourceCppASTNode>("r0", std::vector<std::string>({"r1", "r2"}));
    auto la = std::make_shared<SourceCppASTNode>("la0", std::vector<std::string>({"la1", "la2"}));
    auto lb = std::make_shared<SourceCppASTNode>("lb0", std::vector<std::string>({"lb1"}));
    auto lc = std::make_shared<SourceCppASTNode>("lc0", std::vector<std::string>());
    lb->push_back(lc);
    root->push_back(la);
    root->push_back(lb);
    MemoryIRBuffer buf;
    buf << "\n" << root;
    EXPECT_EQ(buf.GetRawBuffer(), R"(
r0(r1, r2) {
    la0(la1, la2)
    lb0(lb1) {
        lc0()
    }
}
)");
}

ProgramModulePtr CreateAdd() {
    auto prog = std::make_shared<ProgramModule>("main");
    IRBuilder builder(prog);
    IRBuilderContext ctx;

    auto createConst = [&](int n) {
        return builder.CreateConst(ctx, int64_t(n), "const_" + std::to_string(n));
    };

    FunctionSignature sig;

    // ===== Function =====
    auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);
    prog->SetProgramEntry(func);
        // 进入函数体作用域
    builder.EnterFunctionBody(ctx, func);

    auto param = builder.CreateScalar(ctx, DataType::INT64, "param");

    auto pipeV = builder.CreateScalar(ctx, DataType::INT32, "PIPE_V");
    auto pipeMTE2 = builder.CreateScalar(ctx, DataType::INT32, "PIPE_MTE2");
    auto pipeMTE3 = builder.CreateScalar(ctx, DataType::INT32, "PIPE_MTE3");
    auto eventID0 = builder.CreateScalar(ctx, DataType::INT32, "EVENT_ID0");
    auto voidValue = builder.CreateScalar(ctx, DataType::INT32, "_");

    auto sym_72_dim_0 = builder.CreateScalar(ctx, DataType::INT64, "sym_72_dim_0");
    auto sym_72_dim_1 = builder.CreateScalar(ctx, DataType::INT64, "sym_72_dim_1");
    auto sym_876_dim_0 = builder.CreateScalar(ctx, DataType::INT64, "sym_876_dim_0");
    auto sym_876_dim_1 = builder.CreateScalar(ctx, DataType::INT64, "sym_876_dim_1");

    auto gmt5Addr = builder.CreateScalar(ctx, DataType::INT64, "gmt5Addr");
    auto gmt5Rawshape_0 = builder.CreateScalar(ctx, DataType::INT64, "gmt5Rawshape_0");
    auto gmt5Rawshape_1 = builder.CreateScalar(ctx, DataType::INT64, "gmt5Rawshape_1");
    auto gmt5Stride_0 = builder.CreateScalar(ctx, DataType::INT64, "gmt5Stride_0");
    auto gmt5Stride_1 = builder.CreateScalar(ctx, DataType::INT64, "gmt5Stride_1");

    auto gmt1Addr = builder.CreateScalar(ctx, DataType::INT64, "gmt1Addr");
    auto gmt1Rawshape_0 = builder.CreateScalar(ctx, DataType::INT64, "gmt1Rawshape_0");
    auto gmt1Rawshape_1 = builder.CreateScalar(ctx, DataType::INT64, "gmt1Rawshape_1");
    auto gmt1Stride_0 = builder.CreateScalar(ctx, DataType::INT64, "gmt1Stride_0");
    auto gmt1Stride_1 = builder.CreateScalar(ctx, DataType::INT64, "gmt1Stride_1");

    auto copyInOffset_0 = builder.CreateScalar(ctx, DataType::INT64, "copyInOffset_0");
    auto copyInOffset_1 = builder.CreateScalar(ctx, DataType::INT64, "copyInOffset_1");

    auto copyOutOffset_0 = builder.CreateScalar(ctx, DataType::INT64, "copyOutOffset_0");
    auto copyOutOffset_1 = builder.CreateScalar(ctx, DataType::INT64, "copyOutOffset_1");

    auto gmt5 = builder.CreateTensor(ctx, {gmt5Rawshape_0, gmt5Rawshape_1}, {gmt5Stride_0, gmt5Stride_1}, DataType::FP32, "gmt5");
    auto gmt1 = builder.CreateTensor(ctx, {gmt1Rawshape_0, gmt1Rawshape_1}, {gmt1Stride_0, gmt1Stride_1}, DataType::FP32, "gmt1");
    auto ubt0 = builder.CreateTile(ctx, {32, 32}, DataType::FP32, "ubt0");
    ubt0->SetValidShapes({sym_72_dim_0, sym_72_dim_1});

    auto mem = std::make_shared<Memory>(0x1000, MemSpaceKind::UB);
    mem->SetAddr(0x0);

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, param, createConst(1), createConst(10), gmt5Addr,
        "GET_PARAM_ADDR"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(1), createConst(10), createConst(2), createConst(0), gmt5Rawshape_0,
        "GET_PARAM_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(1), createConst(10), createConst(2), createConst(1), gmt5Rawshape_1,
        "GET_PARAM_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(1), createConst(10), createConst(2), createConst(0), gmt5Stride_0,
        "GET_PARAM_STRIDE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(1), createConst(10), createConst(2), createConst(1), gmt5Stride_1,
        "GET_PARAM_STRIDE_BY_IDX"));

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3, param, createConst(0), createConst(1), gmt1Addr,
        "GET_PARAM_ADDR"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(0), createConst(1), createConst(2), createConst(0), gmt1Rawshape_0,
        "GET_PARAM_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(0), createConst(1), createConst(2), createConst(1), gmt1Rawshape_1,
        "GET_PARAM_RAWSHAPE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(0), createConst(1), createConst(2), createConst(0), gmt1Stride_0,
        "GET_PARAM_STRIDE_BY_IDX"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, param, createConst(0), createConst(1), createConst(2), createConst(1), gmt1Stride_1,
        "GET_PARAM_STRIDE_BY_IDX"));

    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(1), createConst(32), createConst(2), createConst(1), createConst(0), sym_72_dim_0,
        "RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST"));
    builder.Emit(ctx, builder.CreateUnaryScalarOp(Opcode::OP_SCALAR_ASSIGN, sym_72_dim_0, sym_72_dim_1));
    builder.Emit(ctx, builder.CreateUnaryScalarOp(Opcode::OP_SCALAR_ASSIGN, sym_72_dim_0, sym_876_dim_0));
    builder.Emit(ctx, builder.CreateUnaryScalarOp(Opcode::OP_SCALAR_ASSIGN, sym_72_dim_0, sym_876_dim_1));

    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(0), createConst(0), createConst(2), createConst(1), createConst(0), copyInOffset_0,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(0), createConst(0), createConst(2), createConst(1), createConst(1), copyInOffset_1,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));

    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(1), createConst(0), createConst(2), createConst(10), createConst(0), copyOutOffset_0,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));
    builder.Emit(ctx, builder.CreateCall5ScalarOp(
        Opcode::OP_SCALAR_CALL_5, createConst(1), createConst(0), createConst(2), createConst(10), createConst(1), copyOutOffset_1,
        "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST"));

    builder.Emit(ctx, builder.CreateUBCopyInOp(Opcode::OP_UB_COPY_IN, gmt1, {copyInOffset_0, copyInOffset_1}, ubt0));

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeMTE2, pipeV, eventID0, voidValue, "set_flag"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeMTE2, pipeV, eventID0, voidValue, "wait_flag"));

    builder.Emit(ctx, builder.CreateBinaryOp(Opcode::OP_ADD, ubt0, ubt0, ubt0));

    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeV, pipeMTE3, eventID0, voidValue, "set_flag"));
    builder.Emit(ctx, builder.CreateCall3ScalarOp(
        Opcode::OP_SCALAR_CALL_3_RETVOID, pipeV, pipeMTE3, eventID0, voidValue, "wait_flag"));

    builder.Emit(ctx, builder.CreateUBCopyOutOp(Opcode::OP_UB_COPY_OUT, ubt0, {copyOutOffset_0, copyOutOffset_1}, gmt5));

    builder.CreateReturn(ctx, {createConst(0)});
    ctx.PopScope(); // function-body
    return prog;
}

TEST_F(IRSerializerTest, Craft) {
    auto prog = CreateAdd();
    SourceCppIRSerializer serializer;

    MemoryIRBuffer buf;
    serializer.Serialize(buf, prog);
    std::cout << buf.GetRawBuffer();
}