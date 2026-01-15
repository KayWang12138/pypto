/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file verifier/ssa_verify.h
 * \brief SSA semantics verification for TileValue and ScalarValue
 */

#pragma once

#include "ir/verifier/verifier.h"
#include "ir/transform/visitor.h"
#include "ir/operation_base.h"

#include <map>

namespace pto {

class ValueSSAVisitor : public IRVisitor {
public:
    explicit ValueSSAVisitor(std::map<const Value *, size_t> *countMap) : valueCountMap_(countMap) {}

    void VisitImplOp(OperationPtr &op) override;

// C++ requires that when a subclass redefines a virtual function, it must redefine all overloaded virtual functions
// with the same name but different parameters; otherwise, a warning will be issued, and in the current version, all
// warnings are treated as errors.
#define DEFOP(name, inherit, opcode, ...) void VisitImplOp(name##Ptr &op) override;
    #include "ir/operation.def"
    #include "ir/tile_graph.def"
#undef DEFOP
private:
    std::map<const Value *, size_t> *valueCountMap_;
};

VerifyResult VerifySSA(ProgramModulePtr program);

// Helper function to format value information string
std::string GetValueInfoString(const Value *valuePtr);

} // namespace pto
