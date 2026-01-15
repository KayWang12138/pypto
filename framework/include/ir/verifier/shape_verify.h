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
 * \file verifier/shape_verify.h
 * \brief Operation shape verification
 */

#pragma once

#include "ir/verifier/verifier.h"
#include "ir/transform/visitor.h"
#include "ir/operation_base.h"
#include "ir/tile_graph.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pto {

class TileOpShapeVisitor : public IRVisitor {
public:
    std::vector<std::string> violations_;

    void VisitImplOp(OperationPtr &op) override;
// C++ requires that when a subclass redefines a virtual function, it must redefine all overloaded virtual functions
// with the same name but different parameters; otherwise, a warning will be issued, and in the current version, all
// warnings are treated as errors.
#define DEFOP(name, inherit, opcode, ...) void VisitImplOp(name##Ptr &op) override;
    #include "ir/operation.def"
    #include "ir/tile_graph.def"
#undef DEFOP

private:
    bool IsShapeCompatibleForBinaryOp(const TileValuePtr &inputTile, const TileValuePtr &outputTile);
    // Vector operation shape verification
    void CheckUnaryOpShape(UnaryOpPtr &unaryOp);
    void CheckBinaryOpShape(BinaryOpPtr &binaryOp);
    void CheckBinaryScalarMixOpShape(BinaryScalarMixOpPtr &binaryScalarMixOp);
    
    // Matmul operation shape verification
    void CheckMatmulLoadOpShape(MatmulLoadOpPtr &op);
    void CheckMatmulExtractOpShape(MatmulExtractOpPtr &op);
    void CheckMatmulMmadOpShape(MatmulMmadOpPtr &op);
    void CheckMatmulAccOpShape(MatmulAccOpPtr &op);
    void CheckMatmulStoreOpShape(MatmulStoreOpPtr &op);
    void CheckMatmulBiasOpShape(MatmulBiasOpPtr &op);
    void CheckMatmulQuantOpShape(MatmulQuantOpPtr &op);
    
    // Helper functions for matmul shape verification
    bool IsTransposeOpcode(Opcode opcode);
    bool IsTransposeA(Opcode opcode);
    bool IsTransposeB(Opcode opcode);
    std::vector<int64_t> GetTransposedShape(const std::vector<int64_t> &shape);
    
    std::string GetShapeStr(const std::vector<int64_t> &shape) const;
};

VerifyResult VerifyOpShape(ProgramModulePtr program);

} // namespace pto
