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
 * \file hash_computer.cpp
 * \brief Implementation of IR hash computer
 */

#include "ir/hash_computer.h"

#include <sstream>

namespace pto {

uint64_t IRHashComputer::VisitOperation(Operation& op) {
    std::stringstream ss;
    
    // Hash opcode
    ss << "opcode=" << static_cast<int>(op.GetOpcode()) << " ";
    
    // Hash input operands
    ss << "inputOperands=" << " ";
    for (size_t i = 0; i < op.GetNumInputOperand(); ++i) {
        auto operand = op.GetInputOperand(i);
        if (operand) {
            ss << HashValue(operand) << " ";
        }
    }
    
    // Hash output operands
    ss << "outputOperands=" << " ";
    for (size_t i = 0; i < op.GetNumOutputOperand(); ++i) {
        auto operand = op.GetOutputOperand(i);
        if (operand) {
            ss << HashValue(operand) << " ";
        }
    }
    
    // Compute hash from string
    return stringHasher_(ss.str());
}

uint64_t IRHashComputer::VisitCompoundStatement(CompoundStatement& cs) {
    std::vector<uint64_t> hashes;
    
    // Hash all statements in the compound
    for (size_t i = 0; i < cs.GetStatementsNum(); ++i) {
        auto stmt = cs.GetStatement(i);
        if (stmt) {
            hashes.push_back(VisitStatement(*stmt));
        }
    }
    
    // Combine all statement hashes
    auto resHash = CombineHashes(hashes);
    cs.SetHash(resHash);
    return resHash;
}

uint64_t IRHashComputer::VisitOpStatement(OpStatement& os) {
    std::vector<uint64_t> hashes;
    hashes.push_back(stringHasher_("op statement"));

    // Hash all operations in the op statement
    for (const auto& op : os.Operations()) {
        if (op) {
            hashes.push_back(VisitOperation(*op));
        }
    }
    
    // Combine all operation hashes
    auto resHash = CombineHashes(hashes);
    os.SetHash(resHash);
    return resHash;
}

uint64_t IRHashComputer::VisitForStatement(ForStatement& fs) {
    std::vector<uint64_t> hashes;
    hashes.push_back(stringHasher_("for statement"));

    // Hash iteration variable
    if (fs.GetIterationVar()) {
        hashes.push_back(HashValue(fs.GetIterationVar()));
    }
    
    // Hash loop range
    if (fs.GetRange()) {
        if (fs.GetStart()) {
            hashes.push_back(HashValue(fs.GetStart()));
        }
        if (fs.GetEnd()) {
            hashes.push_back(HashValue(fs.GetEnd()));
        }
        if (fs.GetStep()) {
            hashes.push_back(HashValue(fs.GetStep()));
        }
    }
    
    // Hash iter args, initValue/value/result type is the same
    for (const auto& iterArg : fs.IterArgs()) {
        if (iterArg.initValue) {
            hashes.push_back(HashValue(iterArg.initValue));
        }
    }
    
    // Hash loop body (compound statement)
    if (fs.GetCompound()) {
        hashes.push_back(VisitCompoundStatement(*fs.GetCompound()));
    }
    
    auto resHash = CombineHashes(hashes);
    fs.SetHash(resHash);
    return resHash;
}

uint64_t IRHashComputer::VisitIfStatement(IfStatement& is) {
    std::vector<uint64_t> hashes;
    hashes.push_back(stringHasher_("if statement"));
    
    // Hash condition
    if (is.GetCondition()) {
        hashes.push_back(HashValue(is.GetCondition()));
    }
    
    hashes.push_back(stringHasher_("then"));
    // Hash then branch
    if (is.GetThenCompound()) {
        hashes.push_back(VisitCompoundStatement(*is.GetThenCompound()));
    }
    
    hashes.push_back(stringHasher_("else"));
    // Hash else branch
    if (is.GetElseCompound()) {
        hashes.push_back(VisitCompoundStatement(*is.GetElseCompound()));
    }
    
    // Hash results
    for (const auto& result : is.Results()) {
        if (result) {
            hashes.push_back(HashValue(result));
        }
    }
    
    auto resHash = CombineHashes(hashes);
    is.SetHash(resHash);
    return resHash;
}

uint64_t IRHashComputer::VisitYieldStatement(YieldStatement& ys) {
    std::vector<uint64_t> hashes;
    hashes.push_back(stringHasher_("yield statement"));

    // Hash yielded values
    for (const auto& value : ys.Values()) {
        if (value) {
            hashes.push_back(HashValue(value));
        }
    }
    
    auto resHash = CombineHashes(hashes);
    ys.SetHash(resHash);
    return resHash;
}

uint64_t IRHashComputer::VisitReturnStatement(ReturnStatement& rs) {
    std::vector<uint64_t> hashes;
    hashes.push_back(stringHasher_("return statement"));

    // Hash returned values
    for (const auto& value : rs.Values()) {
        if (value) {
            hashes.push_back(HashValue(value));
        }
    }
    
    auto resHash = CombineHashes(hashes);
    rs.SetHash(resHash);
    return resHash;
}

uint64_t IRHashComputer::VisitFunction(Function& func) {
    std::vector<uint64_t> hashes;

    // Hash function name
    hashes.push_back(stringHasher_("function: " + func.GetSSAName()));

    // Hash function signature
    hashes.push_back(HashFunctionSignature(func.GetSignature()));
    
    // Hash function kind
    hashes.push_back(intHasher_(static_cast<int>(func.GetKind())));
    
    // Hash function body (compound statement)
    if (func.GetCompound()) {
        hashes.push_back(VisitCompoundStatement(*func.GetCompound()));
    }
    
    return CombineHashes(hashes);
}

uint64_t IRHashComputer::CombineHashes(const std::vector<uint64_t>& hashes) {
    if (hashes.empty()) {
        return 0;
    }
    
    // Use a simple combination strategy: hash of concatenated hash values
    std::stringstream ss;
    for (uint64_t hash : hashes) {
        ss << hash << " ";
    }
    
    std::hash<std::string> hasher;
    return hasher(ss.str());
}

uint64_t IRHashComputer::HashValue(const ValuePtr& value) {
    if (!value) {
        return 0;
    }
    
    std::stringstream ss;
    
    // Hash value kind
    ss << "valueKind=" << static_cast<int>(value->GetValueKind()) << " ";
    
    // Hash data type
    ss << "dataType=" << static_cast<int>(value->GetDataType()) << " ";
    
    // For scalar values, hash immediate value if available
    if (value->GetValueKind() == ValueKind::Scalar) {
        auto scalar = std::dynamic_pointer_cast<ScalarValue>(value);
        if (scalar) {
            ss << "scalarValueKind=" << static_cast<int>(scalar->GetScalarValueKind()) << " ";
            if (scalar->HasImmediateValue()) {
                // Hash immediate value
                ss << "immediateValue=" << std::visit([](const auto& val) -> std::string {
                    return std::to_string(val);
                }, scalar->GetImmediateValue()) << " ";
            }
        }
    }
    // For tensor/tile values, hash shape information if available
    else if (value->GetValueKind() == ValueKind::Tensor) {
        auto tensor = std::dynamic_pointer_cast<TensorValue>(value);
        if (tensor) {
            ss << "tensorShape=" << " ";
            for (const auto& shape : tensor->GetShape()) {
                shape->PrintSSAName(ss);
                ss << " ";
            }
        }
    }
    else if (value->GetValueKind() == ValueKind::Tile) {
        auto tile = std::dynamic_pointer_cast<TileValue>(value);
        if (tile) {
            ss << "tileShape=" << " ";
            for (const auto& shape : tile->GetShape()) {
                ss << shape << " ";
            }
        }
    }

    return stringHasher_(ss.str());
}

uint64_t IRHashComputer::HashFunctionSignature(const FunctionSignature& signature) {
    std::vector<uint64_t> hashes;
    
    // Hash arguments
    for (const auto& arg : signature.arguments) {
        if (arg) {
            hashes.push_back(HashValue(arg));
        }
    }
    
    // Hash results
    for (const auto& result : signature.results) {
        if (result) {
            hashes.push_back(HashValue(result));
        }
    }
    
    return CombineHashes(hashes);
}

} // namespace pto
