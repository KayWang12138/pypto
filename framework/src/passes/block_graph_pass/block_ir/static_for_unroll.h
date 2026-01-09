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
 * \file static_for_unroll.h
 * \brief Static for loop unrolling transformation for PTO IR
 */

#ifndef PASS_STATIC_FOR_UNROLL_H
#define PASS_STATIC_FOR_UNROLL_H

#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "interface/utils/common.h"
#include <unordered_map>

namespace npu {
namespace tile_fwk {

/**
 * \brief Static for loop unrolling transformation
 * 
 * This transformation operates on PTO IR ProgramModule.
 * For each ControlFlow function in the module:
 * - Validates that all ForStatement nodes have static (immediate) iteration counts
 * - Validates that there are no If statements (only static for loops allowed)
 * - Unrolls the loop body for static for loops
 * 
 * The transformation processes functions in-place, replacing ForStatement nodes
 * with unrolled sequences of statements.
 */
class StaticForUnrollTransform {
public:
    StaticForUnrollTransform() = default;
    ~StaticForUnrollTransform() = default;

    /**
     * \brief Apply the transformation to a ProgramModule
     * \param module The ProgramModule to transform
     * \return SUCCESS if transformation succeeds, FAILED otherwise
     */
    Status Apply(const std::shared_ptr<pto::ProgramModule>& module);

private:
    /**
     * \brief Process a single PTO function
     * \param func The PTO IR function to process
     * \return SUCCESS if processing succeeds, FAILED otherwise
     */
    Status ProcessFunction(const std::shared_ptr<pto::Function>& func);
    
    /**
     * \brief Validate and unroll loops in a compound statement
     * \param compound The compound statement to process
     * \return SUCCESS if validation and unrolling succeed, FAILED otherwise
     */
    Status ProcessCompound(const pto::CompoundStatementPtr& compound);
    
    /**
     * \brief Check if a scalar value is a static immediate
     * \param value The scalar value to check
     * \param result Output parameter for the immediate value
     * \return true if the value is a static immediate, false otherwise
     */
    bool IsStaticImmediate(const pto::ScalarValuePtr& value, int64_t& result);
    
    /**
     * \brief Unroll a static for loop
     * \param forStmt The for statement to unroll
     * \param parentCompound The parent compound statement containing the for loop
     * \param stmtIndex The index of the for statement in parent compound
     * \return SUCCESS if unrolling succeeds, FAILED otherwise
     */
    Status UnrollForLoop(const pto::ForStatementPtr& forStmt, 
                        const pto::CompoundStatementPtr& parentCompound,
                        size_t stmtIndex);

    /**
     * \brief Clone a statement, replacing iteration variable with a constant
     * \param stmt The statement to clone
     * \param iterVar The iteration variable to replace
     * \param iterValue The constant value to replace with
     * \param iterIndex The current iteration index (used for generating unique names)
     * \return The cloned statement
     */
    pto::StatementPtr CloneStatement(const pto::StatementPtr& stmt,
                                     const pto::ScalarValuePtr& iterVar,
                                     const pto::ScalarValuePtr& iterValue,
                                     int64_t iterIndex);

    /**
     * \brief Clone a value, replacing iteration variable with a constant
     * \param value The value to clone
     * \param valueMap Map from original values to cloned values (for tracking replacements)
     * \param iterIndex The current iteration index (used for generating unique names)
     * \return The cloned value
     */
    pto::ValuePtr CloneValue(const pto::ValuePtr& value,
                            std::unordered_map<pto::ValuePtr, pto::ValuePtr>& valueMap,
                            int64_t iterIndex);
};

} // namespace tile_fwk
} // namespace npu

#endif // PASS_STATIC_FOR_UNROLL_H
