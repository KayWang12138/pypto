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
 * \file backend_expr_generator.h
 * \brief Expression batch generator for splitting large control flow functions
 */

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {

// Maximum expressions per batch/file
constexpr size_t EXPRS_PER_BATCH = 60;

// Expression batch information
struct ExprBatchInfo {
    int devRootKey;
    size_t batchIndex;
    size_t startExprIndex;
    size_t endExprIndex;
    size_t totalExprs;
    std::string fileName;
    std::string functionName;
    std::vector<std::string> dependArgs;
};

// Generator for expression batches
class ExprBatchGenerator {
public:
    ExprBatchGenerator(const std::string& outputDir, int devRootKey, size_t totalExprs)
        : outputDir_(outputDir), devRootKey_(devRootKey), totalExprs_(totalExprs) {
        CalculateBatches();
    }

    // Get all batch information
    const std::vector<ExprBatchInfo>& GetBatches() const {
        return batches_;
    }

    // Generate header file for all batches
    void GenerateHeaderFile() const {
        std::string headerPath = outputDir_ + "/backend_expr_" + std::to_string(devRootKey_) + ".h";
        std::ofstream header(headerPath);
        if (!header.is_open()) {
            return;
        }

        header << "#pragma once\n"
               << "#include <cstdint>\n\n"
               << "namespace npu::tile_fwk {\n\n";

        for (const auto& batch : batches_) {
            header << "void " << batch.functionName << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs, uint64_t *exprList);\n";
        }

        header << "\n} // namespace npu::tile_fwk\n";
        header.close();
    }

    // Generate a specific batch file with expressions
    template<typename ExpressionSet, typename BuildExpressionFunc>
    void GenerateBatchFile(const ExprBatchInfo& batch, 
                          const ExpressionSet& expressions,
                          const BuildExpressionFunc& buildExpr) {
        std::string filePath = outputDir_ + "/" + batch.fileName;
        std::ofstream out(filePath);
        if (!out.is_open()) {
            return;
        }

        // Write file header
        out << "/**\n"
            << " * Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
            << " * Auto-generated expression batch file\n"
            << " * Batch " << batch.batchIndex << " for devRootKey " << devRootKey_ << "\n"
            << " * Expressions " << batch.startExprIndex << " to " << batch.endExprIndex << "\n"
            << " */\n\n"
            << "#include \"tilefwk/aikernel_data.h\"\n"
            << "#include \"tilefwk/aicpu_runtime.h\"\n"
            << "#include <cstdint>\n\n"
            << "namespace npu::tile_fwk {\n\n"
            << "void " << batch.functionName << "(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs, uint64_t *exprList) {\n";

        // Write expressions for this batch
        size_t exprCount = 0;
        for (const auto& expr : expressions) {
            size_t currentIdx = expressions.GetIndex(expr);
            if (currentIdx >= batch.startExprIndex && currentIdx < batch.endExprIndex) {
                auto exprStr = buildExpr(expr, &batch.dependArgs);
                out << "    RUNTIME_SetExpr(exprList, " << currentIdx << ", " << exprStr << ");\n";
                exprCount++;
            }
            if (currentIdx >= batch.endExprIndex) break;
        }

        out << "}\n\n"
            << "} // namespace npu::tile_fwk\n";
        out.close();
    }

private:
    void CalculateBatches() {
        size_t numBatches = (totalExprs_ + EXPRS_PER_BATCH - 1) / EXPRS_PER_BATCH;
        
        for (size_t i = 0; i < numBatches; ++i) {
            ExprBatchInfo batch;
            batch.devRootKey = devRootKey_;
            batch.batchIndex = i;
            batch.startExprIndex = i * EXPRS_PER_BATCH;
            batch.endExprIndex = std::min(batch.startExprIndex + EXPRS_PER_BATCH, totalExprs_);
            batch.totalExprs = totalExprs_;
            batch.fileName = "backend_expr_" + std::to_string(devRootKey_) + "_" + std::to_string(i) + ".cpp";
            batch.functionName = "SetExprBatch_" + std::to_string(devRootKey_) + "_" + std::to_string(i);
            batches_.push_back(batch);
        }
    }

    std::string outputDir_;
    int devRootKey_;
    size_t totalExprs_;
    std::vector<ExprBatchInfo> batches_;
};

} // namespace npu::tile_fwk
