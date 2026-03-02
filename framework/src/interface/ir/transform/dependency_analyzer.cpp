/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/transform/dependency_analyzer.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/stmt.h"
#include "ir/transform/base/visitor.h"

namespace pypto {
namespace ir {

DependencyGraph DependencyAnalyzer::Analyze(const FunctionPtr &func) {
  INTERNAL_CHECK(func) << "DependencyAnalyzer cannot analyze null function";

  // Step 1: Identify basic blocks from control flow
  std::vector<BasicBlock> blocks = IdentifyBasicBlocks(func->body_);
  LOG_INFO << "Identified " << blocks.size() << " basic blocks";

  // Step 2: Build dependency graph for each basic block
  std::vector<DependencyEdge> allDependencies;
  for (const auto &block : blocks) {
    auto blockDeps = AnalyzeBlockDependencies(block);
    allDependencies.insert(allDependencies.end(), blockDeps.begin(), blockDeps.end());
  }
  LOG_INFO << "Found " << allDependencies.size() << " dependency edges";

  // Step 3: Log dependency statistics
  int rawCount = 0, warCount = 0, wawCount = 0;
  for (const auto &edge : allDependencies) {
    switch (edge.type) {
      case DependencyEdge::RAW:
        rawCount++;
        break;
      case DependencyEdge::WAR:
        warCount++;
        break;
      case DependencyEdge::WAW:
        wawCount++;
        break;
      default:
        break;
    }
  }
  LOG_INFO << "Dependency types: RAW=" << rawCount << ", WAR=" << warCount << ", WAW=" << wawCount;

  return DependencyGraph(std::move(blocks), std::move(allDependencies));
}

std::vector<BasicBlock> DependencyAnalyzer::AnalyzeBasicBlocks(const FunctionPtr &func) {
  INTERNAL_CHECK(func) << "DependencyAnalyzer cannot analyze null function";
  return IdentifyBasicBlocks(func->body_);
}

std::vector<DependencyEdge> DependencyAnalyzer::AnalyzeDependencies(const FunctionPtr &func) {
  INTERNAL_CHECK(func) << "DependencyAnalyzer cannot analyze null function";

  // Create a single basic block containing all statements
  std::vector<BasicBlock> blocks = IdentifyBasicBlocks(func->body_);

  std::vector<DependencyEdge> allDependencies;
  for (const auto &block : blocks) {
    auto blockDeps = AnalyzeBlockDependencies(block);
    allDependencies.insert(allDependencies.end(), blockDeps.begin(), blockDeps.end());
  }

  return allDependencies;
}

std::vector<BasicBlock> DependencyAnalyzer::IdentifyBasicBlocks(const StmtPtr &stmt) {
  std::vector<BasicBlock> blocks;
  int nextId = 0;

  // Helper class to traverse statements and build basic blocks
  class BasicBlockBuilder {
   public:
    explicit BasicBlockBuilder(std::vector<BasicBlock>& blocks, int &nextId)
        : blocks_(blocks), nextId_(nextId) {}

    // Process a statement and create basic blocks
    int ProcessStmt(const StmtPtr &stmt, const std::vector<int>& predecessors) {
      if (!stmt) {
        return -1;
      }

      // Handle different statement types
      if (auto seq = As<SeqStmts>(stmt)) {
        return ProcessSeqStmts(seq, predecessors);
      } else if (auto ifStmt = As<IfStmt>(stmt)) {
        return ProcessIfStmt(ifStmt, predecessors);
      } else if (auto forStmt = As<ForStmt>(stmt)) {
        return ProcessForStmt(forStmt, predecessors);
      } else {
        // Single statement forms a basic block
        return CreateSingleStmtBlock(stmt, predecessors, false);
      }
    }

   private:
    int ProcessSeqStmts(const std::shared_ptr<const SeqStmts>& seq, const std::vector<int>& predecessors) {
      if (seq->stmts_.empty()) {
        return -1;
      }

      // Merge consecutive non-control-flow statements into a single basic block
      // This correctly implements the definition of a basic block: maximal sequence of
      // sequential statements with no branches
      std::vector<int> currentPreds = predecessors;
      int lastBlockId = -1;
      std::vector<StmtPtr> pendingStmts;  // Buffer for consecutive simple statements

      for (const auto &subStmt : seq->stmts_) {
        // Check if this is a control flow statement
        if (IsA<IfStmt>(subStmt) || IsA<ForStmt>(subStmt)) {
          // Flush pending simple statements as one basic block
          if (!pendingStmts.empty()) {
            lastBlockId = CreateMergedBlock(pendingStmts, currentPreds);
            currentPreds = {lastBlockId};
            pendingStmts.clear();
          }

          // Process control flow statement
          lastBlockId = ProcessStmt(subStmt, currentPreds);
          if (lastBlockId != -1) {
            currentPreds = {lastBlockId};
          }
        } else if (auto nestedSeq = As<SeqStmts>(subStmt)) {
          // Flush pending simple statements before processing nested SeqStmts
          if (!pendingStmts.empty()) {
            lastBlockId = CreateMergedBlock(pendingStmts, currentPreds);
            currentPreds = {lastBlockId};
            pendingStmts.clear();
          }

          // Process nested SeqStmts recursively
          lastBlockId = ProcessSeqStmts(nestedSeq, currentPreds);
          if (lastBlockId != -1) {
            currentPreds = {lastBlockId};
          }
        } else {
          // Simple statement (AssignStmt, EvalStmt, ReturnStmt): buffer it
          pendingStmts.push_back(subStmt);
        }
      }

      // Flush remaining simple statements as one basic block
      if (!pendingStmts.empty()) {
        lastBlockId = CreateMergedBlock(pendingStmts, currentPreds);
      }

      return lastBlockId;
    }

    int ProcessIfStmt(const std::shared_ptr<const IfStmt>& ifStmt, const std::vector<int>& predecessors) {
      // Create a block for the condition (if needed)
      // For simplicity, we'll process then/else bodies as separate blocks

      // Process then body
      int thenExit = ProcessStmt(ifStmt->thenBody_, predecessors);

      // Process else body (if exists)
      int elseExit = -1;
      if (ifStmt->elseBody_.has_value()) {
        elseExit = ProcessStmt(*ifStmt->elseBody_, predecessors);
      }

      // Create a virtual merge block (both branches merge here)
      // In a more complete implementation, we would track this merge point
      // For now, we'll return the then_exit (simplified)
      return thenExit;
    }

    int ProcessForStmt(const std::shared_ptr<const ForStmt>& forStmt, const std::vector<int>& predecessors) {
      // Create a basic block for the loop body
      BasicBlock loopBlock;
      loopBlock.id = nextId_++;
      loopBlock.isLoopBody = true;
      loopBlock.predecessors = predecessors;

      // Collect statements from loop body
      CollectStmtsInBlock(forStmt->body_, loopBlock.statements);

      // Loop body can jump back to itself (loop-carried dependency)
      loopBlock.successors.push_back(loopBlock.id);

      // Add a successor for loop exit (next block after loop)
      // We'll model this as continuing to the next block
      int exitId = nextId_;  // The next block would have this ID
      loopBlock.successors.push_back(exitId);

      blocks_.push_back(loopBlock);

      return loopBlock.id;
    }

    int CreateSingleStmtBlock(const StmtPtr &stmt, const std::vector<int>& predecessors, bool isLoop) {
      BasicBlock block;
      block.id = nextId_++;
      block.predecessors = predecessors;
      block.isLoopBody = isLoop;

      // Collect statements
      CollectStmtsInBlock(stmt, block.statements);

      blocks_.push_back(block);

      return block.id;
    }

    int CreateMergedBlock(const std::vector<StmtPtr>& stmts, const std::vector<int>& predecessors) {
      BasicBlock block;
      block.id = nextId_++;
      block.predecessors = predecessors;
      block.isLoopBody = false;

      // Add all statements directly to the block
      for (const auto &stmt : stmts) {
        CollectStmtsInBlock(stmt, block.statements);
      }

      blocks_.push_back(block);

      return block.id;
    }

    void CollectStmtsInBlock(const StmtPtr &stmt, std::vector<StmtPtr>& statements) {
      if (!stmt) {
        return;
      }

      // Recursively collect all non-control-flow statements
      if (auto seq = As<SeqStmts>(stmt)) {
        for (const auto &subStmt : seq->stmts_) {
          CollectStmtsInBlock(subStmt, statements);
        }
      } else if (IsA<IfStmt>(stmt) || IsA<ForStmt>(stmt)) {
        // Control flow statements are not added to the current block
        // They create their own blocks
      } else {
        // Regular statement (AssignStmt, EvalStmt, etc.)
        statements.push_back(stmt);
      }
    }

    std::vector<BasicBlock>& blocks_;
    int &nextId_;
  };

  // Build basic blocks starting from the root statement
  BasicBlockBuilder builder(blocks, nextId);
  builder.ProcessStmt(stmt, {});

  LOG_DEBUG << "Created " << blocks.size() << " basic blocks";

  return blocks;
}

std::vector<DependencyEdge> DependencyAnalyzer::AnalyzeBlockDependencies(const BasicBlock &block) {
  std::vector<DependencyEdge> dependencies;

  // Track last write and last read for each variable
  std::map<std::string, StmtPtr> lastWrite;               // var_name -> stmt that last wrote it
  std::map<std::string, std::vector<StmtPtr>> lastReads;  // var_name -> stmts that read it

  // Helper class to collect variable reads and writes
  class VarCollector : public IRVisitor {
   public:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;
    std::set<VarPtr> readVars;
    std::set<VarPtr> writeVars;

    void VisitExpr_(const VarPtr &var) override {
      readVars.insert(var);
      IRVisitor::VisitExpr_(var);
    }
  };

  // Helper: add RAW dependencies for all read variables
  auto AddRAWDependencies = [&](const std::set<VarPtr>& readVars, const StmtPtr &consumerStmt) {
    for (const auto &readVar : readVars) {
      std::string varName = readVar->name_;
      if (lastWrite.count(varName)) {
        DependencyEdge edge;
        edge.producer = lastWrite[varName];
        edge.consumer = consumerStmt;
        edge.variable = readVar;
        edge.type = DependencyEdge::RAW;
        edge.producerPipe = GetPipeTypeFromStmt(lastWrite[varName]);
        edge.consumerPipe = GetPipeTypeFromStmt(consumerStmt);
        dependencies.push_back(edge);
      }
    }
  };

  // Process each statement in the block
  for (const auto &stmt : block.statements) {
    VarCollector collector;

    // Identify the defined (written) variable and used (read) variables
    if (auto assign = As<AssignStmt>(stmt)) {
      // The assigned variable is written
      VarPtr defVar = assign->var_;

      // Collect variables read from the value expression
      collector.VisitExpr(assign->value_);

      // Check for RAW dependencies (Read-After-Write)
      AddRAWDependencies(collector.readVars, stmt);

      // Check for WAR dependencies (Write-After-Read)
      std::string defName = defVar->name_;
      if (lastReads.count(defName)) {
        for (const auto &readStmt : lastReads[defName]) {
          DependencyEdge edge;
          edge.producer = readStmt;
          edge.consumer = stmt;
          edge.variable = defVar;
          edge.type = DependencyEdge::WAR;
          edge.producerPipe = GetPipeTypeFromStmt(readStmt);
          edge.consumerPipe = GetPipeTypeFromStmt(stmt);
          dependencies.push_back(edge);
        }
      }

      // Check for WAW dependencies (Write-After-Write)
      if (lastWrite.count(defName)) {
        DependencyEdge edge;
        edge.producer = lastWrite[defName];
        edge.consumer = stmt;
        edge.variable = defVar;
        edge.type = DependencyEdge::WAW;
        edge.producerPipe = GetPipeTypeFromStmt(lastWrite[defName]);
        edge.consumerPipe = GetPipeTypeFromStmt(stmt);
        dependencies.push_back(edge);
      }

      // Update tracking: record this write
      lastWrite[defName] = stmt;

      // Record reads for this statement
      for (const auto &readVar : collector.readVars) {
        lastReads[readVar->name_].push_back(stmt);
      }

    } else if (auto evalStmt = As<EvalStmt>(stmt)) {
      // EvalStmt doesn't define variables, only reads
      collector.VisitExpr(evalStmt->expr_);

      // Check for RAW dependencies
      AddRAWDependencies(collector.readVars, stmt);

      // Record reads
      for (const auto &readVar : collector.readVars) {
        lastReads[readVar->name_].push_back(stmt);
      }
    }
  }

  LOG_DEBUG << "Found " << dependencies.size() << " dependencies in block " << block.id;

  return dependencies;
}

// Helper method to extract pipe type from a statement
std::string DependencyAnalyzer::GetPipeTypeFromStmt(const StmtPtr &stmt) {
  if (!stmt) {
    return "UNKNOWN";
  }

  // Try to extract Call expression from the statement
  if (auto assign = As<AssignStmt>(stmt)) {
    if (auto call = As<Call>(assign->value_)) {
      return GetPipeType(call);
    }
  } else if (auto evalStmt = As<EvalStmt>(stmt)) {
    if (auto call = As<Call>(evalStmt->expr_)) {
      return GetPipeType(call);
    }
  }

  return "UNKNOWN";
}

std::vector<DependencyEdge> DependencyAnalyzer::MergeDependencies(
    const std::vector<std::vector<DependencyEdge>>& pathDependencies) {
  std::vector<DependencyEdge> merged;

  // Take union of all dependencies from all paths
  // Use a set to track unique edges (by producer, consumer, variable, type)
  std::set<std::tuple<StmtPtr, StmtPtr, std::string, DependencyEdge::Type>> seen;

  for (const auto &path : pathDependencies) {
    for (const auto &edge : path) {
      auto key = std::make_tuple(edge.producer, edge.consumer, edge.variable->name_, edge.type);
      if (seen.find(key) == seen.end()) {
        seen.insert(key);
        merged.push_back(edge);
      }
    }
  }

  LOG_DEBUG << "Merged " << merged.size() << " unique dependencies from " << pathDependencies.size()
            << " paths";

  return merged;
}

std::string DependencyAnalyzer::GetPipeType(const CallPtr &callExpr) {
  if (!callExpr || !callExpr->op_) {
    return "UNKNOWN";
  }

  // Try to get pipe type from the Op
  auto pipeOpt = callExpr->op_->GetPipe();
  if (pipeOpt.has_value()) {
    // Convert PipeType enum to string
    PipeType pipe = *pipeOpt;
    switch (pipe) {
      case PipeType::M:
        return "CUBE";
      case PipeType::V:
        return "VECTOR";
      case PipeType::MTE1:
        return "MTE1";
      case PipeType::MTE2:
        return "MTE2";
      case PipeType::MTE3:
        return "MTE3";
      case PipeType::S:
        return "SCALAR";
      case PipeType::FIX:
        return "FIX";
      case PipeType::ALL:
        return "ALL";
      default:
        return "UNKNOWN";
    }
  }

  // Try to get pipe_type from kwargs
  if (callExpr->HasKwarg("pipe_type")) {
    try {
      std::string pipeStr = callExpr->GetKwarg<std::string>("pipe_type", "UNKNOWN");
      return pipeStr;
    } catch (...) {
      // If type conversion fails, return UNKNOWN
      return "UNKNOWN";
    }
  }

  return "UNKNOWN";
}

}  // namespace ir
}  // namespace pypto
