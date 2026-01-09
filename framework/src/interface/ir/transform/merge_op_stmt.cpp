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
 * \file merge_op_stmt.cpp
 * \brief Implementation of MergeOpStmt pass
 */

#include "ir/transform/merge_op_stmt.h"
#include "ir/statement.h"

namespace pto {

StatementPtr MergeOpStmt::VisitStmt_(CompoundStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;
  std::vector<StatementPtr> newStatements;

  // Visit and mutate all statements
  for (size_t i = 0; i < stmt->GetStatementsNum(); ++i) {
    StatementPtr stmtPtr = stmt->GetStatement(i);
    StatementPtr newStmt = VisitStmt(stmtPtr);
    newStatements.push_back(newStmt);
    if (newStmt != stmtPtr) {
      changed = true;
    }
  }

  // Merge consecutive OpStatements
  std::vector<StatementPtr> mergedStatements = MergeConsecutiveOpStmts(newStatements);
  
  // Check if merging occurred
  if (mergedStatements.size() != newStatements.size()) {
    changed = true;
  }

  // Create new CompoundStatement if any child was modified
  if (changed) {
    auto newStmt = std::make_shared<CompoundStatement>();
    for (auto& s : mergedStatements) {
      newStmt->AddStatement(s);
    }
    // Copy parent and envTable if needed
    if (auto parent = stmt->GetParent().lock()) {
      newStmt->SetParent(parent);
    }
    newStmt->GetEnvTable() = stmt->GetEnvTable();
    // Copy attributes
    newStmt->Attributes() = stmt->Attributes();
    return newStmt;
  }

  return stmt;
}

StatementPtr MergeOpStmt::VisitStmt_(OpStatementPtr& stmt) {
  // OpStatement itself doesn't need modification, just return it
  return stmt;
}

StatementPtr MergeOpStmt::VisitStmt_(ForStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;

  // Visit and mutate loop body
  CompoundStatementPtr compound = stmt->GetCompound();
  if (compound) {
    StatementPtr compoundStmt = std::static_pointer_cast<Statement>(compound);
    StatementPtr newCompoundStmt = VisitStmt(compoundStmt);
    CompoundStatementPtr newCompound = std::dynamic_pointer_cast<CompoundStatement>(newCompoundStmt);
    if (newCompound != compound) {
      changed = true;
      compound = newCompound;
    }
  }

  // Create new ForStatement if body was modified
  if (changed) {
    auto newForStmt = std::make_shared<ForStatement>(
      stmt->GetIterationVar(),
      stmt->GetStart(),
      stmt->GetEnd(),
      stmt->GetStep()
    );
    // Copy iter args
    for (const auto& iterArg : stmt->IterArgs()) {
      newForStmt->AddIterArg(iterArg.initValue);
    }
    // Set compound
    if (compound) {
      for (size_t i = 0; i < compound->GetStatementsNum(); ++i) {
        newForStmt->GetCompound()->AddStatement(compound->GetStatement(i));
      }
    }
    // Copy attributes
    newForStmt->Attributes() = stmt->Attributes();
    return newForStmt;
  }

  return stmt;
}

StatementPtr MergeOpStmt::VisitStmt_(IfStatementPtr& stmt) {
  if (!stmt) return stmt;

  bool changed = false;

  // Visit and mutate then branch
  CompoundStatementPtr thenCompound = stmt->GetThenCompound();
  if (thenCompound) {
    StatementPtr thenStmt = std::static_pointer_cast<Statement>(thenCompound);
    StatementPtr newThenStmt = VisitStmt(thenStmt);
    CompoundStatementPtr newThenCompound = std::dynamic_pointer_cast<CompoundStatement>(newThenStmt);
    if (newThenCompound != thenCompound) {
      changed = true;
      thenCompound = newThenCompound;
    }
  }

  // Visit and mutate else branch
  CompoundStatementPtr elseCompound = stmt->GetElseCompound();
  if (elseCompound) {
    StatementPtr elseStmt = std::static_pointer_cast<Statement>(elseCompound);
    StatementPtr newElseStmt = VisitStmt(elseStmt);
    CompoundStatementPtr newElseCompound = std::dynamic_pointer_cast<CompoundStatement>(newElseStmt);
    if (newElseCompound != elseCompound) {
      changed = true;
      elseCompound = newElseCompound;
    }
  }

  // Create new IfStatement if branches were modified
  if (changed) {
    auto newIfStmt = std::make_shared<IfStatement>(stmt->GetCondition());
    // Set then compound
    if (thenCompound) {
      for (size_t i = 0; i < thenCompound->GetStatementsNum(); ++i) {
        newIfStmt->GetThenCompound()->AddStatement(thenCompound->GetStatement(i));
      }
    }
    // Set else compound
    if (elseCompound) {
      for (size_t i = 0; i < elseCompound->GetStatementsNum(); ++i) {
        newIfStmt->GetElseCompound()->AddStatement(elseCompound->GetStatement(i));
      }
    }
    // Copy attributes
    newIfStmt->Attributes() = stmt->Attributes();
    return newIfStmt;
  }

  return stmt;
}

StatementPtr MergeOpStmt::VisitStmt_(YieldStatementPtr& stmt) {
  // YieldStatement doesn't contain nested statements, just return it
  return stmt;
}

StatementPtr MergeOpStmt::VisitStmt_(ReturnStatementPtr& stmt) {
  // ReturnStatement doesn't contain nested statements, just return it
  return stmt;
}

StatementPtr MergeOpStmt::VisitStmt_(StatementPtr& stmt) {
  // Default implementation: just return the statement
  return stmt;
}

std::vector<StatementPtr> MergeOpStmt::MergeConsecutiveOpStmts(const std::vector<StatementPtr>& statements) {
  std::vector<StatementPtr> mergedStatements;
  OpStatementPtr currentOpStmt = nullptr;

  for (size_t idx = 0; idx < statements.size(); ++idx) {
    auto& s = statements[idx];
    
    if (auto opStmt = std::dynamic_pointer_cast<OpStatement>(s)) {
      if (currentOpStmt) {
        // Merge operations from this OpStatement into currentOpStmt
        for (auto& op : opStmt->Operations()) {
          currentOpStmt->Operations().push_back(op);
        }
        // Don't add this OpStatement to mergedStatements, as it's been merged
      } else {
        // Start a new OpStatement
        currentOpStmt = opStmt;
        mergedStatements.push_back(s);
      }
    } else {
      // If we have a current OpStatement, finalize it before adding non-op statement
      if (currentOpStmt) {
        currentOpStmt = nullptr;
      }
      mergedStatements.push_back(s);
    }
  }

  // If we still have a current OpStatement at the end, it's already added
  return mergedStatements;
}

} // namespace pto

