/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file function.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <set>
#include <tuple>
#include <map>
#include <unordered_set>
#include "tilefwk/tensor.h"
#include "symbolic_scalar.h"

// Helper macros to count arguments
#define RECORD_FUNC_VAR_NAME_COUNTER_HELPER(var, cnt) var##cnt
#define RECORD_FUNC_VAR_NAME_COUNTER(var, cnt) RECORD_FUNC_VAR_NAME_COUNTER_HELPER(var, cnt)
#define RECORD_FUNC_VAR_NAME(var) RECORD_FUNC_VAR_NAME_COUNTER(var, __COUNTER__)

// Main FUNCTION macro that dispatches based on argument count
#define FUNCTION(name, ...)                                                                       \
    if (auto RECORD_FUNC_VAR_NAME(recordFunc) = npu::tile_fwk::RecordFunc(name, ##__VA_ARGS__); false) { \
    } else

#define LOOP(name, funcType, index, ...) \
    for (auto &index : npu::tile_fwk::RecordLoopFunc(name, funcType, #index, ##__VA_ARGS__))

#define IF(cond) if (npu::tile_fwk::RecordIfBranch(cond, __FILE__, __LINE__))

#define ELSE else

#define UNROLL(X) if (npu::tile_fwk::RecordLoopFunc::MatchUnrollTimes(X))
#define UNROLL_DEFAULT if (npu::tile_fwk::RecordLoopFunc::MatchUnrollTimes(1))

namespace npu::tile_fwk {
class LoopRange;
class DynloopFunctionAttribute;

enum class FunctionType {
    EAGER,
    STATIC,
    DYNAMIC,
    DYNAMIC_LOOP,
    DYNAMIC_LOOP_PATH,
    INVALID,
    MAX,
};

const std::string FUNCTION_PREFIX = "TENSOR_";

class RecordFunc {
public:
    explicit RecordFunc(const std::string &name);
    RecordFunc(const std::string &name, const FunctionType type);
    RecordFunc(const std::string &name, const FunctionType type,
        const std::vector<std::reference_wrapper<Tensor>> &explicitOpArgs);
    RecordFunc(const std::string &name, const FunctionType type,
        const std::vector<std::reference_wrapper<const Tensor>> &startArgsInputTensorList,
        const std::vector<std::reference_wrapper<const Tensor>> &startArgsOutputTensorList,
        const std::vector<std::pair<std::reference_wrapper<const Tensor>, std::reference_wrapper<const Tensor>>> &inplaceArgs = {});

    ~RecordFunc();

private:
    Function *func_{nullptr};
    std::string funcName;
};

class RecordLoopFunc {
public:
    struct IteratorEnd {
        RecordLoopFunc &func;
        SymbolicScalar scalar;
    };

    class Iterator {
    public:
        Iterator(RecordLoopFunc &rlf, const SymbolicScalar &scalar)
            : rlf_(rlf), scalar_(scalar), originalScalar_(scalar) {}

        Iterator operator++();
        bool operator!=(const IteratorEnd &rhs);
        const SymbolicScalar &operator*() const { return scalar_; }
        SymbolicScalar &operator*() { return scalar_; }

    private:
        RecordLoopFunc &rlf_;
        SymbolicScalar scalar_;
        SymbolicScalar originalScalar_;
        int cur_{0};
    };

    explicit RecordLoopFunc(const std::string &name, FunctionType funcType, const std::string &iterName,
        const LoopRange &range, const std::set<int> &unrollList = {}, bool submitBeforeLoop = false);
    ~RecordLoopFunc();

    void BeginLoopFunction();
    void EndLoopFunction();

    std::shared_ptr<DynloopFunctionAttribute> GetLoopAttr();

    Iterator begin();
    IteratorEnd end();
    void IterationBegin();
    void IterationNext();
    bool IterationEnd();
    bool Condition(const SymbolicScalar &cond, const std::string &file, int line);

    const SymbolicScalar &LoopBegin() const;
    const SymbolicScalar &LoopStep() const;
    const SymbolicScalar &LoopEnd() const;

    bool VisitedUnroll(int unrollTimes) const { return visited_.count(unrollTimes) > 0; }
    void VisitUnroll(int unrollTimes);
    bool IsCustomUnrollTimes(int unrollTimes) const {
        return customUnrollTimes_.count(unrollTimes) > 0;
    } // check is user defined
    bool StillHaveUnrollTimes() const { return !unrollTimes_.empty(); }
    int CurUnrollTimes() const;
    void NextUnrollTimes();

    bool CustomUnrollTimesMatched() const { return customUnrollTimes_.count(CurUnrollTimes()) > 0; }
    static bool MatchUnrollTimes(int unrollTimes);

private:
    std::string GetLoopSuffix(int count) { return "_PATH" + std::to_string(count); }
    std::string name_;
    std::string iterName_;
    std::string curPathFuncName_;
    std::shared_ptr<LoopRange> loopRange_;
    bool submitBeforeLoop_;
    FunctionType funcType_{FunctionType::STATIC};
    Function *currentLoopFunc_{nullptr};
    bool dryRun_{false};
    bool hasManualUnroll_{false};
    int endCount_{0};

    std::vector<std::shared_ptr<LoopRange>> rangeOfEaceUnroll_;
    std::set<int, std::greater<>> unrollTimes_;
    std::unordered_set<int> visited_;
    std::unordered_set<int> customUnrollTimes_;

    void GenDefaultUnrollTimes(const std::set<int> &unrollList);
};

class RecordIfBranch {
public:
    explicit RecordIfBranch(const SymbolicScalar &cond, const std::string &file, int line)
        : cond_(cond), file_(file), line_(line) {}

    operator bool() const;

private:
    SymbolicScalar cond_{0};
    std::string file_;
    int line_{-1};
};
} // namespace npu::tile_fwk
