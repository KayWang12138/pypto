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
 * \file operation_interpreter.h
 * \brief
 */
/*for flow verify tool */

#pragma once

#include "interface/interpreter/thread_pool.h"
#include "interface/operation/attribute.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "calc.h"
#include "calculator.h"
#include "tilefwk/data_type.h"
#include "calc.h"
namespace npu::tile_fwk {

constexpr int DATATYPE_EIGHT = 8;

struct FunctionFrame;
class OperationInterpreter;
struct ExecuteOperationContext {
    FunctionFrame *frame;
    OperationInterpreter *opInter;
    Operation *op;
    const std::vector<LogicalTensorDataPtr> *ioperandDataViewList;
    std::vector<LogicalTensorDataPtr> *ooperandDataViewList;
    std::vector<LogicalTensorDataPtr> *ooperandInplaceDataViewList;

    std::string Dump() const;
};

using Funcs = std::function<void(ExecuteOperationContext*)>;

class OperationInterpreter {
public:
    OperationInterpreter(int threadCount) : evaluateSymbol(std::make_shared<EvaluateSymbol>()), pool(threadCount) {}

    std::shared_ptr<EvaluateSymbol> evaluateSymbol;

    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar &ss) {
        return evaluateSymbol->EvaluateSymbolicScalar(ss);
    }
    std::vector<int64_t> EvaluateOffset(const std::vector<int64_t> &offset, const std::vector<SymbolicScalar> &dynOffset) {
        return evaluateSymbol->EvaluateOffset(offset, dynOffset);
    }
    std::vector<int64_t> EvaluateOpImmediate(FunctionFrame *frame, const std::vector<OpImmediate> &opImmList);

    std::vector<int64_t> EvaluateValidShape(const std::vector<SymbolicScalar> &dynValidShape) {
        return evaluateSymbol->EvaluateValidShape(dynValidShape);
    }

    void ExecuteOperation(ExecuteOperationContext *ctx);

    util::ThreadPool &GetPool() { return pool; }

    util::ThreadPool* GetPoolPtr() { return &pool; }

    int GetThreadCount() const { return pool.GetThreadCount(); }

    // 注册默认函数
    static void RegisterFunc(const Opcode opcode, Funcs func) {
        operationInterpreterFuncs_()[opcode] = std::move(func);
    }
private:
    // 调用场景对应的函数 CallOperationInterpreterFunc
    void CallOperationInterpreterFunc(ExecuteOperationContext *ctx) {
        const Opcode opcode = ctx->op->GetOpcode();
        auto it = operationInterpreterFuncs_().find(opcode);
        if (it != operationInterpreterFuncs_().end()) {
            it->second(ctx);
        } else {
            ASSERT(0) << "opcode [" << ctx->op->GetOpcodeStr() << "]'s torch interface implementation is not registered";
        }
    }

    std::vector<LogicalTensorDataPtr> GetValidDataView(const std::vector<LogicalTensorDataPtr> &dataViewList) const {
        std::vector<LogicalTensorDataPtr> result;
        for (auto &dataView : dataViewList) {
            auto &validShape = dataView->GetValidShape();
            if (validShape == dataView->GetShape()) {
                result.emplace_back(dataView);
            } else {
                result.emplace_back(dataView->View(validShape, std::vector<int64_t>(validShape.size(), 0)));
            }
        }
        return result;
    }

    OperationInterpreter() = default;

    static std::unordered_map<Opcode, Funcs>& operationInterpreterFuncs_() {
        static std::unordered_map<Opcode, Funcs> instance;
        return instance;
    }

    util::ThreadPool pool{64};
};

#define REGISTER_CALC_OP(OpCoreStr, OpType, FuncName) \
class OpCoreStr##ClacOpRegister { \
public: \
    OpCoreStr##ClacOpRegister() { \
        OperationInterpreter::RegisterFunc(OpType, FuncName); \
    } \
}; \
static OpCoreStr##ClacOpRegister OpCoreStr##_calcop_register

#undef CASE_DATA_TYPE_DIS
#undef CASE_DATA_TYPE
} // namespace npu::tile_fwk
