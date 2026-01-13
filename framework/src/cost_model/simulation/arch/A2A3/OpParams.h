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
 * \file OpParams.h
 * \brief
 */

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "tilefwk/data_type.h"
#include "interface/operation/opcode.h"

// Forward declarations to avoid heavy includes in header
namespace npu::tile_fwk {
class Operation;
}

using npu::tile_fwk::DataType;

namespace CostModel {

// Interface for Latency Calculation
class LatencyCalculator {
public:
    virtual ~LatencyCalculator() = default;
    virtual int Calculate(const npu::tile_fwk::Operation* op) const = 0;
};

using CalculatorPtr = std::shared_ptr<LatencyCalculator>;

// Registry Singleton
class OpRegistry {
public:
    static OpRegistry& GetInstance() {
        static OpRegistry instance;
        return instance;
    }
    
    void Register(npu::tile_fwk::Opcode opcode, CalculatorPtr calc) {
        registry_[opcode] = calc;
    }

    CalculatorPtr Get(npu::tile_fwk::Opcode opcode) const {
        auto opIt = registry_.find(opcode);
        if (opIt != registry_.end()) {
            return opIt->second;
        }
        return defaultCalc_;
    }

    void SetDefault(CalculatorPtr calc) {
        defaultCalc_ = calc;
    }

    CalculatorPtr GetDefault() const {
        return defaultCalc_;
    }

private:
    OpRegistry() = default;

    std::unordered_map<npu::tile_fwk::Opcode, CalculatorPtr> registry_;
    CalculatorPtr defaultCalc_;
};

// Helper for Auto-Registration
struct OpLatencyRegistrar {
    OpLatencyRegistrar(npu::tile_fwk::Opcode opcode, CalculatorPtr calc) {
        OpRegistry::GetInstance().Register(opcode, calc);
    }
};

#define _OP_PARAMS_CONCAT_IMPL(x, y) x##y
#define _OP_PARAMS_CONCAT(x, y) _OP_PARAMS_CONCAT_IMPL(x, y)

// Macro for Registration
// Usage: OP_LATENCY_REGISTER(Opcode::OP_ADD, LinearShape(0.5, 10.0));
#define OP_LATENCY_REGISTER(opcode, ...) \
    static const CostModel::OpLatencyRegistrar _OP_PARAMS_CONCAT(g_latency_registrar_, __LINE__)( \
        opcode, std::shared_ptr<CostModel::LatencyCalculator>(new __VA_ARGS__))

} // namespace CostModel
