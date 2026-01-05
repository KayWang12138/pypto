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
 * \file vjp_registry.h
 * \brief VJP (Vector-Jacobian Product) registry for autograd.
 */

#ifndef PASS_VJP_REGISTRY_H_
#define PASS_VJP_REGISTRY_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

// Attribute key for marking tensors that require gradients
constexpr const char* ATTR_REQUIRES_GRAD = "requires_grad";
// Attribute key for marking loss tensor
constexpr const char* ATTR_IS_LOSS = "is_loss";

/**
 * @brief Context for VJP computation.
 *
 * Provides access to the operation node, output gradients, saved tensors,
 * and attributes needed to compute input gradients.
 */
struct VJPContext {
    Operation* node;
    std::unordered_map<std::string, LogicalTensorPtr> out_grads;
    std::unordered_map<std::string, LogicalTensorPtr> saved_tensors;
    Function* function;

    /**
     * @brief Get the output gradient for a named output.
     * @param name Output name (e.g., "output", "output0")
     * @return Pointer to gradient tensor, or nullptr if not available
     */
    LogicalTensorPtr GetOutputGrad(const std::string& name) const;

    /**
     * @brief Get a saved tensor by name.
     * @param name Tensor name (e.g., "input", "other")
     * @return Pointer to saved tensor, or nullptr if not available
     */
    LogicalTensorPtr GetSaved(const std::string& name) const;

    /**
     * @brief Get the shape of an input tensor by name.
     * @param name Input name
     * @return Shape vector, empty if input not found
     */
    std::vector<int64_t> GetInputShape(const std::string& name) const;

    /**
     * @brief Get an attribute value with default.
     * @tparam T Attribute type
     * @param name Attribute name
     * @param defaultVal Default value if not found
     * @return Attribute value or default
     */
    template<typename T>
    T GetAttr(const std::string& name, T defaultVal) const {
        if (node == nullptr) {
            return defaultVal;
        }
        T value;
        if (node->GetAttr(OP_ATTR_PREFIX + name, value)) {
            return value;
        }
        return defaultVal;
    }

    /**
     * @brief Get scalar attribute value with default.
     *
     * Scalar attributes are commonly stored under OpAttributeKey::scalar
     * as an Element. This helper also falls back to OP_ATTR_PREFIX + "scalar".
     */
    double GetScalarAttr(double defaultVal) const;

    /**
     * @brief Check if an attribute exists.
     */
    bool HasAttr(const std::string& name) const {
        if (node == nullptr) {
            return false;
        }
        return node->HasAttr(OP_ATTR_PREFIX + name);
    }

    /**
     * @brief Unbroadcast gradient to match target shape.
     *
     * When an input was broadcast during forward pass, the gradient needs
     * to be summed along the broadcast dimensions.
     *
     * @param grad The gradient tensor
     * @param targetShape The original input shape
     * @return Gradient tensor with correct shape
     */
    LogicalTensorPtr Unbroadcast(LogicalTensorPtr grad, const std::vector<int64_t>& targetShape) const;
};

/**
 * @brief Result of a VJP computation.
 *
 * Maps input names to their gradient tensors.
 */
using VJPResult = std::unordered_map<std::string, LogicalTensorPtr>;

/**
 * @brief VJP function signature.
 *
 * Takes a context and returns gradients for each input.
 */
using VJPFunc = std::function<VJPResult(VJPContext&)>;

/**
 * @brief Registry for VJP rules.
 *
 * Singleton that maintains the mapping from Opcode to VJP functions.
 */
class VJPRegistry {
public:
    /**
     * @brief Get the singleton instance.
     */
    static VJPRegistry& Instance();

    /**
     * @brief Register a VJP rule for an opcode.
     * @param op Opcode to register
     * @param vjp VJP function
     */
    void Register(Opcode op, VJPFunc vjp);

    /**
     * @brief Check if a VJP rule exists for an opcode.
     */
    bool HasVJP(Opcode op) const;

    /**
     * @brief Get the VJP rule for an opcode.
     * @throws std::runtime_error if no rule exists
     */
    VJPFunc GetVJP(Opcode op) const;

    /**
     * @brief Get all registered opcodes.
     */
    std::vector<Opcode> GetRegisteredOpcodes() const;

private:
    VJPRegistry() = default;
    VJPRegistry(const VJPRegistry&) = delete;
    VJPRegistry& operator=(const VJPRegistry&) = delete;

    std::unordered_map<Opcode, VJPFunc> registry_;
};

/**
 * @brief Helper class for static VJP registration.
 */
class VJPRegistrar {
public:
    VJPRegistrar(Opcode op, VJPFunc vjp) {
        VJPRegistry::Instance().Register(op, std::move(vjp));
    }
};

/**
 * @brief Macro for registering VJP rules.
 *
 * Usage:
 *   REG_VJP(OP_ADD, vjp_add);
 */
#define REG_VJP(opcode, func) \
    static VJPRegistrar __vjp_registrar_##opcode(Opcode::opcode, func)

} // namespace npu::tile_fwk

#endif // PASS_VJP_REGISTRY_H_
