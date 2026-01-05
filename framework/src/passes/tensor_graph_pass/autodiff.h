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
 * \file autodiff.h
 * \brief Automatic differentiation pass for computing gradients.
 */

#ifndef PASS_AUTODIFF_H_
#define PASS_AUTODIFF_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
#include "vjp_registry.h"

namespace npu::tile_fwk {

/**
 * @brief Pass for automatic differentiation via reverse-mode autodiff (VJP).
 *
 * This pass computes gradients of a loss tensor with respect to all tensors
 * marked with requires_grad=true. It works by:
 * 1. Finding the loss tensor (marked with is_loss=true)
 * 2. Traversing operations in reverse topological order
 * 3. Applying VJP rules to propagate gradients backwards
 * 4. Generating new operations for gradient computation
 *
 * The resulting gradient tensors are stored as attributes on the original
 * tensors and can be accessed after the pass completes.
 */
class AutodiffPass : public Pass {
public:
    AutodiffPass() : Pass("AutodiffPass") {}
    ~AutodiffPass() override = default;

    Status RunOnFunction(Function& function) override;
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;

    // Static gradient tensor registry (persists across TensorMap resets)
    static void RegisterGradientTensor(int magic, LogicalTensorPtr tensor);
    static LogicalTensorPtr GetGradientTensor(int magic);
    static void ClearGradientRegistry();

private:
    /**
     * @brief Find the loss tensor in the function.
     * @return Pointer to loss tensor, or nullptr if not found
     */
    LogicalTensorPtr FindLossTensor(Function& function);

    /**
     * @brief Collect all tensors that require gradients.
     */
    std::vector<LogicalTensorPtr> CollectRequiresGradTensors(Function& function);

    /**
     * @brief Initialize the gradient of the loss tensor.
     */
    Status InitializeLossGrad(Function& function, LogicalTensorPtr loss);

    /**
     * @brief Get operations in reverse topological order.
     */
    std::vector<Operation*> GetReverseTopoOrder(Function& function);

    /**
     * @brief Process a single operation during backward pass.
     */
    Status ProcessOperation(Function& function, Operation* op);

    /**
     * @brief Accumulate gradient for a tensor.
     *
     * If the tensor already has a gradient, adds the new gradient to it.
     * Otherwise, sets the new gradient directly.
     */
    Status AccumulateGrad(Function& function, LogicalTensorPtr tensor, LogicalTensorPtr grad);

    /**
     * @brief Reduce gradient to match original input shape (for broadcasting).
     *
     * When an input was broadcast during forward pass, the gradient needs
     * to be summed along the broadcast dimensions.
     */
    LogicalTensorPtr Unbroadcast(Function& function, LogicalTensorPtr grad,
                                  const std::vector<int64_t>& targetShape);

    /**
     * @brief Build VJP context for an operation.
     */
    VJPContext BuildVJPContext(Function& function, Operation* op);

    /**
     * @brief Check if a tensor's dtype is differentiable (floating point).
     */
    bool IsDifferentiableDtype(DataType dtype) const;

    /**
     * @brief Create a tensor filled with ones (for loss gradient initialization).
     */
    LogicalTensorPtr CreateOnesTensor(Function& function, const Shape& shape, DataType dtype);

    /**
     * @brief Create an add operation for gradient accumulation.
     */
    LogicalTensorPtr CreateAddOp(Function& function, LogicalTensorPtr a, LogicalTensorPtr b);

    /**
     * @brief Create a cast operation for dtype conversion.
     */
    LogicalTensorPtr CreateCastOp(Function& function, LogicalTensorPtr input, DataType targetDtype);

    /**
     * @brief Create a sum operation for unbroadcasting.
     */
    LogicalTensorPtr CreateSumOp(Function& function, LogicalTensorPtr input, int64_t axis, bool keepdim);

    // Gradient storage: tensor magic -> gradient tensor
    std::unordered_map<int, LogicalTensorPtr> valueGrads_;

    // Set of tensors requiring gradients (by magic)
    std::unordered_set<int> requiresGradSet_;

    // Tensors to compute gradients for
    std::vector<LogicalTensorPtr> wrtTensors_;
};

} // namespace npu::tile_fwk

#endif // PASS_AUTODIFF_H_
