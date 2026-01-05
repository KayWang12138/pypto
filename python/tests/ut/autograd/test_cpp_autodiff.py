#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for C++ AutodiffPass tensor attributes and integration."""
import pytest
import pypto


class TestTensorGradientAttributes:
    """Test tensor attributes related to autograd."""

    def test_requires_grad_default_false(self):
        """Test that requires_grad is False by default."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        assert t.requires_grad is False

    def test_requires_grad_set_true(self):
        """Test setting requires_grad to True."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_set_false(self):
        """Test setting requires_grad to False."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        t.requires_grad = True
        t.requires_grad = False
        assert t.requires_grad is False

    def test_is_loss_default_false(self):
        """Test that is_loss is False by default."""
        t = pypto.Tensor([1], pypto.DT_FP32, "loss")
        assert t.is_loss is False

    def test_is_loss_set_true(self):
        """Test setting is_loss to True."""
        t = pypto.Tensor([1], pypto.DT_FP32, "loss")
        t.is_loss = True
        assert t.is_loss is True

    def test_is_loss_set_false(self):
        """Test setting is_loss to False."""
        t = pypto.Tensor([1], pypto.DT_FP32, "loss")
        t.is_loss = True
        t.is_loss = False
        assert t.is_loss is False

    def test_get_gradient_magic_no_gradient(self):
        """Test that get_gradient_magic returns -1 when no gradient exists."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        assert t.get_gradient_magic() == -1

    def test_empty_tensor_requires_grad_raises(self):
        """Test that setting requires_grad on empty tensor raises."""
        t = pypto.Tensor()
        with pytest.raises(ValueError, match="Empty tensor"):
            _ = t.requires_grad

    def test_empty_tensor_is_loss_raises(self):
        """Test that setting is_loss on empty tensor raises."""
        t = pypto.Tensor()
        with pytest.raises(ValueError, match="Empty tensor"):
            _ = t.is_loss

    def test_empty_tensor_get_gradient_magic_raises(self):
        """Test that get_gradient_magic on empty tensor raises."""
        t = pypto.Tensor()
        with pytest.raises(ValueError, match="Empty tensor"):
            t.get_gradient_magic()


class TestTensorAttributeCombinations:
    """Test combinations of tensor attributes for autograd."""

    def test_requires_grad_multiple_tensors(self):
        """Test requires_grad on multiple tensors."""
        t1 = pypto.Tensor([16, 16], pypto.DT_FP32, "input")
        t2 = pypto.Tensor([16, 16], pypto.DT_FP32, "weight")
        t3 = pypto.Tensor([16, 16], pypto.DT_FP32, "output")

        t1.requires_grad = True
        t2.requires_grad = True
        # t3 should remain False

        assert t1.requires_grad is True
        assert t2.requires_grad is True
        assert t3.requires_grad is False

    def test_full_autograd_setup(self):
        """Test complete autograd tensor setup."""
        input_t = pypto.Tensor([4, 8], pypto.DT_FP32, "input")
        weight_t = pypto.Tensor([8, 4], pypto.DT_FP32, "weight")
        output_t = pypto.Tensor([4, 4], pypto.DT_FP32, "output")
        loss_t = pypto.Tensor([1], pypto.DT_FP32, "loss")

        # Mark tensors for autograd
        input_t.requires_grad = True
        weight_t.requires_grad = True
        loss_t.is_loss = True

        # Verify setup
        assert input_t.requires_grad is True
        assert weight_t.requires_grad is True
        assert output_t.requires_grad is False
        assert loss_t.is_loss is True

        # No gradients computed yet
        assert input_t.get_gradient_magic() == -1
        assert weight_t.get_gradient_magic() == -1


class TestDifferentDtypes:
    """Test autograd attributes with different dtypes."""

    def test_requires_grad_fp32(self):
        """Test requires_grad with FP32."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_fp16(self):
        """Test requires_grad with FP16."""
        t = pypto.Tensor([16, 16], pypto.DT_FP16, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_bf16(self):
        """Test requires_grad with BF16."""
        t = pypto.Tensor([16, 16], pypto.DT_BF16, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_int32(self):
        """Test requires_grad with INT32 (non-differentiable but attribute can be set)."""
        t = pypto.Tensor([16, 16], pypto.DT_INT32, "test")
        # The attribute can be set, but AutodiffPass will skip non-differentiable types
        t.requires_grad = True
        assert t.requires_grad is True


class TestDifferentShapes:
    """Test autograd attributes with different tensor shapes."""

    def test_requires_grad_1d(self):
        """Test requires_grad with 1D tensor."""
        t = pypto.Tensor([128], pypto.DT_FP32, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_2d(self):
        """Test requires_grad with 2D tensor."""
        t = pypto.Tensor([64, 128], pypto.DT_FP32, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_3d(self):
        """Test requires_grad with 3D tensor."""
        t = pypto.Tensor([8, 64, 128], pypto.DT_FP32, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_requires_grad_4d(self):
        """Test requires_grad with 4D tensor."""
        t = pypto.Tensor([2, 4, 64, 128], pypto.DT_FP32, "test")
        t.requires_grad = True
        assert t.requires_grad is True

    def test_is_loss_scalar(self):
        """Test is_loss with scalar tensor."""
        t = pypto.Tensor([1], pypto.DT_FP32, "loss")
        t.is_loss = True
        assert t.is_loss is True


class TestGradientInfoAPI:
    """Test the new gradient info retrieval API."""

    def test_get_gradient_output_index_no_gradient(self):
        """Test get_gradient_output_index returns -1 when no gradient computed."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        t.requires_grad = True
        # No function compiled yet, so should return -1
        assert t.get_gradient_output_index() == -1

    def test_get_gradient_info_no_gradient(self):
        """Test get_gradient_info returns None when no gradient computed."""
        t = pypto.Tensor([16, 16], pypto.DT_FP32, "test")
        t.requires_grad = True
        # No function compiled yet, so should return None
        assert t.get_gradient_info() is None


class TestFunctionGradientAPI:
    """Test Function-level gradient API."""

    def test_get_gradient_output_index_method(self):
        """Test Function.get_gradient_output_index method exists."""
        func = pypto.get_last_function()
        if func is not None:
            # Method should exist
            assert hasattr(func.base, 'get_gradient_output_index')

    def test_get_gradient_tensors_info_method(self):
        """Test Function.get_gradient_tensors_info method exists."""
        func = pypto.get_last_function()
        if func is not None:
            # Method should exist
            assert hasattr(func.base, 'get_gradient_tensors_info')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
