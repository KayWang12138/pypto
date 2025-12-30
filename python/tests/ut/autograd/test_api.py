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
"""
Unit tests for autograd train_step API (M3 milestone).

Tests:
- train_step creation and parameter resolution
- wrt and loss validation
- grads dict handling (full/partial/auto-allocation)
- TrainGraphBuilder functionality
"""
import pytest


class TestTrainStepCreation:
    """Test train_step creation and configuration."""

    def test_create_train_step_with_name_wrt(self):
        """Test creating train_step with parameter names for wrt."""
        from pypto.autograd import train_step

        def forward(x, weight, out, loss_out):
            pass

        train = train_step(forward, wrt=("weight",), loss="loss_out")

        assert train.wrt == ("weight",)
        assert train.loss == "loss_out"
        assert train.forward_fn is forward

    def test_create_train_step_with_index_wrt(self):
        """Test creating train_step with parameter indices for wrt."""
        from pypto.autograd import train_step

        def forward(x, weight, out, loss_out):
            pass

        train = train_step(forward, wrt=(1,), loss=3)

        assert train.wrt == (1,)
        assert train.loss == 3

    def test_create_train_step_multiple_wrt(self):
        """Test creating train_step with multiple wrt parameters."""
        from pypto.autograd import train_step

        def forward(x, weight, bias, out, loss_out):
            pass

        train = train_step(forward, wrt=("weight", "bias"), loss="loss_out")

        assert train.wrt == ("weight", "bias")


class TestParamResolution:
    """Test parameter name/index resolution."""

    def test_resolve_name_to_name(self):
        """Test resolving parameter name returns same name."""
        from pypto.autograd import train_step

        def forward(x, weight, out, loss_out):
            pass

        train = train_step(forward, wrt=("weight",), loss="loss_out")

        assert train._resolve_param_name("weight") == "weight"
        assert train._resolve_param_name("loss_out") == "loss_out"

    def test_resolve_index_to_name(self):
        """Test resolving parameter index to name."""
        from pypto.autograd import train_step

        def forward(x, weight, out, loss_out):
            pass

        train = train_step(forward, wrt=(1,), loss=3)

        assert train._resolve_param_name(0) == "x"
        assert train._resolve_param_name(1) == "weight"
        assert train._resolve_param_name(3) == "loss_out"

    def test_invalid_param_name_raises(self):
        """Test that invalid parameter name raises ValueError."""
        from pypto.autograd import TrainGraphBuilder

        def forward(x, weight, out, loss_out):
            pass

        builder = TrainGraphBuilder(forward, wrt=("nonexistent",), loss="loss_out")

        with pytest.raises(ValueError, match="not found"):
            builder._resolve_param_name("nonexistent")

    def test_invalid_param_index_raises(self):
        """Test that invalid parameter index raises ValueError."""
        from pypto.autograd import TrainGraphBuilder

        def forward(x, weight, out, loss_out):
            pass

        builder = TrainGraphBuilder(forward, wrt=(10,), loss="loss_out")

        with pytest.raises(ValueError, match="out of range"):
            builder._resolve_param_index(10)


class TestWrtValidation:
    """Test wrt parameter validation."""

    def test_validate_wrt_differentiable_dtype(self):
        """Test that wrt validation passes for differentiable dtypes."""
        from pypto.autograd import TrainGraphBuilder

        def forward(x, weight, out, loss_out):
            pass

        builder = TrainGraphBuilder(forward, wrt=("weight",), loss="loss_out")

        # Mock tensor with differentiable dtype
        class MockTensor:
            dtype = "DT_FP32"
            shape = [32, 64]

        bound_args = {
            "x": MockTensor(),
            "weight": MockTensor(),
            "out": MockTensor(),
            "loss_out": MockTensor(),
        }

        names = builder.validate_wrt(bound_args)
        assert names == ["weight"]

    def test_validate_wrt_non_differentiable_dtype_raises(self):
        """Test that wrt validation fails for non-differentiable dtypes."""
        from pypto.autograd import TrainGraphBuilder

        def forward(x, weight, out, loss_out):
            pass

        builder = TrainGraphBuilder(forward, wrt=("weight",), loss="loss_out")

        # Mock tensor with non-differentiable dtype
        class MockTensor:
            dtype = "DT_INT32"
            shape = [32, 64]

        class MockFloatTensor:
            dtype = "DT_FP32"
            shape = [32, 64]

        bound_args = {
            "x": MockFloatTensor(),
            "weight": MockTensor(),  # INT32 - not differentiable
            "out": MockFloatTensor(),
            "loss_out": MockFloatTensor(),
        }

        with pytest.raises(TypeError, match="not differentiable"):
            builder.validate_wrt(bound_args)


class TestLossValidation:
    """Test loss parameter validation."""

    def test_validate_scalar_loss(self):
        """Test that scalar loss validation passes."""
        from pypto.autograd import TrainGraphBuilder

        def forward(x, weight, out, loss_out):
            pass

        builder = TrainGraphBuilder(forward, wrt=("weight",), loss="loss_out")

        class MockTensor:
            shape = [1]
            dtype = "DT_FP32"

        bound_args = {
            "x": MockTensor(),
            "weight": MockTensor(),
            "out": MockTensor(),
            "loss_out": MockTensor(),  # Scalar
        }

        name = builder.validate_loss(bound_args)
        assert name == "loss_out"

    def test_validate_non_scalar_loss_raises(self):
        """Test that non-scalar loss raises ValueError."""
        from pypto.autograd import TrainGraphBuilder

        def forward(x, weight, out, loss_out):
            pass

        builder = TrainGraphBuilder(forward, wrt=("weight",), loss="loss_out")

        class MockScalarTensor:
            shape = [1]
            dtype = "DT_FP32"

        class MockNonScalarTensor:
            shape = [32, 64]  # Not scalar
            dtype = "DT_FP32"

        bound_args = {
            "x": MockScalarTensor(),
            "weight": MockScalarTensor(),
            "out": MockScalarTensor(),
            "loss_out": MockNonScalarTensor(),  # Not scalar
        }

        with pytest.raises(ValueError, match="must be scalar"):
            builder.validate_loss(bound_args)


class TestGradsHandling:
    """Test gradient buffer handling."""

    def test_validate_grads_full_provided(self):
        """Test that full grads dict is validated correctly."""
        from pypto.autograd import train_step

        def forward(x, weight, out, loss_out):
            pass

        train = train_step(forward, wrt=("weight",), loss="loss_out")

        class MockTensor:
            dtype = "DT_FP32"
            shape = [32, 64]

        bound_args = {
            "x": MockTensor(),
            "weight": MockTensor(),
            "out": MockTensor(),
            "loss_out": MockTensor(),
        }

        grads = {"weight": MockTensor()}
        validated = train._validate_grads(grads, bound_args)

        assert "weight" in validated
        assert validated["weight"] is grads["weight"]

    def test_validate_grads_partial_allowed(self):
        """Test that partial grads dict is allowed with allow_missing_grads=True."""
        from pypto.autograd import train_step

        def forward(x, weight, bias, out, loss_out):
            pass

        train = train_step(
            forward,
            wrt=("weight", "bias"),
            loss="loss_out",
            allow_missing_grads=True,
        )

        class MockTensor:
            dtype = "DT_FP32"
            shape = [32, 64]

        bound_args = {
            "x": MockTensor(),
            "weight": MockTensor(),
            "bias": MockTensor(),
            "out": MockTensor(),
            "loss_out": MockTensor(),
        }

        # Only provide weight gradient
        grads = {"weight": MockTensor()}
        validated = train._validate_grads(grads, bound_args)

        assert validated["weight"] is grads["weight"]
        assert validated["bias"] is None

    def test_validate_grads_partial_not_allowed_raises(self):
        """Test that partial grads raises when allow_missing_grads=False."""
        from pypto.autograd import train_step

        def forward(x, weight, bias, out, loss_out):
            pass

        train = train_step(
            forward,
            wrt=("weight", "bias"),
            loss="loss_out",
            allow_missing_grads=False,  # Require all grads
        )

        class MockTensor:
            dtype = "DT_FP32"
            shape = [32, 64]

        bound_args = {
            "x": MockTensor(),
            "weight": MockTensor(),
            "bias": MockTensor(),
            "out": MockTensor(),
            "loss_out": MockTensor(),
        }

        # Only provide weight gradient
        grads = {"weight": MockTensor()}

        with pytest.raises(ValueError, match="Missing gradient buffers"):
            train._validate_grads(grads, bound_args)

    def test_invalid_grad_key_raises(self):
        """Test that invalid gradient key raises ValueError."""
        from pypto.autograd import train_step

        def forward(x, weight, out, loss_out):
            pass

        train = train_step(forward, wrt=("weight",), loss="loss_out")

        class MockTensor:
            dtype = "DT_FP32"
            shape = [32, 64]

        bound_args = {
            "x": MockTensor(),
            "weight": MockTensor(),
            "out": MockTensor(),
            "loss_out": MockTensor(),
        }

        # Invalid key
        grads = {"invalid_key": MockTensor()}

        with pytest.raises(ValueError, match="not in wrt parameters"):
            train._validate_grads(grads, bound_args)


class TestJitDecorator:
    """Test the jit decorator syntax sugar."""

    def test_jit_decorator_creates_train_step(self):
        """Test that @jit decorator creates TrainStep."""
        from pypto.autograd import jit, TrainStep

        @jit(wrt=("weight",), loss="loss_out")
        def forward(x, weight, out, loss_out):
            pass

        assert isinstance(forward, TrainStep)
        assert forward.wrt == ("weight",)
        assert forward.loss == "loss_out"


class TestBackwardEngine:
    """Test backward engine functionality."""

    def test_engine_initialization(self):
        """Test BackwardEngine initialization."""
        from pypto.autograd import BackwardEngine, Graph

        graph = Graph("test")
        engine = BackwardEngine(graph)

        assert engine.graph is graph

    def test_compute_gradients_function(self):
        """Test compute_gradients convenience function."""
        from pypto.autograd import compute_gradients, Graph

        graph = Graph("test")
        v1 = graph.create_value(tensor_id=1, shape=[2, 3], name="x")
        v2 = graph.create_value(tensor_id=2, shape=[1], name="loss")

        # For now just verify it doesn't crash
        # Full gradient computation requires real tensors
        result = compute_gradients(graph, v2, [v1])
        assert "x" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
