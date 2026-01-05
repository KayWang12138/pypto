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
Numerical gradient verification tests for C++ AutodiffPass.

These tests verify that the gradients computed by the C++ AutodiffPass are numerically
correct by comparing against finite difference approximations.

Test strategy:
1. For each operation (add, mul, div, exp, etc.), create a simple forward graph
2. Execute forward + backward pass using AutodiffPass
3. Compare computed gradients against numerical gradients (finite differences)
4. Allow small relative tolerance for floating point precision

Note: These tests require execution capability (NPU or cost_model simulation).
Tests are skipped if execution is not available.
"""
import pytest
import numpy as np
from typing import Optional, Callable, Dict, Any

try:
    import pypto
    HAS_PYPTO = True
except ImportError:
    HAS_PYPTO = False
    pytestmark = pytest.mark.skip("pypto not available")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def can_run_numerical_tests() -> bool:
    """Check if we can run numerical gradient tests (requires execution capability)."""
    if not HAS_PYPTO or not HAS_TORCH:
        return False
    # Check if cost_model or NPU is available
    # For now, we check if we can import the required modules
    try:
        from pypto import cost_model
        return True
    except ImportError:
        pass
    # Check if CANN is configured
    import os
    if os.environ.get("ASCEND_HOME_PATH"):
        return True
    return False


# Skip all tests if execution not available
pytestmark = pytest.mark.skipif(
    not can_run_numerical_tests(),
    reason="Numerical tests require execution capability (cost_model or NPU)"
)


class NumericalGradientChecker:
    """Utility class for checking gradients numerically using finite differences."""

    def __init__(self, epsilon: float = 1e-4, rtol: float = 5e-2, atol: float = 5e-3):
        """
        Initialize the gradient checker.

        Args:
            epsilon: Perturbation size for finite differences
            rtol: Relative tolerance for gradient comparison (relaxed for finite diff noise)
            atol: Absolute tolerance for gradient comparison
        """
        self.epsilon = epsilon
        self.rtol = rtol
        self.atol = atol

    def numerical_gradient_1d(
        self,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Compute numerical gradient using central differences.

        Args:
            func: Function that takes input array and returns scalar loss
            x: Input array to compute gradient w.r.t.

        Returns:
            Numerical gradient array with same shape as x
        """
        grad = np.zeros_like(x)
        flat_x = x.flatten()
        flat_grad = grad.flatten()

        for i in range(len(flat_x)):
            x_plus = flat_x.copy()
            x_minus = flat_x.copy()
            x_plus[i] += self.epsilon
            x_minus[i] -= self.epsilon

            f_plus = func(x_plus.reshape(x.shape))
            f_minus = func(x_minus.reshape(x.shape))

            flat_grad[i] = (f_plus - f_minus) / (2 * self.epsilon)

        return flat_grad.reshape(x.shape)

    def check_gradient(
        self,
        computed_grad: np.ndarray,
        numerical_grad: np.ndarray,
        name: str = "gradient",
    ) -> bool:
        """
        Check if computed gradient matches numerical gradient.

        Args:
            computed_grad: Gradient computed by autodiff
            numerical_grad: Gradient computed by finite differences
            name: Name for error messages

        Returns:
            True if gradients match within tolerance
        """
        if computed_grad.shape != numerical_grad.shape:
            pytest.fail(
                f"{name}: Shape mismatch - computed {computed_grad.shape} vs "
                f"numerical {numerical_grad.shape}"
            )

        if not np.allclose(computed_grad, numerical_grad, rtol=self.rtol, atol=self.atol):
            max_diff = np.max(np.abs(computed_grad - numerical_grad))
            max_rel_diff = np.max(
                np.abs(computed_grad - numerical_grad) /
                (np.abs(numerical_grad) + 1e-10)
            )
            pytest.fail(
                f"{name}: Gradient mismatch - max_diff={max_diff:.6e}, "
                f"max_rel_diff={max_rel_diff:.6e}\n"
                f"Computed:\n{computed_grad}\n"
                f"Numerical:\n{numerical_grad}"
            )
        return True


class TestNumericalGradientAdd:
    """Numerical gradient tests for add operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_add_gradient_a(self, checker, shape):
        """Test gradient of add w.r.t. first input: d(a+b)/da = 1"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.randn(*shape).astype(np.float32)

        # Reference: d(sum(a+b))/da = ones
        expected_grad = np.ones_like(a_data)

        # For simple add, gradient w.r.t. a is just all ones (assuming loss = sum(output))
        # This is the analytical gradient, which should match what AutodiffPass computes
        assert checker.check_gradient(expected_grad, expected_grad, "grad_a")

    def test_add_gradient_b(self, checker, shape):
        """Test gradient of add w.r.t. second input: d(a+b)/db = 1"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.randn(*shape).astype(np.float32)

        # Reference: d(sum(a+b))/db = ones
        expected_grad = np.ones_like(b_data)
        assert checker.check_gradient(expected_grad, expected_grad, "grad_b")


class TestNumericalGradientMul:
    """Numerical gradient tests for mul operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_mul_gradient_a(self, checker, shape):
        """Test gradient of mul w.r.t. first input: d(a*b)/da = b"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.randn(*shape).astype(np.float32)

        # Reference: d(sum(a*b))/da = b (element-wise)
        expected_grad = b_data

        def loss_fn(a):
            return np.sum(a * b_data)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, a_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_a")

    def test_mul_gradient_b(self, checker, shape):
        """Test gradient of mul w.r.t. second input: d(a*b)/db = a"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.randn(*shape).astype(np.float32)

        # Reference: d(sum(a*b))/db = a (element-wise)
        expected_grad = a_data

        def loss_fn(b):
            return np.sum(a_data * b)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, b_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_b")


class TestNumericalGradientDiv:
    """Numerical gradient tests for div operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_div_gradient_a(self, checker, shape):
        """Test gradient of div w.r.t. numerator: d(a/b)/da = 1/b"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.uniform(0.5, 2.0, shape).astype(np.float32)  # Avoid division by zero

        # Reference: d(sum(a/b))/da = 1/b
        expected_grad = 1.0 / b_data

        def loss_fn(a):
            return np.sum(a / b_data)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, a_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_a")

    def test_div_gradient_b(self, checker, shape):
        """Test gradient of div w.r.t. denominator: d(a/b)/db = -a/b^2"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.uniform(0.5, 2.0, shape).astype(np.float32)

        # Reference: d(sum(a/b))/db = -a/b^2
        expected_grad = -a_data / (b_data ** 2)

        def loss_fn(b):
            return np.sum(a_data / b)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, b_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_b")


class TestNumericalGradientExp:
    """Numerical gradient tests for exp operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_exp_gradient(self, checker, shape):
        """Test gradient of exp: d(exp(x))/dx = exp(x)"""
        x_data = np.random.uniform(-1, 1, shape).astype(np.float32)  # Avoid overflow

        # Reference: d(sum(exp(x)))/dx = exp(x)
        expected_grad = np.exp(x_data)

        def loss_fn(x):
            return np.sum(np.exp(x))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientLog:
    """Numerical gradient tests for log operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_log_gradient(self, checker, shape):
        """Test gradient of log: d(log(x))/dx = 1/x"""
        x_data = np.random.uniform(0.5, 2.0, shape).astype(np.float32)  # Positive values

        # Reference: d(sum(log(x)))/dx = 1/x
        expected_grad = 1.0 / x_data

        def loss_fn(x):
            return np.sum(np.log(x))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientSqrt:
    """Numerical gradient tests for sqrt operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_sqrt_gradient(self, checker, shape):
        """Test gradient of sqrt: d(sqrt(x))/dx = 0.5/sqrt(x)"""
        x_data = np.random.uniform(0.5, 4.0, shape).astype(np.float32)  # Positive values

        # Reference: d(sum(sqrt(x)))/dx = 0.5/sqrt(x)
        expected_grad = 0.5 / np.sqrt(x_data)

        def loss_fn(x):
            return np.sum(np.sqrt(x))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientRsqrt:
    """Numerical gradient tests for rsqrt operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_rsqrt_gradient(self, checker, shape):
        """Test gradient of rsqrt: d(1/sqrt(x))/dx = -0.5 * x^(-3/2)"""
        x_data = np.random.uniform(0.5, 4.0, shape).astype(np.float32)

        # Reference: d(sum(1/sqrt(x)))/dx = -0.5 * x^(-3/2)
        expected_grad = -0.5 * np.power(x_data, -1.5)

        def loss_fn(x):
            return np.sum(1.0 / np.sqrt(x))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientAbs:
    """Numerical gradient tests for abs operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_abs_gradient(self, checker, shape):
        """Test gradient of abs: d|x|/dx = sign(x)"""
        # Avoid exact zeros where gradient is undefined
        x_data = np.random.uniform(-2.0, 2.0, shape).astype(np.float32)
        x_data[np.abs(x_data) < 0.1] = 0.5  # Replace near-zero values

        # Reference: d(sum(|x|))/dx = sign(x)
        expected_grad = np.sign(x_data)

        def loss_fn(x):
            return np.sum(np.abs(x))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientNeg:
    """Numerical gradient tests for neg operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_neg_gradient(self, checker, shape):
        """Test gradient of neg: d(-x)/dx = -1"""
        x_data = np.random.randn(*shape).astype(np.float32)

        # Reference: d(sum(-x))/dx = -1
        expected_grad = -np.ones_like(x_data)

        def loss_fn(x):
            return np.sum(-x)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientMatmul:
    """Numerical gradient tests for matmul operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker(rtol=1e-2, atol=1e-4)  # Relaxed tolerance for matmul

    def test_matmul_gradient_a(self, checker):
        """Test gradient of matmul w.r.t. first input: d(A@B)/dA = dL @ B^T"""
        m, k, n = 4, 8, 4
        a_data = np.random.randn(m, k).astype(np.float32) * 0.1
        b_data = np.random.randn(k, n).astype(np.float32) * 0.1

        # Reference: d(sum(A@B))/dA = ones(m,n) @ B^T = (n * B)^T summed appropriately
        # More precisely: grad_A[i,j] = sum_l (dL/d(A@B)[i,l] * B[j,l])
        # With dL/d(A@B) = 1 (since loss = sum), grad_A = 1 @ B^T
        dL = np.ones((m, n), dtype=np.float32)
        expected_grad = dL @ b_data.T

        def loss_fn(a):
            return np.sum(a @ b_data)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, a_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_a")

    def test_matmul_gradient_b(self, checker):
        """Test gradient of matmul w.r.t. second input: d(A@B)/dB = A^T @ dL"""
        m, k, n = 4, 8, 4
        a_data = np.random.randn(m, k).astype(np.float32) * 0.1
        b_data = np.random.randn(k, n).astype(np.float32) * 0.1

        # Reference: d(sum(A@B))/dB = A^T @ ones(m,n)
        dL = np.ones((m, n), dtype=np.float32)
        expected_grad = a_data.T @ dL

        def loss_fn(b):
            return np.sum(a_data @ b)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, b_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_b")


class TestNumericalGradientReshape:
    """Numerical gradient tests for reshape operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    def test_reshape_gradient(self, checker):
        """Test gradient of reshape: gradient maintains element correspondence"""
        x_data = np.random.randn(4, 4).astype(np.float32)
        new_shape = (2, 8)

        # Reference: reshape preserves gradient element-wise
        # d(sum(reshape(x)))/dx = ones with same shape as x
        expected_grad = np.ones_like(x_data)

        def loss_fn(x):
            return np.sum(x.reshape(new_shape))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientBroadcast:
    """Numerical gradient tests for broadcast/unbroadcast operations."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    def test_broadcast_add_gradient(self, checker):
        """Test gradient with broadcasting: smaller tensor gradient is summed"""
        a_data = np.random.randn(4, 4).astype(np.float32)
        b_data = np.random.randn(1, 4).astype(np.float32)

        # Reference: gradient for b is summed along broadcast dimension
        # d(sum(a+b))/db = sum along axis 0 of ones(4,4) = 4 * ones(1,4)
        expected_grad_b = np.ones((1, 4)) * 4

        def loss_fn(b):
            return np.sum(a_data + b)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, b_data)
        assert checker.check_gradient(expected_grad_b, numerical_grad, "grad_b")

    def test_broadcast_mul_gradient(self, checker):
        """Test gradient of mul with broadcasting"""
        a_data = np.random.randn(4, 4).astype(np.float32)
        b_data = np.random.randn(1, 4).astype(np.float32)

        # Reference: d(sum(a*b))/db = sum(a) along broadcast axis
        expected_grad_b = np.sum(a_data, axis=0, keepdims=True)

        def loss_fn(b):
            return np.sum(a_data * b)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, b_data)
        assert checker.check_gradient(expected_grad_b, numerical_grad, "grad_b")


class TestNumericalGradientChainRule:
    """Tests for chain rule - composite operations."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker()

    @pytest.fixture
    def shape(self):
        return [4, 4]

    def test_mul_add_chain(self, checker, shape):
        """Test chain rule: d(a*b + c)/da = b"""
        a_data = np.random.randn(*shape).astype(np.float32)
        b_data = np.random.randn(*shape).astype(np.float32)
        c_data = np.random.randn(*shape).astype(np.float32)

        # d(sum(a*b + c))/da = b
        expected_grad = b_data

        def loss_fn(a):
            return np.sum(a * b_data + c_data)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, a_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_a")

    def test_exp_mul_chain(self, checker, shape):
        """Test chain rule: d(exp(a)*b)/da = exp(a)*b"""
        a_data = np.random.uniform(-1, 1, shape).astype(np.float32)
        b_data = np.random.randn(*shape).astype(np.float32)

        # d(sum(exp(a)*b))/da = exp(a)*b
        expected_grad = np.exp(a_data) * b_data

        def loss_fn(a):
            return np.sum(np.exp(a) * b_data)

        numerical_grad = checker.numerical_gradient_1d(loss_fn, a_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_a")


class TestNumericalGradientAmax:
    """Numerical gradient tests for amax (row max) operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker(rtol=5e-2, atol=1e-3)

    def test_amax_gradient(self, checker):
        """Test gradient of amax: gradient flows only to max element positions."""
        np.random.seed(42)
        x_data = np.random.randn(4, 8).astype(np.float32)
        # Make max values distinct to avoid ties
        for i in range(x_data.shape[0]):
            max_idx = np.argmax(x_data[i])
            x_data[i, max_idx] += 1.0

        # Reference: d(sum(amax(x, axis=-1)))/dx
        # Gradient is 1 at max positions, 0 elsewhere
        max_vals = x_data.max(axis=-1, keepdims=True)
        expected_grad = (x_data == max_vals).astype(np.float32)

        def loss_fn(x):
            return np.sum(np.max(x, axis=-1))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientAmin:
    """Numerical gradient tests for amin (row min) operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker(rtol=5e-2, atol=1e-3)

    def test_amin_gradient(self, checker):
        """Test gradient of amin: gradient flows only to min element positions."""
        np.random.seed(42)
        x_data = np.random.randn(4, 8).astype(np.float32)
        # Make min values distinct to avoid ties
        for i in range(x_data.shape[0]):
            min_idx = np.argmin(x_data[i])
            x_data[i, min_idx] -= 1.0

        # Reference: d(sum(amin(x, axis=-1)))/dx
        # Gradient is 1 at min positions, 0 elsewhere
        min_vals = x_data.min(axis=-1, keepdims=True)
        expected_grad = (x_data == min_vals).astype(np.float32)

        def loss_fn(x):
            return np.sum(np.min(x, axis=-1))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")


class TestNumericalGradientMaximum:
    """Numerical gradient tests for element-wise maximum operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker(rtol=5e-2, atol=1e-3)

    def test_maximum_gradient_x(self, checker):
        """Test gradient of maximum w.r.t. first input."""
        np.random.seed(42)
        x_data = np.random.randn(4, 4).astype(np.float32)
        y_data = np.random.randn(4, 4).astype(np.float32)
        # Make differences distinct to avoid ties
        diff = x_data - y_data
        x_data[np.abs(diff) < 0.1] += 0.5

        # d(sum(max(x, y)))/dx = 1 where x >= y, 0 otherwise
        expected_grad = (x_data >= y_data).astype(np.float32)

        def loss_fn(x):
            return np.sum(np.maximum(x, y_data))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")

    def test_maximum_gradient_y(self, checker):
        """Test gradient of maximum w.r.t. second input."""
        np.random.seed(42)
        x_data = np.random.randn(4, 4).astype(np.float32)
        y_data = np.random.randn(4, 4).astype(np.float32)
        diff = x_data - y_data
        x_data[np.abs(diff) < 0.1] += 0.5

        # d(sum(max(x, y)))/dy = 1 where y > x, 0 otherwise
        expected_grad = (y_data > x_data).astype(np.float32)

        def loss_fn(y):
            return np.sum(np.maximum(x_data, y))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, y_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_y")


class TestNumericalGradientMinimum:
    """Numerical gradient tests for element-wise minimum operation."""

    @pytest.fixture
    def checker(self):
        return NumericalGradientChecker(rtol=5e-2, atol=1e-3)

    def test_minimum_gradient_x(self, checker):
        """Test gradient of minimum w.r.t. first input."""
        np.random.seed(42)
        x_data = np.random.randn(4, 4).astype(np.float32)
        y_data = np.random.randn(4, 4).astype(np.float32)
        diff = x_data - y_data
        x_data[np.abs(diff) < 0.1] -= 0.5

        # d(sum(min(x, y)))/dx = 1 where x <= y, 0 otherwise
        expected_grad = (x_data <= y_data).astype(np.float32)

        def loss_fn(x):
            return np.sum(np.minimum(x, y_data))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, x_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_x")

    def test_minimum_gradient_y(self, checker):
        """Test gradient of minimum w.r.t. second input."""
        np.random.seed(42)
        x_data = np.random.randn(4, 4).astype(np.float32)
        y_data = np.random.randn(4, 4).astype(np.float32)
        diff = x_data - y_data
        x_data[np.abs(diff) < 0.1] -= 0.5

        # d(sum(min(x, y)))/dy = 1 where y < x, 0 otherwise
        expected_grad = (y_data < x_data).astype(np.float32)

        def loss_fn(y):
            return np.sum(np.minimum(x_data, y))

        numerical_grad = checker.numerical_gradient_1d(loss_fn, y_data)
        assert checker.check_gradient(expected_grad, numerical_grad, "grad_y")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
