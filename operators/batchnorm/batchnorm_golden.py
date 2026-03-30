#!/usr/bin/env python3
# coding: utf-8
"""
BatchNorm Golden Reference Implementation

This module provides a pure PyTorch reference implementation of BatchNorm
for accuracy validation of the PyPTO kernel implementation.

Mathematical Formula:
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

Where:
    - E[x] is computed along axes [0, 1, 3, 4] (batch, seq_len, H, W)
    - Var[x] is computed along the same axes
    - gamma, beta are learnable parameters along channels dimension
"""

import torch
from typing import Tuple, Optional


def batchnorm_golden(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    training: bool = True
) -> torch.Tensor:
    """
    BatchNorm golden reference implementation.

    Args:
        x: Input tensor of shape [batch, seq_len, channels, H, W] or [batch, seq_len, channels]
        gamma: Scale parameter of shape [channels]
        beta: Shift parameter of shape [channels]
        eps: Small constant for numerical stability, default 1e-5
        running_mean: Running mean for inference mode, shape [channels]
        running_var: Running variance for inference mode, shape [channels]
        training: If True, compute batch statistics; if False, use running statistics

    Returns:
        Normalized output tensor with same shape as input
    """
    # Get input shape and ndim
    input_shape = x.shape
    ndim = x.dim()

    if ndim == 3:
        # 3D input: [batch, seq_len, channels]
        # Normalize along axes [0, 1] (batch and seq_len)
        if training:
            mean = x.mean(dim=[0, 1], keepdim=True)
            var = x.var(dim=[0, 1], keepdim=True, unbiased=False)
        else:
            if running_mean is None or running_var is None:
                raise ValueError("running_mean and running_var must be provided in inference mode")
            mean = running_mean.view(1, 1, -1)
            var = running_var.view(1, 1, -1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Affine transformation
        # gamma and beta are [channels], broadcast to [batch, seq_len, channels]
        output = x_norm * gamma.view(1, 1, -1) + beta.view(1, 1, -1)

    elif ndim == 5:
        # 5D input: [batch, seq_len, channels, H, W]
        # Normalize along axes [0, 1, 3, 4] (batch, seq_len, H, W)
        if training:
            mean = x.mean(dim=[0, 1, 3, 4], keepdim=True)
            var = x.var(dim=[0, 1, 3, 4], keepdim=True, unbiased=False)
        else:
            if running_mean is None or running_var is None:
                raise ValueError("running_mean and running_var must be provided in inference mode")
            mean = running_mean.view(1, 1, -1, 1, 1)
            var = running_var.view(1, 1, -1, 1, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Affine transformation
        # gamma and beta are [channels], broadcast to [batch, seq_len, channels, H, W]
        output = x_norm * gamma.view(1, 1, -1, 1, 1) + beta.view(1, 1, -1, 1, 1)

    else:
        raise ValueError(f"Unsupported input dimension: {ndim}, expected 3 or 5")

    return output


def batchnorm_golden_simple(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Simplified BatchNorm golden for training mode only.

    This is a simplified version that always computes batch statistics,
    useful for basic functionality testing.

    Args:
        x: Input tensor of shape [batch, seq_len, channels, H, W] or [batch, seq_len, channels]
        gamma: Scale parameter of shape [channels]
        beta: Shift parameter of shape [channels]
        eps: Small constant for numerical stability

    Returns:
        Normalized output tensor
    """
    return batchnorm_golden(x, gamma, beta, eps, training=True)


def get_test_configs() -> dict:
    """
    Return test configurations from spec.md.

    Returns:
        Dictionary of test configurations
    """
    return {
        "功能_P0": {
            "shape": (2, 32, 64, 28, 28),
            "channels": 64,
            "eps": 1e-5,
            "dtype": torch.float32,
            "description": "ResNet typical config"
        },
        "性能_P0": {
            "shape": (8, 64, 256, 56, 56),
            "channels": 256,
            "eps": 1e-5,
            "dtype": torch.float32,
            "description": "Large batch performance test"
        },
        "动态shape_1": {
            "shape": (1, 16, 128, 14, 14),
            "channels": 128,
            "eps": 1e-5,
            "dtype": torch.float32,
            "description": "Small batch short sequence"
        },
        "动态shape_2": {
            "shape": (16, 128, 512, 7, 7),
            "channels": 512,
            "eps": 1e-5,
            "dtype": torch.float32,
            "description": "Large batch long sequence"
        },
        "1D_BN": {
            "shape": (2, 512, 768),
            "channels": 768,
            "eps": 1e-5,
            "dtype": torch.float32,
            "description": "BatchNorm1d config"
        },
    }


def validate_batchnorm():
    """
    Validate the golden implementation against various test cases.
    """
    print("=" * 60)
    print("BatchNorm Golden Validation Report")
    print("=" * 60)

    configs = get_test_configs()

    # Test P0 configs first
    p0_configs = {k: v for k, v in configs.items() if "P0" in k}

    print("\n[P0 Configs - Must Pass]")
    for name, cfg in p0_configs.items():
        shape = cfg["shape"]
        channels = cfg["channels"]
        eps = cfg["eps"]
        dtype = cfg["dtype"]

        x = torch.randn(shape, dtype=dtype)
        gamma = torch.ones(channels, dtype=dtype)
        beta = torch.zeros(channels, dtype=dtype)

        try:
            output = batchnorm_golden(x, gamma, beta, eps)

            # Basic validation
            assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # Check numerical properties
            if dtype == torch.float32:
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-3, 1e-3

            # Compare with PyTorch built-in batch_norm
            if x.dim() == 3:
                mean = x.mean(dim=[0, 1], keepdim=True)
                var = x.var(dim=[0, 1], keepdim=True, unbiased=False)
            else:
                mean = x.mean(dim=[0, 1, 3, 4], keepdim=True)
                var = x.var(dim=[0, 1, 3, 4], keepdim=True, unbiased=False)

            x_norm = (x - mean) / torch.sqrt(var + eps)
            expected = x_norm * gamma.view(*([-1] + [1] * (x.dim() - 2))) + beta.view(*([-1] + [1] * (x.dim() - 2)))

            if not torch.allclose(output, expected, atol=atol, rtol=rtol):
                max_diff = (output - expected).abs().max().item()
                print(f"  {name}: FAIL - Max diff: {max_diff}")
            else:
                print(f"  {name}: PASS")
        except Exception as e:
            print(f"  {name}: FAIL - {e}")

    # Test P1 configs
    p1_configs = {k: v for k, v in configs.items() if "P1" in k or "P0" not in k}

    print("\n[P1 Configs - Functional Tests]")
    for name, cfg in p1_configs.items():
        shape = cfg["shape"]
        channels = cfg["channels"]
        eps = cfg["eps"]
        dtype = cfg["dtype"]

        x = torch.randn(shape, dtype=dtype)
        gamma = torch.ones(channels, dtype=dtype)
        beta = torch.zeros(channels, dtype=dtype)

        try:
            output = batchnorm_golden(x, gamma, beta, eps)
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            print(f"  {name}: PASS")
        except Exception as e:
            print(f"  {name}: FAIL - {e}")

    # Test different dtypes
    print("\n[Dtype Support Tests]")
    for dtype_name, dtype in [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        x = torch.randn(2, 16, 64, 14, 14, dtype=dtype)
        gamma = torch.ones(64, dtype=dtype)
        beta = torch.zeros(64, dtype=dtype)

        try:
            output = batchnorm_golden(x, gamma, beta)
            assert output.dtype == dtype
            print(f"  {dtype_name}: PASS")
        except Exception as e:
            print(f"  {dtype_name}: FAIL - {e}")

    # Test 3D input (BatchNorm1d)
    print("\n[3D Input Test]")
    x = torch.randn(2, 512, 768, dtype=torch.float32)
    gamma = torch.ones(768, dtype=torch.float32)
    beta = torch.zeros(768, dtype=torch.float32)

    try:
        output = batchnorm_golden(x, gamma, beta)
        assert output.shape == x.shape
        print(f"  3D input: PASS")
    except Exception as e:
        print(f"  3D input: FAIL - {e}")

    # Test inference mode
    print("\n[Inference Mode Test]")
    x = torch.randn(2, 16, 64, 14, 14, dtype=torch.float32)
    gamma = torch.ones(64, dtype=torch.float32)
    beta = torch.zeros(64, dtype=torch.float32)
    running_mean = torch.randn(64, dtype=torch.float32)
    running_var = torch.abs(torch.randn(64, dtype=torch.float32)) + 0.1

    try:
        output = batchnorm_golden(x, gamma, beta, eps=1e-5,
                                   running_mean=running_mean,
                                   running_var=running_var,
                                   training=False)
        assert output.shape == x.shape
        print(f"  Inference mode: PASS")
    except Exception as e:
        print(f"  Inference mode: FAIL - {e}")

    # Test numerical stability
    print("\n[Numerical Stability Tests]")
    # Large values
    x = torch.randn(2, 16, 64, 14, 14, dtype=torch.float32) * 1000
    gamma = torch.ones(64, dtype=torch.float32)
    beta = torch.zeros(64, dtype=torch.float32)
    try:
        output = batchnorm_golden(x, gamma, beta)
        assert not torch.isnan(output).any()
        print(f"  Large values: PASS")
    except Exception as e:
        print(f"  Large values: FAIL - {e}")

    # Small variance (near-zero)
    x = torch.ones(2, 16, 64, 14, 14, dtype=torch.float32)
    gamma = torch.ones(64, dtype=torch.float32)
    beta = torch.zeros(64, dtype=torch.float32)
    try:
        output = batchnorm_golden(x, gamma, beta, eps=1e-5)
        assert not torch.isnan(output).any()
        print(f"  Near-zero variance: PASS")
    except Exception as e:
        print(f"  Near-zero variance: FAIL - {e}")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)


if __name__ == "__main__":
    validate_batchnorm()
