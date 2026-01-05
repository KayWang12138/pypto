#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""
PyPTO Autograd - C++ AutodiffPass based automatic differentiation.

Usage:
    x.requires_grad = True
    loss.is_loss = True

    @pypto.jit
    def forward(x, loss):
        loss[:] = pypto.sum(x * 2)

    forward(x, loss)
    grad_x = x.get_gradient_tensor()
"""
