#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
测试用例耗时标记装饰器

用于标记长耗时测试用例，支持在 pytest 多进程并行执行时进行用例重排序优化。
"""
import pytest


def time_cost(seconds: int):
    """
    Decorator: annotate a test case with estimated duration (seconds).

    This decorator marks test cases with their expected execution time,
    allowing pytest to reorder tests for optimal parallel execution.

    Args:
        seconds: Estimated execution time in seconds

    Example:
        @time_cost(120)
        def test_something():
            ...
    """
    def decorator(func):
        # Store the time cost as a public attribute on the function
        func.time_cost = seconds
        return func
    return decorator


# def get_time_cost(func):
#     """
#     Get the time cost of a decorated function.

#     Args:
#         func: The decorated function

#     Returns:
#         int or None: Time cost in seconds, or None if not decorated
#     """
#     return getattr(func, 'time_cost', None)
