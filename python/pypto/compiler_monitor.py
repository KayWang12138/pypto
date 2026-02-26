# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compiler monitor: progress and timing for PyPTO compilation."""

from . import pypto_impl


def set_compiler_monitor_options(enable=True, interval_sec=30, timeout_sec=60, total_timeout_sec=120):
    """Set compiler monitor options.

    Args:
        enable: Whether to enable the monitor. Default True.
        interval_sec: Progress print interval in seconds. Default 30.
        timeout_sec: Progress print timeout in seconds. Default 60.
        total_timeout_sec: Progress print total timeout in seconds. Default 120.
    """
    pypto_impl.SetCompilerMonitorOptions(enable=enable, interval_sec=interval_sec,
        timeout_sec=timeout_sec, total_timeout_sec=total_timeout_sec)
