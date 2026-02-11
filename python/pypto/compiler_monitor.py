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
Compiler Monitor API for PyPTO.

This module provides functions to configure compilation progress monitoring
and timeout detection for PyPTO operators.

Example usage:
    >>> import pypto
    >>>
    >>> # Configure monitoring with default settings
    >>> pypto.set_compiler_monitor_options()
    >>>
    >>> # Configure with custom timeout and interval
    >>> pypto.set_compiler_monitor_options(
    ...     interval_sec=30,
    ...     timeout_sec=600,  # 10 minutes
    ...     timeout_action="warn"  # Just warn on timeout
    ... )
    >>>
    >>> # Use fine-grained monitoring (per-pass)
    >>> pypto.set_compiler_monitor_options(stage_mode="fine")
    >>>
    >>> # Define custom stages
    >>> pypto.set_compiler_monitor_options(
    ...     stage_mode="custom",
    ...     custom_stages=["Parse", "Optimize", "CodeGen", "Binary"]
    ... )
"""

from typing import Optional, List

# Import pypto_impl at module level (consistent with other pypto modules)
from . import pypto_impl

# Enum classes for configuration
class TimeoutAction:
    """Timeout action when a compilation stage exceeds timeout."""
    THROW = "throw"   # Throw exception
    WARN = "warn"     # Log warning but continue

class StageMode:
    """Stage granularity mode."""
    COARSE = "coarse"  # 6 major stages
    FINE = "fine"      # Individual passes
    CUSTOM = "custom"  # User-defined stages

# Pre-defined coarse stage names for reference
COARSE_STAGES = [
    "FrontendParser",
    "TensorGraphPass",
    "TileGraphPass",
    "BlockGraphPass",
    "CodeGen",
    "BinaryGeneration"
]

def set_compiler_monitor_options(
    enable: bool = True,
    interval_sec: int = 30,
    timeout_sec: int = 600,
    total_timeout_sec: int = 0,
    timeout_action: str = TimeoutAction.THROW,
    stage_mode: str = StageMode.COARSE,
    custom_stages: Optional[List[str]] = None,
) -> None:
    """Set compiler monitor options.

    Args:
        enable: Whether to enable monitoring. Default: True
        interval_sec: Progress print interval in seconds. Default: 30 (30 seconds)
        timeout_sec: Per-stage timeout threshold in seconds. Default: 600 (10 minutes)
        total_timeout_sec: Total compilation timeout threshold in seconds. Default: 0 (disabled, no limit)
        timeout_action: What to do on timeout. Options:
            - "throw": Raise CompilationTimeoutException
            - "warn": Log warning and continue execution
            Default: "throw"
        stage_mode: Stage granularity mode. Options:
            - "coarse": 6 major high-level stages (default)
            - "fine": Individual pass-level monitoring
            - "custom": User-defined stage names
        custom_stages: List of custom stage names. Required when stage_mode="custom".

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> # Default configuration
        >>> pypto.set_compiler_monitor_options()
        >>>
        >>> # Custom with 5-minute per-stage timeout, 10-minute total timeout
        >>> pypto.set_compiler_monitor_options(
        ...     interval_sec=30,
        ...     timeout_sec=300,
        ...     total_timeout_sec=600
        ... )
        >>>
        >>> # Fine-grained monitoring
        >>> pypto.set_compiler_monitor_options(stage_mode="fine")
        >>>
        >>> # Custom stages
        >>> pypto.set_compiler_monitor_options(
        ...     stage_mode="custom",
        ...     custom_stages=["Parse", "Optimize", "CodeGen"]
        ... )
    """
    # Validate interval_sec
    if not isinstance(interval_sec, int) or interval_sec <= 0:
        raise ValueError(f"interval_sec must be a positive integer, got {interval_sec}")

    # Validate timeout_sec
    if not isinstance(timeout_sec, int) or timeout_sec <= 0:
        raise ValueError(f"timeout_sec must be a positive integer, got {timeout_sec}")

    # Validate total_timeout_sec
    if not isinstance(total_timeout_sec, int) or total_timeout_sec < 0:
        raise ValueError(f"total_timeout_sec must be a non-negative integer, got {total_timeout_sec}")

    # Validate timeout_action
    if timeout_action not in (TimeoutAction.THROW, TimeoutAction.WARN):
        raise ValueError(
            f"timeout_action must be '{TimeoutAction.THROW}' or '{TimeoutAction.WARN}', "
            f"got {timeout_action}"
        )

    # Validate stage_mode
    if stage_mode not in (StageMode.COARSE, StageMode.FINE, StageMode.CUSTOM):
        raise ValueError(
            f"stage_mode must be '{StageMode.COARSE}', '{StageMode.FINE}', or '{StageMode.CUSTOM}', "
            f"got {stage_mode}"
        )

    # Validate custom_stages when using CUSTOM mode
    if stage_mode == StageMode.CUSTOM:
        if custom_stages is None:
            raise ValueError("custom_stages required when stage_mode='custom'")
        if not custom_stages:
            raise ValueError("custom_stages list cannot be empty")
        if any(not isinstance(s, str) or not s.strip() for s in custom_stages):
            raise ValueError("custom_stages must contain non-empty strings")

    # Convert timeout_action to enum value
    timeout_action_int = 0 if timeout_action == TimeoutAction.THROW else 1

    # Set monitor options
    pypto_impl.SetMonitorOptions(
        enable=enable,
        interval_sec=interval_sec,
        timeout_sec=timeout_sec,
        total_timeout_sec=total_timeout_sec,
        timeout_action=timeout_action_int
    )

    # Set stage mode
    stage_mode_int = 0  # COARSE
    if stage_mode == StageMode.FINE:
        stage_mode_int = 1
    elif stage_mode == StageMode.CUSTOM:
        stage_mode_int = 2

    custom_stages_str = ""
    if stage_mode == StageMode.CUSTOM:
        custom_stages_str = ",".join(s.strip() for s in custom_stages)

    pypto_impl.SetMonitorStageMode(stage_mode_int, custom_stages_str)


def initialize_monitor() -> None:
    """Initialize the compiler monitor.

    This function must be called before starting compilation to enable monitoring.
    The monitor options should be set using set_compiler_monitor_options() before
    calling this function.

    Note:
        This is usually called automatically when needed, but can be called
        explicitly for finer control.

    Example:
        >>> pypto.set_compiler_monitor_options()
        >>> pypto.initialize_monitor()
        >>> # ... compile ...
        >>> pypto.shutdown_monitor()
    """
    pypto_impl.InitializeMonitor()


def shutdown_monitor() -> None:
    """Shutdown the compiler monitor and release resources.

    This function should be called after compilation is complete to properly
    clean up monitoring resources (e.g., the background monitoring thread).

    Example:
        >>> pypto.set_compiler_monitor_options()
        >>> pypto.initialize_monitor()
        >>> # ... compile ...
        >>> pypto.shutdown_monitor()
    """
    pypto_impl.ShutdownMonitor()


# Predefined presets for common use cases
def set_monitor_quiet_mode() -> None:
    """Set quiet monitoring mode with longer intervals.

    Prints progress every 5 minutes instead of 1 minute.
    """
    set_compiler_monitor_options(interval_sec=300)


def set_monitor_aggressive_mode() -> None:
    """Set aggressive monitoring mode with shorter intervals and timeouts.

    Prints progress every 30 seconds and times out after 5 minutes.
    """
    set_compiler_monitor_options(
        interval_sec=30,
        timeout_sec=300,
        timeout_action=TimeoutAction.THROW
    )


def set_monitor_warn_only(timeout_sec: int = 600) -> None:
    """Set monitor to warn-only mode.

    Args:
        timeout_sec: Timeout threshold in seconds. Default: 600 (10 minutes)

    The monitor will log a warning when a stage times out, but will not
    interrupt compilation.
    """
    set_compiler_monitor_options(
        timeout_sec=timeout_sec,
        timeout_action=TimeoutAction.WARN
    )


__all__ = [
    # Main API
    "set_compiler_monitor_options",
    "initialize_monitor",
    "shutdown_monitor",
    # Enums
    "TimeoutAction",
    "StageMode",
    # Constants
    "COARSE_STAGES",
    # Presets
    "set_monitor_quiet_mode",
    "set_monitor_aggressive_mode",
    "set_monitor_warn_only",
]
