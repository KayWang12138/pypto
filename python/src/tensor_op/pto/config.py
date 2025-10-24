#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
from typing import List, Union

import pto
from pto import pto_impl


def set_print_options(edge_items: int, precision: int, threshold: int, linewidth: int):
    """
    Set tensor print options.

    Args:
        edge_items (int): print max items in tensor head and tail.
        precision (int): precision print precision.
        threshold (int): threshold threshold to use.
        linewidth (int): linewidth max line width.
    """
    pto_impl.SetPrintOptions(edge_items, precision, threshold, linewidth)


def get_print_options() -> pto_impl.print_options:
    """
    Get current tensor print options.

    Returns:
        pto_impl.print_options: The current print option settings.
    """
    return pto_impl.GetPrintOptions()


def set_pass_option(key: str, value: Union[str, int, List[int], dict[int, int]]) -> None:
    """
    Set pass options.

    Args:
        key (str): Configuration option key. Supported keys include:
            - "cycle_lower_bound": Lower bound of schedule cycles for each subgraph (default: 512)
            - "cycle_upper_bound": Upper bound of schedule cycles for each subgraph (default: 10000)
            ...
        value (Union[str, int, List[int], dict[int, int]]): The configuration option value.
    """
    pto_impl.SetPassOption(key, value)


def get_pass_option(key: str) -> Union[str, int, List[int], dict[int, int]]:
    """
    Get pass option by key.

    Args:
        key (str): Configuration option key.

    Returns:
        Union[str, int, List[int], dict[int, int]]: The value associated with the key.
    """
    return pto_impl.GetPassOption(key)


def set_host_option(key: str, value: Union[str, int, List[int], dict[int, int]]) -> None:
    """
    Set host options.

    Args:
        key (str): Host configuration option key.
        value (Union[str, int, List[int], dict[int, int]]): Host configuration option value.
    """
    pto_impl.SetHostOption(key, value)


def get_host_option(key: str) -> Union[str, int, List[int], dict[int, int]]:
    """
    Get host option by key.

    Args:
        key (str): Host configuration option key.

    Returns:
        Union[str, int, List[int], dict[int, int]]: The value associated with the key.
    """
    return pto_impl.GetHostOption(key)
