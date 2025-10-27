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
from typing import List, Union, Dict

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
    pto_impl.SetOption(f"pass.{key}", value)


def get_pass_option(key: str) -> Union[str, int, List[int], dict[int, int]]:
    """
    Get pass option by key.

    Args:
        key (str): Configuration option key.

    Returns:
        Union[str, int, List[int], dict[int, int]]: The value associated with the key.
    """
    return pto_impl.GetOption(f"pass.{key}")


def set_host_option(key: str, value: Union[str, int, List[int], dict[int, int]]) -> None:
    """
    Set host options.

    Args:
        key (str): Host configuration option key.
        value (Union[str, int, List[int], dict[int, int]]): Host configuration option value.
    """
    pto_impl.SetOption(f"host.{key}", value)


def get_host_option(key: str) -> Union[str, int, List[int], dict[int, int]]:
    """
    Get host option by key.

    Args:
        key (str): Host configuration option key.

    Returns:
        Union[str, int, List[int], dict[int, int]]: The value associated with the key.
    """
    return pto_impl.GetOption(f"host.{key}")


def set_codegen_option(key: str, value: Union[str, int, List[int], Dict[int, int]]) -> None:
    """
    Set codegen options.

    Parameters
    ---------
    key: str
        Config option key.

    value : Union[str, int, List[int], Dict[int, int]]
        Config option value.
    """

    pto_impl.SetOption(f"codegen.{key}", value)

def get_codegen_option(key:str) -> Union[str, int, List[int], Dict[int, int]]:
    """
    Get codegen options.

    Parameters
    ---------
    key: str
        Config option key.

    Returns
    -------
    Union[str, int, List[int], Dict[int, int]]
        Config option value.
    """

    return pto_impl.GetOption(f"codegen.{key}")

def set_runtime_option(key:str, value:Union[str, int, List[int], Dict[int, int]]) -> None:
    """
    Set runtime options.

    Parameters
    ---------
    key: str
        Config option key.

    value : Union[str, int, List[int], Dict[int, int]]
        Config option value.
    """

    pto_impl.SetOption(f"runtime.{key}", value)

def get_runtime_option(key:str) -> Union[str, int, List[int], Dict[int, int]]:
    """
    Get runtime options.

    Parameters
    ---------
    key: str
        Config option key.

    Returns
    -------
    Union[str, int, List[int], Dict[int, int]]
        Config option value.
    """

    return pto_impl.GetOption(f"runtime.{key}")


def set_semantic_label(label: str) -> None:
    """
    Set the semantic label object.

    Parameters
    ---------
    label: str
        Semantic label.
        Note: label will be attached to subsequent operations

    """

    pto_impl.SetSemanticLabel(label)
