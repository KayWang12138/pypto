#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
"""
import inspect
from typing import List, Union, Dict
import inspect

from . import pto_impl


def set_print_options(edge_items: int, precision: int, threshold: int, linewidth: int):
    """
    Set tensor print options.

    Parameters
    ----------
    edge_items : int
        Print max items in tensor head and tail.

    precision : int
        Print precision.

    threshold : int
        Threshold to use.

    linewidth : int
        Max line width.
    """
    pto_impl.SetPrintOptions(edge_items, precision, threshold, linewidth)


def set_pass_options(*,
                     sg_skip_partition: bool = None,
                     cycle_upper_bound: int = None,
                     cycle_lower_bound: int = None,
                     parallel_threshold: int = None,
                     sg_vec_parallel_num: int = None,
                     nbuffer_merge_mode: int = None,
                     vec_nbuffer_map: Dict[int, int] = None,
                     l1_reuse: int = None,
                     l1_reuse_map: Dict[int, int] = None,
                     cube_nbuffer: int = None,
                     cube_nbuffer_map: Dict[int, int] = None,
                     copyin_threshold: int = None,
                     ooo_preschedule_method: str = None
                     ) -> None:
    """
    Set pass options.

    Parameters
    ---------
    sg_skip_partition : bool
        Whether to skip the subgraph partitioning process.

    cycle_upper_bound : int
        Merged graph parameter, used to configure
        the upper bound of subgraph size.

    cycle_lower_bound : int
        Merged graph parameter, used to configure
        the lower bound of subgraph size.

    parallel_threshold : int
        Merged graph parameter, used to configure
        the minimum parallelism of subgraphs with the same structure.

    sg_vec_parallel_num : int
        Merged graph parameter, used to configure
        the minimum parallelism of AIV subgraphs with the same structure.

    nbuffer_merge_mode : int
        Merged graph parameter, used to configure
        the merging strategy for AIV subgraphs with the same structure.

    vec_nbuffer_map : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of AIV subgraphs with the same structure.

    l1_reuse : int
        Merged graph parameter, used to configure
        the merging strategy for subgraphs with the same structure
        and repeated transfer of the same GM data.

    l1_reuse_map : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of subgraphs with the same structure
        and repeated transfer of the same GM data.

    cube_nbuffer : int
        Merged graph parameter, used to configure
        the merging strategy for AIC subgraphs with the same structure.

    cube_nbuffer_map : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of AIC subgraphs with the same structure.

    copyin_threshold : int
        Merged graph parameter, used to configure the merged graph size.

    ooo_preschedule_method : str
        Method for controlling the OoO PreSchedule of specific subgraphs.
    """
    params = locals()
    for name, value in params.items():
        if f"pass.{name}" in pto_impl.GetOptions() and value is not None:
            pto_impl.SetOption(f"pass.{name}", value)


def get_pass_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get pass options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All pass options
    """
    pass_options = {}
    for k in pto_impl.GetOptions():
        if k.startswith("pass."):
            pass_options[k[5:]] = pto_impl.GetOption(k)

    return pass_options


def set_host_options(*, only_codegen: bool = None) -> None:
    """
    Set host options.

    Parameters
    ---------
    only_codegen : bool
        Shield the static on-board process.
    """
    params = locals()
    for name, value in params.items():
        if f"host.{name}" in pto_impl.GetOptions() and value is not None:
            pto_impl.SetOption(f"host.{name}", value)


def get_host_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get host options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All host options
    """
    host_options = {}
    for k in pto_impl.GetOptions():
        if k.startswith("host."):
            host_options[k[5:]] = pto_impl.GetOption(k)

    return host_options


def set_codegen_options(*,
                        support_dynamic_unaligned: bool = None,
                        codegen_expression_fusion: bool = None
                        ) -> None:
    """
    Set codegen options.

    Parameters
    ---------
    support_dynamic_unaligned : bool
        Whether to support dynamic Shape.

    codegen_expression_fusion : bool
        Whether to support executing dynamic
        expression calculation on the device side.
    """
    params = locals()
    for name, value in params.items():
        if f"codegen.{name}" in pto_impl.GetOptions() and value is not None:
            pto_impl.SetOption(f"codegen.{name}", value)


def get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get codegen options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All codegen options
    """
    codegen_options = {}
    for k in pto_impl.GetOptions():
        if k.startswith("codegen."):
            codegen_options[k[8:]] = pto_impl.GetOption(k)

    return codegen_options


def set_runtime_options(*,
                        machine_sched_mode: int = None,
                        workspace_recycle_period: int = None,
                        estimated_stitch_task_max_loop_num: int = None,
                        first_stitch_task_loop_num: int = None,
                        subseq_stitch_task_incr_loop_num: int = None
                        ) -> None:
    """
    Set runtime options.

    Parameters
    ---------
    machine_sched_mode : int
        Set the scheduling mode of the computation subgraph.

    workspace_recycle_period : int
        Parameter for controlling the size of the non-outcast memory pool
        allocated to the root function, where the memory pool size is
        max_root_nonoutcast_workspace *.

    estimated_stitch_task_max_loop_num : int
        Used to evaluate the size of workspace memory required by
        an operator during runtime when compiling the operator.

    first_stitch_task_loop_num : int
        The amount of computation tasks for the first stitch task submitted to
        the scheduling AICPU for processing, controlled in the ctrlflow AICPU
        during machine runtime.

    subseq_stitch_task_incr_loop_num : int
        The computation amount of the processing loop for non-initial
        stitch tasks, controlled in the ctrlflow AICPU during machine runtime.
    """
    params = locals()
    for name, value in params.items():
        if f"runtime.{name}" in pto_impl.GetOptions() and value is not None:
            pto_impl.SetOption(f"runtime.{name}", value)


def get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get runtime options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All runtime options
    """
    runtime_options = {}
    for k in pto_impl.GetOptions():
        if k.startswith("runtime."):
            runtime_options[k[8:]] = pto_impl.GetOption(k)

    return runtime_options


def set_semantic_label(label: str) -> None:
    """
    Set the semantic label object.

    Parameters
    ---------
    label: str
        Semantic label.
        Note: label will be attached to subsequent operations

    """
    pto_impl.SetSemanticLabel(label, inspect.stack()[1].filename, inspect.stack()[1].lineno)


def set_option(key: str, value: Union[str, int, List[int], Dict[int, int]]) -> None:
    """
    Set global options.

    Parameters
    ---------
    key: str
        Config option key.

    value : Union[str, int, List[int], Dict[int, int]]
        Config option value.
    """

    pto_impl.SetOption(key, value)


def get_option(key: str) -> Union[str, int, List[int], Dict[int, int]]:
    """
    Get global options.

    Parameters
    ---------
    key: str
        Config option key.

    Returns
    -------
    Union[str, int, List[int], Dict[int, int]]
        Config option value.
    """

    return pto_impl.GetOption(key)


def reset_options() -> None:
    """
        Reset all configuration items to their default values.
    """
    return pto_impl.Reset()