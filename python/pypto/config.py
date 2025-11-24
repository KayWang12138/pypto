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
"""
import inspect
from typing import List, Union, Dict, Optional
import inspect

from . import pto_impl


class CachedOptions:

    def __init__(self):
        self._options = pto_impl.GetOptions()

    def reset(self):
        self._options = pto_impl.GetOptions()

    def set_options(self, prefix, options):
        for name, value in options.items():
            key = f"{prefix}.{name}"
            if key in self._options and value is not None:
                self._options[key] = value
                pto_impl.SetOption(key, value)

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value
        pto_impl.SetOption(key, value)

    def get_options(self, prefix):
        prefix = f"{prefix}."
        return {k[len(prefix):]: v for k, v in self._options.items() if k.startswith(prefix)}

_pto_options = CachedOptions()


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
                     sg_skip_partition: Optional[bool] = None,
                     cycle_upper_bound: Optional[int] = None,
                     cycle_lower_bound: Optional[int] = None,
                     parallel_threshold: Optional[int] = None,
                     sg_vec_parallel_num: Optional[int] = None,
                     nbuffer_merge_mode: Optional[int] = None,
                     vec_nbuffer_map: Optional[Dict[int, int]] = None,
                     l1_reuse: Optional[int] = None,
                     l1_reuse_map: Optional[Dict[int, int]] = None,
                     cube_nbuffer: Optional[int] = None,
                     cube_nbuffer_map: Optional[Dict[int, int]] = None,
                     copyin_threshold: Optional[int] = None,
                     ooo_preschedule_method: Optional[str] = None
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
    _pto_options.set_options("pass", locals())


def get_pass_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get pass options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All pass options
    """
    return _pto_options.get_options("pass")


def set_host_options(*, only_codegen: Optional[bool] = None) -> None:
    """
    Set host options.

    Parameters
    ---------
    only_codegen : bool
        Shield the static on-board process.
    """
    _pto_options.set_options("host", locals())


def get_host_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get host options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All host options
    """
    return _pto_options.get_options("host")


def set_codegen_options(*,
                        support_dynamic_unaligned: Optional[bool] = None,
                        codegen_expression_fusion: Optional[bool] = None
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
    return _pto_options.set_options("codegen", locals())


def get_codegen_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get codegen options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All codegen options
    """
    return _pto_options.get_options("codegen")


def set_runtime_options(*,
                        machine_sched_mode: Optional[int] = None,
                        workspace_recycle_period: Optional[int] = None,
                        estimated_stitch_task_max_loop_num: Optional[int] = None,
                        first_stitch_task_loop_num: Optional[int] = None,
                        subseq_stitch_task_incr_loop_num: Optional[int] = None,
                        cfgcache_device_task_num: Optional[int] = None,
                        cfgcache_root_task_num: Optional[int] = None,
                        cfgcache_leaf_task_num: Optional[int] = None,
                        stitch_callop_max_num: int = None
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
        
    stitch_callop_max_num: int
        The maximum Callop computation amount per loop for stitch tasks,
        controlled in the ctrlflow AICPU during machine runtime.
    """
    _pto_options.set_options("runtime", locals())


def get_runtime_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get runtime options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All runtime options
    """
    return _pto_options.get_options("runtime")


def set_verify_options(*,
                       verify_tensor_graph: Optional[bool] = None,
                       verify_pass: Optional[bool] = None,
                       check_precision: Optional[bool] = None,
                       dump_tensor: Optional[bool] = None,
                       dump_operation: Optional[bool] = None,
                       profile_enable: Optional[bool] = None,
                       verify_execute_graph: Optional[bool] = None,
                       ) -> None:
    """
    Set verify options.

    Parameters
    ---------
    verify_tensor_graph : bool
        Whether to verify the tensor graph.

    verify_pass : bool
        Whether to verify the pass.

    check_precision : bool
        Whether to check the precision.

    dump_tensor : bool
        Whether to dump the tensor.

    dump_operation : bool
        Whether to dump the operation.
    """
    _pto_options.set_options("verify", locals())


def get_verify_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get verify options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All verify options
    """
    return _pto_options.get_options("verify")


def set_semantic_label(label: str) -> None:
    """
    Set the semantic label object.

    Parameters
    ---------
    label: str
        Semantic label.
        Note: label will be attached to subsequent operations

    """
    pto_impl.SetSemanticLabel(label, inspect.stack()[
                              1].filename, inspect.stack()[1].lineno)


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
    _pto_options[key] = value


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

    return _pto_options[key]


def reset_options() -> None:
    """
        Reset all configuration items to their default values.
    """
    pto_impl.Reset()
    _pto_options.reset()
