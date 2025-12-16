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
from enum import IntEnum
from functools import wraps

from . import pypto_impl


class CachedOptions:

    def __init__(self):
        self._options = pypto_impl.GetOptions()

    def reset(self):
        self._options = pypto_impl.GetOptions()

    def set_options(self, prefix, options):
        for name, value in options.items():
            key = f"{prefix}.{name}"
            if key in self._options and value is not None:
                pypto_impl.SetOption(key, value)
                self._options[key] = value

    def __getitem__(self, key):
        return self._options[key]

    def __setitem__(self, key, value):
        self._options[key] = value
        pypto_impl.SetOption(key, value)

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
    pypto_impl.SetPrintOptions(edge_items, precision, threshold, linewidth)


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
                     cube_nbuffer_merge_mode: Optional[int] = None,
                     cube_nbuffer_map: Optional[Dict[int, int]] = None,
                     copyin_threshold: Optional[int] = None,
                     sg_set_scope: Optional[int] = None,
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

    cube_nbuffer_merge_mode : int
        Merged graph parameter, used to configure
        the merging strategy for AIC subgraphs with the same structure.

    cube_nbuffer_map : Dict[int, int]
        Merged graph parameter, used to configure
        the merging quantity of AIC subgraphs with the same structure.

    copyin_threshold : int
        Merged graph parameter, used to configure the merged graph size.
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
                        stitch_callop_max_num: int = None,
                        run_mode: Optional[int] = None
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
                       cost_model_enable: Optional[bool] = None,
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
    pypto_impl.SetSemanticLabel(label, inspect.stack()[
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
    pypto_impl.Reset()
    _pto_options.reset()


class Options:
    """Configuration options class, supports context manager and decorator modes"""
    INIT_FIELDS = [
        "name", "codegen_options", "host_options", "pass_options",
        "runtime_options", "verify_options", "vec_tile_shapes",
        "cube_tile_shapes", "matrix_size"
    ]

    PREFIX_MAP = {
        "codegen_options": "codegen.",
        "host_options": "host.",
        "pass_options": "pass.",
        "runtime_options": "runtime.",
        "verify_options": "verify."
    }

    def __init__(self, **kwargs):
        for field in self.INIT_FIELDS:
            setattr(self, field, kwargs.get(field, None))

    def prepare_options(self):
        """Convert configuration to target format"""
        opts = {}

        for attr, prefix in self.PREFIX_MAP.items():
            value = getattr(self, attr)
            if isinstance(value, dict):
                opts.update({f"{prefix}{k}": v for k, v in value.items()})

        if self.vec_tile_shapes is not None:
            opts["vec_tile_shapes"] = self.vec_tile_shapes

        if self.cube_tile_shapes is not None:
            if isinstance(self.cube_tile_shapes, CubeTile):
                opts["cube_tile_shapes"] = self.cube_tile_shapes._impl
            else:
                opts["cube_tile_shapes"] = CubeTile(*self.cube_tile_shapes)._impl

        if self.matrix_size is not None:
            opts["matrix_size"] = self.matrix_size

        return opts

    def __enter__(self):
        """Context manager enter logic"""
        opts = self.prepare_options()
        stack_frame = inspect.stack()[1]
        # Use decorator position if available, otherwise use caller position
        filename = getattr(self, 'decorator_filename', stack_frame.filename) or '<unknown>'
        lineno = getattr(self, 'decorator_lineno', stack_frame.lineno) or 0

        pypto_impl.BeginScope(self.name, opts, filename, lineno)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit logic"""
        stack_frame = inspect.stack()[1]
        pypto_impl.EndScope(stack_frame.filename, stack_frame.lineno)

    def __call__(self, func):
        """Decorator mode logic: capture function definition location and wrap"""
        self.decorator_filename = func.__code__.co_filename
        self.decorator_lineno = func.__code__.co_firstlineno

        if not self.name:
            self.name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def options(
    name="",
    codegen_options=None,
    host_options=None,
    pass_options=None,
    runtime_options=None,
    verify_options=None,
    vec_tile_shapes=None,
    cube_tile_shapes=None,
    matrix_size=None,
):
    """
    Create an Options instance. Can be used as decorator or context manager.

    Parameters
    ---------
    name: Scope name
    codegen_options: Code generation options (dict)
    host_options: Host options (dict)
    pass_options: Pass options (dict)
    runtime_options: Runtime options (dict)
    verify_options: Verify options (dict)
    vec_tile_shapes: Vector tile shapes (list)
    cube_tile_shapes: Cube tile shapes (CubeTile instance or list)
    matrix_size: Matrix size (list)

    Returns:
    -------
    Options instance

    Examples:
    -------
    # As decorator
    @pypto.options(pass_options={"l1_reuse": 4})
    def func():
        pass

    # As context manager
    with pypto.options(name="test", cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True]):
        pass
    """
    # Automatically collect parameters and pass them with unpacking (eliminate duplicate parameter writing)
    return Options(**locals())


def get_current_scope():
    """Get current config scope."""
    return pypto_impl.CurrentScope()


def set_options(
    codegen_options=None,
    host_options=None,
    pass_options=None,
    runtime_options=None,
    verify_options=None,
    vec_tile_shapes=None,
    cube_tile_shapes=None,
    matrix_size=None,
):
    """
    Finish the old scope and start a new scope.

    Parameters
    ---------
    codegen_options: Code generation options (dict)
    host_options: Host options (dict)
    pass_options: Pass options (dict)
    runtime_options: Runtime options (dict)
    verify_options: Verify options (dict)
    vec_tile_shapes: Vector tile shapes (list)
    cube_tile_shapes: Cube tile shapes (CubeTile instance or list)
    matrix_size: Matrix size (list)

    Examples:
    ---------
    set_options(pass_options={"l1_reuse": 4})
    set_options(cube_tile_shapes=[[16, 16], [256, 512, 128], [128, 128], True])
    """
    temp_opts = options(**locals())
    opts = temp_opts.prepare_options()

    stack_frame = inspect.stack()[1]
    pypto_impl.SetScope(opts, stack_frame.filename, stack_frame.lineno)


def get_options_tree():
    """Get the tree structure string of configuration options"""
    return pypto_impl.GetOptionsTree()


class CubeTile:
    """CubeTile"""
    def __init__(self, m, k, n, set_l1_tile=False):
        """
        CubeTile tile for matmul operation, m[0], k[0], n[0] for L0 Cache, m[1], k[1], n[1] for L1 Cache

        Parameters
        ---------
        m: list
            tile size for M dimension, must have exactly 2 elements
        k: list
            tile size for K dimension, can have 2 or 3 elements
        n: list
            tile size for N dimension, must have exactly 2 elements
        setL1Tile: bool
            whether to set L1 tile
        """

        if len(m) != 2:
            raise ValueError(f"m must have exactly 2 elements, got {len(m)}")
        if len(n) != 2:
            raise ValueError(f"n must have exactly 2 elements, got {len(n)}")
        if len(k) not in [2, 3]:
            raise ValueError(f"k must have 2 or 3 elements, got {len(k)}")

        k_padded = list(k)
        if len(k_padded) == 2:
            k_padded.append(k_padded[1])  # k[2] = k[1]

        self._impl = pypto_impl.CubeTile(list(m), k_padded, list(n), set_l1_tile)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def __repr__(self):
        return repr(self._impl)

    def __str__(self):
        return str(self._impl)