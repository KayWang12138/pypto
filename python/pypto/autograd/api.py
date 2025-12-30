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
"""Public API for PyPTO autograd."""
import inspect
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING

from .builder import TrainGraphBuilder
from .utils import is_differentiable_dtype

if TYPE_CHECKING:
    from ..tensor import Tensor


class TrainStep:
    """Training step wrapper for automatic differentiation."""

    def __init__(
        self,
        fwd: Callable,
        wrt: Tuple[Union[str, int], ...],
        loss: Union[str, int],
        grads_key: str = "grads",
        allow_missing_grads: bool = True,
        jit_options: Optional[Dict[str, Any]] = None,
    ):
        self.original_fn = fwd
        self.forward_fn = self._unwrap_jit(fwd)
        self.wrt = wrt if isinstance(wrt, tuple) else (wrt,)
        self.loss = loss
        self.grads_key = grads_key
        self.allow_missing_grads = allow_missing_grads
        self.jit_options = jit_options or {}
        self._signature = inspect.signature(self.forward_fn)
        self._param_names = list(self._signature.parameters.keys())
        self._cache: Dict[tuple, Any] = {}
        self._builder = TrainGraphBuilder(
            forward_fn=self.forward_fn,
            wrt=self.wrt,
            loss=self.loss,
        )

    def _unwrap_jit(self, fn: Callable) -> Callable:
        if hasattr(fn, "__wrapped__"):
            return fn.__wrapped__
        if hasattr(fn, "_original_func"):
            return fn._original_func
        return fn

    def _resolve_param_name(self, name_or_index: Union[str, int]) -> str:
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= len(self._param_names):
                raise ValueError(
                    f"Parameter index {name_or_index} out of range "
                    f"(function has {len(self._param_names)} parameters)"
                )
            return self._param_names[name_or_index]

        return name_or_index

    def _get_wrt_names(self) -> Tuple[str, ...]:
        return tuple(self._resolve_param_name(w) for w in self.wrt)

    def _validate_grads(
        self,
        grads: Optional[Dict[str, Optional["Tensor"]]],
        bound_args: Dict[str, Any],
    ) -> Dict[str, Optional["Tensor"]]:
        wrt_names = self._get_wrt_names()
        result: Dict[str, Optional["Tensor"]] = {}

        for name in wrt_names:
            result[name] = None

        if grads is not None:
            for name, grad_buffer in grads.items():
                if name not in wrt_names:
                    raise ValueError(
                        f"Gradient key '{name}' not in wrt parameters. "
                        f"Expected one of: {wrt_names}"
                    )
                if grad_buffer is not None:
                    if name in bound_args:
                        param_tensor = bound_args[name]
                        if hasattr(param_tensor, "dtype") and hasattr(grad_buffer, "dtype"):
                            pass
                result[name] = grad_buffer

        if not self.allow_missing_grads:
            missing = [name for name in wrt_names if result[name] is None]
            if missing:
                raise ValueError(
                    f"Missing gradient buffers for: {missing}. "
                    "Either provide all buffers or set allow_missing_grads=True"
                )

        return result

    def _allocate_missing_grads(
        self,
        grads: Dict[str, Optional["Tensor"]],
        bound_args: Dict[str, Any],
    ) -> Dict[str, "Tensor"]:
        import pypto

        result = dict(grads)
        for name, grad_buffer in result.items():
            if grad_buffer is None:
                param_tensor = bound_args.get(name)
                if param_tensor is not None and hasattr(param_tensor, "shape"):
                    shape = param_tensor.shape
                    dtype = getattr(param_tensor, "dtype", pypto.DT_FP32)
                    result[name] = pypto.tensor(shape, dtype)
        return result

    def _get_cache_key(self, bound_args: Dict[str, Any]) -> tuple:
        key_parts = []

        for name, value in bound_args.items():
            if hasattr(value, "shape"):
                shape = tuple(value.shape)
                dtype = getattr(value, "dtype", None)
                key_parts.append((name, shape, dtype))

        return tuple(sorted(key_parts))

    def __call__(self, *args, **kwargs) -> Dict[str, "Tensor"]:
        """Execute training step."""
        grads = kwargs.pop(self.grads_key, None)
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bound_args = dict(bound.arguments)

        wrt_names = self._builder.validate_wrt(bound_args)
        loss_name = self._builder.validate_loss(bound_args)

        grads = self._validate_grads(grads, bound_args)
        grads = self._allocate_missing_grads(grads, bound_args)

        cache_key = self._get_cache_key(bound_args)
        if cache_key not in self._cache:
            training_fn = self._builder.build_training_function(
                wrt_names=list(wrt_names),
                loss_name=loss_name,
                jit_options=self.jit_options,
            )
            self._cache[cache_key] = training_fn

        training_fn = self._cache[cache_key]
        fwd_positional = [bound_args[name] for name in self._param_names]
        grad_positional = [grads[name] for name in wrt_names]
        training_fn(*fwd_positional, *grad_positional)

        return grads


def train_step(
    fwd: Callable,
    wrt: Tuple[Union[str, int], ...],
    loss: Union[str, int],
    grads_key: str = "grads",
    allow_missing_grads: bool = True,
    jit_options: Optional[Dict[str, Any]] = None,
) -> TrainStep:
    """Create a training step from a forward function."""
    return TrainStep(
        fwd=fwd,
        wrt=wrt,
        loss=loss,
        grads_key=grads_key,
        allow_missing_grads=allow_missing_grads,
        jit_options=jit_options,
    )


def jit(
    wrt: Tuple[Union[str, int], ...],
    loss: Union[str, int],
    grads_key: str = "grads",
    allow_missing_grads: bool = True,
    **jit_options,
) -> Callable:
    """Decorator for JIT-compiled training functions."""
    def decorator(fn: Callable) -> TrainStep:
        return TrainStep(
            fwd=fn,
            wrt=wrt,
            loss=loss,
            grads_key=grads_key,
            allow_missing_grads=allow_missing_grads,
            jit_options=jit_options,
        )

    return decorator
