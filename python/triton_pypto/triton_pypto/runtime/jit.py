from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union, overload
from typing_extensions import ParamSpec, Self, TypeAlias

import numpy as np
import pypto
import torch
import triton

from ..language.compound import TensorPointer
from ..language.operations import Context, HostTensorWrapper
from ..log import get_logger, get_progress_iter
from .mock import mock
from . import device

T = TypeVar("T")
P = ParamSpec("P")

AnyGrid: TypeAlias = Tuple[int, ...]
Grid: TypeAlias = Tuple[int, int, int]

logger = get_logger("triton_pypto.jit", "TRITON_PYPTO")


@dataclass
class JITOptions:
    dynamic: bool = False
    partition: bool = True
    unroll_factor: int = 1


class JITFunction(Generic[P, T]):

    def __init__(self, fn: Callable[P, T], options: Optional[JITOptions] = None) -> None:
        self.fn = fn
        self.grid: Callable[..., AnyGrid] = lambda *_: (1, 1, 1)
        self.options = options if options is not None else JITOptions()
        logger.info("JITFunction %s", fn.__name__)

    def __getitem__(self, grid: Union[AnyGrid, Callable[..., AnyGrid]]) -> Self:
        if callable(grid):
            grid_fn = grid
        else:
            grid_fn = lambda *_: tuple(grid)
        self.grid = grid_fn
        return self

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> None:
        mock()
        device.initialize()
        signature = inspect.signature(self.fn)
        for drop_arg in ["maxnreg", "num_ctas", "num_stages", "num_warps"]:
            if drop_arg not in signature.parameters:
                kwds.pop(drop_arg, None)
        call_args = inspect.getcallargs(self.fn, *args, **kwds)
        grid = self.normalize_grid(self.grid(call_args))
        logger.info("Grid: num_programs %r", grid)
        Context.num_programs = grid
        args = tuple(self.wrap_kernel_arg(arg) for arg in args)
        in_out_tensors = [arg.base.base for arg in args if isinstance(arg, TensorPointer)]
        kwds = {name: self.wrap_kernel_arg(arg) for name, arg in kwds.items()}
        logger.info("Args %s", args)
        logger.info("Kwds %s", kwds)
        logger.info("Options %s", self.options)
        Context.dynamic = self.options.dynamic
        pypto.set_vec_tile_shapes(8) # NOTE: Need for load store
        with pypto.function(self.fn.__name__, *in_out_tensors):
            loop_range = None
            if self.options.dynamic:
                loop_range = self.dynamic_grid_loop(grid, unroll_factor=self.options.unroll_factor)
            else:
                loop_range = self.static_grid_loop(grid)
            pypto.set_pass_options(pg_skip_partition=(not self.options.partition))
            for pid_x, pid_y, pid_z in loop_range:
                self.call_jit_fn((pid_x, pid_y, pid_z), args, kwds)
        device.run_once(*in_out_tensors)

    def call_jit_fn(self, grid: Grid, args: Tuple[Any], kwds: Dict[str, Any]) -> None:
        logger.debug("Grid: program_id %r", grid)
        Context.program_id = grid
        self.fn(*args, **kwds)

    def set_options(self, **kwds) -> None:
        for name, value in kwds.items():
            setattr(self.options, name, value)

    @staticmethod
    def normalize_grid(grid: AnyGrid) -> Grid:
        normal_grid = [grid[0], 1, 1]
        if len(grid) >= 2:
            normal_grid[1] = grid[1]
        if len(grid) >= 3:
            normal_grid[2] = grid[2]
        return tuple(normal_grid)

    @classmethod
    def wrap_kernel_arg(cls, arg: Any) -> Any:
        if isinstance(arg, triton.language.constexpr):
            return cls.wrap_kernel_arg(arg.value)
        if isinstance(arg, torch.Tensor):
            wrapper = HostTensorWrapper(tensor=pypto.from_torch(arg), storage=arg)
            return TensorPointer(wrapper)
        if isinstance(arg, np.ndarray):
            return cls.wrap_kernel_arg(torch.from_numpy(arg))
        return arg

    @staticmethod
    def delinearize_pid(pid: int, grid: Grid):
        _, grid_y, grid_z = grid
        pid_x = pid // (grid_y * grid_z)
        pid_y = (pid // grid_z) % grid_y
        pid_z = pid % grid_z
        return pid_x, pid_y, pid_z

    @classmethod
    def dynamic_grid_loop(cls, grid: Grid, unroll_factor: int = 1):
        grid_x, grid_y, grid_z = grid
        num_programs = grid_x * grid_y * grid_z
        main_trip = num_programs // unroll_factor
        for pid in pypto.loop(0, main_trip, 1, name="dynamic_grid", idx_name="dynamic_pid"):
            for i in range(unroll_factor):
                yield cls.delinearize_pid(pid * unroll_factor + i, grid)
        epilogue_trip = num_programs % unroll_factor
        if epilogue_trip == 0:
            return
        pid_offset = main_trip * unroll_factor
        for i in pypto.loop(0, epilogue_trip, 1, name="dynamic_grid_epilogue", idx_name="dynamic_pid"):
            yield cls.delinearize_pid(pid_offset + i, grid)

    @classmethod
    def static_grid_loop(cls, grid: Grid):
        for _ in pypto.loop(0, 1, 1, name="static_grid", idx_name="static_pid"):
            for pid in get_progress_iter(range(math.prod(grid)), "TRITON_PYPTO_PROGRESS_BAR"):
                yield cls.delinearize_pid(pid, grid)


@overload
def jit(fn: Callable[P, T]) -> JITFunction[P, T]:
    ...


@overload
def jit(*, dynamic: bool = False, unroll_factor: int = 1) -> Callable[[Callable[P, T]], JITFunction[P, T]]:
    ...


def jit(fn: Optional[Callable[P, T]] = None, **options):

    def decorator(fn: Callable[P, T]) -> JITFunction[P, T]:
        assert callable(fn)
        return JITFunction(fn, JITOptions(**options))

    if fn is None:
        return decorator
    return decorator(fn)
