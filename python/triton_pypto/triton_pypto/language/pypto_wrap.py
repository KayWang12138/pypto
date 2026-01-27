from __future__ import annotations

import functools
import io
import math
from typing import Callable, Generic, Iterable, List, TypeVar, overload

import pypto

from ..log import get_logger
from . import dtypes

T = TypeVar("T")

logger = get_logger("triton_pypto.pypto_wrap", "TRITON_PYPTO")


class BaseWrapper(Generic[T]):

    def __init__(self, base: T) -> None:
        self.base = base

    def unwrap(self) -> T:
        return self.base


@overload
def unwrap(obj: BaseWrapper[T]) -> T:
    ...


@overload
def unwrap(obj: T) -> T:
    ...


def unwrap(obj):
    if isinstance(obj, BaseWrapper):
        return obj.unwrap()
    return obj


def managed_call(fn: Callable[..., T], *args, **kwds) -> T:
    args = tuple(unwrap(arg) for arg in args)
    kwds = {name: unwrap(arg) for name, arg in kwds.items()}
    log_str = io.StringIO()
    print(f"↳ pypto.{fn.__name__}", *args, *(f"{name}={arg}" for name, arg in kwds.items()), end="", file=log_str)
    try:
        result = fn(*args, **kwds)
        print(" ->", result, end="", file=log_str)
        return result
    finally:
        logger.debug(log_str.getvalue())


def managed_wrap(fn: T) -> T:

    @functools.wraps(fn)
    def wrapper(*args, **kwds):
        return managed_call(fn, *args, **kwds)

    return wrapper


# Vector
abs = managed_wrap(pypto.abs)
add = managed_wrap(pypto.add)
amax = managed_wrap(pypto.amax)
arange = managed_wrap(pypto.arange)
assemble = managed_wrap(pypto.assemble)
cast = managed_wrap(pypto.cast)
clip = managed_wrap(pypto.clip)
concat = managed_wrap(pypto.concat)
cos = managed_wrap(pypto.cos)
cumsum = managed_wrap(pypto.cumsum)
div = managed_wrap(pypto.div)
eq = managed_wrap(pypto.eq)
exp = managed_wrap(pypto.exp)
expand_clone = managed_wrap(pypto.expand_clone)
full = managed_wrap(pypto.full)
ge = managed_wrap(pypto.ge)
gt = managed_wrap(pypto.gt)
le = managed_wrap(pypto.le)
log = managed_wrap(pypto.log)
logical_and = managed_wrap(pypto.logical_and)
logical_not = managed_wrap(pypto.logical_not)
lt = managed_wrap(pypto.lt)
maximum = managed_wrap(pypto.maximum)
minimum = managed_wrap(pypto.minimum)
mul = managed_wrap(pypto.mul)
ne = managed_wrap(pypto.ne)
reshape = managed_wrap(pypto.reshape)
rsqrt = managed_wrap(pypto.rsqrt)
sigmoid = managed_wrap(pypto.sigmoid)
sin = managed_wrap(pypto.sin)
softmax = managed_wrap(pypto.softmax)
sqrt = managed_wrap(pypto.sqrt)
sub = managed_wrap(pypto.sub)
sum = managed_wrap(pypto.sum)
transpose = managed_wrap(pypto.transpose)
unsqueeze = managed_wrap(pypto.unsqueeze)
view = managed_wrap(pypto.view)
where = managed_wrap(pypto.where)

# Cube
matmul = managed_wrap(pypto.matmul)

# Other
loop = managed_wrap(pypto.loop)
set_cube_tile_shapes = managed_wrap(pypto.set_cube_tile_shapes)
set_vec_tile_shapes = managed_wrap(pypto.set_vec_tile_shapes)

# Platform
l0_size = 64 * 1024
ub_size = 192 * 1024


def reduce_shape(shape: List[int], max_bytes: int, dtype_size: int) -> List[int]:
    current_shape = shape.copy()
    aligned_size_bytes = 32
    reduce_step = aligned_size_bytes // dtype_size

    while dtype_size * math.prod(current_shape) > max_bytes:
        idx = max(range(len(current_shape)), key=current_shape.__getitem__)
        reduction = min(reduce_step, current_shape[idx] - 1)
        current_shape[idx] -= reduction
        if all(d == 1 for d in current_shape):
            break

    return current_shape


def auto_cube_tile(m: List[int], k: List[int], n: List[int], dtype: dtypes.AnyDataType) -> None:
    for name, dims in ('m', m), ('k', k), ('n', n):
        if any(d <= 0 for d in dims):
            raise ValueError(f"All {name} dimensions must be > 0, got {dims}")

    dtype_info = dtypes.to_info(dtype)
    dtype_size = dtype_info.bitwidth // 8

    def process_dim(dims: List[int]) -> List[int]:
        alloc_bytes = dtype_size * math.prod(dims)
        return reduce_shape(dims, l0_size, dtype_size) if alloc_bytes > l0_size else dims.copy()

    new_m = process_dim(m)
    new_k = process_dim(k)
    new_n = process_dim(n)

    set_cube_tile_shapes(new_m, new_k, new_n)


def auto_vec_tile(target_shape: Iterable[int], dtype: dtypes.AnyDataType, buf_num: int = 2) -> None:
    shape_list = list(target_shape)

    if any(d <= 0 for d in shape_list):
        raise ValueError(f"All dimensions must be > 0, got {shape_list}")

    dtype_info = dtypes.to_info(dtype)
    dtype_size = dtype_info.bitwidth // 8
    aligned_size_bytes = 32
    buffer_size = ub_size // buf_num // aligned_size_bytes * aligned_size_bytes

    min_tile_size = aligned_size_bytes // dtype_size
    shape_list[-1] = max(shape_list[-1] // min_tile_size * min_tile_size, min_tile_size)
    alloc_bytes = dtype_size * math.prod(shape_list)
    new_shape = reduce_shape(shape_list, buffer_size, dtype_size) if alloc_bytes > buffer_size else shape_list

    set_vec_tile_shapes(*map(int, new_shape))


@functools.wraps(pypto.tensor.__repr__)
def tensor_repr(self: pypto.tensor) -> str:
    return f"<pypto.tensor {hex(id(self))} {self.shape} {self.dtype.name}>"


pypto.tensor.__repr__ = tensor_repr
